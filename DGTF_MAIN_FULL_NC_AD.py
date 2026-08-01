"""
DGTF multimodal classification pipeline for DTI and fMRI connectivity data.

The script trains modality-specific graph neural networks, estimates node
importance from DTI, transfers the resulting node weights to the fMRI branch,
and evaluates late fusion on subjects shared by both modalities. Clinical and
demographic inputs are optional and controlled through the configuration.
"""

import os
import re
import gc
import json
import copy
import math
import itertools
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# OPTIONAL: TRANSFORMERS (PubMedBERT) FOR CLINICAL TEXT EMBEDDING

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore
    AutoModel = None      # type: ignore

# PyG
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
try:
    from IPython.display import FileLink, display  # type: ignore
    IPYTHON_DISPLAY_AVAILABLE = True
except Exception:
    FileLink = None  # type: ignore
    display = None   # type: ignore
    IPYTHON_DISPLAY_AVAILABLE = False




# DETERMINISTIC SEED


def set_deterministic_mode(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_safe_n_splits(y: np.ndarray, requested_splits: int) -> int:
    y = np.asarray(y, dtype=np.int64)
    if y.size < 2:
        raise ValueError("Need at least 2 samples for cross-validation.")
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size < 2:
        raise ValueError("Need at least 2 classes for stratified cross-validation.")
    min_count = int(counts.min())
    safe = max(2, min(int(requested_splits), min_count))
    return safe


def get_modality_n_splits(cfg, modality_name: str) -> int:
    modality = str(modality_name).strip().lower()
    if modality == "dti":
        return int(getattr(cfg, "dti_n_splits", 10))
    if modality == "fmri":
        return int(getattr(cfg, "fmri_n_splits", 10))
    raise ValueError(f"Unknown modality for n_splits: {modality_name}")


def get_fusion_n_repeats(cfg) -> int:
    return int(getattr(cfg, "fusion_n_repeats", 5))


def get_fusion_test_size(cfg) -> float:
    return float(getattr(cfg, "fusion_test_size", 0.2))


def get_effective_gnn_dropout(cfg) -> float:
    return float(getattr(cfg, "gnn_dropout", 0.0)) if bool(getattr(cfg, "use_gnn_dropout", True)) else 0.0


def get_effective_fusion_dropout(cfg) -> float:
    return float(getattr(cfg, "fusion_dropout", 0.0)) if bool(getattr(cfg, "use_fusion_dropout", True)) else 0.0


def build_early_stopping_from_config(cfg) -> "EarlyStopping":
    if bool(getattr(cfg, "use_early_stopping", True)):
        patience = int(getattr(cfg, "early_stopping_tolerance", 40))
    else:
        patience = int(10**9)
    return EarlyStopping(
        patience=patience,
        min_delta=float(getattr(cfg, "early_stopping_min_delta", 0.0)),
        restore_best_weights=True,
    )


def make_weighted_cross_entropy_from_labels(labels, num_classes: int, device: torch.device):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    weights = np.zeros(int(num_classes), dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = float(labels.size) / (float(int(num_classes)) * counts[present])
    else:
        weights[:] = 1.0
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(int(num_classes), dtype=np.float32)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))



# LOGGER


class UnifiedLogger:
    _instances: Dict[str, logging.Logger] = {}
    _initialized: bool = False

    @staticmethod
    def initialize(log_dir: str = "./logs", level: int = logging.INFO):
        if UnifiedLogger._initialized:
            return
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/multimodal_gnn_{timestamp}.log"
        logging.basicConfig(
            level=level,
            format="%(asctime)s - [%(levelname)-8s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        UnifiedLogger._initialized = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        if not UnifiedLogger._initialized:
            UnifiedLogger.initialize()
        if name not in UnifiedLogger._instances:
            UnifiedLogger._instances[name] = logging.getLogger(name)
        return UnifiedLogger._instances[name]



# CONFIGURATION


@dataclass
class DemographicInfo:
    """Stores demographic information for a single subject."""
    subject_id: str
    age: Optional[float] = None
    gender: Optional[str] = None
    apoe_a1: Optional[int] = None
    apoe_a2: Optional[int] = None
    apoe4_count: int = 0
    apoe4_carrier: bool = False
    diagnosis: Optional[str] = None
    mmse: Optional[float] = None
    global_cdr: Optional[float] = None

    def to_feature_vector(self) -> np.ndarray:
        """5D diagnosis-free clinical covariate vector."""
        age_norm = (self.age - 50) / 50 if self.age is not None else 0.5
        age_norm = np.clip(age_norm, 0, 1)

        gender_enc = 1.0 if self.gender == 'F' else 0.0
        apoe4 = float(self.apoe4_count)

        mmse_norm = self.mmse / 30.0 if self.mmse is not None else 0.5
        mmse_norm = np.clip(mmse_norm, 0, 1)

        cdr_norm = self.global_cdr / 3.0 if self.global_cdr is not None else 0.0
        cdr_norm = np.clip(cdr_norm, 0, 1)

        diagnosis_map = {'NC': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
        diagnosis_num = diagnosis_map.get(self.diagnosis, 0) / 3.0

        return np.array([age_norm, gender_enc, apoe4, mmse_norm, cdr_norm], dtype=np.float32)

    def generate_clinical_description(self) -> str:
        """Generate personalized clinical description for PubMedBERT encoding."""
        parts = []

        if self.age is not None and self.gender is not None:
            age_group = "elderly" if self.age >= 75 else "older adult"
            gender_str = "female" if self.gender == 'F' else "male"
            parts.append(f"A {int(self.age)}-year-old {gender_str} {age_group}")
        elif self.age is not None:
            parts.append(f"A {int(self.age)}-year-old patient")
        else:
            parts.append("An elderly patient")

        diagnosis_descriptions = {
            'NC': "with normal cognitive function and no evidence of dementia",
            'EMCI': "presenting with early mild cognitive impairment, showing subtle memory deficits",
            'LMCI': "diagnosed with late mild cognitive impairment, demonstrating significant memory decline",
            'AD': "diagnosed with Alzheimer's disease dementia, exhibiting progressive cognitive decline"
        }
        if self.diagnosis in diagnosis_descriptions:
            parts.append(diagnosis_descriptions[self.diagnosis])

        if self.apoe4_count > 0:
            if self.apoe4_count == 2:
                parts.append("who is homozygous for the APOE4 allele, conferring high genetic risk")
            else:
                parts.append("who is heterozygous for the APOE4 allele, indicating elevated genetic risk")
        else:
            parts.append("with no APOE4 alleles")

        if self.mmse is not None:
            if self.mmse >= 27:
                parts.append(f"MMSE score of {self.mmse:.0f} indicates intact cognitive function")
            elif self.mmse >= 24:
                parts.append(f"MMSE score of {self.mmse:.0f} suggests mild cognitive impairment")
            elif self.mmse >= 18:
                parts.append(f"MMSE score of {self.mmse:.0f} indicates moderate cognitive impairment")
            else:
                parts.append(f"MMSE score of {self.mmse:.0f} indicates severe cognitive impairment")

        if self.global_cdr is not None:
            cdr_descriptions = {
                0: "CDR of 0 indicates no dementia",
                0.5: "CDR of 0.5 indicates questionable dementia",
                1: "CDR of 1 indicates mild dementia",
                2: "CDR of 2 indicates moderate dementia",
                3: "CDR of 3 indicates severe dementia"
            }
            cdr_key = min(cdr_descriptions.keys(), key=lambda x: abs(x - self.global_cdr))
            parts.append(cdr_descriptions[cdr_key])

        return ". ".join(parts) + "."

@dataclass
class MultiModalGNNConfig:
    experiment_name: str = "DTI_fMRI_MultiModalGNN_Fusion"
    results_base_dir: str = "./multimodal_gnn_results"
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DATA PATHS
    # DTI
    dti_connectivity_dir: str = "/kaggle/input/dti-dataset/DTI_dataset/connectivity_matrices"
    dti_node_features_dir: str = "/kaggle/input/dti-dataset/DTI_dataset/node_features"

    # fMRI DFC dataset (single directory with *_dfc.npy files)
    # Example file: /kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5/002_S_4229_dfc.npy
    fmri_base_dir: str = "/kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5"

    # fMRI DFC dimensions
    fmri_time_steps: int = 30
    fmri_num_regions: int = 90

    # Labels
    diagnostic_json: str = "/kaggle/input/dti-diagnostic-groups/dti_diagnostic_groups.json"
    demographic_excel_path: str = "/kaggle/input/demographic/demographic.xlsx"


    # Demographics (optional covariates)
    
    # DemographicDataLoader (below) is used for LABELS (diagnosis) as a fallback.
    # DemographicFeatureLoader (below) is used for MODEL INPUT (age/sex/education/etc).
    use_demographics: bool = True
    demographics_in_unimodal: bool = False
    demographics_in_fusion: bool = False
    demographic_features_to_use: List[str] = field(default_factory=lambda: ["age", "sex", "education"])
    demographic_feature_dim: int = 5  # fixed (single demographic vector as in full_model_loss.py)

    
    # Clinical embedding via PubMedBERT (optional; subject-level from demographics)
    # Mirrors the "PersonalizedClinicalEncoder" logic in the reference implementation (PubMedBERT CLS embedding).
    # By default we DO NOT include diagnosis text to avoid label leakage.
    use_clinical_embedding: bool = True
    clinical_in_fusion: bool = True
    use_pubmedbert: bool = True
    pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    pubmedbert_local_files_only: bool = False
    clinical_device: str = "cuda" if torch.cuda.is_available() else "cpu"  # default; set to "cpu" if needed
    llm_embedding_dim: int = 768
    use_apoe4: bool = True
    clinical_embedding_mode: str = "clinical_no_diagnosis"  # demographics_only | clinical_no_diagnosis | clinical_with_diagnosis
    mask_diagnosis_in_embedding: bool = True

# PubMed / literature prior over nodes (optional; applied ONLY to SELECTED nodes)
    pubmed_node_prior_path: Optional[str] = None
    pubmed_alpha: float = 0.0  # scales PubMed modulation of selected-node weights

    # Dims (auto-detected after scan)
    dti_num_regions: int = 90
    fmri_num_regions: int = 90
    dti_node_feature_dim: Optional[int] = None
    fmri_node_feature_dim: Optional[int] = None

    # Model Architecture (shared)
    hidden_dim: int = 128
    gnn_num_layers: int = 3
    attention_heads: int = 8
    gnn_dropout: float = 0.15
    pooling: str = "meanmax"  # "meanmax", "mean", "max"

    # Graph construction (per modality thresholds)
    # Used to sparsify each modality graph (abs(conn) > threshold -> edge)
    # We tune DTI and fMRI separately via small grid sweeps (see sweep params below).
    dti_connectivity_threshold: float = 0.05
    fmri_connectivity_threshold: float = 0.05
    use_edge_weights: bool = True
    use_node_features: bool = True

    # HYPERPARAMETER SWEEPS
    # Tune DTI and fMRI thresholds separately (they often differ).
    # Keep these grids small; CV is expensive.
    run_hparam_sweep: bool = False
    sweep_mode: str = "sequential"  # "sequential" or "full"
    primary_sweep_metric: str = "accuracy"  # "accuracy" or "auc" (binary only)

    dti_connectivity_threshold_grid: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    fmri_connectivity_threshold_grid: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])

    node_selection_topk_grid: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 40])
    node_reweight_factor_grid: List[float] = field(default_factory=lambda: [1.25, 1.5, 2.0, 3.0])

    # Safety: cap total sweep configurations (0 => no cap).
    max_sweep_configs: int = 0

    # Training (training)
    batch_size: int = 4
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5

    # Optimizer / training stability
    # We use AdamW (cleaner decoupled weight decay).
    # ReduceLROnPlateau adapts LR based on validation loss.
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-6

    # Gradient clipping (global norm). Set 0.0 to disable.
    gradient_clip_norm: float = 1.0

    # Fusion training (also training)
    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.3
    fusion_epochs: int = 150
    fusion_learning_rate: float = 1e-3
    fusion_weight_decay: float = 5e-5

    # Regularization toggles
    use_gnn_dropout: bool = True
    use_fusion_dropout: bool = True

    # Cross-validation (separate per training stage)
    dti_n_splits: int = 10
    fmri_n_splits: int = 10
    fusion_n_repeats: int = 5
    fusion_test_size: float = 0.2

    # Early stopping controls
    use_early_stopping: bool = True
    early_stopping_tolerance: int = 7
    early_stopping_min_delta: float = 0.0

    # Outer loop iterations (optional)
    n_iterations: int = 10

    # Node selection (DTI-only)
    node_selection_method: str = "gradient"  # "gradient" (DTI saliency) or "centrality" (DTI graph centrality)
    node_selection_topk: int = 25
    node_reweight_factor: float = 2.5
    node_selection_max_batches: int = 0  # 0 => use all batches

    # Multi-class defaults (used before task filtering)
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["NC", "AD", "EMCI", "LMCI"])
    class_mapping: Dict[str, int] = field(default_factory=lambda: {"NC": 0, "AD": 1, "EMCI": 2, "LMCI": 3})

    def __post_init__(self):
        Path(self.results_base_dir).mkdir(parents=True, exist_ok=True)
        for subdir in [
            "figures",
            "subject_lists",
            "models",
            "dataset_analysis",
            "logs",
            "results",
            "checkpoints",
            "node_selection",
        ]:
            Path(f"{self.results_base_dir}/{subdir}").mkdir(parents=True, exist_ok=True)

        set_deterministic_mode(self.random_seed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)



# SUBJECT ID EXTRACTOR


class SubjectIDExtractor:
    """Extracts subject IDs from file paths."""

    def __init__(self):
        self.patterns = [
            re.compile(r'(\d{3}_S_\d{5})'),
            re.compile(r'(\d{3}_S_\d{4})'),
        ]

    def extract(self, filepath: str) -> Optional[str]:
        """Extract subject ID from filepath."""
        basename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)

        for pattern in self.patterns:
            match = pattern.search(basename)
            if match:
                return match.group(1)
            match = pattern.search(dirname)
            if match:
                return match.group(1)
        return None


    def extract(self, filepath: str) -> Optional[str]:
        text = str(filepath)
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None



# DIAGNOSTIC GROUP LOADER


class DiagnosticGroupLoader:
    def __init__(self, json_path: str, source_name: str = "DX"):
        self.logger = UnifiedLogger.get_logger(f"DiagnosticLoader_{source_name}")
        self.json_path = Path(json_path) if json_path else None
        self.source_name = source_name
        self.subject_info: Dict[str, Dict[str, Any]] = {}
        if json_path and Path(json_path).exists():
            self._load_data()
        else:
            self.logger.warning(f"Diagnostic JSON not found: {json_path}")

    def _load_data(self):
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)

            if "details" in data:
                self._parse_details(data["details"])
            elif "groups" in data:
                self._parse_details(data["groups"])
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        self._parse_category(key, value)

            self.logger.info(f"Loaded {len(self.subject_info)} subjects from {self.source_name} diagnostic JSON")
            category_counts = Counter(v["class"] for v in self.subject_info.values())
            for cat, count in sorted(category_counts.items()):
                self.logger.info(f"  {cat}: {count} subjects")

        except Exception as e:
            self.logger.error(f"Failed to load {self.json_path}: {e}")
            import traceback
            traceback.print_exc()

    def _parse_details(self, details_dict):
        for category, subjects_data in details_dict.items():
            self._parse_category(category, subjects_data)

    def _parse_category(self, category, subjects_data):
        if isinstance(subjects_data, dict):
            subjects_list = subjects_data.get("subjects", [])
            if not subjects_list:
                for k, v in subjects_data.items():
                    if isinstance(v, dict) and "subject_id" in v:
                        subjects_list.append(v)
                    elif re.match(r"\d{3}_S_\d{4,5}", str(k)):
                        subjects_list.append({"subject_id": k})
        elif isinstance(subjects_data, list):
            subjects_list = subjects_data
        else:
            return

        for entry in subjects_list:
            if isinstance(entry, dict):
                sid = entry.get("subject_id")
                if sid:
                    self.subject_info[sid] = {"class": category, "quality_score": entry.get("quality_score", 0.85)}
            elif isinstance(entry, str):
                self.subject_info[entry] = {"class": category, "quality_score": 0.85}

    def get_class(self, subject_id: str) -> Optional[str]:
        info = self.subject_info.get(subject_id, {})
        return info.get("class")

    def get_all_subject_ids(self) -> Set[str]:
        return set(self.subject_info.keys())



# DEMOGRAPHIC DATA LOADER


class DemographicDataLoader:
    """Loads and processes demographic data from Excel file (single demographics source)."""

    def __init__(self, excel_path: str):
        self.logger = UnifiedLogger.get_logger("DemographicLoader")
        self.excel_path = Path(excel_path)
        self.subjects: Dict[str, DemographicInfo] = {}
        self.statistics: Dict[str, Any] = {}

        if self.excel_path.exists():
            self._load_data()
        else:
            self.logger.warning(f"Demographic file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        pattern = re.compile(r'(\d{3}_S_\d{4,5})')
        match = pattern.search(raw_id)
        if match:
            return match.group(1)
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_string(self, value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        v = str(value).strip().upper()
        return v[0] if v else None

    def _parse_apoe_genotype(self, apoe_a1: Any, apoe_a2: Any) -> Tuple[Optional[int], Optional[int], int]:
        try:
            a1 = int(float(apoe_a1)) if not pd.isna(apoe_a1) else None
            a2 = int(float(apoe_a2)) if not pd.isna(apoe_a2) else None

            if a1 is not None and a1 not in [2, 3, 4]:
                a1 = None
            if a2 is not None and a2 not in [2, 3, 4]:
                a2 = None

            apoe4_count = 0
            if a1 == 4:
                apoe4_count += 1
            if a2 == 4:
                apoe4_count += 1

            return a1, a2, apoe4_count
        except (ValueError, TypeError):
            return None, None, 0

    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        columns_lower = {col.lower().strip(): col for col in df.columns}

        for pattern in ['subject id', 'subjectid', 'subject_id', 'id', 'ptid', 'rid', 'subject']:
            if pattern in columns_lower:
                mapping['subject_id'] = columns_lower[pattern]
                break

        for pattern in ['age']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['age'] = col
                    break

        for pattern in ['sex', 'gender']:
            if pattern in columns_lower:
                mapping['gender'] = columns_lower[pattern]
                break

        for col_lower, col in columns_lower.items():
            if 'apoe' in col_lower and ('a1' in col_lower or col_lower.endswith('1')):
                mapping['apoe_a1'] = col
            if 'apoe' in col_lower and ('a2' in col_lower or col_lower.endswith('2')):
                mapping['apoe_a2'] = col

        for pattern in ['mmse', 'mini-mental']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['mmse'] = col
                    break

        for pattern in ['cdr', 'global cd']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['cdr'] = col
                    break

        for pattern in ['research group', 'research', 'group', 'diagnosis', 'dx']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['research_group'] = col
                    break
            if 'research_group' in mapping:
                break

        self.logger.info(f"Column mapping: {mapping}")
        return mapping

    def _extract_diagnosis(self, row: pd.Series, column_mapping: Dict[str, str]) -> Optional[str]:
        research_col = column_mapping.get('research_group')
        if research_col is None:
            return None

        value = row.get(research_col, None)
        if pd.isna(value):
            return None

        value = str(value).upper().strip()
        diagnosis_map = {
            'CN': 'NC', 'NC': 'NC', 'NORMAL': 'NC', 'NL': 'NC',
            'EMCI': 'EMCI', 'EARLY MCI': 'EMCI',
            'LMCI': 'LMCI', 'LATE MCI': 'LMCI', 'MCI': 'LMCI',
            'AD': 'AD', 'DEMENTIA': 'AD', 'ALZHEIMER': 'AD'
        }
        for key, diagnosis in diagnosis_map.items():
            if key in value:
                return diagnosis
        return None

    def _load_data(self):
        self.logger.info(f"Loading demographics from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            self.logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
            col_map = self._identify_columns(df)

            valid_count = 0
            for _, row in df.iterrows():
                sid_col = col_map.get('subject_id')
                if sid_col is None:
                    continue

                subject_id = self._normalize_subject_id(row.get(sid_col, None))
                if subject_id is None:
                    continue

                age = self._safe_float(row.get(col_map.get('age', 'Age'), None))
                gender = self._safe_string(row.get(col_map.get('gender', 'Sex'), None))

                a1, a2, apoe4_count = self._parse_apoe_genotype(
                    row.get(col_map.get('apoe_a1', 'APOE A1'), None),
                    row.get(col_map.get('apoe_a2', 'APOE A2'), None),
                )

                diagnosis = self._extract_diagnosis(row, col_map)
                mmse = self._safe_float(row.get(col_map.get('mmse', 'MMSE Total Score'), None))
                cdr = self._safe_float(row.get(col_map.get('cdr', 'Global CDR'), None))

                self.subjects[subject_id] = DemographicInfo(
                    subject_id=subject_id,
                    age=age,
                    gender=gender,
                    apoe_a1=a1,
                    apoe_a2=a2,
                    apoe4_count=apoe4_count,
                    apoe4_carrier=(apoe4_count > 0),
                    diagnosis=diagnosis,
                    mmse=mmse,
                    global_cdr=cdr,
                )
                valid_count += 1

            self.logger.info(f"Loaded {valid_count} subjects with demographics")
        except Exception as e:
            self.logger.error(f"Error loading demographic data: {e}")
            import traceback
            traceback.print_exc()

    def get_subject_info(self, subject_id: str) -> Optional[DemographicInfo]:
        return self.subjects.get(subject_id)

    def get_diagnosis(self, subject_id: str) -> Optional[str]:
        info = self.subjects.get(subject_id)
        return info.diagnosis if info is not None else None

@dataclass
class DemographicInfo:
    subject_id: str
    age: Optional[float] = None
    gender: Optional[str] = None  # 'M' or 'F'
    apoe_a1: Optional[int] = None
    apoe_a2: Optional[int] = None
    apoe4_count: int = 0
    apoe4_carrier: bool = False
    diagnosis: Optional[str] = None
    mmse: Optional[float] = None
    global_cdr: Optional[float] = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert demographic info to a 5-D diagnosis-free clinical covariate vector."""
        age_norm = (self.age - 50.0) / 50.0 if self.age is not None else 0.5
        age_norm = float(np.clip(age_norm, 0.0, 1.0))
        gender_enc = 1.0 if self.gender == 'F' else 0.0
        apoe4 = float(self.apoe4_count)

        mmse_norm = (self.mmse / 30.0) if self.mmse is not None else 0.5
        mmse_norm = float(np.clip(mmse_norm, 0.0, 1.0))

        cdr_norm = (self.global_cdr / 3.0) if self.global_cdr is not None else 0.0
        cdr_norm = float(np.clip(cdr_norm, 0.0, 1.0))

        diagnosis_map = {'NC': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
        diagnosis_num = diagnosis_map.get(self.diagnosis, 0) / 3.0

        return np.asarray([age_norm, gender_enc, apoe4, mmse_norm, cdr_norm], dtype=np.float32)

    def generate_clinical_description(self, use_apoe4: bool = True, mode: str = "clinical_no_diagnosis") -> str:
        """
        mode:
          - "demographics_only": only age/gender + imaging context
          - "clinical_no_diagnosis": + APOE/MMSE/CDR but no diagnosis sentence
          - "clinical_with_diagnosis": includes diagnosis sentence (not used by default; may leak label)
        """
        parts: List[str] = []

        if self.age is not None and self.gender is not None:
            age_group = "elderly" if self.age >= 75 else "older adult"
            gender_str = "female" if self.gender == 'F' else "male"
            parts.append(f"A {int(self.age)}-year-old {gender_str} {age_group}")
        elif self.age is not None:
            parts.append(f"A {int(self.age)}-year-old patient")
        else:
            parts.append("An elderly patient")

        if mode == "clinical_with_diagnosis" and self.diagnosis is not None:
            diagnosis_desc = {
                'NC': "with normal cognitive function and no evidence of dementia",
                'EMCI': "presenting with early mild cognitive impairment",
                'LMCI': "diagnosed with late mild cognitive impairment",
                'AD': "diagnosed with Alzheimer's disease dementia",
            }
            if self.diagnosis in diagnosis_desc:
                parts.append(diagnosis_desc[self.diagnosis])

        if use_apoe4 and mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
            if self.apoe4_count == 2:
                parts.append(
                    "who is homozygous for the APOE4 allele, conferring elevated genetic risk for neurodegenerative disease"
                )
            elif self.apoe4_count == 1:
                parts.append(
                    "who is heterozygous for the APOE4 allele, indicating some genetic risk"
                )
            else:
                parts.append("with no APOE4 alleles")

        if mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
            if self.mmse is not None:
                parts.append(f"with a Mini-Mental State Examination score of {self.mmse:.0f} out of 30")
            if self.global_cdr is not None:
                parts.append(f"and a Clinical Dementia Rating of {self.global_cdr:.1f}")

        parts.append(
            "presenting for multimodal neuroimaging assessment using DTI structural connectivity and fMRI functional connectivity for cognitive evaluation"
        )
        return ". ".join(parts) + "."


class ClinicalDemographicDataLoader:
    """
    Loads richer demographic / phenotype information (age/sex/APOE/MMSE/CDR/diagnosis)
    from the same demographic Excel file.

    This loader is specifically used to build PubMedBERT clinical text embeddings.
    """
    def __init__(self, excel_path: str):
        self.logger = UnifiedLogger.get_logger("ClinicalDemographicLoader")
        self.excel_path = Path(excel_path)
        self.subjects: Dict[str, DemographicInfo] = {}
        if self.excel_path.exists():
            self._load_data()
        else:
            self.logger.warning(f"Demographic file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        for pattern in [re.compile(r"(\d{3}_S_\d{5})"), re.compile(r"(\d{3}_S_\d{4})")]:
            match = pattern.search(raw_id)
            if match:
                return match.group(1)
        return None

    def _parse_apoe_genotype(self, apoe_a1, apoe_a2) -> Tuple[Optional[int], Optional[int], int]:
        try:
            a1 = int(float(apoe_a1)) if not pd.isna(apoe_a1) else None
            a2 = int(float(apoe_a2)) if not pd.isna(apoe_a2) else None
            if a1 is not None and a1 not in [2, 3, 4]:
                a1 = None
            if a2 is not None and a2 not in [2, 3, 4]:
                a2 = None
            apoe4_count = sum(1 for a in [a1, a2] if a == 4)
            return a1, a2, apoe4_count
        except (ValueError, TypeError):
            return None, None, 0

    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        columns_lower = {col.lower().strip(): col for col in df.columns}

        for pattern in ['subject id', 'subjectid', 'subject_id', 'id', 'ptid', 'rid', 'subject']:
            if pattern in columns_lower:
                mapping['subject_id'] = columns_lower[pattern]
                break

        # age
        for col_lower, col in columns_lower.items():
            if 'age' in col_lower:
                mapping['age'] = col
                break

        # gender
        for pattern in ['sex', 'gender']:
            if pattern in columns_lower:
                mapping['gender'] = columns_lower[pattern]
                break

        # APOE alleles
        for col_lower, col in columns_lower.items():
            if 'apoe' in col_lower and ('a1' in col_lower or col_lower.endswith('1')):
                mapping['apoe_a1'] = col
            if 'apoe' in col_lower and ('a2' in col_lower or col_lower.endswith('2')):
                mapping['apoe_a2'] = col

        # MMSE
        for col_lower, col in columns_lower.items():
            if 'mmse' in col_lower:
                mapping['mmse'] = col
                break

        # CDR
        for col_lower, col in columns_lower.items():
            if 'cdr' in col_lower or 'global cd' in col_lower:
                mapping['cdr'] = col
                break

        # diagnosis
        for pattern in ['research', 'group', 'diagnosis', 'dx', 'dx group', 'dx_group', 'label', 'class']:
            for col_lower, col in columns_lower.items():
                if pattern in col_lower:
                    mapping['research_group'] = col
                    break
            if 'research_group' in mapping:
                break

        return mapping

    def _extract_diagnosis(self, row: pd.Series, col_map: Dict[str, str]) -> Optional[str]:
        col = col_map.get('research_group')
        if col is None:
            return None
        value = row.get(col, None)
        if pd.isna(value):
            return None
        value = str(value).upper().strip()
        diagnosis_map = {
            'CN': 'NC', 'NC': 'NC', 'NORMAL': 'NC', 'NL': 'NC',
            'EMCI': 'EMCI', 'EARLY MCI': 'EMCI',
            'LMCI': 'LMCI', 'LATE MCI': 'LMCI', 'MCI': 'LMCI',
            'AD': 'AD', 'DEMENTIA': 'AD', 'ALZHEIMER': 'AD'
        }
        for key, dx in diagnosis_map.items():
            if key in value:
                return dx
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_gender(self, value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        s = str(value).strip().upper()
        if s and s[0] in ('M', 'F'):
            return s[0]
        # numeric fallback
        try:
            v = float(s)
            if v == 1.0:
                return 'M'
            if v == 2.0 or v == 0.0:
                return 'F'
        except Exception:
            pass
        return None

    def _load_data(self):
        self.logger.info(f"Loading clinical demographics from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            self.logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
            col_map = self._identify_columns(df)

            sid_col = col_map.get('subject_id')
            if sid_col is None:
                self.logger.warning("No subject_id column found in demographic sheet (clinical loader).")
                return

            valid_count = 0
            for _, row in df.iterrows():
                sid = self._normalize_subject_id(row.get(sid_col, None))
                if sid is None:
                    continue

                age = self._safe_float(row.get(col_map.get('age', '__NONE__'), None))
                gender = self._safe_gender(row.get(col_map.get('gender', '__NONE__'), None))

                a1, a2, apoe4_count = self._parse_apoe_genotype(
                    row.get(col_map.get('apoe_a1', '__NONE__'), None),
                    row.get(col_map.get('apoe_a2', '__NONE__'), None),
                )

                diagnosis = self._extract_diagnosis(row, col_map)
                mmse = self._safe_float(row.get(col_map.get('mmse', '__NONE__'), None))
                cdr = self._safe_float(row.get(col_map.get('cdr', '__NONE__'), None))

                info = DemographicInfo(
                    subject_id=sid,
                    age=age,
                    gender=gender,
                    apoe_a1=a1,
                    apoe_a2=a2,
                    apoe4_count=int(apoe4_count),
                    apoe4_carrier=bool(apoe4_count > 0),
                    diagnosis=diagnosis,
                    mmse=mmse,
                    global_cdr=cdr,
                )
                self.subjects[sid] = info
                valid_count += 1

            self.logger.info(f"Loaded clinical demographics for {valid_count} subjects")

        except Exception as e:
            self.logger.error(f"Error loading clinical demographics: {e}")
            import traceback
            traceback.print_exc()

    def get_subject_info(self, subject_id: str) -> Optional[DemographicInfo]:
        return self.subjects.get(subject_id)

    def get_diagnosis(self, subject_id: str) -> Optional[str]:
        info = self.subjects.get(subject_id)
        return info.diagnosis if info is not None else None

    def get_all_subject_ids(self) -> Set[str]:
        return set(self.subjects.keys())


class PersonalizedClinicalEncoder:
    """
    PubMedBERT-based clinical embedding (CLS token) with caching.

    - Uses the demographic loader to create per-subject text prompts.
    - Encodes each prompt via PubMedBERT (frozen).
    - If transformers/model not available, falls back to a deterministic numeric embedding.
    """
    def __init__(self, config: "MultiModalGNNConfig", demographic_loader: ClinicalDemographicDataLoader):
        self.config = config
        self.demographic_loader = demographic_loader
        self.logger = UnifiedLogger.get_logger("ClinicalEncoder")

        # Device dedicated for BERT (default: CPU to avoid GPU memory spikes)
        device_str = getattr(config, "clinical_device", None) or getattr(config, "device", "cpu")
        self.device = torch.device(device_str)

        self.tokenizer = None
        self.model = None
        self.embeddings_cache: Dict[str, torch.Tensor] = {}

        if (config.use_clinical_embedding and config.use_pubmedbert and TRANSFORMERS_AVAILABLE):
            self._initialize_model()

        # Safety: prevent diagnosis leakage if requested
        if getattr(config, "mask_diagnosis_in_embedding", True) and getattr(config, "clinical_embedding_mode", "") == "clinical_with_diagnosis":
            self.logger.warning("mask_diagnosis_in_embedding=True -> forcing clinical_embedding_mode='clinical_no_diagnosis'")
            self.config.clinical_embedding_mode = "clinical_no_diagnosis"

    def _initialize_model(self):
        try:
            self.logger.info(f"Loading PubMedBERT: {self.config.pubmedbert_model}")
            local_only = bool(getattr(self.config, "pubmedbert_local_files_only", False))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.pubmedbert_model, local_files_only=local_only)
            self.model = AutoModel.from_pretrained(self.config.pubmedbert_model, local_files_only=local_only)
            self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.logger.info("PubMedBERT loaded and frozen")
        except Exception as e:
            self.logger.error(f"Failed to load PubMedBERT (will use fallback embedding): {e}")
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def encode_subject(self, subject_id: str) -> torch.Tensor:
        if not self.config.use_clinical_embedding:
            return torch.zeros(self.config.llm_embedding_dim, dtype=torch.float32)

        if subject_id in self.embeddings_cache:
            return self.embeddings_cache[subject_id]

        demo = self.demographic_loader.get_subject_info(subject_id)
        if demo is not None:
            description = demo.generate_clinical_description(
                use_apoe4=self.config.use_apoe4,
                mode=self.config.clinical_embedding_mode,
            )
        else:
            description = (
                "An elderly patient presenting for multimodal neuroimaging assessment using "
                "DTI structural connectivity and fMRI functional connectivity for cognitive evaluation."
            )

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().float()  # CLS
        else:
            emb = self._fallback_embedding(demo)

        # Ensure correct dim
        if emb.numel() != int(self.config.llm_embedding_dim):
            emb = emb.reshape(-1)
            if emb.numel() > int(self.config.llm_embedding_dim):
                emb = emb[: int(self.config.llm_embedding_dim)]
            else:
                pad = torch.zeros(int(self.config.llm_embedding_dim) - emb.numel(), dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=0)

        self.embeddings_cache[subject_id] = emb
        return emb

    def _fallback_embedding(self, demo: Optional[DemographicInfo]) -> torch.Tensor:
        """Deterministic fallback embedding when transformers/model is unavailable."""
        if demo is not None:
            features = demo.to_feature_vector()
        else:
            dim = 3 if self.config.use_apoe4 else 2
            if self.config.clinical_embedding_mode in ("clinical_no_diagnosis", "clinical_with_diagnosis"):
                dim += 2
            features = np.full(dim, 0.5, dtype=np.float32)

        emb = np.zeros(int(self.config.llm_embedding_dim), dtype=np.float32)
        chunk = int(self.config.llm_embedding_dim) // max(int(len(features)), 1)
        for i, feat in enumerate(features):
            start = i * chunk
            end = min((i + 1) * chunk, int(self.config.llm_embedding_dim))
            emb[start:end] = float(feat)
        return torch.tensor(emb, dtype=torch.float32)

    def precompute_all_embeddings(self, subject_ids: List[str]) -> int:
        if not self.config.use_clinical_embedding:
            return 0
        self.logger.info(f"Precomputing clinical embeddings for {len(subject_ids)} subjects")
        for sid in subject_ids:
            if sid not in self.embeddings_cache:
                _ = self.encode_subject(sid)
        return len(self.embeddings_cache)


# DEMOGRAPHIC FEATURE LOADER (MODEL COVARIATES)


class DemographicFeatureLoader:
    """
    Loads demographic covariates from the Excel sheet and builds a numeric feature vector
    per subject (e.g., age / sex / education).

    IMPORTANT:
    - This is different from DemographicDataLoader which is used ONLY to recover diagnosis labels.
    - These features are OPTIONAL and are intended to be concatenated at the graph (or fusion) level.
    """

    # Common patterns used in ADNI-like demographic sheets
    _DEFAULT_FEATURE_PATTERNS = {
        "age": ["age", "ptage", "age_bl", "ageat", "age at"],
        "sex": ["sex", "gender", "ptgender"],
        "education": ["educ", "education", "pteducat", "years of education"],
        "pubmed": ["pubmed", "pmid", "literature", "paper", "papers", "article", "articles", "citation", "citations"],
    }

    def __init__(self, excel_path: str, features_to_use: Optional[List[str]] = None):
        self.logger = UnifiedLogger.get_logger("DemographicFeatureLoader")
        self.excel_path = Path(excel_path)
        self.features_to_use = [f.lower().strip() for f in (features_to_use or ["age", "sex", "education"])]

        self.feature_columns: Dict[str, str] = {}  # feature_name -> column name in df
        self.subject_features: Dict[str, np.ndarray] = {}
        self.feature_dim: int = 0

        if self.excel_path.exists():
            self._load()
        else:
            self.logger.warning(f"Demographic feature file not found: {excel_path}")

    def _normalize_subject_id(self, raw_id: Any) -> Optional[str]:
        if pd.isna(raw_id):
            return None
        raw_id = str(raw_id).strip()
        for pattern in [re.compile(r"(\d{3}_S_\d{5})"), re.compile(r"(\d{3}_S_\d{4})")]:
            m = pattern.search(raw_id)
            if m:
                return m.group(1)
        return None

    def _identify_subject_id_column(self, df: pd.DataFrame) -> Optional[str]:
        columns_lower = {col.lower().strip(): col for col in df.columns}
        for pattern in ["subject id", "subjectid", "subject_id", "id", "ptid", "rid", "subject"]:
            if pattern in columns_lower:
                return columns_lower[pattern]
        # fallback: try to find any column containing "ptid"
        for col in df.columns:
            if "ptid" in col.lower():
                return col
        return None

    def _infer_feature_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        cols_l = {col.lower(): col for col in df.columns}
        out: Dict[str, str] = {}

        for feat in self.features_to_use:
            patterns = self._DEFAULT_FEATURE_PATTERNS.get(feat, [feat])
            found = None

            # exact match first
            for p in patterns:
                if p.lower() in cols_l:
                    found = cols_l[p.lower()]
                    break

            # substring match
            if found is None:
                for col in df.columns:
                    col_l = col.lower()
                    if any(p.lower() in col_l for p in patterns):
                        found = col
                        break

            if found is not None:
                out[feat] = found

        return out

    def _encode_value(self, feat: str, value: Any) -> float:
        if pd.isna(value):
            return float("nan")

        if feat == "sex":
            # Common encodings in ADNI-like sheets:
            # M/F, Male/Female, 1/2 (often 1=Male, 2=Female), or 0/1
            if isinstance(value, (int, float, np.integer, np.floating)):
                v = float(value)
                if v in (1.0,):
                    return 1.0
                if v in (2.0, 0.0):
                    return 0.0
                return float("nan")

            s = str(value).strip().upper()
            if s in {"M", "MALE"}:
                return 1.0
            if s in {"F", "FEMALE"}:
                return 0.0
            # try numeric string
            try:
                v = float(s)
                if v == 1.0:
                    return 1.0
                if v in (2.0, 0.0):
                    return 0.0
            except Exception:
                pass
            return float("nan")

        # numeric features (age, education, etc.)
        try:
            return float(value)
        except Exception:
            s = str(value).strip()
            try:
                return float(s)
            except Exception:
                return float("nan")

    def _load(self):
        self.logger.info(f"Loading demographic FEATURES from: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path)
            sid_col = self._identify_subject_id_column(df)
            if sid_col is None:
                self.logger.warning("Could not identify subject id column in demographic sheet.")
                return

            self.feature_columns = self._infer_feature_columns(df)
            if not self.feature_columns:
                self.logger.warning(
                    "No demographic feature columns matched. "
                    "Set config.demographic_features_to_use or adjust patterns."
                )
                return

            self.logger.info(f"Demographic feature columns: {self.feature_columns}")

            # Build per-subject feature vectors
            for _, row in df.iterrows():
                sid = self._normalize_subject_id(row.get(sid_col, None))
                if sid is None:
                    continue

                vec = []
                for feat in self.features_to_use:
                    col = self.feature_columns.get(feat)
                    val = row.get(col, np.nan) if col is not None else np.nan
                    vec.append(self._encode_value(feat, val))

                self.subject_features[sid] = np.asarray(vec, dtype=np.float32)

            self.feature_dim = len(self.features_to_use)
            self.logger.info(f"Loaded demographic FEATURES for {len(self.subject_features)} subjects; dim={self.feature_dim}")

        except Exception as e:
            self.logger.error(f"Error loading demographic FEATURES: {e}")
            import traceback
            traceback.print_exc()

    def get_features(self, subject_id: str) -> Optional[np.ndarray]:
        return self.subject_features.get(subject_id)

    def attach_to_subject_dict(self, subjects: Dict[str, Dict]):
        """Add a 'demographics' key to each subject entry (if available)."""
        for sid, s in subjects.items():
            s["demographics"] = self.get_features(sid)



# PUBMED / LITERATURE NODE PRIOR (OPTIONAL)


class PubMedNodePriorLoader:
    """
    Loads a per-node prior vector (length = num_regions) derived from PubMed / literature.

    Two ways to provide the prior:
      1) External file via `prior_path` :
         - .npy: 1D array
         - .csv: first numeric column used
         - .json: list of scores, or dict with key 'scores'
      2) From the demographic Excel file (same file used for labels/covariates),
         by adding a sheet or columns that contain PubMed node scores.

    Excel conventions supported (auto-detected):
      - A sheet name containing "pubmed" (case-insensitive) with:
          * (expected_len rows) x (>=1 numeric column): uses first numeric column, OR
          * 2 columns: node index + score, OR
          * 1 row with expected_len numeric columns: uses that row as vector, OR
          * columns named like "pubmed_0", "pubmed_1", ... (or ROI/Node/Region variants):
            uses column-wise mean across rows.
      - If no PubMed sheet exists, we also scan the main sheet(s) for columns named like
        "pubmed_0" .. "pubmed_{N-1}" (mean across subjects).

    This prior is applied ONLY to SELECTED nodes (DTI-selected top-k) by scaling their weights:
        w_selected := w_selected * (1 + pubmed_alpha * norm_score)

    Unselected nodes remain unchanged (weight=1).
    """

    def __init__(self, prior_path: Optional[str], demographic_excel_path: Optional[str] = None):
        self.logger = UnifiedLogger.get_logger("PubMedNodePriorLoader")
        self.prior_path = Path(prior_path) if prior_path else None
        self.demographic_excel_path = Path(demographic_excel_path) if demographic_excel_path else None
        self.last_source: Optional[str] = None

    _COLNAME_INDEX_RE = re.compile(
        r"(?:^|[^a-z0-9])(?:pubmed|pmid|literature|node|roi|region)[_\s-]*([0-9]{1,4})(?:$|[^a-z0-9])",
        re.IGNORECASE,
    )

    def load_vector(self, expected_len: int) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray of shape (expected_len,) if a prior is found; otherwise None.

        Priority:
            1) prior_path (file)
            2) demographic_excel_path (Excel)
        """
        # 1) external file
        if self.prior_path is not None:
            vec = self._load_from_path(self.prior_path, expected_len)
            if vec is not None:
                self.last_source = f"file:{self.prior_path}"
                return vec

        # 2) fallback: demographic excel
        if self.demographic_excel_path is not None and self.demographic_excel_path.exists():
            vec = self._load_from_demographic_excel(self.demographic_excel_path, expected_len)
            if vec is not None:
                return vec

        return None

    def _load_from_path(self, p: Path, expected_len: int) -> Optional[np.ndarray]:
        if not p.exists():
            self.logger.warning(f"PubMed prior file not found: {p}")
            return None
        try:
            if p.suffix.lower() == ".npy":
                vec = np.load(p)
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    self.logger.warning("PubMed prior CSV has no numeric columns.")
                    return None
                vec = df[num_cols[0]].values
            elif p.suffix.lower() == ".json":
                with open(p, "r") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    vec = np.asarray(obj)
                elif isinstance(obj, dict):
                    if "scores" in obj and isinstance(obj["scores"], list):
                        vec = np.asarray(obj["scores"])
                    else:
                        # try dict of index->score
                        try:
                            items = sorted(((int(k), float(v)) for k, v in obj.items()), key=lambda t: t[0])
                            vec = np.asarray([v for _, v in items])
                        except Exception:
                            self.logger.warning("Unsupported PubMed prior JSON format.")
                            return None
                else:
                    self.logger.warning("Unsupported PubMed prior JSON format.")
                    return None
            else:
                self.logger.warning(f"Unsupported PubMed prior format: {p.suffix}")
                return None

            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            return self._adapt_len(vec, expected_len)

        except Exception as e:
            self.logger.error(f"Failed to load PubMed prior from file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _adapt_len(self, vec: np.ndarray, expected_len: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != expected_len:
            self.logger.warning(
                f"PubMed prior length {vec.shape[0]} != expected {expected_len}. "
                "Will adapt by truncation/padding."
            )
            if vec.shape[0] > expected_len:
                vec = vec[:expected_len]
            else:
                pad = np.zeros((expected_len,), dtype=np.float32)
                pad[: vec.shape[0]] = vec
                vec = pad
        return vec.astype(np.float32)

    def _extract_index_from_colname(self, colname: str) -> Optional[int]:
        m = self._COLNAME_INDEX_RE.search(str(colname))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _try_vector_from_pubmed_columns(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Columns like pubmed_0..pubmed_{N-1} (or node/roi/region variants). Uses column-wise mean."""
        idx_to_col = {}
        for col in df.columns:
            idx = self._extract_index_from_colname(col)
            if idx is None:
                continue
            if idx < 0 or idx > expected_len:
                continue
            idx_to_col[idx] = col

        if not idx_to_col:
            return None

        # Detect 1-based indexing if all indices are in 1..N and 0 absent
        keys = sorted(idx_to_col.keys())
        one_based = (0 not in idx_to_col) and (min(keys) >= 1) and (max(keys) == expected_len)
        shift = -1 if one_based else 0

        vec = np.zeros((expected_len,), dtype=np.float32)
        filled = 0
        for idx, col in idx_to_col.items():
            j = idx + shift
            if j < 0 or j >= expected_len:
                continue
            col_vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
            mval = float(np.nanmean(col_vals)) if np.isfinite(np.nanmean(col_vals)) else 0.0
            vec[j] = mval
            filled += 1

        if filled == 0:
            return None
        return vec.astype(np.float32)

    def _try_vector_from_node_score_table(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Try to interpret df as a node-index + score table."""
        if df.shape[1] < 2:
            return None

        cols = list(df.columns)

        for idx_col in cols:
            idx_series = pd.to_numeric(df[idx_col], errors="coerce")
            if idx_series.notna().sum() < max(5, expected_len // 3):
                continue

            for score_col in cols:
                if score_col == idx_col:
                    continue

                score_series = pd.to_numeric(df[score_col], errors="coerce")
                if score_series.notna().sum() < max(5, expected_len // 3):
                    continue

                good = np.isfinite(idx_series.values) & np.isfinite(score_series.values)
                idx_good = idx_series.values[good].astype(int)
                score_good = score_series.values[good].astype(np.float32)

                if idx_good.size < max(5, expected_len // 3):
                    continue

                # detect 1-based vs 0-based
                if idx_good.min() == 1 and idx_good.max() == expected_len:
                    idx_good = idx_good - 1

                vec = np.zeros((expected_len,), dtype=np.float32)
                for i, s in zip(idx_good, score_good):
                    if 0 <= i < expected_len:
                        vec[i] = float(s)
                return vec.astype(np.float32)

        return None

    def _try_vector_from_shape(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        """Infer vector from df shape if it looks like (N x 1) or (1 x N)."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            if df.shape[0] == expected_len:
                vec = df[num_cols[0]].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)
            if df.shape[0] == 1 and len(num_cols) == expected_len:
                vec = df[num_cols].iloc[0].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)

        # Try coercion if dtypes are object
        df_num = df.apply(pd.to_numeric, errors="coerce")
        num_cols2 = df_num.columns[df_num.notna().any(axis=0)]
        if len(num_cols2) > 0:
            if df_num.shape[0] == expected_len:
                vec = df_num[num_cols2[0]].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)
            if df_num.shape[0] == 1 and len(num_cols2) == expected_len:
                vec = df_num[num_cols2].iloc[0].values.astype(np.float32)
                return self._adapt_len(vec, expected_len)

        return None

    def _parse_pubmed_prior_from_df(self, df: pd.DataFrame, expected_len: int) -> Optional[np.ndarray]:
        if df is None or df.empty:
            return None

        vec = self._try_vector_from_pubmed_columns(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        vec = self._try_vector_from_node_score_table(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        vec = self._try_vector_from_shape(df, expected_len)
        if vec is not None:
            return self._adapt_len(vec, expected_len)

        return None

    def _load_from_demographic_excel(self, excel_path: Path, expected_len: int) -> Optional[np.ndarray]:
        try:
            xls = pd.ExcelFile(excel_path)
        except Exception as e:
            self.logger.warning(f"Could not open demographic Excel for PubMed prior: {excel_path} ({e})")
            return None

        sheet_names = list(xls.sheet_names)
        pubmed_sheets = [s for s in sheet_names if "pubmed" in s.lower() or "literature" in s.lower()]

        # 1) Try PubMed sheets
        for sname in pubmed_sheets:
            try:
                df = pd.read_excel(xls, sheet_name=sname)
                vec = self._parse_pubmed_prior_from_df(df, expected_len)
                if vec is not None:
                    self.last_source = f"demographic_excel:{excel_path.name}:sheet={sname}"
                    self.logger.info(f"Loaded PubMed node prior from Excel sheet '{sname}' ({excel_path})")
                    return vec.astype(np.float32)
            except Exception:
                continue

        # 2) Fallback: scan all sheets for pubmed_* columns
        for sname in sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sname)
                vec = self._try_vector_from_pubmed_columns(df, expected_len)
                if vec is not None:
                    vec = self._adapt_len(vec, expected_len)
                    self.last_source = f"demographic_excel:{excel_path.name}:columns_in_sheet={sname}"
                    self.logger.info(f"Loaded PubMed node prior from pubmed_* columns in sheet '{sname}' ({excel_path})")
                    return vec.astype(np.float32)
            except Exception:
                continue

        self.logger.warning(
            "No PubMed node prior found in demographic Excel. "
            "Provide config.pubmed_node_prior_path or add a PubMed sheet/columns."
        )
        return None


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x2 = (x - xmin) / (xmax - xmin)
    x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    return x2.astype(np.float32)


def apply_pubmed_prior_to_selected_nodes(
    node_weights: np.ndarray,
    selected_mask: np.ndarray,
    pubmed_prior: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply PubMed prior ONLY to selected nodes (keeps unselected untouched)."""
    if node_weights is None:
        return None
    if pubmed_prior is None:
        return node_weights

    w = np.asarray(node_weights, dtype=np.float32).copy()
    mask = np.asarray(selected_mask, dtype=bool)
    prior = np.asarray(pubmed_prior, dtype=np.float32).reshape(-1)

    if prior.shape[0] != w.shape[0]:
        # safety adapt
        prior = adapt_node_weights(prior, target_num_regions=w.shape[0]).astype(np.float32)

    prior_n = _minmax_norm(prior)
    scale = 1.0 + float(alpha) * prior_n
    # Apply only on selected
    w[mask] = w[mask] * scale[mask]
    return w.astype(np.float32)


def build_demographics_matrix(
    subjects: Dict[str, Dict],
    subject_ids: List[str],
    feature_dim: int,
) -> np.ndarray:
    """Stack demographics vectors in the same order as subject_ids (NaN if missing)."""
    if feature_dim <= 0:
        return np.zeros((len(subject_ids), 0), dtype=np.float32)

    rows = []
    for sid in subject_ids:
        vec = None
        if sid in subjects:
            vec = subjects[sid].get("demographics", None)
        if vec is None:
            vec = np.full((feature_dim,), np.nan, dtype=np.float32)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != feature_dim:
            # adapt by trunc/pad with nan
            out = np.full((feature_dim,), np.nan, dtype=np.float32)
            out[: min(feature_dim, vec.shape[0])] = vec[: min(feature_dim, vec.shape[0])]
            vec = out
        rows.append(vec)
    return np.stack(rows, axis=0).astype(np.float32)


def impute_and_scale_demographics(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute NaNs using train medians then StandardScale using train statistics."""
    if X_train.size == 0:
        return X_train, X_val

    Xtr = np.asarray(X_train, dtype=np.float32).copy()
    Xva = np.asarray(X_val, dtype=np.float32).copy()

    # median imputation (train only)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    # fill NaNs
    inds_tr = np.where(~np.isfinite(Xtr))
    if inds_tr[0].size > 0:
        Xtr[inds_tr] = np.take(med, inds_tr[1])

    inds_va = np.where(~np.isfinite(Xva))
    if inds_va[0].size > 0:
        Xva[inds_va] = np.take(med, inds_va[1])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    return Xtr_s, Xva_s


# CLINICAL (PubMedBERT) EMBEDDINGS HELPERS (FUSION INPUT)


def build_clinical_embedding_matrix(
    clinical_encoder: Optional[Any],
    subject_ids: List[str],
    embed_dim: int,
) -> np.ndarray:
    """Stack PubMedBERT embeddings in the same order as subject_ids."""
    if clinical_encoder is None or embed_dim <= 0:
        return np.zeros((len(subject_ids), 0), dtype=np.float32)

    rows: List[np.ndarray] = []
    for sid in subject_ids:
        emb_t = clinical_encoder.encode_subject(sid)  # torch.Tensor [D]
        emb = emb_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if emb.shape[0] != embed_dim:
            # safety truncate/pad
            out = np.zeros((embed_dim,), dtype=np.float32)
            out[: min(embed_dim, emb.shape[0])] = emb[: min(embed_dim, emb.shape[0])]
            emb = out
        rows.append(emb)
    return np.stack(rows, axis=0).astype(np.float32)


def scale_dense_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """StandardScale using TRAIN statistics (no leakage)."""
    if X_train.size == 0:
        return X_train, X_val
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train, dtype=np.float32)).astype(np.float32)
    Xva = scaler.transform(np.asarray(X_val, dtype=np.float32)).astype(np.float32)
    return Xtr, Xva




# SMART PATH DISCOVERY (optional helper)


class SmartPathDiscovery:
    def __init__(self):
        self.logger = UnifiedLogger.get_logger("PathDiscovery")

    def find_connectivity_dir(self, base_paths: List[str]) -> Optional[str]:
        candidates = []
        for base in base_paths:
            candidates.extend([
                base,
                os.path.join(base, "connectivity_matrices"),
                os.path.join(base, "connectivity"),
                os.path.join(base, "DTI_dataset", "connectivity_matrices"),
                os.path.join(base, "fMRI_dataset", "connectivity_matrices"),
                os.path.join(base, "dataset", "connectivity_matrices"),
            ])

        for candidate in candidates:
            p = Path(candidate)
            if p.exists() and p.is_dir():
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                if subdirs:
                    csv_files = list(subdirs[0].glob("*.csv"))
                    npy_files = list(subdirs[0].glob("*.npy"))
                    if csv_files or npy_files:
                        self.logger.info(f"Found connectivity dir: {candidate} ({len(subdirs)} subject dirs)")
                        return str(p)

                csv_files = list(p.glob("*.csv"))
                npy_files = list(p.glob("*.npy"))
                if csv_files or npy_files:
                    self.logger.info(f"Found connectivity dir (flat): {candidate}")
                    return str(p)

        self.logger.warning("Could not find connectivity directory")
        return None

    def find_node_features_dir(self, base_paths: List[str]) -> Optional[str]:
        candidates = []
        for base in base_paths:
            candidates.extend([
                base,
                os.path.join(base, "node_features"),
                os.path.join(base, "DTI_dataset", "node_features"),
                os.path.join(base, "fMRI_dataset", "node_features"),
            ])

        for candidate in candidates:
            p = Path(candidate)
            if p.exists() and p.is_dir():
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                csv_files = list(p.glob("**/*.csv"))
                if subdirs or csv_files:
                    self.logger.info(f"Found node features dir: {candidate}")
                    return str(p)

        self.logger.info("Node features directory not found (optional)")
        return None



# GENERIC CONNECTIVITY DATA SCANNER (DTI / fMRI)


class ConnectivityDataScanner:
    """
    Scan a modality dataset directory, load connectivity matrices and optional node features,
    validate numeric integrity, and assign labels.
    """

    def __init__(self, modality_name: str, connectivity_dir: str, node_features_dir: str,
                 config: MultiModalGNNConfig, connectivity_threshold: float):
        self.modality_name = modality_name
        self.connectivity_dir = Path(connectivity_dir)
        self.node_features_dir = Path(node_features_dir) if node_features_dir else Path("")
        self.config = config
        self.connectivity_threshold = connectivity_threshold

        self.logger = UnifiedLogger.get_logger(f"{modality_name}.DataScanner")
        self.id_extractor = SubjectIDExtractor()

    def scan_and_validate(self, diagnostic_loader: DiagnosticGroupLoader, demographic_loader: DemographicDataLoader) -> Dict[str, Dict]:
        self.logger.info("=" * 60)
        self.logger.info(f"SCANNING AND VALIDATING {self.modality_name} DATASET")
        self.logger.info("=" * 60)
        self.logger.info(f"Connectivity dir: {self.connectivity_dir} | exists={self.connectivity_dir.exists()}")
        self.logger.info(f"Node features dir: {self.node_features_dir} | exists={self.node_features_dir.exists()}")

        # Smart discovery if missing
        if not self.connectivity_dir.exists():
            self.logger.warning(f"Primary connectivity path not found: {self.connectivity_dir}")
            self.logger.info("Attempting smart path discovery...")

            discoverer = SmartPathDiscovery()
            found_conn = discoverer.find_connectivity_dir([
                str(self.connectivity_dir),
                "/kaggle/input",
                "/kaggle/input/dti-dataset",
                "/kaggle/input/fmri-dataset",
                "/kaggle/input/fMRI-dataset",
                "/kaggle/input/dataset",
            ])
            if found_conn:
                self.connectivity_dir = Path(found_conn)
                self.logger.info(f"Auto-discovered connectivity dir: {found_conn}")
            else:
                self.logger.error("Could not find connectivity matrices directory!")
                return {}

            found_nf = discoverer.find_node_features_dir([
                str(self.node_features_dir),
                "/kaggle/input",
                "/kaggle/input/dti-dataset",
                "/kaggle/input/fmri-dataset",
            ])
            if found_nf:
                self.node_features_dir = Path(found_nf)
                self.logger.info(f"Auto-discovered node features dir: {found_nf}")

        all_discovered = self._discover_subjects()
        if not all_discovered:
            self.logger.error("No subjects discovered!")
            return {}

        self.logger.info(f"Discovered {len(all_discovered)} subjects in {self.modality_name}")

        labeled_subjects = self._assign_labels(all_discovered, diagnostic_loader, demographic_loader)
        if not labeled_subjects:
            self.logger.error("No subjects could be labeled!")
            return {}

        valid_subjects = self._validate_data(labeled_subjects)
        if not valid_subjects:
            self.logger.error("No subjects passed validation!")
            return {}

        class_counts = Counter(v["class"] for v in valid_subjects.values())
        self.logger.info(f"FINAL CLASS DISTRIBUTION ({self.modality_name}):")
        for cls_name in ["NC", "AD", "EMCI", "LMCI"]:
            self.logger.info(f"  {cls_name}: {class_counts.get(cls_name, 0)}")

        return valid_subjects

    # discovery
    def _discover_subjects(self) -> Dict[str, Dict]:
        all_discovered: Dict[str, Dict] = {}

        if not self.connectivity_dir.exists():
            return all_discovered

        # Strategy 1: subject subdirectories
        subject_dirs = [d for d in self.connectivity_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(subject_dirs)} subdirectories in connectivity dir")

        for subject_dir in subject_dirs:
            subject_id = self.id_extractor.extract(subject_dir.name)
            if subject_id is None:
                dirname = subject_dir.name.strip()
                if re.match(r"\d{3}_S_\d{4,5}", dirname):
                    subject_id = dirname
                else:
                    continue

            conn_file = self._find_connectivity_file(subject_dir, subject_id)
            if conn_file is None:
                continue

            node_features_path = self._find_node_features(subject_id)
            all_discovered[subject_id] = {
                "subject_id": subject_id,
                "connectivity_path": str(conn_file),
                "node_features_path": node_features_path,
            }

        # Strategy 2: flat files
        if not all_discovered:
            csv_files = list(self.connectivity_dir.glob("*.csv"))
            npy_files = list(self.connectivity_dir.glob("*.npy"))
            all_files = csv_files + npy_files
            self.logger.info(f"Found {len(all_files)} files in connectivity dir (flat)")
            for f in all_files:
                subject_id = self.id_extractor.extract(f.name)
                if subject_id is None:
                    continue
                node_features_path = self._find_node_features(subject_id)
                all_discovered[subject_id] = {
                    "subject_id": subject_id,
                    "connectivity_path": str(f),
                    "node_features_path": node_features_path,
                }

        # Strategy 3: recursive
        if not all_discovered:
            files = list(self.connectivity_dir.rglob("*.csv")) + list(self.connectivity_dir.rglob("*.npy"))
            self.logger.info(f"Found {len(files)} files recursively")
            for f in files:
                subject_id = self.id_extractor.extract(str(f))
                if subject_id is None:
                    continue
                if subject_id not in all_discovered:
                    node_features_path = self._find_node_features(subject_id)
                    all_discovered[subject_id] = {
                        "subject_id": subject_id,
                        "connectivity_path": str(f),
                        "node_features_path": node_features_path,
                    }

        return all_discovered

    def _find_connectivity_file(self, subject_dir: Path, subject_id: str) -> Optional[Path]:
        candidates = [
            subject_dir / f"{subject_id}_connectivity_matrix.csv",
            subject_dir / f"{subject_id}_connectivity.csv",
            subject_dir / f"{subject_id}.csv",
            subject_dir / "connectivity_matrix.csv",
            subject_dir / "connectivity.csv",
            subject_dir / f"{subject_id}_connectivity_matrix.npy",
            subject_dir / f"{subject_id}.npy",
            subject_dir / "connectivity_matrix.npy",
            subject_dir / "connectivity.npy",
        ]
        for c in candidates:
            if c.exists():
                return c

        # fallback: first csv or npy
        csvs = list(subject_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
        npys = list(subject_dir.glob("*.npy"))
        if npys:
            return npys[0]
        return None

    def _find_node_features(self, subject_id: str) -> Optional[str]:
        if not self.node_features_dir.exists():
            return None

        subject_nf_dir = self.node_features_dir / subject_id
        if subject_nf_dir.exists():
            candidates = [
                subject_nf_dir / f"{subject_id}_node_features.csv",
                subject_nf_dir / f"{subject_id}.csv",
                subject_nf_dir / "node_features.csv",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
            csv_files = list(subject_nf_dir.glob("*.csv"))
            if csv_files:
                return str(csv_files[0])

        flat_candidates = [
            self.node_features_dir / f"{subject_id}_node_features.csv",
            self.node_features_dir / f"{subject_id}.csv",
        ]
        for c in flat_candidates:
            if c.exists():
                return str(c)

        return None

    # labels
    def _assign_labels(self, all_discovered: Dict[str, Dict],
                       diagnostic_loader: DiagnosticGroupLoader,
                       demographic_loader: DemographicDataLoader) -> Dict[str, Dict]:
        labeled_subjects: Dict[str, Dict] = {}
        label_source_counts = Counter()

        for subject_id, info in all_discovered.items():
            label = None
            source = None

            diag_class = diagnostic_loader.get_class(subject_id)
            if diag_class is not None and diag_class in self.config.class_mapping:
                label = diag_class
                source = "diagnostic_json"

            if label is None:
                demo_class = demographic_loader.get_diagnosis(subject_id)
                if demo_class is not None and demo_class in self.config.class_mapping:
                    label = demo_class
                    source = "demographic"

            if label is None:
                continue

            info["class"] = label
            info["label"] = self.config.class_mapping[label]
            info["label_source"] = source
            labeled_subjects[subject_id] = info
            label_source_counts[source] += 1

        self.logger.info(f"Labeled subjects: {len(labeled_subjects)} | label sources: {dict(label_source_counts)}")
        return labeled_subjects

    # validation
    def _validate_data(self, labeled_subjects: Dict[str, Dict]) -> Dict[str, Dict]:
        valid_subjects: Dict[str, Dict] = {}
        reference_shape = None

        for subject_id, info in labeled_subjects.items():
            connectivity = self._load_and_validate_connectivity(info["connectivity_path"])
            if connectivity is None:
                continue

            if reference_shape is None:
                reference_shape = connectivity.shape
            elif connectivity.shape != reference_shape:
                connectivity = self._resize_matrix(connectivity, reference_shape[0])

            node_features = None
            if self.config.use_node_features and info.get("node_features_path"):
                node_features = self._load_and_validate_node_features(info["node_features_path"], connectivity.shape[0])

            info["connectivity"] = connectivity
            info["node_features"] = node_features
            valid_subjects[subject_id] = info

        # Infer dims (stored back to config by caller)
        return valid_subjects

    def _load_and_validate_connectivity(self, filepath: str) -> Optional[np.ndarray]:
        try:
            fp = Path(filepath)
            if fp.suffix.lower() == ".npy":
                connectivity = np.load(fp).astype(np.float32)
            else:
                # CSV
                try:
                    df = pd.read_csv(fp, index_col=0)
                except Exception:
                    df = pd.read_csv(fp)

                df_numeric = df.apply(pd.to_numeric, errors="coerce")
                df_numeric = df_numeric.dropna(axis=0, how="all")
                df_numeric = df_numeric.dropna(axis=1, how="all")
                if df_numeric.empty:
                    return None

                if df_numeric.shape[0] != df_numeric.shape[1]:
                    common = df_numeric.index.intersection(df_numeric.columns)
                    if len(common) > 0:
                        df_numeric = df_numeric.loc[common, common]

                connectivity = df_numeric.values.astype(np.float32)

            # Ensure 2D square
            if connectivity.ndim != 2:
                return None
            if connectivity.shape[0] != connectivity.shape[1]:
                min_dim = min(connectivity.shape)
                connectivity = connectivity[:min_dim, :min_dim]
                if min_dim < 10:
                    return None

            if connectivity.shape[0] < 10:
                return None

            if np.any(np.isinf(connectivity)):
                connectivity = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
            connectivity = np.nan_to_num(connectivity, nan=0.0)

            if np.allclose(connectivity, 0.0):
                return None

            # Symmetrize
            connectivity = (connectivity + connectivity.T) / 2.0
            return connectivity

        except Exception:
            return None

    def _load_and_validate_node_features(self, filepath: str, num_nodes: int) -> Optional[np.ndarray]:
        try:
            fp = Path(filepath)
            # Only CSV for node features
            try:
                df = pd.read_csv(fp, index_col=0)
            except Exception:
                df = pd.read_csv(fp)

            if df.shape[1] > 1 and df.iloc[:, 0].dtype == object:
                df = df.iloc[:, 1:]

            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return None

            node_features = df[numeric_columns].values.astype(np.float32)
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)

            # shape fix
            if node_features.shape[0] != num_nodes:
                if node_features.shape[0] > num_nodes:
                    node_features = node_features[:num_nodes, :]
                else:
                    padded = np.zeros((num_nodes, node_features.shape[1]), dtype=np.float32)
                    padded[:node_features.shape[0], :] = node_features
                    node_features = padded

            scaler = StandardScaler()
            node_features = scaler.fit_transform(node_features)
            return node_features.astype(np.float32)

        except Exception:
            return None

    def _resize_matrix(self, matrix: np.ndarray, target_size: int) -> np.ndarray:
        current_size = matrix.shape[0]
        if current_size == target_size:
            return matrix
        elif current_size > target_size:
            return matrix[:target_size, :target_size]
        else:
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            padded[:current_size, :current_size] = matrix
            return padded



# TASK FILTERING / RELABELING


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "NC_AD": {
        "display_name": "NC vs AD",
        "class_names": ["NC", "AD"],
        "source_to_target": {"NC": "NC", "AD": "AD"},
    },
    "NC_MCI": {
        "display_name": "NC vs MCI",
        "class_names": ["NC", "MCI"],
        "source_to_target": {"NC": "NC", "EMCI": "MCI", "LMCI": "MCI"},
    },
    "EMCI_LMCI": {
        "display_name": "EMCI vs LMCI",
        "class_names": ["EMCI", "LMCI"],
        "source_to_target": {"EMCI": "EMCI", "LMCI": "LMCI"},
    },
    "NC_EMCI": {
        "display_name": "NC vs EMCI",
        "class_names": ["NC", "EMCI"],
        "source_to_target": {"NC": "NC", "EMCI": "EMCI"},
    },
    "NC_EMCI_LMCI": {
        "display_name": "NC vs EMCI vs LMCI",
        "class_names": ["NC", "EMCI", "LMCI"],
        "source_to_target": {"NC": "NC", "EMCI": "EMCI", "LMCI": "LMCI"},
    },
    "MCI_AD": {
        "display_name": "MCI vs AD",
        "class_names": ["MCI", "AD"],
        "source_to_target": {"EMCI": "MCI", "LMCI": "MCI", "AD": "AD"},
    },
}

TASK_ALIASES: Dict[str, str] = {
    "NC_AD": "NC_AD",
    "NCVSAD": "NC_AD",
    "NC-AD": "NC_AD",
    "NC_VS_AD": "NC_AD",
    "NC_MCI": "NC_MCI",
    "NCVSMCI": "NC_MCI",
    "NC-MCI": "NC_MCI",
    "NC_VS_MCI": "NC_MCI",
    "EMCI_LMCI": "EMCI_LMCI",
    "EMCIVSLMCI": "EMCI_LMCI",
    "EMCI-LMCI": "EMCI_LMCI",
    "EMCI_VS_LMCI": "EMCI_LMCI",
    "NC_EMCI": "NC_EMCI",
    "NCVSEMCI": "NC_EMCI",
    "NC-EMCI": "NC_EMCI",
    "NC_VS_EMCI": "NC_EMCI",
    "NC_EMCI_LMCI": "NC_EMCI_LMCI",
    "NCVSEMCIVSLMCI": "NC_EMCI_LMCI",
    "NC-EMCI-LMCI": "NC_EMCI_LMCI",
    "NC_VS_EMCI_VS_LMCI": "NC_EMCI_LMCI",
    "MCI_AD": "MCI_AD",
    "MCIVSAD": "MCI_AD",
    "MCI-AD": "MCI_AD",
    "MCI_VS_AD": "MCI_AD",
}


def normalize_task_name(task: str) -> str:
    task_norm = str(task).upper().strip()
    task_norm = task_norm.replace("/", "_")
    task_norm = task_norm.replace(" ", "")
    task_norm = task_norm.replace("VS", "_VS_")
    task_norm = task_norm.replace("-", "_")
    while "__" in task_norm:
        task_norm = task_norm.replace("__", "_")
    if task_norm in TASK_SPECS:
        return task_norm
    if task_norm in TASK_ALIASES:
        return TASK_ALIASES[task_norm]
    compact = task_norm.replace("_", "")
    if compact in TASK_ALIASES:
        return TASK_ALIASES[compact]
    raise ValueError(
        f"Unknown task '{task}'. Use one of: {', '.join(TASK_SPECS.keys())}"
    )


def get_task_display_name(task: str) -> str:
    task_key = normalize_task_name(task)
    return TASK_SPECS[task_key]["display_name"]



def build_task_subjects(valid_subjects: Dict[str, Dict], task: str) -> Tuple[Dict[str, Dict], List[str], Dict[str, int]]:
    """
    Supported task options:
      - "NC_AD"      : NC vs AD
      - "NC_MCI"     : NC vs (EMCI + LMCI) mapped to MCI
      - "EMCI_LMCI"  : EMCI vs LMCI
      - "MCI_AD"     : (EMCI + LMCI) mapped to MCI vs AD

    The function keeps the original subject payload and only rewrites the
    task-specific class / label when required.
    """
    task_key = normalize_task_name(task)
    spec = TASK_SPECS[task_key]
    class_names = list(spec["class_names"])
    mapping = {name: idx for idx, name in enumerate(class_names)}
    source_to_target = dict(spec["source_to_target"])

    out: Dict[str, Dict] = {}
    for sid, s in valid_subjects.items():
        src_class = s.get("class")
        if src_class not in source_to_target:
            continue
        target_class = source_to_target[src_class]
        s2 = copy.deepcopy(s)
        s2["original_class"] = src_class
        s2["class"] = target_class
        s2["label"] = mapping[target_class]
        out[sid] = s2

    return out, class_names, mapping



# DATASET + COLLATE (generic for connectivity graphs)


class FMRIDFCScanner:
    """
    Scan fMRI DFC dataset where each subject is stored as a single NumPy file:
        <SUBJECT_ID>_dfc.npy
    with shape (T, N, N).

    We convert the DFC sequence into a *static* connectivity matrix for this pipeline by
    averaging over time: connectivity = mean(dfc[:T], axis=0).

    This matches the rest of the codebase which expects one connectivity matrix per subject.
    """

    def __init__(self, base_dir: str, config: MultiModalGNNConfig):
        self.base_dir = Path(base_dir)
        self.config = config
        self.logger = UnifiedLogger.get_logger("FMRIDFCScanner")
        self._id_patterns = [
            re.compile(r"(\d{3}_S_\d{5})"),
            re.compile(r"(\d{3}_S_\d{4})"),
        ]

    def _extract_subject_id(self, filename: str) -> Optional[str]:
        for pat in self._id_patterns:
            m = pat.search(filename)
            if m:
                return m.group(1)
        return None

    def scan_and_validate(self, diagnostic_loader: DiagnosticGroupLoader, demographic_loader: Optional[DemographicDataLoader] = None) -> Dict[str, Dict]:
        if not self.base_dir.exists():
            self.logger.error(f"fMRI base dir not found: {self.base_dir}")
            return {}

        dfc_files = sorted(self.base_dir.glob("*_dfc.npy"))
        if len(dfc_files) == 0:
            # Some datasets might use uppercase or other suffix conventions
            dfc_files = sorted(self.base_dir.glob("*dfc*.npy"))

        self.logger.info(f"[fMRI] Found {len(dfc_files)} DFC files in {self.base_dir}")

        subjects: Dict[str, Dict] = {}
        for fpath in dfc_files:
            subject_id = self._extract_subject_id(fpath.name)
            if subject_id is None:
                continue

            # label (match DTI logic: use diagnostic JSON first, fallback to demographics)
            cls = None
            diag_class = diagnostic_loader.get_class(subject_id)
            if diag_class is not None and diag_class in self.config.class_mapping:
                cls = diag_class
            elif demographic_loader is not None:
                demo_class = demographic_loader.get_diagnosis(subject_id)
                if demo_class is not None and demo_class in self.config.class_mapping:
                    cls = demo_class

            if cls is None:
                continue

            try:
                dfc = np.load(str(fpath))
            except Exception as e:
                self.logger.warning(f"[fMRI] Could not load {fpath}: {e}")
                continue

            if not (isinstance(dfc, np.ndarray) and dfc.ndim == 3 and dfc.shape[1] == dfc.shape[2]):
                self.logger.warning(f"[fMRI] Unexpected shape for {fpath.name}: {getattr(dfc, 'shape', None)}")
                continue

            # Time steps: truncate/pad to config.fmri_time_steps
            T_target = int(getattr(self.config, "fmri_time_steps", 30))
            if dfc.shape[0] > T_target:
                dfc = dfc[:T_target]
            elif dfc.shape[0] < T_target:
                # pad by repeating last frame
                pad = np.repeat(dfc[-1:,:,:], T_target - dfc.shape[0], axis=0)
                dfc = np.concatenate([dfc, pad], axis=0)

            # Regions: take first 90 to match DTI (and/or pad if needed)
            R_target = int(getattr(self.config, "fmri_num_regions", 90))
            if dfc.shape[1] > R_target:
                dfc = dfc[:, :R_target, :R_target]
            elif dfc.shape[1] < R_target:
                tmp = np.zeros((dfc.shape[0], R_target, R_target), dtype=np.float32)
                r = dfc.shape[1]
                tmp[:, :r, :r] = dfc
                dfc = tmp

            dfc = dfc.astype(np.float32, copy=False)
            dfc = np.nan_to_num(dfc, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert dynamic FC -> static connectivity for this model
            connectivity = dfc.mean(axis=0)
            # squash outliers slightly
            connectivity = np.tanh(connectivity)

            # Ensure symmetric (some DFC estimators yield small asymmetries)
            connectivity = (connectivity + connectivity.T) / 2.0
            np.fill_diagonal(connectivity, 0.0)

            demo_vec = None
            if demographic_loader is not None:
                demo_info = demographic_loader.get_subject_info(subject_id)
                if demo_info is not None:
                    demo_vec = demo_info.to_feature_vector()
                else:
                    demo_vec = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)

            subjects[subject_id] = {
                "subject_id": subject_id,
                "connectivity": connectivity,
                "node_features": None,  # dataset has no separate node features
                "class": cls,
                "label": int(self.config.class_mapping[cls]),
                "demographics": demo_vec,
                "source_path": str(fpath),
                "dfc_shape": tuple(dfc.shape),
            }

        self.logger.info(f"[fMRI] Valid subjects with labels: {len(subjects)}")
        return subjects




class ConnectivityGraphDataset(Dataset):
    def __init__(
        self,
        subjects: Dict[str, Dict],
        use_node_features: bool = True,
        use_demographics: bool = False,
        demographic_dim: int = 0,
    ):
        self.valid_data = []
        self.labels = []
        self.use_node_features = bool(use_node_features)
        self.use_demographics = bool(use_demographics and demographic_dim > 0)
        self.demographic_dim = int(demographic_dim)

        for subject_id, subject_data in subjects.items():
            demo_vec = None
            if self.use_demographics:
                raw = subject_data.get("demographics", None)
                if raw is None:
                    demo_vec = np.full((self.demographic_dim,), np.nan, dtype=np.float32)
                else:
                    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
                    if raw.shape[0] != self.demographic_dim:
                        tmp = np.full((self.demographic_dim,), np.nan, dtype=np.float32)
                        tmp[: min(self.demographic_dim, raw.shape[0])] = raw[: min(self.demographic_dim, raw.shape[0])]
                        demo_vec = tmp
                    else:
                        demo_vec = raw.astype(np.float32)

            self.valid_data.append({
                "connectivity": subject_data["connectivity"],
                "node_features": subject_data.get("node_features", None),
                "demographics": demo_vec,
                "subject_id": subject_id,
                "class": subject_data["class"],
                "label": int(subject_data["label"]),
            })
            self.labels.append(int(subject_data["label"]))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        data = self.valid_data[idx]
        connectivity = torch.FloatTensor(data["connectivity"])
        label = torch.LongTensor([data["label"]])[0]

        node_features = None
        if self.use_node_features and data["node_features"] is not None:
            node_features = torch.FloatTensor(data["node_features"])

        demo = None
        if self.use_demographics and data["demographics"] is not None:
            demo = torch.FloatTensor(data["demographics"])

        return connectivity, node_features, demo, label, data["subject_id"]


def collate_connectivity_batch(batch):
    """Collate: (connectivity, node_features, demographics, label, subject_id)."""
    if not batch:
        return torch.tensor([]), None, None, torch.tensor([]), []

    connectivities, node_features_list, demo_list, labels, subject_ids = [], [], [], [], []

    has_node_features = (batch[0][1] is not None)
    has_demo = (batch[0][2] is not None)

    for connectivity, node_features, demo, label, subject_id in batch:
        connectivities.append(connectivity)

        if has_node_features and node_features is not None:
            node_features_list.append(node_features)

        if has_demo and demo is not None:
            demo_list.append(demo)

        labels.append(label)
        subject_ids.append(subject_id)

    connectivities = torch.stack(connectivities)
    labels = torch.stack(labels)

    node_features_batch = None
    if has_node_features and len(node_features_list) > 0:
        node_features_batch = torch.stack(node_features_list)

    demo_batch = None
    if has_demo and len(demo_list) > 0:
        demo_batch = torch.stack(demo_list)

    return connectivities, node_features_batch, demo_batch, labels, subject_ids




# EARLY STOPPING (validation-loss based)


class EarlyStopping:
    """
    Early stopping on **best validation loss** .

    Stops training when `val_loss` has not improved by at least `min_delta`
    for `patience` consecutive epochs.

    Optionally caches & restores the best model weights (by val_loss).
    """
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)

        self.counter = 0
        self.early_stop = False

        self.best_val_loss = float("inf")
        self.best_state_dict = None  # type: ignore
        self.best_epoch = -1

    def __call__(self, train_loss: float, validation_loss: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None):
        # We keep the (train_loss, validation_loss) signature for minimal changes,
        # but only val_loss is used.
        val = float(validation_loss)

        improved = val < (self.best_val_loss - self.min_delta)

        if improved:
            self.best_val_loss = val
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)

            if self.restore_best_weights and (model is not None):
                # deepcopy is important; state_dict tensors are references.
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: nn.Module):
        """Restore cached best weights into `model` (if available)."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return model



# MODEL (generic connectivity GNN, supports node reweighting)


class ConnectivityGNN(nn.Module):
    def __init__(
        self,
        num_regions: int,
        node_feature_dim: Optional[int],
        num_classes: int,
        connectivity_threshold: float,
        config: MultiModalGNNConfig,
        demographic_dim: int = 0,
    ):
        super().__init__()
        self.config = config

        self.num_regions = int(num_regions)
        self.node_feature_dim = node_feature_dim
        self.num_classes = int(num_classes)
        self.connectivity_threshold = float(connectivity_threshold)

        # Demographics (optional)
        self.demographic_dim = int(demographic_dim) if demographic_dim else 0
        self.use_demographics = self.demographic_dim > 0

        self.use_node_features = config.use_node_features and (node_feature_dim is not None)
        input_dim = node_feature_dim if self.use_node_features else self.num_regions

        hidden_dim = config.hidden_dim
        heads = config.attention_heads

        # If we use edge weights, we treat them as 1D edge attributes (edge_dim=1).
        edge_dim = 1 if config.use_edge_weights else None

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=get_effective_gnn_dropout(config),
                edge_dim=edge_dim,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(config.gnn_num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=get_effective_gnn_dropout(config),
                    edge_dim=edge_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if config.pooling == "meanmax":
            pooled_dim = hidden_dim * 2
        elif config.pooling in {"mean", "max"}:
            pooled_dim = hidden_dim
        else:
            raise ValueError(f"Unknown pooling: {config.pooling}")

        classifier_in_dim = pooled_dim + (self.demographic_dim if self.use_demographics else 0)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(get_effective_gnn_dropout(config)),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(get_effective_gnn_dropout(config)),

            nn.Linear(hidden_dim // 2, self.num_classes)
        )

    def create_graph_from_connectivity(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor],
        batch_size: int,
        node_weights: Optional[torch.Tensor] = None,
    ):
        """
        IMPORTANT: we do NOT remove unselected nodes.
        We ONLY REWEIGHT nodes by scaling their node features.
        """
        num_regions = connectivity_matrices.shape[1]
        device = connectivity_matrices.device

        # node features
        if self.use_node_features and node_features_batch is not None:
            x = node_features_batch.reshape(-1, node_features_batch.shape[-1])
        else:
            # fallback: each node gets its connectivity profile (row)
            x = connectivity_matrices.reshape(batch_size, num_regions, -1)
            x = x.reshape(-1, num_regions)

        # apply node reweighting (selected nodes get higher weights)
        if node_weights is not None:
            # node_weights should be shape [num_regions]
            if node_weights.numel() != num_regions:
                raise ValueError(f"node_weights has {node_weights.numel()} elements, but num_regions={num_regions}")
            w = node_weights.to(device).reshape(1, num_regions).repeat(batch_size, 1).reshape(-1, 1)
            x = x * w

        # edges
        edge_index_list = []
        edge_attr_list = []
        batch_vector = torch.repeat_interleave(torch.arange(batch_size, device=device), num_regions)

        thr = float(self.connectivity_threshold)
        for b in range(batch_size):
            offset = b * num_regions
            conn = connectivity_matrices[b]

            rows, cols = torch.where(torch.abs(conn) > thr)
            if rows.numel() == 0:
                continue
            # remove self-loops
            keep = rows != cols
            rows = rows[keep]
            cols = cols[keep]
            if rows.numel() == 0:
                continue

            ei = torch.stack([rows + offset, cols + offset], dim=0)
            edge_index_list.append(ei)

            if self.config.use_edge_weights:
                ew = conn[rows, cols].reshape(-1, 1)
                edge_attr_list.append(ew)

        # safety edges if graph ended up empty (e.g., too high threshold)
        if len(edge_index_list) == 0:
            # add a small chain per graph
            chain_i = torch.arange(0, min(num_regions - 1, 5), device=device)
            chain_j = chain_i + 1
            for b in range(batch_size):
                offset = b * num_regions
                ei = torch.stack(
                    [torch.cat([chain_i + offset, chain_j + offset]),
                     torch.cat([chain_j + offset, chain_i + offset])],
                    dim=0,
                )
                edge_index_list.append(ei)
                if self.config.use_edge_weights:
                    edge_attr_list.append(torch.full((ei.size(1), 1), 0.1, device=device))

        edge_index = torch.cat(edge_index_list, dim=1).long().contiguous()
        batch = batch_vector.long().contiguous()

        edge_attr = None
        if self.config.use_edge_weights:
            edge_attr = torch.cat(edge_attr_list, dim=0).float().contiguous() if len(edge_attr_list) > 0 else None
        return x, edge_index, edge_attr, batch

    def encode(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor] = None,
        node_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns graph-level embedding (pooled representation) before classifier."""
        batch_size = connectivity_matrices.shape[0]
        x, edge_index, edge_attr, batch = self.create_graph_from_connectivity(
            connectivity_matrices, node_features_batch, batch_size, node_weights=node_weights
        )

        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=get_effective_gnn_dropout(self.config), training=self.training)

        if self.config.pooling == "meanmax":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_pooled = torch.cat([x_mean, x_max], dim=1)
        elif self.config.pooling == "mean":
            x_pooled = global_mean_pool(x, batch)
        elif self.config.pooling == "max":
            x_pooled = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        return x_pooled

    def forward(
        self,
        connectivity_matrices: torch.Tensor,
        node_features_batch: Optional[torch.Tensor] = None,
        demographics_batch: Optional[torch.Tensor] = None,
        node_weights: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ):
        emb_graph = self.encode(connectivity_matrices, node_features_batch, node_weights=node_weights)

        if self.use_demographics:
            if demographics_batch is None:
                demo = torch.zeros((emb_graph.size(0), self.demographic_dim), device=emb_graph.device, dtype=emb_graph.dtype)
            else:
                demo = demographics_batch.to(emb_graph.device)
                if demo.dim() == 1:
                    demo = demo.unsqueeze(0)
                if demo.size(0) != emb_graph.size(0):
                    raise ValueError("demographics_batch batch size does not match connectivity batch size")
                if demo.size(1) != self.demographic_dim:
                    raise ValueError(f"demographics_batch has dim={demo.size(1)} but expected {self.demographic_dim}")
            emb_for_cls = torch.cat([emb_graph, demo], dim=1)
        else:
            emb_for_cls = emb_graph

        logits = self.classifier(emb_for_cls)
        if return_embedding:
            return logits, emb_graph
        return logits


# TRAINING (AGM style) - UPDATED to accept node_weights


def train_gnn(model, optimizer, criterion, dataloader, train_dataset, device, node_weights=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for connectivity, node_features, demo, labels, _ in dataloader:
        connectivity = connectivity.to(device)
        if node_features is not None:
            node_features = node_features.to(device)
        if demo is not None:
            demo = demo.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        # Gradient clipping for stability
        clip_norm = float(getattr(model.config, "gradient_clip_norm", 0.0))
        if clip_norm and clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        running_loss += loss.item() * connectivity.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_acc, epoch_loss


def eval_gnn(model, criterion, dataloader, valid_dataset, device, node_weights=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs_pos = []  # for binary ROC/AUC

    with torch.no_grad():
        for connectivity, node_features, demo, labels, _ in dataloader:
            connectivity = connectivity.to(device)
            if node_features is not None:
                node_features = node_features.to(device)
            if demo is not None:
                demo = demo.to(device)
            labels = labels.to(device)

            outputs = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * connectivity.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            if outputs.size(1) == 2:
                all_probs_pos.append(probs[:, 1].detach().cpu())

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)
    print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])

    n_classes = int(getattr(model, "num_classes", 2))
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(n_classes))) if labels_np.size else np.zeros((n_classes, n_classes), dtype=int)

    fpr = np.array([])
    tpr = np.array([])
    if all_probs_pos and labels_np.size:
        probs_pos_np = torch.cat(all_probs_pos).numpy()
        try:
            fpr, tpr, _ = roc_curve(labels_np, probs_pos_np)
        except Exception:
            pass

    probs_pos_np = torch.cat(all_probs_pos).numpy() if (all_probs_pos and labels_np.size) else np.array([])
    return epoch_acc, epoch_loss, cm, fpr, tpr, labels_np, preds_np, probs_pos_np


def train_and_evaluate_gnn(
    model,
    optimizer,
    criterion,
    trainloader,
    valloader,
    train_dataset,
    valid_dataset,
    early_stopping: EarlyStopping,
    device,
    num_epochs: int = 150,
    node_weights=None,
    scheduler=None,
):
    """Train a unimodal GNN with:
      - AdamW optimizer (passed in)
      - ReduceLROnPlateau scheduler (optional; passed in)
      - Early stopping on best validation loss (see EarlyStopping)
      - Optional node reweighting via `node_weights`
      - Optional gradient clipping (configured via model.config.gradient_clip_norm)
    """
    epoch_acc_train = []
    epoch_acc_val = []
    epoch_loss_train = []
    epoch_loss_val = []

    for epoch in range(int(num_epochs)):
        print(f"Epoch {epoch}/{int(num_epochs) - 1}")
        print("-" * 10)

        acc_tr, loss_tr = train_gnn(
            model, optimizer, criterion, trainloader, train_dataset, device, node_weights=node_weights
        )
        epoch_acc_train.append(float(acc_tr.detach().cpu()))
        epoch_loss_train.append(float(loss_tr))

        acc_va, loss_va, _, _, _, _, _, _ = eval_gnn(
            model, criterion, valloader, valid_dataset, device, node_weights=node_weights
        )
        epoch_acc_val.append(float(acc_va.detach().cpu()))
        epoch_loss_val.append(float(loss_va))

        # LR scheduler (val-loss driven)
        if scheduler is not None:
            try:
                scheduler.step(float(loss_va))
            except TypeError:
                # Some schedulers use step() without metric
                scheduler.step()

        # Early stopping (best val loss)
        early_stopping(loss_tr, loss_va, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print(f"Early stop at epoch: {epoch} | best_val_loss={early_stopping.best_val_loss:.6f}")
            break

    # Restore best weights (by val loss)
    model = early_stopping.restore(model)

    # Final eval at best weights
    acc_final, loss_final, cm, fpr, tpr, _, _, _ = eval_gnn(
        model, criterion, valloader, valid_dataset, device, node_weights=node_weights
    )

    return (
        model,
        epoch_acc_train,
        epoch_acc_val,
        cm,
        fpr,
        tpr,
        epoch_loss_train,
        epoch_loss_val,
    )



# NODE SELECTION (DTI ONLY) - GRADIENT SALIENCY ON INPUT NODE FEATURES


class DTINodeSelector:
    """
    Computes node importance scores using gradient saliency w.r.t. the
    input node features x (before first GAT layer), averaged over batches.

    Selection is computed ONLY on DTI .
    """
    def __init__(self, topk: int = 20, max_batches: int = 0):
        self.topk = int(topk)
        self.max_batches = int(max_batches)

    @torch.no_grad()
    def _init_scores(self, num_regions: int):
        scores = torch.zeros(num_regions, dtype=torch.float32)
        counts = torch.zeros(num_regions, dtype=torch.float32)
        return scores, counts

    def compute_node_scores(
        self,
        model: ConnectivityGNN,
        dataloader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        num_regions = model.num_regions

        scores, counts = self._init_scores(num_regions)
        scores = scores.to(device)
        counts = counts.to(device)

        # We need grads => no torch.no_grad()
        batch_seen = 0
        for connectivity, node_features, demo, labels, _ in dataloader:
            batch_seen += 1
            if self.max_batches > 0 and batch_seen > self.max_batches:
                break

            connectivity = connectivity.to(device)
            if node_features is not None:
                node_features = node_features.to(device)
            labels = labels.to(device)

            batch_size = connectivity.shape[0]

            # Build graph and get input node feature matrix x
            x, edge_index, edge_attr, batch_vec = model.create_graph_from_connectivity(
                connectivity, node_features, batch_size, node_weights=None
            )

            # Enable grads on x only
            x = x.detach().clone().requires_grad_(True)

            # Forward manually through GNN to keep grad on x
            h = x
            for i, (gat, bn) in enumerate(zip(model.gat_layers, model.batch_norms)):
                h = gat(h, edge_index, edge_attr=edge_attr)
                h = bn(h)
                h = F.elu(h)
                if i < len(model.gat_layers) - 1:
                    h = F.dropout(h, p=get_effective_gnn_dropout(model.config), training=False)

            if model.config.pooling == "meanmax":
                h_mean = global_mean_pool(h, batch_vec)
                h_max = global_max_pool(h, batch_vec)
                h_pooled = torch.cat([h_mean, h_max], dim=1)
            elif model.config.pooling == "mean":
                h_pooled = global_mean_pool(h, batch_vec)
            elif model.config.pooling == "max":
                h_pooled = global_max_pool(h, batch_vec)
            else:
                raise ValueError(f"Unknown pooling: {model.config.pooling}")

            if getattr(model, "use_demographics", False) and getattr(model, "demographic_dim", 0) > 0:
                demo0 = torch.zeros((h_pooled.size(0), int(model.demographic_dim)), device=h_pooled.device, dtype=h_pooled.dtype)
                h_for_cls = torch.cat([h_pooled, demo0], dim=1)
            else:
                h_for_cls = h_pooled

            logits = model.classifier(h_for_cls)
            loss = F.cross_entropy(logits, labels)
            # Backprop to x
            loss.backward()

            grad = x.grad  # [batch_size*num_regions, feat_dim]
            if grad is None:
                continue

            grad = grad.detach().abs()

            # Aggregate per node within each graph
            grad = grad.view(batch_size, num_regions, -1)  # [B, R, F]
            node_scores_batch = grad.mean(dim=2)  # [B, R]
            scores += node_scores_batch.sum(dim=0)
            counts += float(batch_size)

            # cleanup grads
            model.zero_grad(set_to_none=True)

        scores = scores / torch.clamp(counts, min=1.0)
        return scores.detach().cpu().numpy()

    def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if scores.ndim != 1:
            raise ValueError("scores must be 1D (num_regions,)")

        topk = min(self.topk, scores.shape[0])
        idx = np.argsort(scores)[::-1][:topk]
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[idx] = True
        return idx, mask

    def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
        w = np.ones(mask.shape[0], dtype=np.float32)
        w[mask] = float(reweight_factor)
        return w


# NODE SELECTION (DTI ONLY) - CENTRALITY (attached-style)


class CentralityNodeSelector:
    """Centrality-based node importance computed from DTI connectivity only.

    This mirrors the *idea* in the attached full_model_loss.py:
      - Build a graph from the DTI connectivity matrix (thresholded)
      - Compute per-node scores (degree + PageRank + betweenness)
      - Average scores across TRAIN subjects
      - Select TOP-K nodes

    It is deterministic (given threshold) and faster than gradient saliency.
    """

    def __init__(self, topk: int = 20, threshold: float = 0.1, max_subjects: int = 0):
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.max_subjects = int(max_subjects)  # 0 => use all

    def compute_node_scores(self, subjects_train: Dict[str, Dict], num_regions: int) -> np.ndarray:
        try:
            import networkx as nx
        except Exception as e:
            raise RuntimeError("Centrality selection requires networkx. Install: pip install networkx") from e

        ids = list(subjects_train.keys())
        if self.max_subjects > 0:
            ids = ids[: self.max_subjects]
        if len(ids) == 0:
            return np.zeros((num_regions,), dtype=np.float32)

        scores_sum = np.zeros((num_regions,), dtype=np.float64)
        used = 0

        for sid in ids:
            conn = subjects_train[sid].get("connectivity", None)
            if conn is None:
                continue
            conn = np.asarray(conn, dtype=np.float32)
            if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
                continue
            n = conn.shape[0]
            if n != num_regions:
                # adapt by trunc/pad
                tmp = np.zeros((num_regions, num_regions), dtype=np.float32)
                m = min(num_regions, n)
                tmp[:m, :m] = conn[:m, :m]
                conn = tmp

            # Build graph with abs(conn) > threshold
            G = nx.Graph()
            G.add_nodes_from(range(num_regions))
            mat = np.abs(conn)
            # Avoid self loops
            np.fill_diagonal(mat, 0.0)
            rows, cols = np.where(mat > self.threshold)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if i >= j:
                    continue
                w = float(mat[i, j])
                if w <= 0:
                    continue
                G.add_edge(i, j, weight=w)

            # Safety: if graph has no edges, skip
            if G.number_of_edges() == 0:
                continue

            # Centralities
            deg = np.array([d for _, d in G.degree(weight='weight')], dtype=np.float64)
            try:
                pr = nx.pagerank(G, weight='weight')
                pr = np.array([pr[i] for i in range(num_regions)], dtype=np.float64)
            except Exception:
                pr = np.zeros((num_regions,), dtype=np.float64)

            try:
                btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
                btw = np.array([btw[i] for i in range(num_regions)], dtype=np.float64)
            except Exception:
                btw = np.zeros((num_regions,), dtype=np.float64)

            # Normalize each component to [0,1] then average
            def _norm01(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                lo = np.min(x)
                hi = np.max(x)
                if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                    return np.zeros_like(x)
                return (x - lo) / (hi - lo)

            s = (_norm01(deg) + _norm01(pr) + _norm01(btw)) / 3.0
            scores_sum += s
            used += 1

        if used == 0:
            return np.zeros((num_regions,), dtype=np.float32)

        scores = (scores_sum / float(used)).astype(np.float32)
        return scores

    def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        topk = min(self.topk, scores.shape[0])
        idx = np.argsort(scores)[::-1][:topk]
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[idx] = True
        return idx, mask

    def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
        w = np.ones(mask.shape[0], dtype=np.float32)
        w[mask] = float(reweight_factor)
        return w



def adapt_node_weights(weights: np.ndarray, target_num_regions: int) -> np.ndarray:
    """
    If fMRI has different num regions than DTI, adapt weights by truncation/padding.
    Unseen nodes get weight=1.0.
    """
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.shape[0] == target_num_regions:
        return w
    if w.shape[0] > target_num_regions:
        return w[:target_num_regions].copy()
    out = np.ones(target_num_regions, dtype=np.float32)
    out[:w.shape[0]] = w
    return out



# EMBEDDING EXTRACTION (for fusion)


@torch.no_grad()
def extract_embeddings(
    model: ConnectivityGNN,
    subjects: Dict[str, Dict],
    config: MultiModalGNNConfig,
    device: torch.device,
    num_regions: int,
    node_feature_dim: Optional[int],
    connectivity_threshold: float,
    node_weights: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      embeddings: [N, D]
      labels: [N]
      subject_ids: list length N
    """
    model.eval()

    dataset = ConnectivityGraphDataset(
        subjects,
        use_node_features=(config.use_node_features and node_feature_dim is not None),
        use_demographics=(config.use_demographics and config.demographics_in_unimodal and config.demographic_feature_dim > 0),
        demographic_dim=int(config.demographic_feature_dim),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

    embs = []
    labels_out = []
    ids_out = []

    for connectivity, node_features, demo, labels, subject_ids in loader:
        connectivity = connectivity.to(device)
        if node_features is not None:
            node_features = node_features.to(device)
        if demo is not None:
            demo = demo.to(device)

        logits, emb = model(connectivity, node_features, demographics_batch=demo, node_weights=node_weights, return_embedding=True)
        embs.append(emb.detach().cpu().numpy())
        labels_out.append(labels.detach().cpu().numpy())
        ids_out.extend(subject_ids)

    if embs:
        embs = np.concatenate(embs, axis=0)
        labels_out = np.concatenate(labels_out, axis=0)
    else:
        embs = np.zeros((0, config.hidden_dim * 2), dtype=np.float32)
        labels_out = np.zeros((0,), dtype=np.int64)

    return embs, labels_out, ids_out



# FUSION CLASSIFIER (MLP on concatenated embeddings)



class FusionMLP(nn.Module):
    """Fusion MLP with optional PubMedBERT clinical embedding projection.

    Input ordering MUST be:
      [DTI_embedding | fMRI_embedding | demographics(optional) | clinical_embedding(optional)]

    Where:
      - DTI_embedding is extracted from the weighted DTI model (DTI-only node selection).
      - fMRI_embedding is extracted from the weighted fMRI model (imported DTI weights).
      - demographics is the numeric covariates vector (optional).
      - clinical_embedding is PubMedBERT CLS embedding (optional).
    """

    def __init__(
        self,
        dti_dim: int,
        fmri_dim: int,
        demo_dim: int,
        clinical_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.dti_dim = int(dti_dim)
        self.fmri_dim = int(fmri_dim)
        self.demo_dim = int(demo_dim)
        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)

        self.use_demo = self.demo_dim > 0
        self.use_clinical = self.clinical_dim > 0

        fusion_in = self.dti_dim + self.fmri_dim + (self.demo_dim if self.use_demo else 0)

        # If clinical embedding is enabled, we project it down before fusion (attached-code style).
        if self.use_clinical:
            self.clinical_proj = nn.Sequential(
                nn.Linear(self.clinical_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            fusion_in += hidden_dim
        else:
            self.clinical_proj = None

        self.net = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0

        dti = x[:, start:start + self.dti_dim]
        start += self.dti_dim

        fmri = x[:, start:start + self.fmri_dim]
        start += self.fmri_dim

        parts = [dti, fmri]

        if self.use_demo:
            demo = x[:, start:start + self.demo_dim]
            start += self.demo_dim
            parts.append(demo)

        if self.use_clinical:
            clinical = x[:, start:start + self.clinical_dim]
            start += self.clinical_dim
            clinical_h = self.clinical_proj(clinical)  # type: ignore
            parts.append(clinical_h)

        x_fused = torch.cat(parts, dim=1)

        return self.net(x_fused)

class FusionEmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: List[str]):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.subject_ids = subject_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_ids[idx]


def collate_fusion(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    X, y, ids = zip(*batch)
    return torch.stack(X), torch.stack(y), list(ids)


def train_fusion_epoch(model, optimizer, criterion, loader, dataset_len, device, grad_clip_norm: float = 0.0):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        _, preds = torch.max(out, 1)
        loss.backward()
        # Gradient clipping for stability
        if grad_clip_norm and float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)

    epoch_loss = running_loss / max(dataset_len, 1)
    epoch_acc = running_corrects.double() / max(dataset_len, 1)
    print(f"Fusion Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_acc, epoch_loss


@torch.no_grad()
def eval_fusion_epoch(model, criterion, loader, dataset_len, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs_pos = []

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        loss = criterion(out, y)

        probs = torch.softmax(out, dim=1)
        _, preds = torch.max(out, 1)

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)

        all_labels.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        if out.size(1) == 2:
            all_probs_pos.append(probs[:, 1].detach().cpu())

    epoch_loss = running_loss / max(dataset_len, 1)
    epoch_acc = running_corrects.double() / max(dataset_len, 1)
    print(f"Fusion Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    n_classes = int(getattr(model, "num_classes", 2))
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(n_classes))) if labels_np.size else np.zeros((n_classes, n_classes), dtype=int)

    fpr = np.array([])
    tpr = np.array([])
    if all_probs_pos and labels_np.size:
        probs_pos_np = torch.cat(all_probs_pos).numpy()
        try:
            fpr, tpr, _ = roc_curve(labels_np, probs_pos_np)
        except Exception:
            pass

    probs_pos_np = torch.cat(all_probs_pos).numpy() if (all_probs_pos and labels_np.size) else np.array([])
    return epoch_acc, epoch_loss, cm, fpr, tpr, labels_np, preds_np, probs_pos_np


def train_and_evaluate_fusion(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    train_len,
    val_len,
    early_stopping: EarlyStopping,
    device,
    num_epochs: int,
    scheduler=None,
    grad_clip_norm: float = 0.0,
):
    """Train fusion MLP with:
      - AdamW optimizer (passed in)
      - ReduceLROnPlateau scheduler (optional; passed in)
      - Early stopping on best validation loss
      - Optional gradient clipping
    """
    epoch_acc_train = []
    epoch_acc_val = []
    epoch_loss_train = []
    epoch_loss_val = []
    for epoch in range(int(num_epochs)):
        print(f"Fusion Epoch {epoch}/{int(num_epochs) - 1}")
        print("-" * 10)

        acc_tr, loss_tr = train_fusion_epoch(
            model, optimizer, criterion, train_loader, train_len, device, grad_clip_norm=grad_clip_norm
        )
        epoch_acc_train.append(float(acc_tr.detach().cpu() if hasattr(acc_tr, "detach") else acc_tr))
        epoch_loss_train.append(float(loss_tr))
        acc_va, loss_va, _, _, _, _, _, _ = eval_fusion_epoch(model, criterion, val_loader, val_len, device)
        epoch_acc_val.append(float(acc_va.detach().cpu() if hasattr(acc_va, "detach") else acc_va))
        epoch_loss_val.append(float(loss_va))

        # LR scheduler (val-loss driven)
        if scheduler is not None:
            try:
                scheduler.step(float(loss_va))
            except TypeError:
                scheduler.step()

        # Early stopping (best val loss)
        early_stopping(loss_tr, loss_va, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print(f"Fusion early stop at epoch: {epoch} | best_val_loss={early_stopping.best_val_loss:.6f}")
            break

    # Restore best weights (by val loss)
    model = early_stopping.restore(model)

    acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos = eval_fusion_epoch(model, criterion, val_loader, val_len, device)
    return model, acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos, {"train_acc": epoch_acc_train, "val_acc": epoch_acc_val, "train_loss": epoch_loss_train, "val_loss": epoch_loss_val}




# MULTI-MODAL CROSS-VALIDATION (splits on COMMON subjects only)


def run_multimodal_cv_for_task(
    dti_subjects_task: Dict[str, Dict],
    fmri_subjects_task: Dict[str, Dict],
    cfg: MultiModalGNNConfig,
    task_name: str,
    pubmed_prior_dti: Optional[np.ndarray] = None,
    clinical_encoder: Optional[Any] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Cross-validate the *fusion* classifier using only common subjects.
    For each fold:
      1) Train base DTI model (no node weighting) on DTI train subjects.
      2) Compute DTI-only node selection (top-k) on DTI train set.
      3) Train weighted DTI model (reweight selected nodes; keep all nodes).
      4) Train weighted fMRI model using the *same* node weights.
      5) Extract embeddings for common train/val subjects.
      6) Train fusion MLP on train embeddings, eval on val embeddings.
    """
    tag_str = f".{run_tag}" if run_tag else ""
    logger = UnifiedLogger.get_logger(f"MultimodalCV.{task_name}{tag_str}")

    # Intersection of IDs for fusion
    common_ids = sorted(list(set(dti_subjects_task.keys()) & set(fmri_subjects_task.keys())))
    if len(common_ids) < 4:
        raise RuntimeError(f"Too few common subjects between DTI and fMRI for task={task_name}: n={len(common_ids)}")

    y_common = np.array([int(dti_subjects_task[sid]["label"]) for sid in common_ids], dtype=np.int64)

    dti_ids_all = sorted(list(dti_subjects_task.keys()))
    y_dti_all = np.array([int(dti_subjects_task[sid]["label"]) for sid in dti_ids_all], dtype=np.int64)
    fmri_ids_all = sorted(list(fmri_subjects_task.keys()))
    y_fmri_all = np.array([int(fmri_subjects_task[sid]["label"]) for sid in fmri_ids_all], dtype=np.int64)

    effective_fusion_repeats = int(get_fusion_n_repeats(cfg))
    effective_dti_splits = get_safe_n_splits(y_dti_all, get_modality_n_splits(cfg, "dti"))
    effective_fmri_splits = get_safe_n_splits(y_fmri_all, get_modality_n_splits(cfg, "fmri"))

    test_size = float(get_fusion_test_size(cfg))
    min_class_count_common = int(np.bincount(y_common).min())
    if min_class_count_common < 2:
        raise RuntimeError(f"Need at least 2 samples per class in common subjects for StratifiedShuffleSplit: min_class_count={min_class_count_common}")
    max_test_fraction = (min_class_count_common - 1) / float(len(common_ids))
    test_size = min(test_size, max_test_fraction)
    test_size = max(test_size, 1.0 / float(len(common_ids)))

    fusion_sss = StratifiedShuffleSplit(
        n_splits=effective_fusion_repeats,
        test_size=test_size,
        random_state=0,
    )
    dti_skf = StratifiedKFold(n_splits=effective_dti_splits, shuffle=True, random_state=0)
    fmri_skf = StratifiedKFold(n_splits=effective_fmri_splits, shuffle=True, random_state=0)

    fusion_split_plan = list(fusion_sss.split(np.arange(len(common_ids)), y_common))
    dti_split_plan = list(dti_skf.split(np.arange(len(dti_ids_all)), y_dti_all))
    fmri_split_plan = list(fmri_skf.split(np.arange(len(fmri_ids_all)), y_fmri_all))

    fold_results = []
    tsne_chunks = []
    tsne_labels = []
    tsne_ids: List[str] = []
    conn_store: Dict[str, Any] = {}
    cms_fusion = []
    aucs_fusion = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    device = torch.device(cfg.device)

    for fold_idx, (fusion_tr_idx, fusion_va_idx) in enumerate(fusion_split_plan, start=1):
        dti_plan_idx = (fold_idx - 1) % len(dti_split_plan)
        fmri_plan_idx = (fold_idx - 1) % len(fmri_split_plan)
        dti_tr_idx, dti_va_idx = dti_split_plan[dti_plan_idx]
        fmri_tr_idx, fmri_va_idx = fmri_split_plan[fmri_plan_idx]

        print("\n" + "=" * 90)
        print(
            f"[TASK {task_name}] REPEAT {fold_idx}/{effective_fusion_repeats} "
            f"| DTI fold {dti_plan_idx + 1}/{effective_dti_splits} "
            f"| fMRI fold {fmri_plan_idx + 1}/{effective_fmri_splits}"
        )
        print("=" * 90)

        train_common_ids = [common_ids[i] for i in fusion_tr_idx]
        val_common_ids = [common_ids[i] for i in fusion_va_idx]

        dti_train_ids = [dti_ids_all[i] for i in dti_tr_idx]
        dti_val_ids = [dti_ids_all[i] for i in dti_va_idx]
        fmri_train_ids = [fmri_ids_all[i] for i in fmri_tr_idx]
        fmri_val_ids = [fmri_ids_all[i] for i in fmri_va_idx]

        # Build unimodal train/val sets from EACH modality's own split plan.
        # Fusion keeps its own independent split on common subjects.
        dti_train_subjects = {sid: dti_subjects_task[sid] for sid in dti_train_ids}
        dti_val_subjects = {sid: dti_subjects_task[sid] for sid in dti_val_ids}

        fmri_train_subjects = {sid: fmri_subjects_task[sid] for sid in fmri_train_ids}
        fmri_val_subjects = {sid: fmri_subjects_task[sid] for sid in fmri_val_ids}

        
        # Stage 1: Train base DTI model (no weights) for node selection
        
        dti_train_ds = ConnectivityGraphDataset(
            dti_train_subjects,
            use_node_features=(cfg.use_node_features and cfg.dti_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )
        dti_val_ds = ConnectivityGraphDataset(
            dti_val_subjects,
            use_node_features=(cfg.use_node_features and cfg.dti_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )

        dti_train_loader = DataLoader(dti_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        dti_val_loader = DataLoader(dti_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

        base_dti_model = ConnectivityGNN(
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)

        criterion = make_weighted_cross_entropy_from_labels([dti_train_subjects[sid]["label"] for sid in dti_train_subjects], cfg.num_classes, device)
        optimizer = optim.AdamW(base_dti_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        base_dti_model, _, _, _, _, _, _, _ = train_and_evaluate_gnn(
            base_dti_model, optimizer, criterion,
            dti_train_loader, dti_val_loader,
            dti_train_ds, dti_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=None,
            scheduler=scheduler,
        )

        
        # Stage 2: DTI-only node selection (TOP-K) using base DTI model
        
        selector_method = str(getattr(cfg, 'node_selection_method', 'gradient')).strip().lower()
        if selector_method == 'centrality':
            selector = CentralityNodeSelector(
                topk=cfg.node_selection_topk,
                threshold=float(cfg.dti_connectivity_threshold),
                max_subjects=int(getattr(cfg, 'centrality_max_subjects', 0) or 0),
            )
            scores = selector.compute_node_scores(dti_train_subjects, num_regions=int(cfg.dti_num_regions))
        else:
            selector = DTINodeSelector(topk=cfg.node_selection_topk, max_batches=cfg.node_selection_max_batches)
            scores = selector.compute_node_scores(base_dti_model, dti_train_loader, device=device)
        top_idx, mask = selector.select_topk(scores)
        node_weights_np = selector.make_node_weights(mask, reweight_factor=cfg.node_reweight_factor)

        # Optional: modulate SELECTED node weights using an external PubMed prior
        if pubmed_prior_dti is not None and float(getattr(cfg, "pubmed_alpha", 0.0)) > 0.0:
            node_weights_np = apply_pubmed_prior_to_selected_nodes(
                node_weights_np,
                selected_mask=mask,
                pubmed_prior=pubmed_prior_dti,
                alpha=float(cfg.pubmed_alpha),
            )

        # Save node selection per fold
        fold_ns_path = Path(cfg.results_base_dir) / "node_selection" / f"node_selection_{task_name}{('_' + run_tag) if run_tag else ''}_fold{fold_idx}.json"
        with open(fold_ns_path, "w") as f:
            json.dump({
                "task": task_name,
                "repeat": fold_idx,
                "run_tag": run_tag,
                "dti_connectivity_threshold": float(cfg.dti_connectivity_threshold),
                "fmri_connectivity_threshold": float(cfg.fmri_connectivity_threshold),
                "topk": int(cfg.node_selection_topk),
                "reweight_factor": float(cfg.node_reweight_factor),
                "selected_indices": top_idx.tolist(),
                "scores": scores.tolist(),
            }, f, indent=2)
        print(f"[NodeSelection] Saved: {fold_ns_path}")

        # Prepare node weights tensors for DTI and fMRI
        node_weights_dti = torch.FloatTensor(node_weights_np).to(device)

        node_weights_fmri_np = adapt_node_weights(node_weights_np, target_num_regions=cfg.fmri_num_regions)
        node_weights_fmri = torch.FloatTensor(node_weights_fmri_np).to(device)

        
        # Stage 3: Train weighted DTI model (from scratch) using node_weights_dti
        
        weighted_dti_model = ConnectivityGNN(
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)
        optimizer = optim.AdamW(weighted_dti_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        weighted_dti_model, dti_epoch_acc_train, dti_epoch_acc_val, _, _, _, dti_epoch_loss_train, dti_epoch_loss_val = train_and_evaluate_gnn(
            weighted_dti_model, optimizer, criterion,
            dti_train_loader, dti_val_loader,
            dti_train_ds, dti_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=node_weights_dti,
            scheduler=scheduler,
        )

        
        # Stage 4: Train weighted fMRI model using imported DTI node weights
        
        fmri_train_ds = ConnectivityGraphDataset(
            fmri_train_subjects,
            use_node_features=(cfg.use_node_features and cfg.fmri_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )
        fmri_val_ds = ConnectivityGraphDataset(
            fmri_val_subjects,
            use_node_features=(cfg.use_node_features and cfg.fmri_node_feature_dim is not None),
            use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0),
            demographic_dim=int(cfg.demographic_feature_dim),
        )

        fmri_train_loader = DataLoader(fmri_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        fmri_val_loader = DataLoader(fmri_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)

        weighted_fmri_model = ConnectivityGNN(
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            num_classes=cfg.num_classes,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            config=cfg,
            demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0),
        ).to(device)
        optimizer = optim.AdamW(weighted_fmri_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        early_stopping = build_early_stopping_from_config(cfg)

        weighted_fmri_model, fmri_epoch_acc_train, fmri_epoch_acc_val, _, _, _, fmri_epoch_loss_train, fmri_epoch_loss_val = train_and_evaluate_gnn(
            weighted_fmri_model, optimizer, criterion,
            fmri_train_loader, fmri_val_loader,
            fmri_train_ds, fmri_val_ds,
            early_stopping, device,
            num_epochs=cfg.epochs,
            node_weights=node_weights_fmri,
            scheduler=scheduler,
        )

        
        # Stage 5: Extract embeddings for common train/val subjects (using weighted models)
        
        dti_common_train = {sid: dti_subjects_task[sid] for sid in train_common_ids}
        dti_common_val = {sid: dti_subjects_task[sid] for sid in val_common_ids}

        fmri_common_train = {sid: fmri_subjects_task[sid] for sid in train_common_ids}
        fmri_common_val = {sid: fmri_subjects_task[sid] for sid in val_common_ids}

        dti_emb_tr, y_tr, ids_tr = extract_embeddings(
            weighted_dti_model, dti_common_train, cfg, device,
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            node_weights=node_weights_dti,
        )
        fmri_emb_tr, y_tr2, ids_tr2 = extract_embeddings(
            weighted_fmri_model, fmri_common_train, cfg, device,
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            node_weights=node_weights_fmri,
        )

        dti_emb_va, y_va, ids_va = extract_embeddings(
            weighted_dti_model, dti_common_val, cfg, device,
            num_regions=cfg.dti_num_regions,
            node_feature_dim=cfg.dti_node_feature_dim,
            connectivity_threshold=cfg.dti_connectivity_threshold,
            node_weights=node_weights_dti,
        )
        fmri_emb_va, y_va2, ids_va2 = extract_embeddings(
            weighted_fmri_model, fmri_common_val, cfg, device,
            num_regions=cfg.fmri_num_regions,
            node_feature_dim=cfg.fmri_node_feature_dim,
            connectivity_threshold=cfg.fmri_connectivity_threshold,
            node_weights=node_weights_fmri,
        )

        # Safety checks: IDs should match ordering
        assert ids_tr == ids_tr2, "DTI and fMRI train IDs are not aligned."
        assert ids_va == ids_va2, "DTI and fMRI val IDs are not aligned."
        assert np.array_equal(y_tr, y_tr2), "DTI and fMRI train labels differ."
        assert np.array_equal(y_va, y_va2), "DTI and fMRI val labels differ."

        X_tr = np.concatenate([dti_emb_tr, fmri_emb_tr], axis=1)
        X_va = np.concatenate([dti_emb_va, fmri_emb_va], axis=1)

        # Track fusion input segment dims (ordering must match FusionMLP slicing)
        dti_dim = int(dti_emb_tr.shape[1])
        fmri_dim = int(fmri_emb_tr.shape[1])
        demo_dim = 0
        clinical_dim = 0

        # Optional: add demographics covariates at fusion stage
        if cfg.use_demographics and cfg.demographics_in_fusion and int(cfg.demographic_feature_dim) > 0:
            Xdem_tr_raw = build_demographics_matrix(dti_subjects_task, ids_tr, int(cfg.demographic_feature_dim))
            Xdem_va_raw = build_demographics_matrix(dti_subjects_task, ids_va, int(cfg.demographic_feature_dim))
            Xdem_tr, Xdem_va = impute_and_scale_demographics(Xdem_tr_raw, Xdem_va_raw)
            demo_dim = int(Xdem_tr.shape[1])
            X_tr = np.concatenate([X_tr, Xdem_tr], axis=1)
            X_va = np.concatenate([X_va, Xdem_va], axis=1)

        # Optional: add PubMedBERT clinical embeddings at fusion stage (attached-code style)
        if (
            getattr(cfg, "use_clinical_embedding", False)
            and getattr(cfg, "clinical_in_fusion", True)
            and clinical_encoder is not None
            and int(getattr(cfg, "llm_embedding_dim", 0)) > 0
        ):
            Xclin_tr_raw = build_clinical_embedding_matrix(clinical_encoder, ids_tr, int(cfg.llm_embedding_dim))
            Xclin_va_raw = build_clinical_embedding_matrix(clinical_encoder, ids_va, int(cfg.llm_embedding_dim))
            Xclin_tr, Xclin_va = scale_dense_embeddings(Xclin_tr_raw, Xclin_va_raw)
            clinical_dim = int(Xclin_tr.shape[1])
            X_tr = np.concatenate([X_tr, Xclin_tr], axis=1)
            X_va = np.concatenate([X_va, Xclin_va], axis=1)


        # Stage 6: Train fusion classifier on common train subjects, eval on common val
        
        fusion_train_ds = FusionEmbeddingDataset(X_tr, y_tr, ids_tr)
        fusion_val_ds = FusionEmbeddingDataset(X_va, y_va, ids_va)

        fusion_train_loader = DataLoader(fusion_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fusion)
        fusion_val_loader = DataLoader(fusion_val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fusion)

        fusion_model = FusionMLP(
            dti_dim=dti_dim,
            fmri_dim=fmri_dim,
            demo_dim=demo_dim,
            clinical_dim=clinical_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            dropout=get_effective_fusion_dropout(cfg),
            num_classes=cfg.num_classes,
        ).to(device)

        fusion_criterion = make_weighted_cross_entropy_from_labels(y_tr, cfg.num_classes, device)
        fusion_optimizer = optim.AdamW(fusion_model.parameters(), lr=cfg.fusion_learning_rate, weight_decay=cfg.fusion_weight_decay)
        fusion_scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            fusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                fusion_optimizer,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
                min_lr=float(cfg.lr_scheduler_min_lr),
            )
        fusion_early_stopping = build_early_stopping_from_config(cfg)

        fusion_model, acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos, fusion_history = train_and_evaluate_fusion(
            fusion_model,
            fusion_optimizer,
            fusion_criterion,
            fusion_train_loader,
            fusion_val_loader,
            train_len=len(fusion_train_ds),
            val_len=len(fusion_val_ds),
            early_stopping=fusion_early_stopping,
            device=device,
            num_epochs=cfg.fusion_epochs,
            scheduler=fusion_scheduler,
            grad_clip_norm=float(cfg.gradient_clip_norm),
        )

        if X_va is not None and len(X_va) > 0:
            tsne_chunks.append(np.asarray(X_va, dtype=np.float32))
            tsne_labels.append(np.asarray(y_va, dtype=np.int64))
            tsne_ids.extend(list(ids_va))
        _update_connectivity_aggregate(conn_store, dti_subjects_task, val_common_ids, "dti", list(getattr(cfg, "class_names", [])))
        _update_connectivity_aggregate(conn_store, fmri_subjects_task, val_common_ids, "fmri", list(getattr(cfg, "class_names", [])))

        # Compute fold AUC (binary only)
        fold_auc = float("nan")
        if cfg.num_classes == 2 and len(fpr) > 0 and len(tpr) > 0:
            try:
                fold_auc = auc(fpr, tpr)
                aucs_fusion.append(fold_auc)
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
            except Exception:
                pass


        # compute fold metrics for this fold (fusion)
        metrics_fold = compute_binary_metrics(y_true, y_pred, y_prob_pos, cm)
        prec = metrics_fold["precision"] if metrics_fold["precision"] is not None else float("nan")
        rec = metrics_fold["recall"] if metrics_fold["recall"] is not None else float("nan")
        f1v = metrics_fold["f1"] if metrics_fold["f1"] is not None else float("nan")
        sensitivity = metrics_fold["sensitivity"] if metrics_fold["sensitivity"] is not None else float("nan")
        specificity = metrics_fold["specificity"] if metrics_fold["specificity"] is not None else float("nan")
        auc_value = metrics_fold["auc"]
        print(f"[Fusion Fold {fold_idx}] Acc={float(acc_final):.4f} | AUC={auc_value if auc_value is not None else fold_auc} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1v:.4f} | Spec={specificity:.4f}")

        fold_results.append({
            "repeat": fold_idx,
            "n_common_train": len(train_common_ids),
            "n_common_val": len(val_common_ids),
            "fusion_acc": float(acc_final),
            "fusion_loss": float(loss_final),
            "fusion_auc": auc_value,
            "fusion_precision": prec,
            "fusion_recall": rec,
            "fusion_sensitivity": sensitivity,
            "fusion_specificity": specificity,
            "fusion_f1": f1v,
            "fusion_confusion_matrix": cm.tolist(),
            "fusion_fpr": fpr.tolist() if hasattr(fpr, "tolist") else list(fpr),
            "fusion_tpr": tpr.tolist() if hasattr(tpr, "tolist") else list(tpr),
            "selected_nodes": top_idx.tolist(),
            "dti_train_history": {"acc": [float(v) for v in dti_epoch_acc_train], "loss": [float(v) for v in dti_epoch_loss_train]},
            "dti_val_history": {"acc": [float(v) for v in dti_epoch_acc_val], "loss": [float(v) for v in dti_epoch_loss_val]},
            "fmri_train_history": {"acc": [float(v) for v in fmri_epoch_acc_train], "loss": [float(v) for v in fmri_epoch_loss_train]},
            "fmri_val_history": {"acc": [float(v) for v in fmri_epoch_acc_val], "loss": [float(v) for v in fmri_epoch_loss_val]},
            "fusion_train_history": {"acc": [float(v) for v in fusion_history.get("train_acc", [])], "loss": [float(v) for v in fusion_history.get("train_loss", [])]},
            "fusion_val_history": {"acc": [float(v) for v in fusion_history.get("val_acc", [])], "loss": [float(v) for v in fusion_history.get("val_loss", [])]}
        })

        cms_fusion.append(cm)

        # cleanup
        del base_dti_model, weighted_dti_model, weighted_fmri_model, fusion_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # mean AUC curve
    if len(aucs_fusion) > 0:
        mean_tpr /= len(aucs_fusion)
        mean_tpr[-1] = 1.0
        mean_auc = float(auc(mean_fpr, mean_tpr))
    else:
        mean_auc = float("nan")

    # aggregate acc
    accs = [fr["fusion_acc"] for fr in fold_results]
    mean_acc = float(np.mean(accs)) if accs else float("nan")
    std_acc = float(np.std(accs)) if accs else float("nan")

    result = {
        "task": task_name,
        "run_tag": run_tag,
        "hyperparams": {
            "dti_connectivity_threshold": float(cfg.dti_connectivity_threshold),
            "fmri_connectivity_threshold": float(cfg.fmri_connectivity_threshold),
            "node_selection_topk": int(cfg.node_selection_topk),
            "node_reweight_factor": float(cfg.node_reweight_factor),
            "learning_rate": 1e-3,
            "weight_decay": float(cfg.weight_decay),
            "fusion_learning_rate": 1e-3,
            "fusion_weight_decay": float(cfg.fusion_weight_decay),
            "lr_scheduler_patience": int(getattr(cfg, "lr_scheduler_patience", 0)),
            "lr_scheduler_factor": float(getattr(cfg, "lr_scheduler_factor", 0.0)),
            "gradient_clip_norm": float(getattr(cfg, "gradient_clip_norm", 0.0)),
        },
        "n_common_subjects": len(common_ids),
        "dti_n_splits": int(getattr(cfg, "dti_n_splits", 10)),
        "fmri_n_splits": int(getattr(cfg, "fmri_n_splits", 10)),
        "fusion_n_repeats": int(getattr(cfg, "fusion_n_repeats", 5)),
        "fusion_test_size": float(getattr(cfg, "fusion_test_size", 0.2)),
        "fold_results": fold_results,
        "mean_fusion_accuracy": mean_acc,
        "std_fusion_accuracy": std_acc,
        "mean_fusion_auc": mean_auc if not math.isnan(mean_auc) else None,
    }

    payload: Dict[str, Any] = {"connectivity": _finalize_connectivity_aggregate(conn_store)}
    if tsne_chunks:
        payload["tsne_X"] = np.concatenate(tsne_chunks, axis=0).tolist()
        payload["tsne_y"] = np.concatenate(tsne_labels, axis=0).tolist()
        payload["tsne_ids"] = list(tsne_ids)
    result["_plot_payload"] = payload
    return result


# HYPERPARAMETER SWEEP (DTI/fMRI thresholds + node selection params)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sweep_score(result: Dict[str, Any], cfg: MultiModalGNNConfig) -> float:
    """Higher is better."""
    metric = str(getattr(cfg, "primary_sweep_metric", "accuracy")).strip().lower()

    if metric == "auc":
        auc_v = result.get("mean_fusion_auc", None)
        if auc_v is not None:
            v = _safe_float(auc_v)
            if math.isfinite(v):
                return v

    v = _safe_float(result.get("mean_fusion_accuracy", float("nan")))
    return v


def _fmt_tag_float(x: Any, ndp: int = 2) -> str:
    try:
        return f"{float(x):.{ndp}f}".replace(".", "p")
    except Exception:
        return "nan"


def make_run_tag(cfg: MultiModalGNNConfig, prefix: str = "") -> str:
    """Stable run tag used to avoid overwriting artifacts during sweeps."""
    parts = []
    if prefix:
        parts.append(prefix)

    parts.append(f"dtiT{_fmt_tag_float(cfg.dti_connectivity_threshold)}")
    parts.append(f"fmriT{_fmt_tag_float(cfg.fmri_connectivity_threshold)}")
    parts.append(f"k{int(cfg.node_selection_topk)}")
    parts.append(f"rf{_fmt_tag_float(cfg.node_reweight_factor)}")
    return "_".join(parts)


def run_hparam_sweep_for_task(
    dti_subjects_task: Dict[str, Dict],
    fmri_subjects_task: Dict[str, Dict],
    base_cfg: MultiModalGNNConfig,
    task_name: str,
    pubmed_prior_dti: Optional[np.ndarray] = None,
    clinical_encoder: Optional[Any] = None,
) -> Tuple[MultiModalGNNConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Runs a small sweep over:
      1) DTI threshold x fMRI threshold
      2) node_selection_topk x node_reweight_factor

    Two modes:
      - sequential : first tune thresholds using baseline (topk, rf),
        then tune (topk, rf) using best thresholds.
      - full: single sweep over all four hyperparams (expensive).

    Returns:
      best_cfg, best_result, sweep_summary
    """
    logger = UnifiedLogger.get_logger(f"Sweep.{task_name}")

    sweep_mode = str(getattr(base_cfg, "sweep_mode", "sequential")).strip().lower()
    max_cfg = int(getattr(base_cfg, "max_sweep_configs", 0) or 0)

    # Always deterministic per sweep run
    set_deterministic_mode(int(getattr(base_cfg, "random_seed", 42)))

    dti_thr_grid = list(getattr(base_cfg, "dti_connectivity_threshold_grid", [base_cfg.dti_connectivity_threshold]))
    fmri_thr_grid = list(getattr(base_cfg, "fmri_connectivity_threshold_grid", [base_cfg.fmri_connectivity_threshold]))
    topk_grid = list(getattr(base_cfg, "node_selection_topk_grid", [base_cfg.node_selection_topk]))
    rf_grid = list(getattr(base_cfg, "node_reweight_factor_grid", [base_cfg.node_reweight_factor]))

    metric_name = str(getattr(base_cfg, "primary_sweep_metric", "accuracy")).strip().lower()
    logger.info(
        f"[Sweep {task_name}] mode={sweep_mode} | metric={metric_name} | "
        f"thr_grid: DTI={dti_thr_grid} fMRI={fmri_thr_grid} | topk={topk_grid} | rf={rf_grid}"
    )

    sweep_summary: Dict[str, Any] = {
        "task": task_name,
        "sweep_mode": sweep_mode,
        "primary_metric": metric_name,
        "threshold_grid": {"dti": dti_thr_grid, "fmri": fmri_thr_grid},
        "topk_grid": topk_grid,
        "reweight_factor_grid": rf_grid,
        "max_sweep_configs": max_cfg,
        "runs": [],
        "best": None,
    }

    best_cfg: Optional[MultiModalGNNConfig] = None
    best_result: Optional[Dict[str, Any]] = None
    best_score: float = float("-inf")

    def _eval_one(cfg_run: MultiModalGNNConfig, prefix: str) -> Dict[str, Any]:
        tag = make_run_tag(cfg_run, prefix=prefix)
        logger.info(f"[Sweep {task_name}] RUN {tag}")
        # Ensure deterministic behavior across runs
        set_deterministic_mode(int(getattr(cfg_run, "random_seed", 42)))

        res = run_multimodal_cv_for_task(
            dti_subjects_task,
            fmri_subjects_task,
            cfg_run,
            task_name=task_name,
            pubmed_prior_dti=pubmed_prior_dti,
            clinical_encoder=clinical_encoder,
            run_tag=tag,
        )
        score = _sweep_score(res, cfg_run)

        sweep_summary["runs"].append({
            "run_tag": tag,
            "score": float(score) if math.isfinite(float(score)) else None,
            "mean_fusion_accuracy": float(res.get("mean_fusion_accuracy", float("nan"))),
            "mean_fusion_auc": res.get("mean_fusion_auc", None),
            "hyperparams": res.get("hyperparams", {}),
        })

        return res

    if sweep_mode == "full":
        combos = list(itertools.product(dti_thr_grid, fmri_thr_grid, topk_grid, rf_grid))
        if max_cfg > 0:
            combos = combos[:max_cfg]

        for dti_thr, fmri_thr, topk, rf in combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(dti_thr)
            cfg_run.fmri_connectivity_threshold = float(fmri_thr)
            cfg_run.node_selection_topk = int(topk)
            cfg_run.node_reweight_factor = float(rf)

            res = _eval_one(cfg_run, prefix="FULL")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_score:
                best_score = float(score)
                best_cfg = cfg_run
                best_result = res

    else:
        
        # Stage 1: thresholds sweep
        
        thr_combos = list(itertools.product(dti_thr_grid, fmri_thr_grid))
        if max_cfg > 0:
            thr_combos = thr_combos[:max_cfg]

        best_thr_cfg: Optional[MultiModalGNNConfig] = None
        best_thr_result: Optional[Dict[str, Any]] = None
        best_thr_score: float = float("-inf")

        for dti_thr, fmri_thr in thr_combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(dti_thr)
            cfg_run.fmri_connectivity_threshold = float(fmri_thr)

            # baseline node selection params for threshold tuning
            cfg_run.node_selection_topk = int(base_cfg.node_selection_topk)
            cfg_run.node_reweight_factor = float(base_cfg.node_reweight_factor)

            res = _eval_one(cfg_run, prefix="THR")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_thr_score:
                best_thr_score = float(score)
                best_thr_cfg = cfg_run
                best_thr_result = res

        if best_thr_cfg is None or best_thr_result is None:
            raise RuntimeError("Threshold sweep failed to produce a valid run/result.")

        logger.info(
            f"[Sweep {task_name}] Best thresholds: "
            f"DTI={best_thr_cfg.dti_connectivity_threshold} | fMRI={best_thr_cfg.fmri_connectivity_threshold} | "
            f"score={best_thr_score:.6f}"
        )

        
        # Stage 2: (topk, reweight_factor) sweep
        
        ns_combos = list(itertools.product(topk_grid, rf_grid))
        if max_cfg > 0:
            ns_combos = ns_combos[:max_cfg]

        for topk, rf in ns_combos:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run.dti_connectivity_threshold = float(best_thr_cfg.dti_connectivity_threshold)
            cfg_run.fmri_connectivity_threshold = float(best_thr_cfg.fmri_connectivity_threshold)
            cfg_run.node_selection_topk = int(topk)
            cfg_run.node_reweight_factor = float(rf)

            res = _eval_one(cfg_run, prefix="NS")
            score = _sweep_score(res, cfg_run)

            if math.isfinite(score) and score > best_score:
                best_score = float(score)
                best_cfg = cfg_run
                best_result = res

    if best_cfg is None or best_result is None:
        raise RuntimeError("Hyperparameter sweep did not find any valid configuration.")

    sweep_summary["best"] = {
        "score": float(best_score),
        "metric": metric_name,
        "run_tag": best_result.get("run_tag", None),
        "hyperparams": best_result.get("hyperparams", {}),
        "mean_fusion_accuracy": float(best_result.get("mean_fusion_accuracy", float("nan"))),
        "mean_fusion_auc": best_result.get("mean_fusion_auc", None),
    }

    # Save sweep summary
    out_path = Path(base_cfg.results_base_dir) / "results" / f"hparam_sweep_{task_name}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        logger.info(f"[Sweep {task_name}] Saved sweep summary -> {out_path}")
    except Exception as e:
        logger.warning(f"[Sweep {task_name}] Could not save sweep summary: {e}")

    return best_cfg, best_result, sweep_summary


# MAIN


def main():
    print("\n" + "=" * 80)
    print("MULTI-MODAL DTI + fMRI CONNECTIVITY GNN (DTI-ONLY NODE SELECTION)")
    print(" - Independent unimodal training (DTI, fMRI)")
    print(" - Node selection on DTI only, imported to fMRI (reweight only)")
    print(" - Fusion classifier on common subjects only")
    print(" - Training/optimization: AdamW + CE + ReduceLROnPlateau + grad clipping")
    print("=" * 80 + "\n")

    
    # EDIT THESE:
    
    TASKS_TO_RUN = ["NC_AD", "NC_MCI", "EMCI_LMCI", "MCI_AD"]  # choose any subset of the four binary tasks
    RESULTS_DIR = "./multimodal_dti_fmri_results"

    cfg = MultiModalGNNConfig(
        results_base_dir=RESULTS_DIR,

        # DTI paths
        dti_connectivity_dir="/kaggle/input/dti-dataset/DTI_dataset/connectivity_matrices",
        dti_node_features_dir="/kaggle/input/dti-dataset/DTI_dataset/node_features",
        # fMRI DFC path
        fmri_base_dir="/kaggle/input/fmri-data-size-30-step-5/fmri_data_size_30_step_5",

        # Label sources
        diagnostic_json="/kaggle/input/dti-diagnostic-groups/dti_diagnostic_groups.json",
        demographic_excel_path="/kaggle/input/demographic/demographic.xlsx",

        # Clinical embedding (PubMedBERT; derived from demographics -> text -> CLS embedding)
        use_clinical_embedding=True,
        clinical_in_fusion=False,
        use_pubmedbert=True,
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        pubmedbert_local_files_only=False,
        clinical_device=("cuda" if torch.cuda.is_available() else "cpu"),  # default; set "cpu" if needed
        llm_embedding_dim=768,
        use_apoe4=True,
        clinical_embedding_mode="clinical_no_diagnosis",  # do NOT include diagnosis text (avoid leakage)
        mask_diagnosis_in_embedding=True,


        # Model
        hidden_dim=128,
        gnn_num_layers=3,
        attention_heads=8,
        gnn_dropout=0.3,
        pooling="meanmax",

        # Graph
        dti_connectivity_threshold=0.1,
        fmri_connectivity_threshold=0.1,
        use_edge_weights=True,
        use_node_features=True,

        # Train
        batch_size=8,
        epochs=80,
        learning_rate=1e-3,
        weight_decay=5e-4,

        # Fusion
        fusion_hidden_dim=128,
        fusion_dropout=0.3,
        fusion_epochs=80,
        fusion_learning_rate=1e-3,
        fusion_weight_decay=2e-3,

        # CV + early stop
        dti_n_splits=10,
        fmri_n_splits=10,
        fusion_n_repeats=5,
        fusion_test_size=0.2,
        use_early_stopping=True,
        early_stopping_tolerance=7,
        early_stopping_min_delta=0.0,

        # Node selection
        node_selection_topk=25,
        node_reweight_factor=2.5,
        node_selection_max_batches=50,

        n_iterations=1,
        random_seed=42,
    )

    UnifiedLogger.initialize(f"{cfg.results_base_dir}/logs")
    logger = UnifiedLogger.get_logger("Main")
    cfg.save(f"{cfg.results_base_dir}/config.json")

    device = torch.device(cfg.device)
    print("CUDA available:", torch.cuda.is_available(), "| Using device:", device)

    # label sources
    diagnostic_loader = DiagnosticGroupLoader(cfg.diagnostic_json, "DX")
    demographic_loader = DemographicDataLoader(cfg.demographic_excel_path)

    # scan DTI
    dti_scanner = ConnectivityDataScanner(
        modality_name="DTI",
        connectivity_dir=cfg.dti_connectivity_dir,
        node_features_dir=cfg.dti_node_features_dir,
        config=cfg,
        connectivity_threshold=cfg.dti_connectivity_threshold,
    )
    dti_subjects = dti_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(dti_subjects) == 0:
        logger.error("No valid DTI subjects found.")
        return None

    # infer DTI dims
    dti_sample = next(iter(dti_subjects.values()))
    cfg.dti_num_regions = int(dti_sample["connectivity"].shape[0])
    if dti_sample.get("node_features") is not None:
        cfg.dti_node_feature_dim = int(dti_sample["node_features"].shape[1])
    else:
        cfg.dti_node_feature_dim = None

    # scan fMRI (DFC: *_dfc.npy)
    fmri_scanner = FMRIDFCScanner(
        base_dir=cfg.fmri_base_dir,
        config=cfg,
    )
    fmri_subjects = fmri_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(fmri_subjects) == 0:
        logger.error("No valid fMRI subjects found. (Check fmri_* paths)")
        return None

    # infer fMRI dims
    fmri_sample = next(iter(fmri_subjects.values()))
    cfg.fmri_num_regions = int(fmri_sample["connectivity"].shape[0])
    if fmri_sample.get("node_features") is not None:
        cfg.fmri_node_feature_dim = int(fmri_sample["node_features"].shape[1])
    else:
        cfg.fmri_node_feature_dim = None
        # demographic covariates (MODEL INPUT)
        # Single demographics source (same Excel) for both DTI and fMRI, following full_model_loss.py.
        # Vectors are 5-D: [age_norm, gender_enc, apoe4, mmse_norm, cdr_norm]
        if cfg.use_demographics:
            cfg.demographic_feature_dim = 5
            # Ensure every subject has a demographics vector (or None if missing)
            for sdict in (dti_subjects, fmri_subjects):
                for sid, s in sdict.items():
                    if s.get("demographics") is None and demographic_loader is not None:
                        demo_info = demographic_loader.get_subject_info(sid)
                        if demo_info is not None:
                            s["demographics"] = demo_info.to_feature_vector()
                        else:
                            s["demographics"] = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
            print(f"[Demographics] Using single demographic Excel. dim={cfg.demographic_feature_dim} | path={cfg.demographic_excel_path}")
        else:
            cfg.demographic_feature_dim = 0
            for s in dti_subjects.values():
                s["demographics"] = None
            for s in fmri_subjects.values():
                s["demographics"] = None
            print("[Demographics] Disabled by config. Will ignore.")

        # optional PubMed / literature prior over nodes

    pubmed_prior_dti: Optional[np.ndarray] = None
    # If pubmed_alpha > 0, we try to load a per-node prior vector either from:
    # (a) cfg.pubmed_node_prior_path (preferred), or
    # (b) the demographic Excel file (auto-detected sheets/columns).
    if float(cfg.pubmed_alpha) > 0.0:
        pubmed_loader = PubMedNodePriorLoader(
            prior_path=cfg.pubmed_node_prior_path,
            demographic_excel_path=cfg.demographic_excel_path,
        )
        pubmed_prior_dti = pubmed_loader.load_vector(expected_len=int(cfg.dti_num_regions))
        if pubmed_prior_dti is None:
            print("[PubMedPrior] Not loaded (no prior found in file or demographic Excel). Will ignore.")
        else:
            print(f"[PubMedPrior] Loaded ({pubmed_loader.last_source}) | len={pubmed_prior_dti.shape[0]} | alpha={cfg.pubmed_alpha}")

    
    # optional PubMedBERT clinical embedding (from demographics text; like reference implementation)
    # This is used in the fusion classifier when cfg.clinical_in_fusion=True.
    clinical_encoder = None
    if getattr(cfg, "use_clinical_embedding", False) and getattr(cfg, "clinical_in_fusion", True):
        clinical_encoder = PersonalizedClinicalEncoder(cfg, demographic_loader)

        # Cache embeddings once (reused across folds/tasks)
        all_ids_for_clinical = sorted(list(set(dti_subjects.keys()) | set(fmri_subjects.keys())))
        n_cached = clinical_encoder.precompute_all_embeddings(all_ids_for_clinical)

        model_status = "loaded" if getattr(clinical_encoder, "model", None) is not None else "FALLBACK"
        print(
            f"[ClinicalEmbedding] Cached={n_cached} | transformers={TRANSFORMERS_AVAILABLE} | PubMedBERT={model_status} | mode={cfg.clinical_embedding_mode}"
        )
    else:
        print("[ClinicalEmbedding] Disabled by config.")

# quick summary
    dti_dist = Counter(v["class"] for v in dti_subjects.values())
    fmri_dist = Counter(v["class"] for v in fmri_subjects.values())
    print("\nDTI distribution:", dict(dti_dist))
    print("fMRI distribution:", dict(fmri_dist))
    print(f"DTI num_regions={cfg.dti_num_regions} | fMRI num_regions={cfg.fmri_num_regions}")

    
    
    # MULTI-ITERATION EXPERIMENTS (10 iterations x 10 folds)
    
    # We repeat the entire CV procedure `cfg.n_iterations` times with different seeds
    # (methodology unchanged) and aggregate metrics across all folds & iterations.

    n_iters = int(getattr(cfg, "n_iterations", 1) or 1)
    all_task_results: Dict[str, Any] = {}

    for task in TASKS_TO_RUN:
        task_iteration_results: List[Dict[str, Any]] = []

        for it in range(n_iters):
            # Different seed each iteration (keeps methodology identical; only randomness changes)
            iter_cfg = copy.deepcopy(cfg)
            iter_cfg.random_seed = int(cfg.random_seed) + int(it)
            set_deterministic_mode(iter_cfg.random_seed)

            logger.info(f"[{get_task_display_name(task)} | {task}] Iteration {it+1}/{n_iters} | seed={iter_cfg.random_seed}")

            # Filter per task
            dti_task, class_names, mapping = build_task_subjects(dti_subjects, task)
            fmri_task, class_names2, mapping2 = build_task_subjects(fmri_subjects, task)

            if class_names != class_names2:
                raise RuntimeError("Task class_names mismatch between modalities.")
            if mapping != mapping2:
                raise RuntimeError("Task label mapping mismatch between modalities.")

            if len(dti_task) == 0 or len(fmri_task) == 0:
                logger.warning(f"[{task}] No data after filtering for at least one modality. Skipping iteration.")
                continue

            # Update config for task (binary)
            task_cfg = copy.deepcopy(iter_cfg)
            task_cfg.num_classes = len(class_names)
            task_cfg.class_names = class_names
            task_cfg.class_mapping = mapping

            # Run fusion CV (optionally with hyperparameter sweep)
            if getattr(task_cfg, "run_hparam_sweep", False):
                best_cfg, result, sweep_summary = run_hparam_sweep_for_task(
                    dti_task,
                    fmri_task,
                    task_cfg,
                    task_name=task,
                    pubmed_prior_dti=pubmed_prior_dti,
                    clinical_encoder=clinical_encoder,
                )
                task_cfg = best_cfg
                result["sweep_summary_path"] = str(Path(task_cfg.results_base_dir) / "results" / f"hparam_sweep_{task}.json")
            else:
                result = run_multimodal_cv_for_task(
                    dti_task,
                    fmri_task,
                    task_cfg,
                    task_name=task,
                    pubmed_prior_dti=pubmed_prior_dti,
                    clinical_encoder=clinical_encoder,
                    run_tag=f"iter{it+1:02d}",
                )

            result["iteration"] = int(it)
            result["seed"] = int(task_cfg.random_seed)
            task_iteration_results.append(result)

            # Save per-iteration
            out_path_it = Path(task_cfg.results_base_dir) / "results" / f"fusion_results_{task}_iter{it+1:02d}.json"
            with open(out_path_it, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved iteration results: {out_path_it}")

        
        # Aggregate over iterations
        
        if not task_iteration_results:
            logger.warning(f"[{task}] No iteration results. Skipping aggregation.")
            continue

        # Collect per-fold metrics across all iterations (total = n_iters * n_splits folds)
        all_fold_metrics = {
            "accuracy": [],
            "auc": [],
            "precision": [],
            "recall": [],
            "sensitivity": [],
            "specificity": [],
            "f1": [],
        }
        # ROC averaging grid (across all iters+folds)
        mean_fpr = np.linspace(0, 1, 200)
        tprs_all = []
        aucs_all = []

        for r in task_iteration_results:
            for fr in r.get("fold_results", []):
                if "fusion_acc" in fr:
                    all_fold_metrics["accuracy"].append(float(fr["fusion_acc"]))
                if fr.get("fusion_auc", None) is not None:
                    all_fold_metrics["auc"].append(float(fr["fusion_auc"]))
                if fr.get("fusion_precision", None) is not None:
                    all_fold_metrics["precision"].append(float(fr["fusion_precision"]))
                if fr.get("fusion_recall", None) is not None:
                    all_fold_metrics["recall"].append(float(fr["fusion_recall"]))
                if fr.get("fusion_sensitivity", None) is not None:
                    all_fold_metrics["sensitivity"].append(float(fr["fusion_sensitivity"]))
                if fr.get("fusion_specificity", None) is not None:
                    all_fold_metrics["specificity"].append(float(fr["fusion_specificity"]))
                if fr.get("fusion_f1", None) is not None:
                    all_fold_metrics["f1"].append(float(fr["fusion_f1"]))

                # ROC curve (if present)
                fpr = np.asarray(fr.get("fusion_fpr", []), dtype=float)
                tpr = np.asarray(fr.get("fusion_tpr", []), dtype=float)
                if fpr.size > 1 and tpr.size > 1:
                    # sort + interpolate
                    order = np.argsort(fpr)
                    fpr_s = fpr[order]
                    tpr_s = tpr[order]
                    tpr_i = np.interp(mean_fpr, fpr_s, tpr_s)
                    tpr_i[0] = 0.0
                    tprs_all.append(tpr_i)

        def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not vals:
                return None, None
            arr = np.asarray(vals, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        agg = {}
        for k, v in all_fold_metrics.items():
            mu, sd = _mean_std(v)
            agg[k] = {"mean": mu, "std": sd, "n": int(len(v))}

        # Mean ROC + std band
        roc_figs = {}
        if tprs_all:
            tprs_all = np.stack(tprs_all, axis=0)
            mean_tpr = np.mean(tprs_all, axis=0)
            std_tpr = np.std(tprs_all, axis=0)
            mean_tpr[-1] = 1.0

            # AUC of mean curve
            try:
                mean_auc = float(auc(mean_fpr, mean_tpr))
            except Exception:
                mean_auc = None

            roc_figs["mean_auc_from_mean_curve"] = mean_auc

            import matplotlib.pyplot as plt

            # Plot: overall mean ROC with std band
            plt.figure()
            plt.plot(mean_fpr, mean_tpr)
            plt.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.2)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Mean ROC over {n_iters} iterations x {get_fusion_n_repeats(cfg)} fusion repeats ({get_task_display_name(task)})")
            roc_path = Path(cfg.results_base_dir) / "figures" / f"roc_mean_{task}_{n_iters}iters_{get_fusion_n_repeats(cfg)}fusionrepeats.png"
            plt.savefig(roc_path, dpi=200, bbox_inches="tight")
            plt.close()
            roc_figs["mean_roc_path"] = str(roc_path)

        # Plot: per-iteration mean ROC (one curve per iteration)
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            for r in task_iteration_results:
                # average folds inside this iteration
                fprs = []
                tprs = []
                for fr in r.get("fold_results", []):
                    fpr = np.asarray(fr.get("fusion_fpr", []), dtype=float)
                    tpr = np.asarray(fr.get("fusion_tpr", []), dtype=float)
                    if fpr.size > 1 and tpr.size > 1:
                        order = np.argsort(fpr)
                        fpr_s = fpr[order]
                        tpr_s = tpr[order]
                        tpr_i = np.interp(mean_fpr, fpr_s, tpr_s)
                        tpr_i[0] = 0.0
                        tprs.append(tpr_i)
                if tprs:
                    tprs = np.stack(tprs, axis=0)
                    mean_tpr_it = np.mean(tprs, axis=0)
                    mean_tpr_it[-1] = 1.0
                    plt.plot(mean_fpr, mean_tpr_it)

            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Per-iteration Mean ROC ({get_task_display_name(task)})")
            roc_it_path = Path(cfg.results_base_dir) / "figures" / f"roc_per_iteration_{task}_{n_iters}iters.png"
            plt.savefig(roc_it_path, dpi=200, bbox_inches="tight")
            plt.close()
            roc_figs["per_iteration_roc_path"] = str(roc_it_path)
        except Exception:
            pass

        aggregated_result = {
            "task": task,
            "n_iterations": n_iters,
            "dti_n_splits": int(getattr(cfg, "dti_n_splits", 10)),
            "fmri_n_splits": int(getattr(cfg, "fmri_n_splits", 10)),
            "fusion_n_repeats": int(getattr(cfg, "fusion_n_repeats", 5)),
        "fusion_test_size": float(getattr(cfg, "fusion_test_size", 0.2)),
            "n_total_repeats": int(get_fusion_n_repeats(cfg)) * int(n_iters),
            "aggregate_metrics_over_all_folds": agg,
            "roc_figures": roc_figs,
            "iterations": task_iteration_results,
        }

        # Save aggregated JSON
        out_path_agg = Path(cfg.results_base_dir) / "results" / f"fusion_results_{task}_AGG_{n_iters}iters.json"
        with open(out_path_agg, "w") as f:
            json.dump(aggregated_result, f, indent=2, default=str)

        all_task_results[task] = aggregated_result

        print("\n" + "-" * 90)
        print(f"[{task}] Aggregated over {n_iters} iterations x {get_fusion_n_repeats(cfg)} fusion repeats (N={get_fusion_n_repeats(cfg)*n_iters})")
        for k in ["accuracy", "auc", "precision", "recall", "f1", "sensitivity", "specificity"]:
            mu = aggregated_result["aggregate_metrics_over_all_folds"][k]["mean"]
            sd = aggregated_result["aggregate_metrics_over_all_folds"][k]["std"]
            n = aggregated_result["aggregate_metrics_over_all_folds"][k]["n"]
            if mu is not None:
                print(f"  {k:12s}: {mu:.4f} ± {sd:.4f}   (n={n})")
        if roc_figs.get("mean_roc_path"):
            print(f"  ROC(mean): {roc_figs['mean_roc_path']}")
        if roc_figs.get("per_iteration_roc_path"):
            print(f"  ROC(iters): {roc_figs['per_iteration_roc_path']}")
        print("-" * 90 + "\n")
    combined_path = Path(cfg.results_base_dir) / "results" / "all_tasks_fusion_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_task_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("DONE. Saved combined results to:")
    print(combined_path)
    print("=" * 80 + "\n")
    
    return all_task_results





# SELF-CONTAINED SINGLE-ABLATION RUNNER ()


from contextlib import contextmanager
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RESULTS_ROOT = "./task_specific_ablation_results"
TASK_NAME = "NC_AD"
ABLATION_NAME = "dgtf_full"
ABLATION_CFG = {'mode': 'multimodal', 'use_demographics': True, 'demographics_in_unimodal': False, 'demographics_in_fusion': True, 'use_clinical_embedding': True, 'clinical_in_fusion': True, 'clinical_embedding_mode': 'clinical_no_diagnosis', 'pubmed_alpha': 0.0, 'node_selection_method': 'gradient', 'node_selection_topk': 25, 'node_reweight_factor': 2.5}

def auto_detect_flat_fmri_dir() -> str:
    explicit = os.environ.get("FMRI_BASE_DIR", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    preferred_names = [
        "fmri_data_size_30_step_5",
        "fmri_data_size_20_step_5",
        "fmri_data_size_40_step_5",
        "fmri_data_size_30_step_3",
        "fmri_data_size_30_step_10",
    ]
    search_roots = [Path("/kaggle/input"), Path("D:/Data/fmri_data"), Path("/mnt/d/Data/fmri_data"), Path(".")]

    for root in search_roots:
        if not root.exists():
            continue
        for name in preferred_names:
            p = root / name
            if p.exists() and p.is_dir():
                return str(p)
        for name in preferred_names:
            for p in root.rglob(name):
                if p.is_dir():
                    return str(p)

    pat = re.compile(r"^fmri_data_size_\d+_step_\d+$")
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and pat.match(p.name):
                return str(p)

    return "/kaggle/input/fmri-data/fmri_data_size_30_step_5"

COMMON_OVERRIDES = {'n_iterations': 10, 'dti_n_splits': 10, 'fmri_n_splits': 10, 'fusion_n_repeats': 10, 'fusion_test_size': 0.2, 'batch_size': 8, 'epochs': 120, 'fusion_epochs': 120, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'fusion_learning_rate': 0.001, 'fusion_weight_decay': 0.0005, 'node_selection_topk': 25, 'node_reweight_factor': 2.5, 'node_selection_max_batches': 50, 'dti_connectivity_threshold': 0.1, 'fmri_connectivity_threshold': 0.1, 'random_seed': 42}
DATA_PATH_OVERRIDES = {'fmri_base_dir': auto_detect_flat_fmri_dir()}

def ensure_results_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "results", "node_selection", "logs", "models", "checkpoints", "subject_lists", "dataset_analysis", "artifacts", "artifacts/history", "artifacts/embeddings", "artifacts/connectivity"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)




def _safe_connectivity_array(subject_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        conn = np.asarray(subject_entry.get("connectivity"), dtype=np.float32)
        if conn.ndim == 2 and conn.shape[0] == conn.shape[1] and conn.size > 0:
            return conn
    except Exception:
        return None
    return None


def _update_connectivity_aggregate(store: Dict[str, Any], subject_map: Dict[str, Dict], subject_ids: List[str], prefix: str, class_names: Optional[List[str]] = None) -> None:
    if prefix not in store:
        store[prefix] = {"sum": None, "count": 0, "class_sums": {}, "class_counts": {}}
    bucket = store[prefix]
    for sid in subject_ids:
        subj = subject_map.get(sid)
        if not isinstance(subj, dict):
            continue
        conn = _safe_connectivity_array(subj)
        if conn is None:
            continue
        if bucket["sum"] is None:
            bucket["sum"] = np.zeros_like(conn, dtype=np.float64)
        bucket["sum"] += conn.astype(np.float64)
        bucket["count"] += 1
        label = subj.get("label", None)
        if label is None:
            continue
        try:
            label_idx = int(label)
        except Exception:
            continue
        label_name = class_names[label_idx] if class_names and 0 <= label_idx < len(class_names) else str(label_idx)
        if label_name not in bucket["class_sums"]:
            bucket["class_sums"][label_name] = np.zeros_like(conn, dtype=np.float64)
            bucket["class_counts"][label_name] = 0
        bucket["class_sums"][label_name] += conn.astype(np.float64)
        bucket["class_counts"][label_name] += 1


def _finalize_connectivity_aggregate(store: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for prefix, bucket in store.items():
        entry: Dict[str, Any] = {}
        if bucket.get("sum") is not None and int(bucket.get("count", 0)) > 0:
            entry["overall_mean"] = (bucket["sum"] / float(bucket["count"])).astype(np.float32)
            entry["count"] = int(bucket["count"])
        class_means: Dict[str, np.ndarray] = {}
        for cname, csum in bucket.get("class_sums", {}).items():
            ccount = int(bucket.get("class_counts", {}).get(cname, 0))
            if ccount > 0:
                class_means[cname] = (csum / float(ccount)).astype(np.float32)
        if class_means:
            entry["class_means"] = class_means
            entry["class_counts"] = {k: int(v) for k, v in bucket.get("class_counts", {}).items()}
        if entry:
            out[prefix] = entry
    return out


def save_iteration_plot_artifacts(result: Dict[str, Any], results_base_dir: str, iteration: int, task_name: str, ablation_name: str, class_names: Optional[List[str]] = None) -> Dict[str, str]:
    base_dir = Path(results_base_dir)
    artifact_dir = base_dir / "artifacts"
    history_dir = artifact_dir / "history"
    embed_dir = artifact_dir / "embeddings"
    conn_dir = artifact_dir / "connectivity"
    for d in [artifact_dir, history_dir, embed_dir, conn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, str] = {}

    history_payload = {
        "task": task_name,
        "ablation_name": ablation_name,
        "iteration": int(iteration),
        "class_names": list(class_names or []),
        "fold_histories": [],
    }
    for fr in result.get("fold_results", []) or []:
        item = {
            "fold": fr.get("fold", fr.get("repeat")),
            "n_train": fr.get("n_train", fr.get("n_common_train")),
            "n_val": fr.get("n_val", fr.get("n_common_val")),
        }
        found = False
        for key in ["train_history", "val_history", "dti_train_history", "dti_val_history", "fmri_train_history", "fmri_val_history", "fusion_train_history", "fusion_val_history"]:
            if key in fr and fr[key]:
                item[key] = fr[key]
                found = True
        if found:
            history_payload["fold_histories"].append(item)
    if history_payload["fold_histories"]:
        history_path = history_dir / f"iter{int(iteration):02d}_epoch_history.json"
        save_json(history_path, history_payload)
        manifest["epoch_history_json"] = str(history_path)

    plot_payload = result.pop("_plot_payload", None) or {}
    X = plot_payload.get("tsne_X")
    y = plot_payload.get("tsne_y")
    ids = plot_payload.get("tsne_ids")
    if X is not None and y is not None:
        try:
            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y)
            if X_arr.ndim == 2 and X_arr.shape[0] >= 5 and y_arr.shape[0] == X_arr.shape[0]:
                embed_path = embed_dir / f"iter{int(iteration):02d}_fusion_embeddings.npz"
                np.savez_compressed(embed_path, embeddings=X_arr, labels=y_arr, subject_ids=np.asarray(ids if ids is not None else []), task=task_name, ablation=ablation_name, iteration=int(iteration), class_names=np.asarray(class_names or []))
                manifest["embedding_npz"] = str(embed_path)
        except Exception:
            pass

    connectivity = plot_payload.get("connectivity") or {}
    if connectivity:
        conn_arrays = {}
        meta = {"task": task_name, "ablation_name": ablation_name, "iteration": int(iteration), "class_names": list(class_names or [])}
        for prefix, entry in connectivity.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("overall_mean") is not None:
                conn_arrays[f"{prefix}_overall_mean"] = np.asarray(entry["overall_mean"], dtype=np.float32)
            for cname, arr in (entry.get("class_means") or {}).items():
                safe_cname = re.sub(r'[^A-Za-z0-9_]+', '_', str(cname))
                conn_arrays[f"{prefix}_class_{safe_cname}"] = np.asarray(arr, dtype=np.float32)
            meta[f"{prefix}_count"] = int(entry.get("count", 0))
            meta[f"{prefix}_class_counts"] = entry.get("class_counts", {})
        if conn_arrays:
            conn_path = conn_dir / f"iter{int(iteration):02d}_mean_connectivity.npz"
            np.savez_compressed(conn_path, **conn_arrays)
            meta_path = conn_dir / f"iter{int(iteration):02d}_mean_connectivity_meta.json"
            save_json(meta_path, meta)
            manifest["connectivity_npz"] = str(conn_path)
            manifest["connectivity_meta_json"] = str(meta_path)

    result["artifact_manifest"] = manifest
    return manifest

def make_cfg(results_dir: Path, extra_overrides: Optional[Dict[str, Any]] = None):
    cfg = MultiModalGNNConfig(results_base_dir=str(results_dir))
    for k, v in COMMON_OVERRIDES.items():
        setattr(cfg, k, v)
    for k, v in DATA_PATH_OVERRIDES.items():
        if v is not None:
            setattr(cfg, k, v)
    if extra_overrides is not None:
        for k, v in extra_overrides.items():
            if k in {"mode", "modality", "selector_patch"}:
                continue
            setattr(cfg, k, v)
    ensure_results_dirs(results_dir)
    return cfg


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray, cm: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = np.asarray(cm)
    metrics: Dict[str, Optional[float]] = {}

    unique_classes = np.unique(np.concatenate([y_true, y_pred])) if (y_true.size or y_pred.size) else np.array([], dtype=np.int64)
    average_mode = "binary" if unique_classes.size <= 2 else "weighted"

    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, average=average_mode, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average_mode, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average_mode, zero_division=0))
    except Exception:
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1"] = None

    metrics["sensitivity"] = metrics["recall"]

    try:
        if cm.ndim == 2 and cm.shape[0] == cm.shape[1] and cm.shape[0] > 0:
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel().tolist()
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
            else:
                specificities = []
                total = float(cm.sum())
                for i in range(cm.shape[0]):
                    tp = float(cm[i, i])
                    fn = float(cm[i, :].sum() - tp)
                    fp = float(cm[:, i].sum() - tp)
                    tn = float(total - tp - fn - fp)
                    denom = tn + fp
                    if denom > 0:
                        specificities.append(tn / denom)
                metrics["specificity"] = float(np.mean(specificities)) if specificities else None
        else:
            metrics["specificity"] = None
    except Exception:
        metrics["specificity"] = None

    try:
        if (
            y_prob_pos is not None
            and len(y_prob_pos) == len(y_true)
            and len(np.unique(y_true)) == 2
        ):
            metrics["auc"] = float(roc_auc_score(y_true, y_prob_pos))
        else:
            metrics["auc"] = None
    except Exception:
        metrics["auc"] = None
    return metrics


def aggregate_scalar_metrics(values: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, vals in values.items():
        clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
        if clean:
            arr = np.asarray(clean, dtype=float)
            out[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": int(arr.size)}
        else:
            out[k] = {"mean": None, "std": None, "n": 0}
    return out


def plot_mean_roc_from_fold_results(fold_results: List[Dict[str, Any]], title: str, out_path: Path):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    for fr in fold_results:
        fpr = np.asarray(fr.get("fpr", []), dtype=float)
        tpr = np.asarray(fr.get("tpr", []), dtype=float)
        auc_val = fr.get("auc", None)
        if fpr.size > 1 and tpr.size > 1:
            order = np.argsort(fpr)
            fpr = fpr[order]
            tpr = tpr[order]
            tpr_i = np.interp(mean_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)
            if auc_val is not None:
                aucs.append(float(auc_val))
    if not tprs:
        return {}
    tprs = np.stack(tprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs)) if aucs else None
    std_auc = float(np.std(aucs)) if aucs else None
    plt.figure(figsize=(6, 5))
    label = f"AUC = {mean_auc:.4f} ± {std_auc:.4f}" if mean_auc is not None and std_auc is not None else "Mean ROC"
    plt.plot(mean_fpr, mean_tpr, lw=2, label=label)
    plt.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return {"mean_roc_path": str(out_path), "mean_auc": mean_auc, "std_auc": std_auc}


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path, class_names: List[str]):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_path)


@contextmanager
def maybe_patch_random_selector(use_random: bool, seed: int = 42):
    if not use_random:
        yield
        return
    original_selector = DTINodeSelector
    class RandomNodeSelector:
        def __init__(self, topk: int = 20, max_batches: int = 0):
            self.topk = int(topk)
            self.max_batches = int(max_batches)
            self.rng = np.random.RandomState(seed)
        def compute_node_scores(self, model, dataloader, device=None):
            return self.rng.rand(int(model.num_regions)).astype(np.float32)
        def select_topk(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            topk = min(self.topk, scores.shape[0])
            if topk <= 0:
                return np.array([], dtype=np.int64), np.zeros(scores.shape[0], dtype=bool)
            idx = np.argsort(scores)[::-1][:topk]
            mask = np.zeros(scores.shape[0], dtype=bool)
            mask[idx] = True
            return idx, mask
        def make_node_weights(self, mask: np.ndarray, reweight_factor: float) -> np.ndarray:
            w = np.ones(mask.shape[0], dtype=np.float32)
            w[mask] = float(reweight_factor)
            return w
    globals()["DTINodeSelector"] = RandomNodeSelector
    try:
        yield
    finally:
        globals()["DTINodeSelector"] = original_selector


def prepare_shared_objects(cfg):
    UnifiedLogger.initialize(f"{cfg.results_base_dir}/logs")
    diagnostic_loader = DiagnosticGroupLoader(cfg.diagnostic_json, "DX")
    demographic_loader = DemographicDataLoader(cfg.demographic_excel_path)
    dti_scanner = ConnectivityDataScanner(modality_name="DTI", connectivity_dir=cfg.dti_connectivity_dir, node_features_dir=cfg.dti_node_features_dir, config=cfg, connectivity_threshold=cfg.dti_connectivity_threshold)
    dti_subjects = dti_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(dti_subjects) == 0:
        raise RuntimeError("No valid DTI subjects found.")
    dti_sample = next(iter(dti_subjects.values()))
    cfg.dti_num_regions = int(dti_sample["connectivity"].shape[0])
    cfg.dti_node_feature_dim = int(dti_sample["node_features"].shape[1]) if dti_sample.get("node_features") is not None else None
    fmri_scanner = FMRIDFCScanner(base_dir=cfg.fmri_base_dir, config=cfg)
    fmri_subjects = fmri_scanner.scan_and_validate(diagnostic_loader, demographic_loader)
    if len(fmri_subjects) == 0:
        raise RuntimeError("No valid fMRI subjects found.")
    fmri_sample = next(iter(fmri_subjects.values()))
    cfg.fmri_num_regions = int(fmri_sample["connectivity"].shape[0])
    cfg.fmri_node_feature_dim = int(fmri_sample["node_features"].shape[1]) if fmri_sample.get("node_features") is not None else None
    cfg.demographic_feature_dim = 5
    for sdict in (dti_subjects, fmri_subjects):
        for sid, s in sdict.items():
            if s.get("demographics") is None and demographic_loader is not None:
                demo_info = demographic_loader.get_subject_info(sid)
                if demo_info is not None:
                    s["demographics"] = demo_info.to_feature_vector()
                else:
                    s["demographics"] = np.array([0.5, 0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    return dti_subjects, fmri_subjects, demographic_loader


def maybe_load_pubmed_prior(cfg) -> Optional[np.ndarray]:
    if float(getattr(cfg, "pubmed_alpha", 0.0)) <= 0.0:
        return None
    loader = PubMedNodePriorLoader(prior_path=cfg.pubmed_node_prior_path, demographic_excel_path=cfg.demographic_excel_path)
    return loader.load_vector(expected_len=int(cfg.dti_num_regions))


def maybe_build_clinical_encoder(cfg, demographic_loader, dti_subjects, fmri_subjects):
    if not (getattr(cfg, "use_clinical_embedding", False) and getattr(cfg, "clinical_in_fusion", True)):
        return None
    encoder = PersonalizedClinicalEncoder(cfg, demographic_loader)
    all_ids = sorted(list(set(dti_subjects.keys()) | set(fmri_subjects.keys())))
    encoder.precompute_all_embeddings(all_ids)
    return encoder


def get_common_task_subjects(dti_subjects, fmri_subjects, task_name: str):
    dti_task, class_names, mapping = build_task_subjects(dti_subjects, task_name)
    fmri_task, class_names2, mapping2 = build_task_subjects(fmri_subjects, task_name)
    if class_names != class_names2 or mapping != mapping2:
        raise RuntimeError("Task metadata mismatch between modalities.")
    common_ids = sorted(list(set(dti_task.keys()) & set(fmri_task.keys())))
    if len(common_ids) < 4:
        raise RuntimeError(f"Too few common subjects for task={task_name}: n={len(common_ids)}")
    dti_common = {sid: dti_task[sid] for sid in common_ids}
    fmri_common = {sid: fmri_task[sid] for sid in common_ids}
    return dti_common, fmri_common, class_names, mapping, common_ids


def run_unimodal_cv(subjects_task: Dict[str, Dict], cfg, task_name: str, modality_name: str, run_tag: str):
    ids = sorted(list(subjects_task.keys()))
    y = np.array([int(subjects_task[sid]["label"]) for sid in ids], dtype=np.int64)
    effective_splits = get_safe_n_splits(y, get_modality_n_splits(cfg, modality_name))
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=0)
    device = torch.device(cfg.device)
    if modality_name.lower() == "dti":
        num_regions = int(cfg.dti_num_regions)
        node_feature_dim = cfg.dti_node_feature_dim
        conn_threshold = float(cfg.dti_connectivity_threshold)
    else:
        num_regions = int(cfg.fmri_num_regions)
        node_feature_dim = cfg.fmri_node_feature_dim
        conn_threshold = float(cfg.fmri_connectivity_threshold)
    fold_results = []
    tsne_chunks = []
    tsne_labels = []
    tsne_ids: List[str] = []
    conn_store: Dict[str, Any] = {}
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(ids)), y), start=1):
        train_ids = [ids[i] for i in tr_idx]
        val_ids = [ids[i] for i in va_idx]
        train_subjects = {sid: subjects_task[sid] for sid in train_ids}
        val_subjects = {sid: subjects_task[sid] for sid in val_ids}
        train_ds = ConnectivityGraphDataset(train_subjects, use_node_features=(cfg.use_node_features and node_feature_dim is not None), use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0), demographic_dim=int(cfg.demographic_feature_dim))
        val_ds = ConnectivityGraphDataset(val_subjects, use_node_features=(cfg.use_node_features and node_feature_dim is not None), use_demographics=(cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0), demographic_dim=int(cfg.demographic_feature_dim))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_connectivity_batch)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_connectivity_batch)
        model = ConnectivityGNN(num_regions=num_regions, node_feature_dim=node_feature_dim, num_classes=cfg.num_classes, connectivity_threshold=conn_threshold, config=cfg, demographic_dim=(int(cfg.demographic_feature_dim) if (cfg.use_demographics and cfg.demographics_in_unimodal and cfg.demographic_feature_dim > 0) else 0)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = None
        if getattr(cfg, "use_lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(cfg.lr_scheduler_factor), patience=int(cfg.lr_scheduler_patience), min_lr=float(cfg.lr_scheduler_min_lr))
        early_stopping = EarlyStopping(patience=cfg.early_stopping_tolerance, min_delta=cfg.early_stopping_min_delta, restore_best_weights=True)
        model, epoch_acc_train, epoch_acc_val, _, _, _, epoch_loss_train, epoch_loss_val = train_and_evaluate_gnn(model, optimizer, criterion, train_loader, val_loader, train_ds, val_ds, early_stopping, device, num_epochs=cfg.epochs, node_weights=None, scheduler=scheduler)
        acc_final, loss_final, cm, fpr, tpr, y_true, y_pred, y_prob_pos = eval_gnn(model, criterion, val_loader, val_ds, device, node_weights=None)
        emb_va, y_emb, ids_emb = extract_embeddings(model, val_subjects, cfg, device, num_regions=num_regions, node_feature_dim=node_feature_dim, connectivity_threshold=conn_threshold, node_weights=None)
        if emb_va is not None and len(emb_va) > 0:
            tsne_chunks.append(np.asarray(emb_va, dtype=np.float32))
            tsne_labels.append(np.asarray(y_emb, dtype=np.int64))
            tsne_ids.extend(list(ids_emb))
        _update_connectivity_aggregate(conn_store, subjects_task, val_ids, modality_name.lower(), list(getattr(cfg, "class_names", [])))
        metrics = compute_binary_metrics(y_true, y_pred, y_prob_pos, cm)
        fold_results.append({"fold": fold_idx, "n_train": len(train_ids), "n_val": len(val_ids), "accuracy": float(acc_final), "loss": float(loss_final), "auc": metrics["auc"], "precision": metrics["precision"], "recall": metrics["recall"], "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"], "f1": metrics["f1"], "confusion_matrix": cm.tolist(), "fpr": fpr.tolist() if hasattr(fpr, "tolist") else list(fpr), "tpr": tpr.tolist() if hasattr(tpr, "tolist") else list(tpr), "modality": modality_name.lower(), "train_history": {"acc": [float(v) for v in epoch_acc_train], "loss": [float(v) for v in epoch_loss_train]}, "val_history": {"acc": [float(v) for v in epoch_acc_val], "loss": [float(v) for v in epoch_loss_val]}})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {"task": task_name, "run_tag": run_tag, "fold_results": fold_results}
    payload: Dict[str, Any] = {"connectivity": _finalize_connectivity_aggregate(conn_store)}
    if tsne_chunks:
        payload["tsne_X"] = np.concatenate(tsne_chunks, axis=0).tolist()
        payload["tsne_y"] = np.concatenate(tsne_labels, axis=0).tolist()
        payload["tsne_ids"] = list(tsne_ids)
    result["_plot_payload"] = payload
    return result


def run_one_iteration(dti_common, fmri_common, cfg, demographic_loader, iter_idx: int):
    cfg_iter = copy.deepcopy(cfg)
    cfg_iter.random_seed = int(cfg.random_seed) + iter_idx
    set_deterministic_mode(cfg_iter.random_seed)
    run_tag = f"{ABLATION_NAME}_iter{iter_idx+1:02d}"
    if ABLATION_CFG["mode"] == "unimodal":
        modality = ABLATION_CFG["modality"]
        subjects_task = dti_common if modality == "dti" else fmri_common
        result = run_unimodal_cv(subjects_task, cfg_iter, TASK_NAME, modality_name=modality, run_tag=run_tag)
        result["iteration"] = iter_idx + 1
        result["seed"] = cfg_iter.random_seed
        save_iteration_plot_artifacts(result, cfg_iter.results_base_dir, iter_idx + 1, TASK_NAME, ABLATION_NAME, list(getattr(cfg_iter, "class_names", [])))
        return result
    pubmed_prior = maybe_load_pubmed_prior(cfg_iter)
    clinical_encoder = maybe_build_clinical_encoder(cfg_iter, demographic_loader, dti_common, fmri_common)
    use_random = ABLATION_CFG.get("selector_patch", None) == "random"
    with maybe_patch_random_selector(use_random=use_random, seed=int(cfg_iter.random_seed)):
        result = run_multimodal_cv_for_task(dti_common, fmri_common, cfg_iter, task_name=TASK_NAME, pubmed_prior_dti=pubmed_prior, clinical_encoder=clinical_encoder, run_tag=run_tag)
    result["iteration"] = iter_idx + 1
    result["seed"] = cfg_iter.random_seed
    return result


def aggregate_iterations(iter_results: List[Dict[str, Any]], out_dir: Path, class_names: List[str]):
    metric_store = {
        "accuracy": [], "auc": [], "precision": [], "recall": [],
        "sensitivity": [], "specificity": [], "f1": []
    }
    roc_fold_results = []
    confusion_sum = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    all_fold_results = []
    iter_summaries = []

    for result in iter_results:
        fold_results = result.get("fold_results", [])
        per_iter_metrics = {k: [] for k in metric_store}
        for fr in fold_results:
            if ABLATION_CFG["mode"] == "unimodal":
                metric_pairs = {
                    "accuracy": fr.get("accuracy"),
                    "auc": fr.get("auc"),
                    "precision": fr.get("precision"),
                    "recall": fr.get("recall"),
                    "sensitivity": fr.get("sensitivity"),
                    "specificity": fr.get("specificity"),
                    "f1": fr.get("f1"),
                }
                cm = np.asarray(fr.get("confusion_matrix", np.zeros((len(class_names), len(class_names)), dtype=np.int64).tolist()), dtype=np.int64)
                roc_fold_results.append({"fpr": fr.get("fpr", []), "tpr": fr.get("tpr", []), "auc": fr.get("auc", None)})
            else:
                metric_pairs = {
                    "accuracy": fr.get("fusion_acc"),
                    "auc": fr.get("fusion_auc"),
                    "precision": fr.get("fusion_precision"),
                    "recall": fr.get("fusion_recall"),
                    "sensitivity": fr.get("fusion_sensitivity"),
                    "specificity": fr.get("fusion_specificity"),
                    "f1": fr.get("fusion_f1"),
                }
                cm = np.asarray(fr.get("fusion_confusion_matrix", np.zeros((len(class_names), len(class_names)), dtype=np.int64).tolist()), dtype=np.int64)
                roc_fold_results.append({"fpr": fr.get("fusion_fpr", []), "tpr": fr.get("fusion_tpr", []), "auc": fr.get("fusion_auc", None)})
            if cm.shape != confusion_sum.shape:
                cm_fixed = np.zeros_like(confusion_sum)
                r = min(confusion_sum.shape[0], cm.shape[0])
                c = min(confusion_sum.shape[1], cm.shape[1])
                cm_fixed[:r, :c] = cm[:r, :c]
                cm = cm_fixed
            confusion_sum += cm
            for k, v in metric_pairs.items():
                if v is not None and not math.isnan(float(v)):
                    metric_store[k].append(float(v))
                    per_iter_metrics[k].append(float(v))
            all_fold_results.append(fr)
        iter_summaries.append({
            "iteration": result.get("iteration"),
            "seed": result.get("seed"),
            "aggregate_metrics": aggregate_scalar_metrics(per_iter_metrics),
        })

    final_metrics = aggregate_scalar_metrics(metric_store)
    roc_info = plot_mean_roc_from_fold_results(roc_fold_results, title=f"{TASK_NAME} - {ABLATION_NAME}", out_path=out_dir / "figures" / f"roc_mean_{ABLATION_NAME}.png")
    cm_path = plot_confusion_matrix(confusion_sum, title=f"{TASK_NAME} - {ABLATION_NAME}", out_path=out_dir / "figures" / f"confusion_matrix_sum_{ABLATION_NAME}.png", class_names=class_names)
    summary = {
        "task": TASK_NAME,
        "ablation_name": ABLATION_NAME,
        "mode": ABLATION_CFG["mode"],
        "n_iterations": int(COMMON_OVERRIDES["n_iterations"]),
        "dti_n_splits": int(COMMON_OVERRIDES["dti_n_splits"]),
        "fmri_n_splits": int(COMMON_OVERRIDES["fmri_n_splits"]),
        "fusion_n_repeats": int(COMMON_OVERRIDES["fusion_n_repeats"]),
        "n_total_folds": int(len(all_fold_results)),
        "aggregate_metrics_over_all_folds": final_metrics,
        "iteration_summaries": iter_summaries,
        "roc_figures": roc_info,
        "confusion_matrix_sum": confusion_sum.tolist(),
        "class_names": list(class_names),
        "confusion_matrix_figure": cm_path,
        "iterations": iter_results,
    }
    return summary


def main_single_ablation():
    scan_cfg_dir = Path(RESULTS_ROOT) / "_shared_scan_setup"
    scan_cfg = make_cfg(scan_cfg_dir, extra_overrides={"use_demographics": True, "use_clinical_embedding": False, "clinical_in_fusion": False, "pubmed_alpha": 0.0})
    dti_subjects, fmri_subjects, demographic_loader = prepare_shared_objects(scan_cfg)
    inferred = {"dti_num_regions": int(scan_cfg.dti_num_regions), "fmri_num_regions": int(scan_cfg.fmri_num_regions), "dti_node_feature_dim": scan_cfg.dti_node_feature_dim, "fmri_node_feature_dim": scan_cfg.fmri_node_feature_dim}
    ablation_dir = Path(RESULTS_ROOT) / TASK_NAME / ABLATION_NAME
    cfg_block = copy.deepcopy(ABLATION_CFG)
    cfg_block.update(inferred)
    cfg = make_cfg(ablation_dir, extra_overrides=cfg_block)
    dti_common, fmri_common, class_names, mapping, common_ids = get_common_task_subjects(dti_subjects, fmri_subjects, TASK_NAME)
    cfg.class_names = class_names
    cfg.class_mapping = mapping
    cfg.num_classes = len(class_names)
    cfg.experiment_name = f"{TASK_NAME}_{ABLATION_NAME}"
    cfg.save(str(ablation_dir / "config.json"))

    print("\n" + "#" * 100)
    print(f"Running task: {TASK_NAME}")
    print(f"Ablation: {ABLATION_NAME}")
    print(f"Common subjects: {len(common_ids)}")
    print(f"Iterations x split plans: {COMMON_OVERRIDES['n_iterations']} x (DTI={COMMON_OVERRIDES['dti_n_splits']}, fMRI={COMMON_OVERRIDES['fmri_n_splits']}, Fusion={COMMON_OVERRIDES['fusion_n_repeats']})")
    print("#" * 100 + "\n")

    iter_results = []
    for iter_idx in range(int(COMMON_OVERRIDES["n_iterations"])):
        print(f"\n>>> ITERATION {iter_idx+1}/{COMMON_OVERRIDES['n_iterations']} <<<\n")
        result = run_one_iteration(dti_common, fmri_common, cfg, demographic_loader, iter_idx)
        iter_results.append(result)
        save_json(ablation_dir / "results" / f"iter{iter_idx+1:02d}.json", result)

    final_summary = aggregate_iterations(iter_results, ablation_dir, class_names)
    save_json(ablation_dir / "results" / "final_summary.json", final_summary)
    downloadable_archive = create_downloadable_results_artifact(ablation_dir)

    print("\n" + "=" * 100)
    print(f"FINAL SUMMARY | {TASK_NAME} | {ABLATION_NAME}")
    for k in ["accuracy", "auc", "precision", "recall", "sensitivity", "specificity", "f1"]:
        blk = final_summary["aggregate_metrics_over_all_folds"][k]
        if blk["mean"] is not None:
            print(f"{k:12s}: {blk['mean']:.4f} ± {blk['std']:.4f}   (n={blk['n']})")
    print(f"Confusion matrix figure: {final_summary['confusion_matrix_figure']}")
    print(f"ROC figure: {final_summary['roc_figures'].get('mean_roc_path', 'N/A')}")
    print(f"Saved results to: {ablation_dir}")
    if downloadable_archive is not None:
        print(f"Downloadable archive path: {downloadable_archive}")
    print("=" * 100 + "\n")



def create_downloadable_results_artifact(ablation_dir: Path) -> Optional[str]:
    """Zip this ablation folder into /kaggle/working and show a Kaggle/Jupyter FileLink."""
    try:
        import os
        import zipfile

        ablation_dir = Path(ablation_dir)
        working_dir = Path("/kaggle/working")
        if not working_dir.exists():
            working_dir = Path.cwd()

        zip_name = f"{ablation_dir.parent.name}_{ablation_dir.name}.zip"
        zip_path = working_dir / zip_name

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(ablation_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = os.path.relpath(file_path, ablation_dir.parent)
                    zipf.write(file_path, arcname)

        print(f"\n✅ Created: {zip_path}")
        print(f"📦 Total size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

        if IPYTHON_DISPLAY_AVAILABLE and display is not None and FileLink is not None:
            try:
                display(FileLink(zip_name))
            except Exception as display_error:
                print(f"Could not render clickable download link: {display_error}")

        return str(zip_path)
    except Exception as e:
        print(f"Could not create downloadable archive for {ablation_dir}: {e}")
        return None

if __name__ == "__main__":
    main_single_ablation()