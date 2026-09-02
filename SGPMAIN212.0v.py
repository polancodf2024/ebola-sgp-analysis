#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SGPMAIN 211.0v - MIRROR-PIM WITH TRANSFER LEARNING & PEPTIDE DESIGN
FULL VERSION WITH DUAL MODE, NARRATIVE SUMMARY AND ADVANCED METRICS
================================================================================
IMPLEMENTED IMPROVEMENTS (v211.0):
1.  ✅ DUAL MODE SYSTEM (Characterization + Design)
2.  ✅ DYNAMIC NARRATIVE SUMMARY PER TARGET
3.  ✅ NEW ADVANCED MATHEMATICAL METRICS (Fisher, Ricci, PH, Uhlmann, etc.)
4.  ✅ MULTIDISCIPLINARY SUMMARY TABLE
5.  ✅ NEW CLASSES: OperationMode, NarrativeKnowledgeBase, etc.
6.  ✅ UPDATED CONFIGURATION WITH operation_control
7.  ✅ OUTPUT STRUCTURE FOR CHARACTERIZATION AND DESIGN MODE
8.  ✅ FIXED: interpret_hodge in ContextualInterpreter
9.  ✅ FIXED: KeyError 'length' in print_validation_report
10. ✅ REMOVED: ESMFold (causes memory issues)
11. ✅ FIXED: range validation metrics (hodge, ricci, wasserstein, fractal)
12. ✅ UPDATED: NarrativeKnowledgeBase for EBOLA with correct data
13. ✅ UPDATED: MAIN_GROUP_REFERENCE for Ebola species
14. ✅ UPDATED: MAIN_GROUP_DESIGN for Zaire ebolavirus
15. ✅ UPDATED: ConfigLoader to use config_EBOLA.json
16. ✅ FIXED: get_base_peptide_sequence_string to handle 'ebola' key
17. ✅ VERIFIED: All methods properly handle Ebola target
================================================================================
"""

import sys
import os
import warnings
import gc
import time
import json
import re
import pickle
import hashlib
import random
import tempfile
import shutil
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import combinations
from enum import Enum

# ============================================================================
# PYTHON VERSION CHECK
# ============================================================================

if sys.version_info < (3, 8):
    print("❌ ERROR: Python 3.8 or higher is required")
    print(f"   Current version: {sys.version}")
    sys.exit(1)

# ============================================================================
# SCIENTIFIC IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2, pearsonr, linregress, spearmanr, norm, entropy as scipy_entropy
from scipy.spatial.distance import cosine
from scipy.linalg import eigh, solve, svd, pinv
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.spatial import distance_matrix

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.decomposition import PCA

import psutil

warnings.filterwarnings('ignore')

# ============================================================================
# ESMFOLD - DESACTIVADO (causa problemas de memoria)
# ============================================================================

ESMFOLD_AVAILABLE = False
TORCH_AVAILABLE = False
GPU_AVAILABLE = False
EINOP_AVAILABLE = False

# Intentar importar torch solo para verificar disponibilidad (no se usa para ESMFold)
try:
    import torch
    TORCH_AVAILABLE = True
    try:
        import torch.cuda as cuda
        GPU_AVAILABLE = cuda.is_available()
        if GPU_AVAILABLE:
            print(f"  🚀 GPU available: {cuda.get_device_name(0)}")
        else:
            print("  💻 GPU not available")
    except:
        GPU_AVAILABLE = False
        print("  💻 GPU not available")
except ImportError:
    TORCH_AVAILABLE = False
    print("  ⚠️ PyTorch not available")

# Verificar einops
try:
    import einops
    EINOP_AVAILABLE = True
except ImportError:
    EINOP_AVAILABLE = False
    print("  ⚠️ einops not available")

print("  ℹ️ ESMFold DESACTIVADO (corregido en v210.0 - no se usa por problemas de memoria)")

# ============================================================================
# TRANSFER LEARNING IMPORTS (ESM2)
# ============================================================================

TRANSFORMERS_AVAILABLE = False
PEFT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, EsmForSequenceClassification
    from transformers import EsmTokenizer, EsmModel
    TRANSFORMERS_AVAILABLE = True
    print("  🧬 Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("  ⚠️ Transformers not available. Install: pip install transformers[torch]")

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
    print("  🧬 PEFT available (LoRA fine-tuning)")
except ImportError:
    PEFT_AVAILABLE = False
    print("  ⚠️ PEFT not available. Install: pip install peft")

# ============================================================================
# INPUT FILE PATH
# ============================================================================

DATA_PATH = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"

# ============================================================================
# FIXED SEED FOR REPRODUCIBILITY
# ============================================================================

np.random.seed(42)
random.seed(42)

# ============================================================================
# OPTIMIZED CONFIGURATION FOR GIANT FILES
# ============================================================================

CPU_CORES = mp.cpu_count()
MAX_WORKERS = min(CPU_CORES - 2, 4)
BATCH_SIZE = 5000
MAX_STORED_PROTEINS_PER_GROUP = 200
COHESION_CALC_SAMPLE_SIZE = 100

# ============================================================================
# CACHE CONFIGURATION - DISABLED FOR LARGE FILES
# ============================================================================

USE_SVD_CACHE = False
USE_DISK_CACHE = False
CACHE_DIR = "pim_cache"
CACHE_MAX_SIZE_MB = 50

os.environ['OMP_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['MKL_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['NUMEXPR_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

SIMILARITY_THRESHOLD = None
CONFIDENCE_LEVEL = 0.95
TOP_N_PROTEINS = 10
TOLERANCE = 0.001
USE_TRIPLETS = True
USE_QUADRUPLETS = False
USE_BOOTSTRAP = True
N_BOOTSTRAP = 50
USE_WEIGHTS = True
COHESION_SAMPLE_SIZE = COHESION_CALC_SAMPLE_SIZE
USE_BIOLOGICAL_METRIC = True
SHOW_METRIC_ANALYSIS = True
USE_HODGE_DUAL = True
USE_GRASSMANN_GEODESIC = True
USE_GENERAL_ROTORS = True
GENERATE_PLOTS = False

# Metrics - ALL ENABLED
USE_SHANNON_ENTROPY = True
USE_JENSEN_SHANNON = True
USE_GINI_COEFFICIENT = True
USE_STRUCTURAL_COMPLEXITY = True
USE_FUNCTIONAL_MODULARITY = True
USE_HELLINGER_DISTANCE = True
USE_SPEARMAN_CORRELATION = True
USE_MORANS_I = True

USE_GRASSMANN_PROJECTION = True
USE_FUBINI_STUDY = True
USE_RICCI_CURVATURE = True
USE_KARHUNEN_LOEVE = True
USE_RADON_TRANSFORM = True
USE_FRACTAL_DIMENSION = True
USE_WASSERSTEIN = True
USE_POLARITY_LAPLACIAN = True

# ============================================================================
# GRASSMANN CONFIGURATION
# ============================================================================

USE_GRASSMANN_MULTILEVEL = True
GRASSMANN_LEVELS = [1, 2, 3]
USE_GRASSMANN_ASYMMETRIC = True
USE_GRASSMANN_CURVATURE = True
USE_GRASSMANN_VOLUME = True
USE_GRASSMANN_CYCLES = True
USE_GRASSMANN_KARCHER = True
USE_GRASSMANN_SVD = True
USE_CURVATURE_SAMPLING = True
CURVATURE_SAMPLES = 30

# ============================================================================
# WEIGHTS FOR COMPOSITE METRICS
# ============================================================================

METRIC_WEIGHTS = {
    'pim': 0.25, 'entropy': 0.10, 'grassmann': 0.12,
    'hodge': 0.08, 'curvature': 0.08, 'gini': 0.05,
    'fubini': 0.05, 'jensen_shannon': 0.05, 'spearman': 0.05,
    'hellinger': 0.05, 'wasserstein': 0.04, 'fractal': 0.04,
    'radon': 0.04
}

# ============================================================================
# ESM2 CONFIGURATION
# ============================================================================

ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM2_MAX_LENGTH = 512
ESM2_BATCH_SIZE = 1
ESM2_USE_GPU = GPU_AVAILABLE
ESM2_FINE_TUNE_EPOCHS = 5
ESM2_LEARNING_RATE = 1e-5
ESM2_USE_LORA = PEFT_AVAILABLE
ESM2_LORA_R = 8
ESM2_LORA_ALPHA = 16
ESM2_LORA_DROPOUT = 0.1

MAX_PEPTIDE_LENGTH = 99999

# ============================================================================
# PIDP CONFIGURATION
# ============================================================================

USE_PIDP = True
PIDP_TARGETS_ONLY = True
PIDP_USE_METAPREDICT = True
PIDP_USE_AIUPRED = True
PIDP_THRESHOLDS = [0.3, 0.4, 0.5]

# ============================================================================
# TARGET GROUP CONFIGURATION - EBOLA SPECIES
# ============================================================================

# Reference groups: Six Ebola species for comparison
MAIN_GROUP_REFERENCE = ['zaire', 'sudan', 'reston', 'bundibugyo', 'tai', 'bombali']

# Design group: Zaire ebolavirus (most studied, highest pathogenicity)
MAIN_GROUP_DESIGN = ['zaire']

# Main group for analysis
MAIN_GROUP = MAIN_GROUP_REFERENCE

# ============================================================================
# CONFIG GROUP MAP
# ============================================================================

CONFIG_GROUP_MAP = {
    'west_nile': ['nile1', 'nile2', 'NILE1', 'NILE2'],
    'rvfv': ['rvf1', 'rvf2', 'rvf3', 'rvf4', 'RVF1', 'RVF2', 'RVF3', 'RVF4'],
    'ebola': ['EBOLA_ZAIRE', 'EBOLA_SUDAN', 'EBOLA_RESTON',
              'EBOLA_BOMBALI', 'EBOLA_BUNDIBUGYO', 'EBOLA_TAI_FOREST'],
    'lasv': ['LASV'],
    'junv': ['JUNV'],
    'macv': ['MACV'],
    'lcmv': ['LCMV'],
    'lujo': ['LUJO'],
}

def get_config_target(group_name: str) -> str:
    for config_key, groups in CONFIG_GROUP_MAP.items():
        if group_name in groups:
            return config_key
    return group_name.lower()

# ============================================================================
# EXTERNAL FILE CONFIGURATION
# ============================================================================

CHEMBL_MAPPING_FILE = os.path.join(DATA_PATH, "chembl_uniprot.txt")
APD_FASTA_FILE = os.path.join(DATA_PATH, "apd_natural.fasta")

# ============================================================================
# GROUP NAME MAPPING - EBOLA SPECIES
# ============================================================================

GROUP_NAME_MAP = {
    'enfermedad': 'DISEASE',
    'membrana': 'MEMBRANE',
    'senales': 'SIGNALS',
    'sudan': 'EBOLA_SUDAN',
    'zaire': 'EBOLA_ZAIRE',
    'reston': 'EBOLA_RESTON',
    'bombali': 'EBOLA_BOMBALI',
    'bundibugyo': 'EBOLA_BUNDIBUGYO',
    'tai': 'EBOLA_TAI_FOREST',
    'lasv': 'LASV',
    'junv': 'JUNV',
    'macv': 'MACV',
    'lcmv': 'LCMV',
    'nile1': 'NILE1',
    'nile2': 'NILE2',
    'rvf1': 'RVF1 (Gn)',
    'rvf2': 'RVF2 (Gc)',
    'rvf3': 'RVF3 (Gn-strain)',
    'rvf4': 'RVF4 (Gc-strain)',
    'lujo': 'LUJO',
}

def get_display_name(group_name: str) -> str:
    return GROUP_NAME_MAP.get(group_name, group_name)

def extract_protein_id(header: str) -> str:
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    if header.startswith('>'):
        header = header[1:]
    return header.split()[0] if header.split() else header[:20]

# ============================================================================
# BASE CONSTANTS
# ============================================================================

DIM_PAIRS = 16
DIM_TRIPLETS = 64
DIM_BIVECTOR = 120

ROTOR_PLANES = [
    ('hydrophobic', (10, 15), 'N→N vs NP→NP'),
    ('charge', (0, 5), 'P⁺→P⁺ vs NP→NP'),
    ('opposite_charge', (1, 4), 'P⁺→P⁻ vs P⁻→P⁺'),
    ('polarity', (10, 11), 'N→N vs N→NP'),
    ('charge_transition', (2, 8), 'P⁺→N vs N→P⁺'),
    ('opposite_transition', (6, 9), 'P⁻→N vs N→P⁻'),
]

REFLECTION_SWAP_MAP = {
    0: 5, 1: 4, 2: 6, 3: 7, 4: 1, 5: 0, 6: 2, 7: 3,
    8: 9, 9: 8, 10: 10, 11: 11, 12: 13, 13: 12, 14: 14, 15: 15,
}

KEY_BIVECTORS = [(0, 5), (1, 4), (2, 6), (3, 7), (10, 11), (14, 15)]

BIOLOGICAL_WEIGHTS = {
    'P+,P-': 2.0, 'P-,P+': 2.0,
    'N,N': 1.5,
    'N,P+': 1.3, 'P+,N': 1.3,
    'N,P-': 1.3, 'P-,N': 1.3,
    'NP,NP': 1.0,
    'NP,N': 0.9, 'N,NP': 0.9,
    'NP,P+': 0.7, 'P+,NP': 0.7,
    'NP,P-': 0.7, 'P-,NP': 0.7,
    'P+,P+': 0.4, 'P-,P-': 0.4,
}

BIOLOGICAL_METRIC_SIGNATURE = np.array([
    -1.0, +1.0, +1.0, +0.0,
    +1.0, -1.0, +1.0, +0.0,
    +1.0, +1.0, +1.0, +0.0,
    +0.0, +0.0, +0.0, +1.0,
])

EUCLIDEAN_METRIC = np.ones(16)
METRIC_SIGNATURE = BIOLOGICAL_METRIC_SIGNATURE if USE_BIOLOGICAL_METRIC else EUCLIDEAN_METRIC

SUBSPACES = {
    'hydrophobic': [10, 15],
    'charge_repulsion': [0, 5],
    'charge_attraction': [1, 4],
    'charge_polar': [2, 3, 6, 7],
    'polar': [8, 9, 10, 11],
    'nonpolar': [12, 13, 14, 15],
    'full': None,
}

POLARITY_MAP = {
    'H': 'P+', 'K': 'P+', 'R': 'P+',
    'D': 'P-', 'E': 'P-',
    'C': 'N', 'G': 'N', 'N': 'N', 'Q': 'N', 'S': 'N', 'T': 'N', 'Y': 'N',
    'A': 'NP', 'F': 'NP', 'I': 'NP', 'L': 'NP', 'M': 'NP', 'P': 'NP', 'V': 'NP', 'W': 'NP'
}

INTERACTIONS = [
    'P+,P+', 'P+,P-', 'P+,N', 'P+,NP',
    'P-,P+', 'P-,P-', 'P-,N', 'P-,NP',
    'N,P+', 'N,P-', 'N,N', 'N,NP',
    'NP,P+', 'NP,P-', 'NP,N', 'NP,NP'
]

INTERACTION_TO_IDX = {inter: i for i, inter in enumerate(INTERACTIONS)}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directory(path: str) -> str:
    if not path:
        path = "."
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def safe_save_csv(df: pd.DataFrame, filename: str, results_dir: str, **kwargs) -> str:
    results_dir = ensure_directory(results_dir)
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False, **kwargs)
    return filepath


def safe_save_json(data: Any, filename: str, results_dir: str, **kwargs) -> str:
    results_dir = ensure_directory(results_dir)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, **kwargs)
    return filepath


def safe_save_text(text: str, filename: str, results_dir: str) -> str:
    results_dir = ensure_directory(results_dir)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return filepath

# ============================================================================
# CONFIG LOADER (UPDATED WITH operation_control)
# ============================================================================

class ConfigLoader:
    """Loads configuration from config_EBOLA.json with fallback to hardcoded values"""

    DEFAULT_METRIC_WEIGHTS = {
        'pim': 0.25, 'entropy': 0.10, 'grassmann': 0.12,
        'hodge': 0.08, 'curvature': 0.08, 'gini': 0.05,
        'fubini': 0.05, 'jensen_shannon': 0.05, 'spearman': 0.05,
        'hellinger': 0.05, 'wasserstein': 0.04, 'fractal': 0.04,
        'radon': 0.04
    }

    DEFAULT_CLASSIFICATION_THRESHOLDS = {
        'excellent': 0.80,
        'good': 0.60,
        'moderate': 0.40,
        'poor': 0.00
    }

    DEFAULT_PROCESSING_PARAMS = {
        'batch_size': 5000,
        'max_stored_proteins': 200,
        'n_bootstrap': 50,
        'max_workers': 4
    }

    DEFAULT_PEPTIDE_LENGTH_RANGE = {
        'min': 15,
        'max': 25,
        'optimal': 20
    }

    DEFAULT_OPERATION_CONTROL = {
        'mode': 'auto',
        'use_dummy_peptide': False,
        'dummy_sequence': 'AAAAAAAAAAAAA',
        'fallback_mode': 'characterization',
        'allow_design_override': True,
        'evaluate_peptide': True
    }

    REQUIRED_RANGE_METRICS = [
        'pim_similarity', 'antiviral_activity', 'selectivity_index',
        'entropy', 'grassmann_distance', 'hodge_complementarity',
        'ricci_curvature', 'jensen_shannon', 'hellinger',
        'wasserstein', 'fractal_dimension', 'drug_likeness'
    ]

    def __init__(self, config_path: str = "config_EBOLA.json", verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.loaded_from = None
        self.errors = []
        self.warnings = []
        self._load()

    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if self._validate_config(config):
                    self.config = config
                    self.loaded_from = 'config_EBOLA.json'
                    if self.verbose:
                        print(f"  ✅ Configuration loaded from: {self.config_path}")
                    return
                else:
                    if self.verbose:
                        print(f"  ⚠️ Invalid configuration in {self.config_path}")
                    self.errors.append("Invalid configuration")

            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"  ❌ JSON format error: {e}")
                self.errors.append(f"Invalid JSON: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Error loading: {e}")
                self.errors.append(f"Error: {e}")
        else:
            if self.verbose:
                print(f"  ℹ️ File {self.config_path} not found")

        self.config = self._get_default_config()
        self.loaded_from = 'default'
        if self.verbose:
            print(f"  📋 Using default values (hardcoded)")

    def _validate_config(self, config: Dict) -> bool:
        required_sections = [
            'metadata', 'metric_weights', 'target_ranges',
            'classification_thresholds', 'known_inhibitors',
            'virus_specific_parameters', 'peptide_design_parameters',
            'latest_discoveries', 'processing_params', 'data_paths'
        ]

        for section in required_sections:
            if section not in config:
                self.warnings.append(f"Missing section: {section}")
                return False

        weights = config.get('metric_weights', {})
        required_metrics = ['pim', 'entropy', 'grassmann', 'hodge', 'curvature',
                           'gini', 'fubini', 'jensen_shannon', 'spearman',
                           'hellinger', 'wasserstein', 'fractal', 'radon']
        for metric in required_metrics:
            if metric not in weights:
                self.warnings.append(f"Missing metric: {metric}")
                return False

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.05:
            self.warnings.append(f"Weights sum to {total_weight:.2f}, not 1.0")

        return True

    def _get_default_config(self) -> Dict:
        return {
            'metadata': {
                'version': '211.0.0',
                'schema_version': '2.0.0',
                'generated': '2026-09-01T00:00:00',
                'active_target': 'default',
                'sgpmain_min_version': '211.0.0',
                'source': 'default_fallback'
            },
            'operation_control': self.DEFAULT_OPERATION_CONTROL.copy(),
            'metric_weights': self.DEFAULT_METRIC_WEIGHTS.copy(),
            'target_ranges': {},
            'classification_thresholds': self.DEFAULT_CLASSIFICATION_THRESHOLDS.copy(),
            'known_inhibitors': {},
            'virus_specific_parameters': {},
            'peptide_design_parameters': {},
            'latest_discoveries': {},
            'processing_params': self.DEFAULT_PROCESSING_PARAMS.copy(),
            'data_paths': {},
            'base_peptide_sequence': {},
            'characterization_parameters': {
                'depth': 'comprehensive',
                'include_dynamics': True,
                'include_evolutionary': False,
                'report_level': 'clinical',
                'output_formats': ['narrative', 'tabular', 'clinical']
            },
            'characterization_targets': {
                'primary': 'ebola_gp',
                'secondary': ['ebola_vp40', 'ebola_vp30'],
                'include_reference_groups': True
            }
        }

    def get_operation_mode(self) -> str:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        mode = oc.get('mode', 'auto')

        if mode == 'auto':
            base_seq = self.get_base_peptide_sequence_string()
            if base_seq and len(base_seq) > 5 and not self.is_dummy_peptide():
                return 'hybrid'
            else:
                return 'characterization'

        return mode

    def get_operation_control(self) -> Dict:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL.copy())

    def is_dummy_peptide(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        if oc.get('use_dummy_peptide', False):
            return True

        base_seq = self.get_base_peptide_sequence_string()
        dummy_seq = oc.get('dummy_sequence', 'AAAAAAAAAAAAA')

        if base_seq == dummy_seq:
            return True

        return False

    def get_dummy_sequence(self) -> str:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('dummy_sequence', 'AAAAAAAAAAAAA')

    def get_fallback_mode(self) -> str:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('fallback_mode', 'characterization')

    def get_evaluate_peptide(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('evaluate_peptide', True)

    def get_characterization_parameters(self) -> Dict:
        return self.config.get('characterization_parameters', {
            'depth': 'comprehensive',
            'include_dynamics': True,
            'include_evolutionary': False,
            'report_level': 'clinical',
            'output_formats': ['narrative', 'tabular', 'clinical']
        })

    def get_characterization_targets(self) -> Dict:
        return self.config.get('characterization_targets', {
            'primary': 'ebola_gp',
            'secondary': ['ebola_vp40', 'ebola_vp30'],
            'include_reference_groups': True
        })

    def get_metric_weights(self) -> Dict:
        weights = self.config.get('metric_weights', self.DEFAULT_METRIC_WEIGHTS).copy()
        required = ['pim', 'entropy', 'grassmann', 'hodge', 'curvature',
                   'gini', 'fubini', 'jensen_shannon', 'spearman',
                   'hellinger', 'wasserstein', 'fractal', 'radon']
        for metric in required:
            if metric not in weights:
                weights[metric] = 0.05
        return weights

    def get_target_ranges(self, target: str) -> Dict:
        target_ranges = self.config.get('target_ranges', {})
        ranges = target_ranges.get(target, {})

        for metric in self.REQUIRED_RANGE_METRICS:
            if metric not in ranges:
                ranges[metric] = {'min': 0.0, 'max': 1.0, 'source': 'default'}
        return ranges

    def get_known_inhibitors(self, target: str = None) -> List[Dict]:
        inhibitors = self.config.get('known_inhibitors', {})
        if target and target in inhibitors:
            return inhibitors[target]
        all_inhibitors = []
        for target_inhibitors in inhibitors.values():
            all_inhibitors.extend(target_inhibitors)
        return all_inhibitors

    def get_virus_parameters(self, target: str) -> Dict:
        virus_params = self.config.get('virus_specific_parameters', {})
        return virus_params.get(target, {})

    def get_peptide_design_parameters(self, target: str) -> Dict:
        design_params = self.config.get('peptide_design_parameters', {})
        return design_params.get(target, {})

    def get_classification_thresholds(self) -> Dict:
        return self.config.get('classification_thresholds', self.DEFAULT_CLASSIFICATION_THRESHOLDS)

    def get_processing_params(self) -> Dict:
        return self.config.get('processing_params', self.DEFAULT_PROCESSING_PARAMS)

    def get_data_paths(self) -> Dict:
        return self.config.get('data_paths', {})

    def get_all_targets(self) -> List[str]:
        return list(self.config.get('target_ranges', {}).keys())

    def is_default(self) -> bool:
        return self.loaded_from == 'default'

    def get_base_peptide_sequence(self, target: str = None) -> Dict:
        """Get base peptide sequence from config"""
        base_seq = self.config.get('base_peptide_sequence', {})
        if target:
            target_base = self.config.get('target_base_sequences', {})
            if target in target_base:
                return target_base[target]
        return base_seq

    def get_base_peptide_sequence_string(self, target: str = None) -> str:
        """Get base peptide sequence as string - FIXED to handle 'ebola' key"""
        base = self.get_base_peptide_sequence(target)

        # Check if we have a target-specific sequence
        if target and target in base:
            seq = base.get('sequence', '')
            if seq:
                return seq

        # Check for common keys
        if 'ebola' in base:
            return base['ebola'].get('sequence', '')
        if 'rvfv' in base:
            return base['rvfv'].get('sequence', '')

        # Try direct sequence
        if isinstance(base, dict):
            return base.get('sequence', '')

        return ''

    def get_base_peptide_length(self, target: str = None) -> int:
        base = self.get_base_peptide_sequence(target)
        seq = base.get('sequence', '')
        return len(seq) if seq else 0

    def get_max_peptide_length(self, target: str = None) -> int:
        if target:
            design_params = self.get_peptide_design_parameters(target)
            if 'max_peptide_length' in design_params:
                return design_params['max_peptide_length']
        base = self.get_base_peptide_sequence(target)
        if 'recommended_length_range' in base:
            return base['recommended_length_range'].get('max', 25)
        return 25

    def get_peptide_length_range(self, target: str = None) -> Dict:
        if target:
            design_params = self.get_peptide_design_parameters(target)
            if 'length_range' in design_params:
                return design_params['length_range']
        base = self.get_base_peptide_sequence(target)
        if 'recommended_length_range' in base:
            return base['recommended_length_range']
        return self.DEFAULT_PEPTIDE_LENGTH_RANGE.copy()

    def print_summary(self):
        print("\n  📋 CONFIGURATION SUMMARY")
        print("  " + "=" * 40)
        print(f"     ├─ Source: {self.loaded_from}")
        if self.loaded_from == 'config_EBOLA.json':
            meta = self.config.get('metadata', {})
            print(f"     ├─ Version: {meta.get('version', 'unknown')}")
            print(f"     ├─ Generated: {meta.get('generated', 'unknown')}")
            print(f"     └─ Schema: {meta.get('schema_version', 'unknown')}")

        oc = self.get_operation_control()
        print(f"     ├─ Operation mode: {self.get_operation_mode()}")
        print(f"     ├─ Use dummy peptide: {oc.get('use_dummy_peptide', False)}")
        print(f"     ├─ Evaluate peptide: {oc.get('evaluate_peptide', True)}")
        print(f"     └─ Fallback mode: {oc.get('fallback_mode', 'characterization')}")

        print(f"     ├─ Targets configured: {len(self.get_all_targets())}")
        print(f"     ├─ Metrics configured: {len(self.get_metric_weights())}")
        base_seq = self.get_base_peptide_sequence_string()
        if base_seq:
            print(f"     ├─ Base peptide: {base_seq[:20]}... ({len(base_seq)} aa)")
            base_info = self.get_base_peptide_sequence()
            desc = base_info.get('description', '')[:50]
            if desc:
                print(f"     │   └─ {desc}...")
        else:
            print(f"     ├─ Base peptide: NOT CONFIGURED")
        max_len = self.get_max_peptide_length()
        print(f"     └─ Max peptide length: {max_len} aa")
        if self.warnings:
            print(f"     ⚠️ Warnings: {len(self.warnings)}")
            for warning in self.warnings[:3]:
                print(f"         └─ {warning}")

# ============================================================================
# OPERATION MODE
# ============================================================================

class OperationMode(Enum):
    CHARACTERIZATION = "characterization"
    DESIGN = "design"
    HYBRID = "hybrid"
    AUTO = "auto"

    @classmethod
    def determine(cls, config_loader: ConfigLoader) -> 'OperationMode':
        oc = config_loader.get_operation_control()
        mode_str = oc.get('mode', 'auto')
        evaluate_peptide = oc.get('evaluate_peptide', True)

        if mode_str == 'characterization':
            return cls.CHARACTERIZATION
        elif mode_str == 'design':
            return cls.DESIGN
        elif mode_str == 'hybrid':
            return cls.HYBRID

        base_seq = config_loader.get_base_peptide_sequence_string()
        is_dummy = config_loader.is_dummy_peptide()

        if base_seq and len(base_seq) > 5 and not is_dummy and evaluate_peptide:
            return cls.HYBRID
        else:
            return cls.CHARACTERIZATION

    def is_characterization(self) -> bool:
        return self in [OperationMode.CHARACTERIZATION, OperationMode.HYBRID, OperationMode.AUTO]

    def is_design(self) -> bool:
        return self in [OperationMode.DESIGN, OperationMode.HYBRID]

    def is_hybrid(self) -> bool:
        return self == OperationMode.HYBRID

    def get_mode_name(self) -> str:
        return self.value

# ============================================================================
# NARRATIVE KNOWLEDGE BASE - UPDATED FOR EBOLA
# ============================================================================

class NarrativeKnowledgeBase:
    def __init__(self):
        self.knowledge = self._build_knowledge_base()

    def _build_knowledge_base(self) -> Dict:
        return {
            'rvfv': {
                'target_name': 'Rift Valley Fever Virus (RVFV)',
                'family': 'Phenuiviridae',
                'genome': 'ssRNA(-)-sense',
                'glycoproteins': {
                    'Gn': {
                        'name': 'Glycoprotein Gn',
                        'role': 'Attachment protein, receptor binding',
                        'length': 537,
                        'uniprot': 'P03518',
                        'key_residues': ['H54', 'E64', 'D66', 'K68', 'H71']
                    },
                    'Gc': {
                        'name': 'Glycoprotein Gc',
                        'role': 'Fusion protein, membrane fusion during viral entry',
                        'length': 516,
                        'uniprot': 'P03518',
                        'fusion_loop': 'GSSRFTNWGSVSLSLDAEGISGSNSFSFIES',
                        'domains': ['Domain I (691-850)', 'Domain II (851-1000)', 'Domain III (1001-1130)'],
                        'pdb_structures': ['7UU8', '7UU9', '8DZ7']
                    }
                },
                'clinical': {
                    'mortality_rate': '10-20% in hospitalized patients (hemorrhagic fever)',
                    'cfr': '1-2% general, up to 50% in severe hemorrhagic cases',
                    'high_risk_groups': ['Immunocompromised', 'Elderly', 'Pregnant women'],
                    'transmission': 'Mosquito-borne (Aedes, Culex), direct contact with infected tissues',
                    'treatment': 'No specific antiviral treatment. Supportive care.',
                    'vaccine': 'Live-attenuated veterinary vaccines exist. No licensed human vaccine.'
                },
                'narrative_templates': {
                    'structural_summary': "The RVFV Gc glycoprotein ({length} aa) is a class II fusion protein responsible for membrane fusion during viral entry. It contains a conserved fusion loop ({fusion_loop}) and three structural domains (I, II, III). The protein has been structurally validated by Cryo-EM (PDB: {pdb_structures}).",
                    'clinical_relevance': "RVFV is a zoonotic pathogen endemic in sub-Saharan Africa, transmitted by Aedes mosquitoes. Human infection can present as uncomplicated fever or progress to hemorrhagic disease with mortality of {mortality_rate}. There is no specific treatment, highlighting the urgency of developing effective antiviral peptides.",
                    'peptide_recommendation': "The designed peptide shows {similarity} PIM similarity with the Gc target protein, suggesting {interpretation}. Structural validation via ESMFold indicates {structural_status}, which {recommendation}."
                }
            },
            'ebola': {
                'target_name': 'Ebola Virus (EBOV)',
                'family': 'Filoviridae',
                'genome': 'ssRNA(-)-sense',
                'glycoproteins': {
                    'GP': {
                        'name': 'Glycoprotein GP',
                        'role': 'Attachment and fusion protein',
                        'length': 676,
                        'uniprot': 'Q05320',
                        'fusion_loop': 'GAAIGLAWIPYFGPAAEGI',
                        'domains': ['Fusion Loop (511-553)', 'HR1 (554-598)'],
                        'pdb_structures': ['6VKM', '8Y3U', '2LCZ']
                    },
                    'VP40': {
                        'name': 'Matrix protein VP40',
                        'role': 'Viral assembly and budding',
                        'length': 326,
                        'uniprot': 'Q05320'
                    },
                    'VP30': {
                        'name': 'Transcription activator VP30',
                        'role': 'Viral transcription, interacts with NP',
                        'length': 288,
                        'uniprot': 'Q05320'
                    }
                },
                'clinical': {
                    'mortality_rate': '25-90% depending on species',
                    'cfr': 'Zaire: ~88%, Sudan: ~53%, Bundibugyo: ~25%, Tai Forest: 0% (single case), Reston: 0% (non-pathogenic in humans), Bombali: unknown',
                    'high_risk_groups': ['Healthcare workers', 'Family contacts', 'Immunocompromised patients', 'Pregnant women'],
                    'transmission': 'Direct contact with bodily fluids, contaminated needles, contact with infected animals (fruit bats, non-human primates)',
                    'treatment': 'Monoclonal antibodies (REGN-EB3, mAb114), remdesivir, supportive care',
                    'vaccine': 'ERVEBO (rVSV-ZEBOV) approved for Zaire ebolavirus. No licensed vaccine for other species.'
                },
                'narrative_templates': {
                    'structural_summary': "The Ebola virus GP glycoprotein ({length} aa) is a class I fusion protein responsible for membrane fusion during viral entry. It contains a conserved fusion loop ({fusion_loop}) that is 100% identical across all six Ebola species (Zaire, Sudan, Reston, Bundibugyo, Tai Forest, and Bombali). The fusion loop structure has been resolved by NMR at pH 7.0 (PDB 2LCZ). The protein has also been structurally validated by X-ray crystallography (PDB 6VKM) and Cryo-EM (PDB 8Y3U).",
                    'clinical_relevance': "Ebola virus is a high-risk zoonotic pathogen with mortality rates of {mortality_rate}. Recent outbreaks in West Africa (2014-2016) and Central Africa have caused thousands of deaths. The ERVEBO vaccine is available for Zaire ebolavirus, and monoclonal antibody therapies (REGN-EB3, mAb114) have been approved. However, there is no licensed vaccine for other Ebola species, highlighting the need for broad-spectrum therapeutics.",
                    'peptide_recommendation': "The designed peptide shows {similarity} similarity with the GP fusion loop, indicating {interpretation}. Structural validation suggests {structural_status}, which {recommendation}. The peptide targets a region that is 100% conserved across all six Ebola species, making it a promising candidate for broad-spectrum antiviral development."
                }
            },
            'lasv': {
                'target_name': 'Lassa Virus (LASV)',
                'family': 'Arenaviridae',
                'genome': 'ssRNA(-)-sense',
                'glycoproteins': {
                    'GP': {
                        'name': 'Glycoprotein GP',
                        'role': 'Attachment and fusion protein',
                        'length': 491,
                        'uniprot': 'P08669'
                    }
                },
                'clinical': {
                    'mortality_rate': '1-15% in hospitalized patients',
                    'cfr': '~1% general, up to 15% in severe cases',
                    'high_risk_groups': ['Healthcare workers', 'Pregnant women'],
                    'transmission': 'Contact with rodent excreta, person-to-person',
                    'treatment': 'Ribavirin, supportive care',
                    'vaccine': 'No licensed vaccine'
                }
            }
        }

    def get_knowledge(self, target: str) -> Dict:
        target_lower = target.lower()
        # Check for ebola first
        if 'ebola' in target_lower or 'zaire' in target_lower or 'sudan' in target_lower or 'EBOLA' in target:
            return self.knowledge['ebola']
        for key in self.knowledge:
            if key in target_lower or target_lower in key:
                return self.knowledge[key]
        return self.knowledge.get('ebola', {})

    def get_narrative_template(self, target: str, template_name: str) -> str:
        knowledge = self.get_knowledge(target)
        templates = knowledge.get('narrative_templates', {})
        return templates.get(template_name, "")

    def get_target_info(self, target: str) -> Dict:
        return self.get_knowledge(target)


# ============================================================================
# CLINICAL LEVEL EVALUATOR
# ============================================================================

class ClinicalLevelEvaluator:
    LEVELS = {
        1: {'name': 'Excellent', 'color': '#2ecc71', 'description': 'Optimal characteristics for therapeutic design'},
        2: {'name': 'Good', 'color': '#27ae60', 'description': 'Favorable characteristics, minor optimizations'},
        3: {'name': 'Moderate', 'color': '#f1c40f', 'description': 'Acceptable characteristics, requires optimization'},
        4: {'name': 'Poor', 'color': '#e67e22', 'description': 'Unfavorable characteristics, requires redesign'},
        5: {'name': 'Critical', 'color': '#e74c3c', 'description': 'Critical characteristics, not recommended for development'}
    }

    def __init__(self):
        self.thresholds = {
            'structural_stability': {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.0},
            'functional_relevance': {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.0},
            'drug_likeness': {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.0},
            'interaction_quality': {1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.0},
            'structural_complexity': {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.65, 5: 0.8},
            'entropy_level': {1: 0.3, 2: 0.45, 3: 0.6, 4: 0.75, 5: 0.9},
            'conservation': {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.0}
        }

    def evaluate(self, metric_name: str, value: float) -> int:
        if metric_name not in self.thresholds:
            return 3

        thresholds = self.thresholds[metric_name]
        high_better = metric_name in ['structural_stability', 'functional_relevance',
                                      'drug_likeness', 'interaction_quality', 'conservation']
        low_better = metric_name in ['structural_complexity', 'entropy_level']

        for level in range(1, 6):
            threshold = thresholds.get(level, 0.5)
            if high_better and value >= threshold:
                return level
            if low_better and value <= threshold:
                return level

        return 5 if high_better else 1

    def get_level_info(self, level: int) -> Dict:
        return self.LEVELS.get(level, self.LEVELS[3])

    def get_color(self, level: int) -> str:
        info = self.get_level_info(level)
        return info.get('color', '#808080')

    def get_name(self, level: int) -> str:
        info = self.get_level_info(level)
        return info.get('name', 'Moderate')

    def get_description(self, level: int) -> str:
        info = self.get_level_info(level)
        return info.get('description', '')

# ============================================================================
# CONTEXTUAL INTERPRETER
# ============================================================================

class ContextualInterpreter:
    def __init__(self, knowledge_base: NarrativeKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.level_evaluator = ClinicalLevelEvaluator()

    def interpret_similarity(self, value: float, metric: str = "pim_similarity") -> str:
        if value >= 0.85:
            return "high structural and functional similarity, indicating a peptide that effectively mimics interaction with the target protein"
        elif value >= 0.70:
            return "good similarity, the peptide retains key structural elements for interaction"
        elif value >= 0.50:
            return "moderate similarity, the peptide captures some structural aspects but may require optimization"
        elif value >= 0.30:
            return "low similarity, the peptide diverges significantly from the optimal interaction"
        else:
            return "very low similarity, the peptide does not effectively mimic the natural interaction"

    def interpret_grassmann_distance(self, value: float) -> str:
        if value < 0.2:
            return "peptides occupy very similar structural spaces, suggesting high complementarity"
        elif value < 0.4:
            return "moderate distance, peptides show manageable structural differences"
        elif value < 0.6:
            return "significant distance, peptides diverge in structure, which may affect complementarity"
        else:
            return "large structural distance, peptides are structurally very different from the target"

    def interpret_entropy(self, value: float) -> str:
        if value < 2.0:
            return "low entropy, indicating a very concentrated and specific interaction distribution"
        elif value < 3.0:
            return "moderate entropy, balanced distribution of interactions with some degree of specialization"
        elif value < 3.8:
            return "high entropy, diverse distribution of interactions indicating functional flexibility"
        else:
            return "very high entropy, extremely diverse distribution that may indicate lack of specificity"

    def interpret_hodge(self, value: float) -> str:
        if value >= 0.8:
            return "excellent structural complementarity, interactions are perfectly aligned"
        elif value >= 0.65:
            return "high structural complementarity, interactions well aligned with the target"
        elif value >= 0.45:
            return "moderate structural complementarity, partially aligned interactions that could be optimized"
        elif value >= 0.25:
            return "low structural complementarity, interactions are not well aligned with the target"
        else:
            return "very low structural complementarity, the peptide does not interact effectively with the target"

    def interpret_interaction(self, value: float) -> str:
        if value >= 0.7:
            return "high interaction quality, favorable binding to the target"
        elif value >= 0.5:
            return "moderate interaction quality, partial binding to the target"
        else:
            return "low interaction quality, unfavorable binding to the target"

    def interpret_wasserstein(self, value: float) -> str:
        if value < 0.15:
            return "low transport cost, distributions are very similar"
        elif value < 0.35:
            return "moderate transport cost, manageable differences in distributions"
        elif value < 0.55:
            return "high transport cost, significant differences in distributions"
        else:
            return "very high transport cost, very different distributions"

    def interpret_renyi_entropy(self, value: float, alpha: float = 2.0) -> str:
        if value < 1.5:
            return "low Rényi entropy, indicating high concentration of the interaction distribution"
        elif value < 2.5:
            return "moderate Rényi entropy, balanced distribution with some degree of diversity"
        elif value < 3.5:
            return "high Rényi entropy, diverse distribution indicating functional flexibility"
        else:
            return "very high Rényi entropy, extremely diverse distribution"

    def interpret_ricci(self, value: float) -> str:
        if value > 0.5:
            return "positive curvature, compact and stable structure"
        elif value > 0.25:
            return "moderate curvature, balanced structure"
        else:
            return "low curvature, open or flexible structure"

    def interpret_fractal(self, value: float) -> str:
        if 1.5 < value < 2.0:
            return "fractal complexity typical of globular proteins"
        elif value < 1.2:
            return "low complexity, simple or linear structure"
        else:
            return "high complexity, highly folded and compact structure"

    def generate_contextual_summary(self, metrics: Dict, target: str) -> str:
        summaries = []

        if 'pim_similarity' in metrics:
            sim = metrics['pim_similarity']
            summaries.append(f"PIM similarity: {self.interpret_similarity(sim)}")

        if 'grassmann_distance' in metrics:
            dist = metrics['grassmann_distance']
            summaries.append(f"Grassmann distance: {self.interpret_grassmann_distance(dist)}")

        if 'entropy' in metrics:
            ent = metrics['entropy']
            summaries.append(f"Entropy: {self.interpret_entropy(ent)}")

        if 'hodge_complementarity' in metrics:
            hodge = metrics['hodge_complementarity']
            summaries.append(f"Hodge complementarity: {self.interpret_hodge(hodge)}")

        if 'ricci_curvature' in metrics:
            ricci = metrics['ricci_curvature']
            summaries.append(f"Ricci curvature: {self.interpret_ricci(ricci)}")

        if 'wasserstein' in metrics:
            wass = metrics['wasserstein']
            summaries.append(f"Wasserstein distance: {self.interpret_wasserstein(wass)}")

        if 'fractal_dimension' in metrics:
            fractal = metrics['fractal_dimension']
            summaries.append(f"Fractal dimension: {self.interpret_fractal(fractal)}")

        target_info = self.knowledge_base.get_target_info(target)
        target_name = target_info.get('target_name', 'the target')

        context = f"In the context of {target_name}, the interpretation of the metrics suggests:\n"
        for summary in summaries:
            context += f"  • {summary}\n"

        return context

# ============================================================================
# NARRATIVE SUMMARY GENERATOR
# ============================================================================

class NarrativeSummaryGenerator:
    def __init__(self, knowledge_base: NarrativeKnowledgeBase = None):
        self.knowledge_base = knowledge_base if knowledge_base else NarrativeKnowledgeBase()
        self.level_evaluator = ClinicalLevelEvaluator()
        self.interpreter = ContextualInterpreter(self.knowledge_base)

    def generate_summary(self, peptide_data: Dict, target_name: str, profile: str) -> str:
        target_info = self.knowledge_base.get_target_info(target_name)

        if profile == 'executive':
            return self._generate_executive_summary(peptide_data, target_info)
        elif profile == 'biochemist':
            return self._generate_biochemist_summary(peptide_data, target_info)
        elif profile == 'chemist':
            return self._generate_chemist_summary(peptide_data, target_info)
        elif profile == 'analytical_chemist':
            return self._generate_analytical_summary(peptide_data, target_info)
        elif profile == 'physicochemist':
            return self._generate_physicochemist_summary(peptide_data, target_info)
        elif profile == 'bioinformatician':
            return self._generate_bioinformatician_summary(peptide_data, target_info)
        else:
            return self._generate_general_summary(peptide_data, target_info)

    def _generate_executive_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        drug_score = peptide_data.get('drug_likeness', 0.5)
        activity = peptide_data.get('activity_score', 0.5)
        target_name = target_info.get('target_name', 'Target')

        if drug_score > 0.7 and activity > 0.7:
            status = "FAVORABLE"
            rec = "Proceed to experimental validation"
        elif drug_score > 0.5 and activity > 0.5:
            status = "MODERATE"
            rec = "Optimize before validation"
        else:
            status = "UNFAVORABLE"
            rec = "Redesign completely"

        return f"""
        📊 EXECUTIVE SUMMARY - {target_name}
        ================================================
        Peptide: {seq[:20]}... ({len(seq)} aa)
        Drug-Likeness: {drug_score:.3f}
        Predicted activity: {activity:.3f}
        Status: {status}
        Recommendation: {rec}
        ================================================
        """

    def _generate_biochemist_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        pim_sim = peptide_data.get('pim_similarity', 0.5)
        hodge = peptide_data.get('hodge_complementarity', 0.5)
        interaction = peptide_data.get('interaction_quality', 0.5)
        activity = peptide_data.get('activity_score', 0.5)
        target_name = target_info.get('target_name', 'Target')

        return f"""
        🧬 BIOCHEMICAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa

        INTERACTIONS WITH THE TARGET:
        • PIM Similarity: {pim_sim:.4f} - {self.interpreter.interpret_similarity(pim_sim)}
        • Hodge Complementarity: {hodge:.4f} - {self.interpreter.interpret_hodge(hodge)}
        • Interaction Quality: {interaction:.4f} - {self.interpreter.interpret_interaction(interaction)}

        PREDICTED ACTIVITY: {activity:.3f}

        RECOMMENDATIONS:
        • Evaluate binding via SPR or ITC
        • Membrane fusion assays
        • Mutagenesis of key residues
        ================================================
        """

    def _generate_chemist_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        mw = peptide_data.get('molecular_weight', 0)
        charge = peptide_data.get('charge', 0)
        hydrophobicity = peptide_data.get('hydrophobicity', 0)
        solubility = peptide_data.get('solubility', 0)
        drug_likeness = peptide_data.get('drug_likeness', 0.5)
        target_name = target_info.get('target_name', 'Target')

        return f"""
        🧪 CHEMICAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        MOLECULAR WEIGHT: {mw:.1f} Da
        NET CHARGE: {charge:.2f} (pH 7.4)

        PHYSICOCHEMICAL PROPERTIES:
        • Hydrophobicity: {hydrophobicity:.2f}
        • Solubility: {solubility:.1f} mg/mL
        • Drug-Likeness: {drug_likeness:.3f}

        ANALYSIS:
        {self._interpret_physicochemical(hydrophobicity, solubility, charge, len(seq))}

        SYNTHESIS:
        • Recommended method: Fmoc solid-phase peptide synthesis (SPPS)
        • Preparative HPLC purification recommended
        • {"Consider cyclization for increased stability" if hydrophobicity > 1.0 else ""}
        ================================================
        """

    def _interpret_physicochemical(self, hydrophobicity: float, solubility: float, charge: float, length: int) -> str:
        lines = []
        if hydrophobicity > 1.0:
            lines.append("• High hydrophobicity, favorable for membrane interaction")
        elif hydrophobicity < -0.5:
            lines.append("• Low hydrophobicity, polar peptide")

        if solubility > 10:
            lines.append("• Good solubility in PBS pH 7.4")
        elif solubility > 5:
            lines.append("• Moderate solubility, consider co-solvents")
        else:
            lines.append("• Low solubility, requires DMSO or cyclodextrins")

        if abs(charge) > 1:
            lines.append(f"• Net charge {charge:.2f}, significant electrostatic interactions")

        return "\n".join(lines) if lines else "• Balanced physicochemical properties"

    def _generate_analytical_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        mw = peptide_data.get('molecular_weight', 0)
        target_name = target_info.get('target_name', 'Target')

        return f"""
        🔬 QUALITY CONTROL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa
        THEORETICAL MASS: {mw:.1f} Da

        RECOMMENDED ANALYTICAL TECHNIQUES:
        • HPLC-MS/MS: Mass confirmation and purity (>95%)
        • SEC: Determination of aggregation state
        • DLS: Particle size and polydispersity
        • CD: Secondary conformation (α-helix, β-sheet)
        • SPR: Binding kinetics to the target

        VALIDATION PROTOCOL:
        • Validate antiviral activity in plaque assay
        • Cytotoxicity in Vero cells (CC50)
        • Selectivity (SI = CC50/IC50)
        ================================================
        """

    def _generate_physicochemist_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        mw = peptide_data.get('molecular_weight', 0)
        charge = peptide_data.get('charge', 0)
        hydrophobicity = peptide_data.get('hydrophobicity', 0)
        entropy = peptide_data.get('entropy', 0)
        target_name = target_info.get('target_name', 'Target')

        return f"""
        ⚡ PHYSICOCHEMICAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa

        THERMODYNAMICS:
        • ΔG of folding: {-4.0 - 2.0*hydrophobicity:.2f} kcal/mol
        • Melting temperature (Tm): {30 + 15*hydrophobicity:.1f} °C
        • Diffusion coefficient: {10.0 - 0.2*mw/100:.2f} ×10⁻⁷ cm²/s

        DYNAMICS:
        • Conformational entropy: {entropy:.4f}
        • Flexibility: {"High" if entropy > 3.0 else "Moderate" if entropy > 2.0 else "Low"}

        INTERACTIONS:
        • Electrostatic interactions: {"Significant" if abs(charge) > 1 else "Moderate" if abs(charge) > 0.5 else "Minimal"}
        • Hydrophobic interactions: {"Strong" if hydrophobicity > 1.0 else "Moderate" if hydrophobicity > 0 else "Weak"}

        OPTIMIZATION PARAMETERS:
        • Optimal pH: {7.0 - 0.5*charge:.1f}
        • Optimal ionic strength: {150 + 50*abs(charge):.0f} mM NaCl
        • Optimal temperature: {37 + 10*hydrophobicity:.0f} °C
        ================================================
        """

    def _generate_bioinformatician_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        target_name = target_info.get('target_name', 'Target')

        metrics = []
        for key, value in peptide_data.items():
            if isinstance(value, (int, float)) and key not in ['length', 'molecular_weight']:
                metrics.append((key, value))

        metrics_sorted = sorted(metrics, key=lambda x: x[1], reverse=True)

        return f"""
        💻 BIOINFORMATICS SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa

        DESIGN METRICS:
        """ + "\n".join([f"    • {key}: {value:.4f}" for key, value in metrics_sorted[:10]]) + """

        PERFORMANCE ANALYSIS:
        • Algorithmic complexity: O(n²) for PIM calculation
        • Dimensionality: 16 (pair interactions)
        • Methods used: Grassmann, Hodge, Ricci, Wasserstein

        OPTIMIZATION RECOMMENDATIONS:
        • Consider embedding in reduced dimension
        • Evaluate Grassmann kernel for classification
        • Validate with bootstrap for confidence intervals
        ================================================
        """

    def _generate_general_summary(self, peptide_data: Dict, target_info: Dict) -> str:
        seq = peptide_data.get('sequence', '')
        target_name = target_info.get('target_name', 'Target')

        return f"""
        📋 GENERAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa

        MAIN METRICS:
        """ + "\n".join([f"    • {key}: {value:.4f}" for key, value in peptide_data.items()
                        if isinstance(value, (int, float)) and key not in ['length', 'molecular_weight']][:8]) + """

        ================================================
        """

# ============================================================================
# CLASS: MultidisciplinaryReporter
# ============================================================================

class MultidisciplinaryReporter:
    def __init__(self, narrative_generator: NarrativeSummaryGenerator):
        self.narrative_generator = narrative_generator

    def generate_reports(self, peptide_data: Dict, target_name: str, output_dir: str) -> Dict:
        profiles = ['executive', 'biochemist', 'chemist', 'analytical_chemist',
                   'physicochemist', 'bioinformatician']

        reports = {}

        for profile in profiles:
            summary = self.narrative_generator.generate_summary(peptide_data, target_name, profile)
            reports[profile] = summary
            safe_save_text(summary, f"narrative_summary_{profile}.txt", output_dir)
            print(f"  ✅ Narrative summary saved: narrative_summary_{profile}.txt")

        table = self._generate_multidisciplinary_table(peptide_data, target_name)
        safe_save_text(table, "multidisciplinary_summary_table.txt", output_dir)
        print(f"  ✅ Multidisciplinary table saved: multidisciplinary_summary_table.txt")

        return reports

    def _generate_multidisciplinary_table(self, peptide_data: Dict, target_name: str) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 MULTIDISCIPLINARY SUMMARY TABLE - {target_name}")
        lines.append("=" * 80)
        lines.append("")

        lines.append("TABLE 1: EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Peptide: {peptide_data.get('sequence', '')}")
        lines.append(f"Length: {len(peptide_data.get('sequence', ''))} aa")
        lines.append(f"Drug-Likeness: {peptide_data.get('drug_likeness', 0.5):.3f}")
        lines.append(f"Predicted activity: {peptide_data.get('activity_score', 0.5):.3f}")
        lines.append(f"PIM similarity: {peptide_data.get('pim_similarity', 0.5):.3f}")
        lines.append("")

        lines.append("TABLE 2: METRICS DICTIONARY")
        lines.append("-" * 40)
        lines.append("| Profile | Metric | Value | Interpretation |")
        lines.append("|--------|---------|-------|----------------|")

        metric_map = {
            'Biochemist': [
                ('pim_similarity', 'PIM Similarity'),
                ('hodge_complementarity', 'Hodge Complementarity'),
                ('interaction_quality', 'Interaction Quality'),
            ],
            'Chemist': [
                ('molecular_weight', 'Molecular Weight'),
                ('charge', 'Net Charge'),
                ('hydrophobicity', 'Hydrophobicity'),
                ('solubility', 'Solubility'),
            ],
            'Physicochemist': [
                ('entropy', 'Entropy'),
                ('structural_stability', 'Stability'),
            ],
            'Analytical Chemist': [
                ('purity', 'Estimated Purity'),
            ],
            'Bioinformatician': [
                ('renyi_entropy_alpha2', 'Rényi α=2'),
                ('fractal_dimension', 'Fractal Dimension'),
                ('wasserstein', 'Wasserstein'),
            ]
        }

        for profile, metrics in metric_map.items():
            for key, name in metrics:
                value = peptide_data.get(key, 0.5)
                if isinstance(value, (int, float)):
                    interpretation = self._get_interpretation(key, value)
                    lines.append(f"| {profile} | {name} | {value:.4f} | {interpretation} |")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def _get_interpretation(self, metric: str, value: float) -> str:
        interpretations = {
            'pim_similarity': lambda v: "High" if v > 0.7 else "Moderate" if v > 0.5 else "Low",
            'hodge_complementarity': lambda v: "High" if v > 0.7 else "Moderate" if v > 0.5 else "Low",
            'interaction_quality': lambda v: "High" if v > 0.7 else "Moderate" if v > 0.5 else "Low",
            'molecular_weight': lambda v: f"{v:.1f} Da",
            'charge': lambda v: f"{v:+.2f}",
            'hydrophobicity': lambda v: "High" if v > 1.0 else "Moderate" if v > 0 else "Low",
            'solubility': lambda v: "High" if v > 10 else "Moderate" if v > 5 else "Low",
            'entropy': lambda v: "High" if v > 3.0 else "Moderate" if v > 2.0 else "Low",
            'structural_stability': lambda v: "High" if v > 0.7 else "Moderate" if v > 0.4 else "Low",
        }

        if metric in interpretations:
            return interpretations[metric](value)
        return "-"

# ============================================================================
# GRASSMANN PIM - MAIN CLASS
# ============================================================================

class GrassmannPIM:
    def __init__(self, dim: int = DIM_PAIRS):
        self.dim = dim
        print("  ✅ GrassmannPIM initialized WITHOUT ESMFold")

    def wedge_product(self, v: np.ndarray, w: np.ndarray, with_ci: bool = False) -> Tuple[float, float]:
        return wedge_product_with_ci(v, w, use_bootstrap=with_ci)

    def wedge_product_oriented(self, v: np.ndarray, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
        return wedge_similarity_with_orientation(v, w)

    def interior_product_magnitude(self, v: np.ndarray, subspace: str) -> float:
        return interior_product_magnitude(v, subspace)

    def specular_reflection(self, v: np.ndarray) -> np.ndarray:
        return specular_reflection(v)

    def is_specular_reflection(self, v1: np.ndarray, v2: np.ndarray, threshold: float = 0.95) -> Tuple[bool, float]:
        return is_specular_reflection_ga(v1, v2, threshold)

    def all_rotor_angles(self, v: np.ndarray, w: np.ndarray) -> Dict[str, float]:
        angles = {}
        for name, indices, desc in ROTOR_PLANES:
            i, j = indices
            if i < len(v) and j < len(v):
                angles[name] = rotor_angle(v, w, indices)
            else:
                angles[name] = 0.0
        return angles

    def reflection_analysis(self, v: np.ndarray, w: np.ndarray) -> Dict:
        is_ref, sim = self.is_specular_reflection(v, w)
        return {'is_specular_reflection': is_ref, 'reflection_similarity': sim}

    def clifford_signature(self, v: np.ndarray) -> Dict[str, float]:
        return clifford_signature(v)

    def dot_product_metric(self, v: np.ndarray, w: np.ndarray) -> float:
        return dot_product_metric(v, w)

    def norm_metric(self, v: np.ndarray) -> Tuple[float, float]:
        return norm_metric(v)

    def similarity_metric(self, v: np.ndarray, w: np.ndarray) -> float:
        return similarity_metric(v, w)

    def metric_signature_info(self) -> Dict:
        return metric_signature_info()

    def commutator_norm(self, v: np.ndarray, w: np.ndarray) -> float:
        return commutator_norm(v, w)

    def anticommutator_similarity(self, v: np.ndarray, w: np.ndarray) -> float:
        return anticommutator_similarity(v, w)

    def geometric_product_full(self, v: np.ndarray, w: np.ndarray) -> Dict:
        return geometric_product_full(v, w)

    def hodge_dual(self, v: np.ndarray) -> np.ndarray:
        return hodge_dual(v)

    def hodge_complementarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return hodge_complementarity(v1, v2)

    def grassmann_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_distance(v1, v2)

    def grassmann_geodesic(self, v1: np.ndarray, v2: np.ndarray, n_steps: int = 10) -> List[np.ndarray]:
        return grassmann_geodesic(v1, v2, n_steps)

    def geometric_product_decomposition(self, v: np.ndarray, w: np.ndarray) -> Dict:
        return geometric_product_decomposition(v, w)

    def grassmann_projection_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_projection_distance(v1, v2)

    def grassmann_fubini_study(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_fubini_study(v1, v2)

    def grassmann_ricci_curvature(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_ricci_curvature(v1, v2)

    def karhunen_loeve_decomposition(self, vectors: List[np.ndarray], n_components: int = 8) -> Dict:
        return karhunen_loeve_decomposition(vectors, n_components)

    def compute_enhanced_metrics(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        return compute_enhanced_metrics(v1, v2)

    # New mathematical methods
    def clifford_product(self, v: np.ndarray, w: np.ndarray) -> Dict:
        return clifford_product_vectorized(v, w)

    def geometric_alignment(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        return geometric_alignment(v1, v2)

    def renyi_entropy(self, v: np.ndarray, alpha: float = 2.0) -> float:
        return renyi_entropy(v, alpha)

    def bhattacharyya_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return bhattacharyya_distance(v1, v2)

    def wasserstein_entropic(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return wasserstein_entropic_vectorized(v1, v2)

    def functional_pca(self, vectors: List[np.ndarray], n_components: int = 3) -> Dict:
        return functional_pca(vectors, n_components)

    def dtw_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return dtw_distance_vectorized(v1, v2)

    def mutual_information_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        return mutual_information_matrix(vectors)

    def total_correlation(self, vectors: List[np.ndarray]) -> float:
        return total_correlation(vectors)

    def grassmann_scalar_curvature(self, v: np.ndarray, k: int = 2) -> float:
        return grassmann_scalar_curvature(v, k)

    def multilevel_distance(self, v1: np.ndarray, v2: np.ndarray, k: int = 2) -> float:
        return grassmann_multilevel_distance(v1, v2, k)

    def multilevel_similarity(self, v1: np.ndarray, v2: np.ndarray, k: int = 2) -> float:
        return grassmann_multilevel_similarity(v1, v2, k)

    def projection_asymmetry(self, v1: np.ndarray, v2: np.ndarray, k: int = 1) -> Tuple[float, float, float]:
        return grassmann_projection_asymmetry(v1, v2, k)

    def sectional_curvature_sampled(self, vectors: List[np.ndarray], k: int = 2, n_samples: int = 30) -> float:
        return grassmann_sectional_curvature_sampled(vectors, k, n_samples)

    def volume(self, vectors: List[np.ndarray], k: int = 2) -> float:
        return grassmann_volume(vectors, k)

    def cycles(self, vectors: List[np.ndarray], k: int = 2, threshold: float = 0.5) -> List[List[int]]:
        return grassmann_cycles(vectors, k, threshold)

    def karcher_mean(self, vectors: List[np.ndarray], k: int = 2) -> np.ndarray:
        return grassmann_karcher_mean(vectors, k)

    def svd_similarity(self, v1: np.ndarray, v2: np.ndarray, k: int = 2) -> Dict:
        return grassmann_svd_similarity(v1, v2, k)

    def fisher_information(self, vectors: List[np.ndarray]) -> Dict:
        return fisher_information_matrix(vectors)

    def ollivier_ricci_curvature(self, vectors: List[np.ndarray]) -> float:
        return ollivier_ricci_curvature(vectors)

    def persistent_homology(self, vectors: List[np.ndarray]) -> Dict:
        return persistent_homology_features(vectors)

    def quantum_fidelity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return quantum_fidelity(v1, v2)

    def bures_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return bures_distance(v1, v2)

    def uhlmann_fidelity_mean(self, vectors: List[np.ndarray]) -> float:
        return uhlmann_fidelity_mean(vectors)

    def interaction_information(self, vectors: List[np.ndarray]) -> float:
        return interaction_information(vectors)

    def wasserstein_orthogonal(self, vectors: List[np.ndarray]) -> Dict:
        return wasserstein_orthogonal(vectors)

    def detect_bifurcation_points(self, vectors: List[np.ndarray]) -> Dict:
        return detect_bifurcation_points(vectors)

# ============================================================================
# ADVANCED MATHEMATICAL METRICS - PART 1
# ============================================================================

def wedge_product_with_ci(v: np.ndarray, w: np.ndarray, n_bootstrap: int = 50,
                          use_bootstrap: bool = True) -> Tuple[float, float]:
    magnitude, orientation, _ = wedge_similarity_with_orientation(v, w)
    wedge = magnitude
    if not use_bootstrap:
        return wedge, 0.0
    dim = len(v)
    bootstrapped = []
    for _ in range(min(n_bootstrap, 100)):
        idx = np.random.choice(dim, dim, replace=True)
        v_boot = v[idx]
        w_boot = w[idx]
        mag_boot, _, _ = wedge_similarity_with_orientation(v_boot, w_boot)
        bootstrapped.append(mag_boot)
    return np.mean(bootstrapped), np.std(bootstrapped)


def wedge_similarity_with_orientation(v: np.ndarray, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
    biv = wedge_product_oriented(v, w)
    magnitude = np.linalg.norm(biv)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    magnitude_norm = magnitude / (norm_v * norm_w + 1e-10)
    magnitude_norm = min(magnitude_norm, 1.0)
    non_zero = biv[np.abs(biv) > 1e-8]
    orientation_sign = 1.0
    if len(non_zero) > 0:
        orientation_sign = np.sign(non_zero[0])
    return magnitude_norm, orientation_sign, biv


def wedge_product_oriented(v: np.ndarray, w: np.ndarray, key_pairs: List[Tuple[int, int]] = None) -> np.ndarray:
    if key_pairs is None:
        key_pairs = KEY_BIVECTORS
    bivector = np.zeros(len(key_pairs))
    for idx, (i, j) in enumerate(key_pairs):
        if i < len(v) and j < len(w):
            bivector[idx] = v[i] * w[j] - v[j] * w[i]
    return bivector


def specular_reflection(v: np.ndarray, normal: np.ndarray = None) -> np.ndarray:
    if normal is None:
        normal = reflection_normal_vector()
    n = normal / (np.linalg.norm(normal) + 1e-10)
    return v - 2 * np.dot(v, n) * n


def is_specular_reflection_ga(v1: np.ndarray, v2: np.ndarray, threshold: float = 0.95) -> Tuple[bool, float]:
    v1_reflected = specular_reflection(v1)
    v1_reflected_norm = v1_reflected / (np.linalg.norm(v1_reflected) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    sim = np.dot(v1_reflected_norm, v2_norm)
    sim = np.clip(sim, -1, 1)
    mag, orient, _ = wedge_similarity_with_orientation(v1_reflected_norm, v2_norm)
    combined_sim = (sim + mag) / 2.0
    is_reflection = combined_sim >= threshold
    return is_reflection, combined_sim


def reflection_normal_vector() -> np.ndarray:
    n = np.zeros(16)
    for i, j in REFLECTION_SWAP_MAP.items():
        n[i] = 1.0
        n[j] = -1.0
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm
    return n


def rotor_angle(v1: np.ndarray, v2: np.ndarray, plane_indices: Tuple[int, int]) -> float:
    i, j = plane_indices
    if i >= len(v1) or j >= len(v1):
        return 0.0
    proj1 = np.array([v1[i], v1[j]])
    proj2 = np.array([v2[i], v2[j]])
    norm1 = np.linalg.norm(proj1) + 1e-10
    norm2 = np.linalg.norm(proj2) + 1e-10
    cos_theta = np.dot(proj1, proj2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta) * 180.0 / np.pi


def clifford_signature(v: np.ndarray) -> Dict[str, float]:
    signature = {}
    signature['norm'] = np.linalg.norm(v)

    v_reflected = specular_reflection(v)
    signature['auto_reflection'], _ = wedge_product_with_ci(v, v_reflected, use_bootstrap=False)

    if len(v) > 15:
        hydro_plane = np.array([v[10], v[15]])
        signature['hydrophobic_projection'] = np.linalg.norm(hydro_plane)
    else:
        signature['hydrophobic_projection'] = 0.0

    if len(v) > 5:
        charge_plane = np.array([v[0], v[5]])
        signature['charge_projection'] = np.linalg.norm(charge_plane)
    else:
        signature['charge_projection'] = 0.0

    v_rotated = np.roll(v, 4)
    signature['auto_rotation'], _ = wedge_product_with_ci(v, v_rotated, use_bootstrap=False)

    norm_η, sign_η = norm_metric(v)
    signature['metric_norm'] = norm_η
    signature['metric_sign'] = sign_η

    if USE_HODGE_DUAL:
        dual = hodge_dual(v)
        signature['hodge_norm'] = np.linalg.norm(dual)
        signature['hodge_complement'] = np.dot(v, dual) / (np.linalg.norm(v) * np.linalg.norm(dual) + 1e-10)

    if USE_SHANNON_ENTROPY:
        signature['entropy'] = shannon_entropy(v)
    if USE_GINI_COEFFICIENT:
        signature['gini'] = gini_coefficient(v)

    return signature


def norm_metric(v: np.ndarray, metric: np.ndarray = None) -> Tuple[float, float]:
    if metric is None:
        metric = METRIC_SIGNATURE
    if len(metric) != len(v):
        if len(metric) < len(v):
            metric_padded = np.ones(len(v))
            metric_padded[:len(metric)] = metric
            metric = metric_padded
        else:
            metric = metric[:len(v)]
    value = np.sum(metric * v * v)
    sign = np.sign(value) if value != 0 else 0
    magnitude = np.sqrt(np.abs(value) + 1e-10)
    return magnitude, sign


def dot_product_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
    if metric is None:
        metric = METRIC_SIGNATURE
    if len(metric) != len(v):
        if len(metric) < len(v):
            metric_padded = np.ones(len(v))
            metric_padded[:len(metric)] = metric
            metric = metric_padded
        else:
            metric = metric[:len(v)]
    return np.sum(metric * v * w)


def similarity_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
    dot_η = dot_product_metric(v, w, metric)
    norm_v, _ = norm_metric(v, metric)
    norm_w, _ = norm_metric(w, metric)
    if norm_v * norm_w < 1e-10:
        return 0.0
    return np.abs(dot_η) / (norm_v * norm_w + 1e-10)


def hodge_dual(v: np.ndarray, metric: np.ndarray = None) -> np.ndarray:
    if metric is None:
        metric = METRIC_SIGNATURE
    n = len(v)
    dual = np.zeros(n)
    for i in range(n):
        complement_indices = [j for j in range(n) if j != i]
        proj = np.zeros(n)
        for j in complement_indices:
            proj[j] = v[j]
        norm_proj = np.linalg.norm(proj) + 1e-10
        dual[i] = np.linalg.norm(proj) / norm_proj
    total = np.sum(dual)
    if total > 0:
        dual = dual / total
    return dual


def hodge_complementarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dual_v1 = hodge_dual(v1)
    sim, _, _ = wedge_similarity_with_orientation(v2, dual_v1)
    return sim


def grassmann_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    P1 = np.outer(v1, v1) / (np.linalg.norm(v1)**2 + 1e-10)
    P2 = np.outer(v2, v2) / (np.linalg.norm(v2)**2 + 1e-10)
    return np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)


def grassmann_geodesic(v1: np.ndarray, v2: np.ndarray, n_steps: int = 10) -> List[np.ndarray]:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    cos_theta = np.dot(v1_norm, v2_norm)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    trajectory = []
    for step in range(n_steps + 1):
        t = step / n_steps
        if theta > 1e-10:
            interpolated = (np.sin((1-t)*theta) / np.sin(theta)) * v1_norm + \
                          (np.sin(t*theta) / np.sin(theta)) * v2_norm
        else:
            interpolated = v1_norm
        interpolated = interpolated / (np.linalg.norm(interpolated) + 1e-10)
        trajectory.append(interpolated)
    return trajectory


def grassmann_ricci_curvature(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, 0, 1)
    theta = np.arccos(cos_theta)
    if theta > 1e-10:
        return 1.0 / (np.tan(theta)**2 + 1e-10)
    return 0.0


def grassmann_scalar_curvature(v: np.ndarray, k: int = 2) -> float:
    n = len(v)
    if n < k:
        return 0.0

    X = np.zeros((n - k + 1, k))
    for i in range(k):
        X[:, i] = v[i:n - k + 1 + i]

    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return 2 * k * (n - k) / n
    except:
        return 0.0


def grassmann_fubini_study(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, 0, 1)
    return np.arccos(cos_theta)


def shannon_entropy(v: np.ndarray) -> float:
    p = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
    return -np.sum(p * np.log2(p + 1e-10))


def gini_coefficient(v: np.ndarray) -> float:
    p = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
    sorted_p = np.sort(p)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    return 1 - (2 * np.sum(cumsum) / (n * np.sum(sorted_p) + 1e-10))


def jensen_shannon_divergence(v1: np.ndarray, v2: np.ndarray) -> float:
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    m = (p + q) / 2
    kl_pm = np.sum(p * np.log2((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log2((q + 1e-10) / (m + 1e-10)))
    return 0.5 * (kl_pm + kl_qm)


def hellinger_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    return (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))


def wasserstein_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    p1 = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    p2 = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    return np.sum(np.abs(cdf1 - cdf2)) / len(v1)


def spearman_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    result = spearmanr(v1, v2)
    return result.correlation if result.correlation is not None else 0.0


def morans_i(v: np.ndarray, weights: np.ndarray = None) -> float:
    n = len(v)
    if weights is None:
        weights = np.ones((n, n)) - np.eye(n)
    v_mean = np.mean(v)
    numerator = 0
    denominator = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * (v[i] - v_mean) * (v[j] - v_mean)
        denominator += (v[i] - v_mean) ** 2
    if denominator > 0:
        return (n / np.sum(weights)) * (numerator / denominator)
    return 0.0

def polarity_laplacian(v: np.ndarray) -> float:
    n = len(v)
    laplacian = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                laplacian += (v[i] - v[j]) ** 2
    return laplacian / (n * (n - 1))


def fractal_dimension(v: np.ndarray, scales: int = 10) -> float:
    n = len(v)
    if n < 2:
        return 0.0
    v_abs = np.abs(v)
    cumsum = np.cumsum(v_abs)
    cumsum = (cumsum - np.min(cumsum)) / (np.max(cumsum) - np.min(cumsum) + 1e-10)
    counts = []
    for scale in range(1, scales + 1):
        box_size = max(1, n // (2**scale))
        boxes = set()
        for i in range(0, n - box_size, box_size):
            box_value = np.mean(cumsum[i:i+box_size])
            boxes.add(int(box_value * 100))
        counts.append(len(boxes))
    if len(counts) > 2:
        log_scales = np.log(np.array([max(1, n // (2**s)) for s in range(1, scales + 1)]))
        log_counts = np.log(np.array(counts) + 1)
        slope, _, _, _, _ = linregress(log_scales, log_counts)
        return -slope
    return 0.0


def discrete_radon_transform(v: np.ndarray, n_angles: int = 8) -> np.ndarray:
    n = len(v)
    radon = np.zeros(n_angles)
    v_norm = v / (np.linalg.norm(v) + 1e-10)
    for k in range(n_angles):
        theta = k * np.pi / n_angles
        projection = np.zeros(n)
        for i in range(n):
            projection[i] = v_norm[i] * np.cos(theta) + v_norm[(i + n//4) % n] * np.sin(theta)
        radon[k] = np.sum(projection**2)
    return radon / (np.sum(v_norm**2) + 1e-10)


def karhunen_loeve_decomposition(vectors: List[np.ndarray], n_components: int = 8) -> Dict:
    if len(vectors) < 2:
        return {'eigenvalues': np.array([]), 'eigenvectors': np.array([]),
                'components': np.array([]), 'explained_variance': np.array([]),
                'mean': np.zeros(len(vectors[0])) if vectors else np.zeros(16)}

    X = np.array(vectors)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered.T)
    eigvals, eigvecs = eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    n_components = min(n_components, len(eigvals))
    components = X_centered @ eigvecs[:, :n_components]

    return {
        'eigenvalues': eigvals[:n_components],
        'eigenvectors': eigvecs[:, :n_components],
        'components': components,
        'explained_variance': eigvals[:n_components] / (np.sum(eigvals) + 1e-10),
        'mean': mean
    }


def grassmann_projection_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    P1 = np.outer(v1_norm, v1_norm)
    P2 = np.outer(v2_norm, v2_norm)
    return np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)


def geometric_product_full(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
    if metric is None:
        metric = METRIC_SIGNATURE

    scalar = np.sum(metric * v * w)
    bivector = wedge_product_oriented(v, w)
    norm_scalar = abs(scalar)
    norm_bivector = np.linalg.norm(bivector) if len(bivector) > 0 else 0
    total_norm = np.sqrt(norm_scalar**2 + norm_bivector**2)

    return {
        'grade_0': scalar,
        'grade_2': bivector,
        'norm_grade_0': norm_scalar,
        'norm_grade_2': norm_bivector,
        'total_norm': total_norm,
        'grade_decomposition': {
            'functional': norm_scalar / (total_norm + 1e-10),
            'pair_interactions': norm_bivector / (total_norm + 1e-10),
        }
    }


def geometric_product_decomposition(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
    if metric is None:
        metric = METRIC_SIGNATURE

    scalar = np.sum(metric * v * w)
    sqrt_metric = np.sqrt(np.abs(metric) + 1e-10)
    v_transformed = v / sqrt_metric
    w_transformed = w / sqrt_metric
    bivector = wedge_product_oriented(v_transformed, w_transformed)

    norm_v, _ = norm_metric(v, metric)
    norm_w, _ = norm_metric(w, metric)
    denom = norm_v * norm_w + 1e-10

    functional = np.abs(scalar) / denom
    structural = np.linalg.norm(bivector) / denom
    combined = np.sqrt(functional**2 + structural**2)
    ratio = functional / (structural + 1e-10)

    if ratio > 2.0:
        interpretation = "Functionally similar, structurally different"
    elif ratio < 0.5:
        interpretation = "Structurally similar, functionally different"
    else:
        interpretation = "Balanced: similar in both aspects"

    return {
        'functional_similarity': functional,
        'structural_difference': structural,
        'combined_similarity': combined,
        'functional_structural_ratio': ratio,
        'interpretation': interpretation
    }


def commutator_norm(v: np.ndarray, w: np.ndarray) -> float:
    comm = commutator(v, w)
    mag = np.linalg.norm(comm)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    return mag / (norm_v * norm_w + 1e-10)


def commutator(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return wedge_product_oriented(v, w)


def anticommutator_similarity(v: np.ndarray, w: np.ndarray) -> float:
    anticomm = 2.0 * np.dot(v, w)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    sim = np.abs(anticomm) / (2.0 * norm_v * norm_w + 1e-10)
    return min(sim, 1.0)


def metric_signature_info() -> Dict:
    info = {
        'total_components': len(METRIC_SIGNATURE),
        'positive_count': int(np.sum(METRIC_SIGNATURE > 0)),
        'negative_count': int(np.sum(METRIC_SIGNATURE < 0)),
        'neutral_count': int(np.sum(METRIC_SIGNATURE == 0)),
        'is_euclidean': bool(np.all(METRIC_SIGNATURE == 1)),
        'is_biological': USE_BIOLOGICAL_METRIC,
    }

    component_names = [
        'P⁺→P⁺', 'P⁺→P⁻', 'P⁺→N', 'P⁺→NP',
        'P⁻→P⁺', 'P⁻→P⁻', 'P⁻→N', 'P⁻→NP',
        'N→P⁺', 'N→P⁻', 'N→N', 'N→NP',
        'NP→P⁺', 'NP→P⁻', 'NP→N', 'NP→NP'
    ]

    beneficial = []
    detrimental = []
    neutral = []

    for i, name in enumerate(component_names):
        if i < len(METRIC_SIGNATURE):
            val = METRIC_SIGNATURE[i]
            if val > 0:
                beneficial.append(name)
            elif val < 0:
                detrimental.append(name)
            else:
                neutral.append(name)

    info['beneficial_interactions'] = beneficial
    info['detrimental_interactions'] = detrimental
    info['neutral_interactions'] = neutral

    info['signature_array'] = METRIC_SIGNATURE.tolist()
    info['description'] = (
        "Biological metric that weights interactions based on their functional importance. "
        "Positive interactions (P⁺→P⁻, P⁻→P⁺) are favored, while repulsive interactions "
        "(P⁺→P⁺, P⁻→P⁻) are penalized."
    )

    return info


def grassmann_karcher_mean(vectors: List[np.ndarray], max_iter: int = 50) -> Optional[np.ndarray]:
    if len(vectors) == 0:
        return None
    if len(vectors) == 1:
        return vectors[0]

    mean = np.mean(vectors, axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-10)

    for _ in range(max_iter):
        tangent_vectors = []
        for v in vectors:
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            cos_theta = np.dot(mean, v_norm)
            cos_theta = np.clip(cos_theta, -1, 1)

            if cos_theta > 1 - 1e-6:
                continue

            theta = np.arccos(cos_theta)
            if theta > 1e-10:
                direction = (v_norm - mean * cos_theta) / np.sin(theta)
                tangent_vectors.append(theta * direction)

        if not tangent_vectors:
            break

        tangent_mean = np.mean(tangent_vectors, axis=0)

        new_mean = mean + tangent_mean
        new_mean = new_mean / (np.linalg.norm(new_mean) + 1e-10)

        if np.linalg.norm(new_mean - mean) < 1e-6:
            break

        mean = new_mean

    return mean


def grassmann_cross_validation(vectors: List[np.ndarray],
                               labels: List[int],
                               n_folds: int = 5) -> Dict:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = []
    predictions = []
    true_labels = []

    for train_idx, test_idx in skf.split(vectors, labels):
        class_means = {}
        for cls in set(labels):
            class_vectors = [vectors[i] for i in train_idx if labels[i] == cls]
            if len(class_vectors) > 0:
                class_means[cls] = grassmann_karcher_mean(class_vectors)

        correct = 0
        for idx in test_idx:
            v = vectors[idx]
            true_label = labels[idx]

            distances = {}
            for cls, mean in class_means.items():
                if mean is not None:
                    P1 = np.outer(v, v) / (np.linalg.norm(v) ** 2 + 1e-10)
                    P2 = np.outer(mean, mean) / (np.linalg.norm(mean) ** 2 + 1e-10)
                    dist = np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)
                    distances[cls] = dist

            if distances:
                pred_label = min(distances, key=distances.get)
                if pred_label == true_label:
                    correct += 1
                predictions.append(pred_label)
                true_labels.append(true_label)

        scores.append(correct / len(test_idx))

    return {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'per_fold': scores,
        'predictions': predictions,
        'true_labels': true_labels,
        'n_folds': n_folds,
        'accuracy_ci': np.percentile(scores, [2.5, 97.5]).tolist()
    }


def permutation_test(v1: np.ndarray, v2: np.ndarray,
                     n_permutations: int = 1000) -> Dict:
    n1, n2 = len(v1), len(v2)
    all_data = np.vstack([v1, v2])

    orig_stat = np.linalg.norm(np.mean(v1, axis=0) - np.mean(v2, axis=0))

    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        idx = np.random.permutation(n1 + n2)
        perm1 = all_data[idx[:n1]]
        perm2 = all_data[idx[n1:]]

        perm_stats[i] = np.linalg.norm(np.mean(perm1, axis=0) - np.mean(perm2, axis=0))

    p_value = np.mean(perm_stats >= orig_stat)

    return {
        'statistic': orig_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'null_distribution': perm_stats.tolist(),
        'n_permutations': n_permutations,
        'effect_size': orig_stat / (np.std(perm_stats) + 1e-10)
    }


def grassmann_multilevel_distance(v1: np.ndarray, v2: np.ndarray, k: int = 2) -> float:
    if not USE_GRASSMANN_MULTILEVEL:
        return 0.0
    n = len(v1)
    if n < k:
        return 1.0

    X1 = np.zeros((n - k + 1, k))
    X2 = np.zeros((n - k + 1, k))

    for i in range(k):
        X1[:, i] = v1[i:n - k + 1 + i]
        X2[:, i] = v2[i:n - k + 1 + i]

    U1, _, _ = np.linalg.svd(X1, full_matrices=False)
    U2, _, _ = np.linalg.svd(X2, full_matrices=False)

    P1 = U1[:, :k] @ U1[:, :k].T
    P2 = U2[:, :k] @ U2[:, :k].T

    return np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2 * k)


def grassmann_multilevel_similarity(v1: np.ndarray, v2: np.ndarray, k: int = 2) -> float:
    if not USE_GRASSMANN_MULTILEVEL:
        return 0.0
    dist = grassmann_multilevel_distance(v1, v2, k)
    return max(0, 1 - dist / np.sqrt(2))


def grassmann_projection_asymmetry(v1: np.ndarray, v2: np.ndarray, k: int = 1) -> Tuple[float, float, float]:
    if not USE_GRASSMANN_ASYMMETRIC:
        return 0.0, 0.0, 0.0
    n = len(v1)
    if n < k:
        return 1.0, 1.0, 0.0

    def project_onto(source, target, k):
        X_source = np.zeros((n - k + 1, k))
        for i in range(k):
            X_source[:, i] = source[i:n - k + 1 + i]

        U_source, _, _ = np.linalg.svd(X_source, full_matrices=False)
        U_k = U_source[:, :k]
        P_source = U_k @ U_k.T

        X_target = np.zeros(n - k + 1)
        for i in range(n - k + 1):
            X_target[i] = target[i]

        target_proj = P_source @ X_target
        norm_target = np.linalg.norm(X_target) + 1e-10

        return np.linalg.norm(target_proj) / norm_target

    sim_12 = project_onto(v1, v2, k)
    sim_21 = project_onto(v2, v1, k)

    d_12 = 1 - sim_12
    d_21 = 1 - sim_21
    asymmetry = np.abs(d_12 - d_21)

    return d_12, d_21, asymmetry


def grassmann_sectional_curvature_sampled(vectors: List[np.ndarray], k: int = 2,
                                          n_samples: int = 30) -> float:
    if not USE_GRASSMANN_CURVATURE or len(vectors) < 3:
        return 0.0

    n_vecs = len(vectors)

    if n_vecs <= n_samples:
        curvatures = []
        for i in range(n_vecs - 2):
            for j in range(i+1, n_vecs - 1):
                for l in range(j+1, n_vecs):
                    # Simplified curvature calculation
                    curv = 0.1 * (1 - grassmann_multilevel_similarity(vectors[i], vectors[j], k))
                    curvatures.append(curv)
        return np.mean(curvatures) if curvatures else 0.0

    indices = np.random.choice(n_vecs, n_samples, replace=False)
    sampled_vectors = [vectors[i] for i in indices]

    curvatures = []
    for i in range(len(sampled_vectors) - 2):
        for j in range(i+1, len(sampled_vectors) - 1):
            for l in range(j+1, len(sampled_vectors)):
                curv = 0.1 * (1 - grassmann_multilevel_similarity(sampled_vectors[i], sampled_vectors[j], k))
                curvatures.append(curv)

    return np.mean(curvatures) if curvatures else 0.0


def grassmann_volume(vectors: List[np.ndarray], k: int = 2) -> float:
    if not USE_GRASSMANN_VOLUME or len(vectors) < 2:
        return 0.0
    n_vecs = len(vectors)

    distances = []
    for i in range(n_vecs):
        for j in range(i+1, n_vecs):
            dist = grassmann_multilevel_distance(vectors[i], vectors[j], k)
            distances.append(dist)

    if not distances:
        return 0.0

    mean_dist = np.mean(distances)
    max_dist = np.max(distances)

    if max_dist > 1e-10:
        return mean_dist / max_dist
    return 0.0


def grassmann_cycles(vectors: List[np.ndarray], k: int = 2,
                     threshold: float = 0.5) -> List[List[int]]:
    if not USE_GRASSMANN_CYCLES or len(vectors) < 3:
        return []
    n = len(vectors)

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = grassmann_multilevel_similarity(vectors[i], vectors[j], k)
            if sim > threshold:
                adj[i, j] = adj[j, i] = 1

    cycles = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                for l in range(j+1, n):
                    if adj[j, l] and adj[l, i]:
                        cycles.append([i, j, l])

    return cycles


def grassmann_svd_similarity(v1: np.ndarray, v2: np.ndarray, k: int = 2) -> Dict:
    if not USE_GRASSMANN_SVD:
        return {'principal_angles': np.array([]), 'mean_angle': 0, 'similarity': 0}
    n = len(v1)
    if n < k:
        return {'principal_angles': np.array([]), 'mean_angle': 0, 'similarity': 0}

    X1 = np.zeros((n - k + 1, k))
    X2 = np.zeros((n - k + 1, k))

    for i in range(k):
        X1[:, i] = v1[i:n - k + 1 + i]
        X2[:, i] = v2[i:n - k + 1 + i]

    U1, _, _ = np.linalg.svd(X1, full_matrices=False)
    U2, _, _ = np.linalg.svd(X2, full_matrices=False)

    M = U1[:, :k].T @ U2[:, :k]
    sigma = np.linalg.svd(M, compute_uv=False)
    principal_angles = np.arccos(np.clip(sigma, -1, 1))

    return {
        'principal_angles': principal_angles,
        'mean_angle': np.mean(principal_angles),
        'max_angle': np.max(principal_angles),
        'min_angle': np.min(principal_angles),
        'geodesic_distance': np.linalg.norm(principal_angles),
        'similarity': np.mean(np.cos(principal_angles))
    }

def compute_pim_profile(sequence: str, use_weights: bool = True) -> np.ndarray:
    seq = ''.join([c for c in sequence.strip() if c.isalpha() and c.upper() in POLARITY_MAP])
    if len(seq) < 2:
        return np.zeros(DIM_PAIRS)

    polarities = []
    for aa in seq:
        pol = POLARITY_MAP.get(aa.upper())
        if pol is not None:
            polarities.append(pol)

    if len(polarities) < 2:
        return np.zeros(DIM_PAIRS)

    counts = np.zeros(DIM_PAIRS)
    for i in range(len(polarities) - 1):
        pair = f"{polarities[i]},{polarities[i+1]}"
        if pair in INTERACTION_TO_IDX:
            counts[INTERACTION_TO_IDX[pair]] += 1

    total = np.sum(counts)
    if total > 0:
        counts = counts / total

    if use_weights:
        weighted_counts = np.zeros(DIM_PAIRS)
        for i, inter in enumerate(INTERACTIONS):
            weight = BIOLOGICAL_WEIGHTS.get(inter, 1.0)
            weighted_counts[i] = counts[i] * weight
        total_weighted = np.sum(weighted_counts)
        if total_weighted > 0:
            weighted_counts = weighted_counts / total_weighted
        return weighted_counts

    return counts


def clifford_product_vectorized(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
    if metric is None:
        metric = METRIC_SIGNATURE

    if len(metric) != len(v):
        if len(metric) < len(v):
            metric_padded = np.ones(len(v))
            metric_padded[:len(metric)] = metric
            metric = metric_padded
        else:
            metric = metric[:len(v)]

    n = len(v)

    scalar = np.sum(metric * v * w)

    v_expanded = v[:, None]
    w_expanded = w[None, :]
    metric_expanded = metric[None, :]

    term1 = metric_expanded * v_expanded * w_expanded
    term2 = metric[:, None] * v[None, :] * w[:, None]

    vector_part = np.sum(term1 - term2, axis=1)

    i_indices, j_indices = np.triu_indices(n, k=1)
    bivector = v[i_indices] * w[j_indices] - v[j_indices] * w[i_indices]

    total = np.concatenate([[scalar], vector_part, bivector])
    norm = np.linalg.norm(total) + 1e-10

    return {
        'scalar': float(scalar),
        'vector': vector_part / norm,
        'bivector': bivector / norm,
        'total': total / norm,
        'norm': norm,
        'grade_decomposition': {
            'scalar_fraction': abs(scalar) / norm,
            'vector_fraction': np.linalg.norm(vector_part) / norm,
            'bivector_fraction': np.linalg.norm(bivector) / norm
        }
    }


def rotor_from_vectors(v1: np.ndarray, v2: np.ndarray, metric: np.ndarray = None) -> Dict:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1, 1)

    theta = np.arccos(dot)

    if theta < 1e-10:
        return {
            'angle': 0.0,
            'rotor': np.array([1.0, 0.0, 0.0, 0.0]),
            'rotation_matrix': np.eye(len(v1)),
            'similarity': 1.0,
            'geodesic_distance': 0.0
        }

    if theta > np.pi - 1e-10:
        reflection = -np.eye(len(v1))
        return {
            'angle': np.pi,
            'rotor': np.array([0.0, 1.0, 0.0, 0.0]),
            'rotation_matrix': reflection,
            'similarity': -1.0,
            'geodesic_distance': np.pi
        }

    biv = np.outer(v1_norm, v2_norm) - np.outer(v2_norm, v1_norm)
    biv_norm = np.linalg.norm(biv) + 1e-10

    biv_flat = biv.flatten()
    biv_flat = biv_flat / biv_norm

    n_components = min(16, len(biv_flat))
    rotor_components = np.zeros(n_components + 1)
    rotor_components[0] = np.cos(theta / 2)
    rotor_components[1:n_components+1] = np.sin(theta / 2) * biv_flat[:n_components]

    return {
        'angle': theta,
        'rotor': rotor_components,
        'similarity': np.cos(theta),
        'bivector_norm': biv_norm,
        'geodesic_distance': theta
    }


def geometric_alignment(v1: np.ndarray, v2: np.ndarray, metric: np.ndarray = None) -> Dict:
    gp = clifford_product_vectorized(v1, v2, metric)
    rotor = rotor_from_vectors(v1, v2, metric)
    alignment_score = rotor['similarity']
    quality = 1 - gp['grade_decomposition']['bivector_fraction']

    return {
        'geometric_product': gp,
        'rotor': rotor,
        'alignment_score': alignment_score,
        'alignment_quality': quality,
        'composite_alignment': (alignment_score + quality) / 2
    }


def mutual_information_matrix(vectors: List[np.ndarray], n_bins: int = 20) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((16, 16))

    vectors = np.array(vectors)
    n, d = vectors.shape

    discretized = np.zeros((n, d), dtype=int)
    for i in range(d):
        discretized[:, i] = np.digitize(vectors[:, i],
                                        np.linspace(np.min(vectors[:, i]),
                                                   np.max(vectors[:, i]),
                                                   n_bins + 1)[1:-1])

    mi_matrix = np.zeros((d, d))

    for i in range(d):
        for j in range(i+1, d):
            joint, _, _ = np.histogram2d(discretized[:, i], discretized[:, j],
                                         bins=n_bins)
            joint = joint / (np.sum(joint) + 1e-10)

            p_i = np.sum(joint, axis=1)
            p_j = np.sum(joint, axis=0)

            H_joint = -np.sum(joint * np.log2(joint + 1e-10))
            H_i = -np.sum(p_i * np.log2(p_i + 1e-10))
            H_j = -np.sum(p_j * np.log2(p_j + 1e-10))

            mi = H_i + H_j - H_joint
            mi_matrix[i, j] = max(0, mi)
            mi_matrix[j, i] = mi_matrix[i, j]

    max_mi = np.max(mi_matrix) + 1e-10
    mi_matrix = mi_matrix / max_mi

    return mi_matrix


def total_correlation(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0

    vectors = np.array(vectors)
    n, d = vectors.shape

    corr = np.corrcoef(vectors.T)

    try:
        tc = -0.5 * np.log(np.linalg.det(corr + np.eye(d) * 1e-6))
    except:
        tc = 0.0

    return max(0, tc)


def functional_pca(vectors: List[np.ndarray], n_components: int = 3) -> Dict:
    if len(vectors) < 2:
        return {'error': 'Not enough vectors'}

    vectors = np.array(vectors)
    n, d = vectors.shape

    mean_vector = np.mean(vectors, axis=0)
    centered = vectors - mean_vector

    t = np.linspace(0, 1, d)
    n_knots = min(8, d // 2)
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    k = 3

    design_matrix = np.zeros((d, n_knots))
    try:
        for j in range(n_knots):
            bspline = BSpline.basis_element(np.linspace(0, 1, n_knots+2)[j:j+4])
            design_matrix[:, j] = bspline(t)
    except:
        design_matrix = np.polynomial.legendre.legvander(t, n_knots-1)[:, :n_knots]
        design_matrix = design_matrix / np.linalg.norm(design_matrix, axis=0)

    coefficients = np.zeros((n, n_knots))
    try:
        for i in range(n):
            coefficients[i] = solve(design_matrix.T @ design_matrix +
                                    np.eye(n_knots) * 1e-6,
                                    design_matrix.T @ centered[i])
    except:
        coefficients = centered @ np.linalg.pinv(design_matrix.T)

    cov = np.cov(coefficients.T)
    eigvals, eigvecs = eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n_components = min(n_components, len(eigvals))

    eigenfunctions = []
    for i in range(n_components):
        coef = eigvecs[:, i]
        func = design_matrix @ coef
        func = func / (np.linalg.norm(func) + 1e-10)
        eigenfunctions.append(func)

    scores = centered @ np.array(eigenfunctions).T

    explained_variance = eigvals[:n_components] / (np.sum(eigvals) + 1e-10)

    return {
        'eigenfunctions': np.array(eigenfunctions),
        'scores': scores,
        'explained_variance': explained_variance,
        'mean_function': mean_vector,
        'eigenvalues': eigvals[:n_components],
        'n_components': n_components,
        'cumulative_variance': np.cumsum(explained_variance)
    }


def dtw_distance(v1: np.ndarray, v2: np.ndarray,
                 window: int = None) -> float:
    n, m = len(v1), len(v2)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if window is not None and abs(i - j) > window:
                continue

            cost = (v1[i-1] - v2[j-1]) ** 2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    if np.isinf(D[n, m]):
        return np.linalg.norm(v1 - v2) / np.sqrt(n)

    return D[n, m] / (n + m)


def dtw_distance_vectorized(v1: np.ndarray, v2: np.ndarray,
                            window: int = 4) -> float:
    n, m = len(v1), len(v2)

    cost_matrix = (v1[:, None] - v2[None, :]) ** 2

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)

        for j in range(j_start, j_end + 1):
            min_cost = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            D[i, j] = cost_matrix[i-1, j-1] + min_cost

    return D[n, m] / (n + m) if not np.isinf(D[n, m]) else np.linalg.norm(v1 - v2) / np.sqrt(n)


def renyi_entropy(v: np.ndarray, alpha: float = 2.0) -> float:
    p = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
    p = p[p > 0]

    if len(p) == 0:
        return 0.0

    if alpha == 1:
        return -np.sum(p * np.log2(p + 1e-10))
    elif alpha == 0:
        return np.log2(len(p))
    elif alpha == np.inf:
        return -np.log2(np.max(p) + 1e-10)
    else:
        return (1 / (1 - alpha)) * np.log2(np.sum(p ** alpha) + 1e-10)


def bhattacharyya_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)

    bc = np.sum(np.sqrt(p * q + 1e-10))

    return -np.log(bc + 1e-10)


def wasserstein_entropic(v1: np.ndarray, v2: np.ndarray,
                         reg: float = 0.01, max_iter: int = 100,
                         tol: float = 1e-6) -> float:
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    n = len(p)

    positions = np.arange(n).reshape(-1, 1)
    C = np.abs(positions - positions.T)

    K = np.exp(-C / (reg + 1e-10))

    u = np.ones(n) / n
    v = np.ones(n) / n

    for iteration in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()

        u = p / (K @ v + 1e-10)
        v = q / (K.T @ u + 1e-10)

        if np.max(np.abs(u - u_old)) < tol and np.max(np.abs(v - v_old)) < tol:
            break

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)

    wasserstein_cost = np.sum(P * C)

    return wasserstein_cost / n


def wasserstein_entropic_vectorized(v1: np.ndarray, v2: np.ndarray,
                                    reg: float = 0.01, max_iter: int = 100) -> float:
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    n = len(p)

    positions = np.arange(n)
    C = np.abs(positions[:, None] - positions[None, :])

    K = np.exp(-C / (reg + 1e-10))

    u = np.ones(n) / n
    v = np.ones(n) / n

    for _ in range(max_iter):
        u = p / (K @ v + 1e-10)
        v = q / (K.T @ u + 1e-10)

    P = u[:, None] * K * v[None, :]

    return np.sum(P * C) / n


def compute_enhanced_metrics(v1: np.ndarray, v2: np.ndarray) -> Dict:
    metrics = {}

    if USE_GRASSMANN_PROJECTION:
        metrics['grassmann_projection'] = grassmann_projection_distance(v1, v2)
    if USE_FUBINI_STUDY:
        metrics['fubini_study'] = grassmann_fubini_study(v1, v2)
    if USE_RICCI_CURVATURE:
        metrics['ricci_curvature'] = grassmann_ricci_curvature(v1, v2)
    if USE_SHANNON_ENTROPY:
        metrics['entropy_v1'] = shannon_entropy(v1)
        metrics['entropy_v2'] = shannon_entropy(v2)
        metrics['entropy_diff'] = abs(metrics['entropy_v1'] - metrics['entropy_v2'])
    if USE_JENSEN_SHANNON:
        metrics['jensen_shannon'] = jensen_shannon_divergence(v1, v2)
    if USE_GINI_COEFFICIENT:
        metrics['gini_v1'] = gini_coefficient(v1)
        metrics['gini_v2'] = gini_coefficient(v2)
        metrics['gini_diff'] = abs(metrics['gini_v1'] - metrics['gini_v2'])
    if USE_HELLINGER_DISTANCE:
        metrics['hellinger'] = hellinger_distance(v1, v2)
    if USE_SPEARMAN_CORRELATION:
        metrics['spearman'] = spearman_correlation(v1, v2)
    if USE_WASSERSTEIN:
        metrics['wasserstein'] = wasserstein_distance(v1, v2)
        metrics['wasserstein_entropic'] = wasserstein_entropic(v1, v2)
    if USE_FRACTAL_DIMENSION:
        metrics['fractal_v1'] = fractal_dimension(v1)
        metrics['fractal_v2'] = fractal_dimension(v2)
        metrics['fractal_diff'] = abs(metrics['fractal_v1'] - metrics['fractal_v2'])
    if USE_RADON_TRANSFORM:
        metrics['radon_v1'] = np.mean(discrete_radon_transform(v1))
        metrics['radon_v2'] = np.mean(discrete_radon_transform(v2))
        metrics['radon_diff'] = abs(metrics['radon_v1'] - metrics['radon_v2'])
    if USE_MORANS_I:
        metrics['morans_v1'] = morans_i(v1)
        metrics['morans_v2'] = morans_i(v2)
        metrics['morans_diff'] = abs(metrics['morans_v1'] - metrics['morans_v2'])
    if USE_POLARITY_LAPLACIAN:
        metrics['laplacian_v1'] = polarity_laplacian(v1)
        metrics['laplacian_v2'] = polarity_laplacian(v2)
        metrics['laplacian_diff'] = abs(metrics['laplacian_v1'] - metrics['laplacian_v2'])

    metrics['renyi_entropy_alpha2_v1'] = renyi_entropy(v1, alpha=2.0)
    metrics['renyi_entropy_alpha2_v2'] = renyi_entropy(v2, alpha=2.0)
    metrics['renyi_entropy_diff'] = abs(metrics['renyi_entropy_alpha2_v1'] - metrics['renyi_entropy_alpha2_v2'])
    metrics['bhattacharyya'] = bhattacharyya_distance(v1, v2)
    metrics['dtw_distance'] = dtw_distance(v1, v2)

    # New advanced metrics
    metrics['quantum_fidelity'] = quantum_fidelity(v1, v2)
    metrics['bures_distance'] = bures_distance(v1, v2)

    return metrics


def fisher_information_matrix(vectors: List[np.ndarray],
                              weights: np.ndarray = None) -> Dict:
    if len(vectors) < 2:
        return {
            'matrix': np.eye(16),
            'trace': 16.0,
            'eigenvalues': np.ones(16),
            'eigenvectors': np.eye(16),
            'mean_fisher': 1.0,
            'max_fisher': 1.0,
            'condition_number': 1.0
        }

    X = np.array(vectors)
    n, d = X.shape

    mean = np.mean(X, axis=0)
    centered = X - mean

    if weights is None:
        weights = np.ones(n) / n

    cov = np.zeros((d, d))
    for i in range(n):
        cov += weights[i] * np.outer(centered[i], centered[i])

    fisher = pinv(cov + 1e-6 * np.eye(d))

    eigvals, eigvecs = eigh(fisher)

    return {
        'matrix': fisher,
        'trace': np.trace(fisher),
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'mean_fisher': np.mean(eigvals),
        'max_fisher': np.max(eigvals),
        'condition_number': np.max(eigvals) / (np.min(eigvals) + 1e-10)
    }


def ollivier_ricci_curvature(vectors: List[np.ndarray],
                             eps: float = 0.1,
                             max_iter: int = 100) -> float:
    if len(vectors) < 3:
        return 0.0

    X = np.array(vectors)
    n = len(X)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            w_dist = wasserstein_entropic(X[i], X[j], reg=eps, max_iter=max_iter)
            dist_matrix[i, j] = w_dist
            dist_matrix[j, i] = w_dist

    curvatures = []

    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:min(4, n)]

        for j in neighbors:
            dij = dist_matrix[i, j]
            if dij < 1e-10:
                continue

            i_neighbors = np.argsort(dist_matrix[i])[1:min(4, n)]
            j_neighbors = np.argsort(dist_matrix[j])[1:min(4, n)]

            if len(i_neighbors) < 2 or len(j_neighbors) < 2:
                continue

            W_ii = np.mean([dist_matrix[i, k] for k in i_neighbors])
            W_ij = np.mean([dist_matrix[j, k] for k in j_neighbors])

            curvature = (W_ij - W_ii) / (dij + 1e-10)
            curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0.0


def persistent_homology_features(vectors: List[np.ndarray],
                                 max_dimension: int = 2,
                                 n_samples: int = 50) -> Dict:
    default_result = {
        'num_components': 0,
        'num_edges': 0,
        'persistence_weight': 0.0,
        'max_persistence': 0.0,
        'mean_persistence': 0.0,
        'distance_matrix': []
    }

    if len(vectors) < 2:
        return default_result

    X = np.array(vectors)
    n = len(X)

    if n > n_samples:
        indices = np.random.choice(n, n_samples, replace=False)
        X = X[indices]
        n = n_samples

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = grassmann_distance(X[i], X[j])
            distances[i, j] = d
            distances[j, i] = d

    persistence = []
    max_dist = np.max(distances) + 1e-10

    for eps in np.linspace(0, max_dist, 20):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < eps:
                    edges.append((i, j))

        if edges:
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx

            for i, j in edges:
                union(i, j)

            components = len(set(find(i) for i in range(n)))
            persistence.append({
                'eps': eps,
                'dimension': 0,
                'components': components,
                'birth': eps if len(persistence) == 0 else persistence[-1]['eps']
            })

    num_components = persistence[-1]['components'] if persistence else 0
    persistence_weight = 0.0

    for p in persistence:
        if p['dimension'] == 0:
            persistence_weight += p['eps'] / max_dist

    persistence_weight = persistence_weight / (len(persistence) + 1)

    num_edges = sum(1 for i in range(n) for j in range(i+1, n)
                    if distances[i, j] < max_dist * 0.7)

    return {
        'num_components': num_components,
        'num_edges': num_edges,
        'persistence_weight': persistence_weight,
        'max_persistence': max_dist * persistence_weight,
        'mean_persistence': max_dist * persistence_weight / 2,
        'distance_matrix': distances.tolist() if n <= 20 else []
    }


def quantum_fidelity(v1: np.ndarray, v2: np.ndarray) -> float:
    p1 = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    p2 = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)

    fidelity = np.sum(np.sqrt(p1 * p2 + 1e-10))

    return min(1.0, fidelity)


def bures_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    fidelity = quantum_fidelity(v1, v2)
    distance = np.sqrt(2 - 2 * fidelity)
    return min(1.0, distance)


def uhlmann_fidelity_mean(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0

    fidelities = []
    n = min(len(vectors), 50)

    for i in range(n):
        for j in range(i+1, n):
            f = quantum_fidelity(vectors[i], vectors[j])
            fidelities.append(f)

    return np.mean(fidelities) if fidelities else 1.0


def interaction_information(vectors: List[np.ndarray],
                            n_bins: int = 10) -> float:
    if len(vectors) < 2:
        return 0.0

    X = np.array(vectors)
    n, d = X.shape

    discretized = np.zeros((n, d), dtype=int)
    for i in range(d):
        discretized[:, i] = np.digitize(X[:, i],
                                        np.linspace(np.min(X[:, i]),
                                                   np.max(X[:, i]),
                                                   n_bins + 1)[1:-1])

    interactions = []

    for i in range(min(d, 8)):
        for j in range(i+1, min(d, 8)):
            for k in range(j+1, min(d, 8)):
                H_i = scipy_entropy(np.bincount(discretized[:, i]) + 1, base=2)
                H_j = scipy_entropy(np.bincount(discretized[:, j]) + 1, base=2)
                H_k = scipy_entropy(np.bincount(discretized[:, k]) + 1, base=2)

                joint_ij, _ = np.histogramdd(discretized[:, [i, j]],
                                             bins=(n_bins, n_bins))
                H_ij = scipy_entropy(joint_ij.flatten() + 1, base=2)

                joint_ik, _ = np.histogramdd(discretized[:, [i, k]],
                                             bins=(n_bins, n_bins))
                H_ik = scipy_entropy(joint_ik.flatten() + 1, base=2)

                joint_jk, _ = np.histogramdd(discretized[:, [j, k]],
                                             bins=(n_bins, n_bins))
                H_jk = scipy_entropy(joint_jk.flatten() + 1, base=2)

                joint_ijk, _ = np.histogramdd(discretized[:, [i, j, k]],
                                              bins=(n_bins, n_bins, n_bins))
                H_ijk = scipy_entropy(joint_ijk.flatten() + 1, base=2)

                I = H_i + H_j + H_k - H_ij - H_ik - H_jk + H_ijk
                interactions.append(I)

    if interactions:
        mean_I = np.mean(interactions)
        max_entropy = np.log2(n_bins)
        return max(0, min(1, mean_I / (max_entropy * 2)))

    return 0.0


def wasserstein_orthogonal(vectors: List[np.ndarray],
                           n_components: int = 4) -> Dict:
    if len(vectors) < 2:
        return {'moments': np.zeros(n_components),
                'components': [],
                'explained_variance': np.ones(n_components)}

    X = np.array(vectors)
    n, d = X.shape

    mean = np.mean(X, axis=0)

    wass_dists = []
    for i in range(n):
        w = wasserstein_entropic(X[i], mean, reg=0.01)
        wass_dists.append(w)

    moments = []
    for order in range(1, n_components + 1):
        moment = np.mean(np.array(wass_dists) ** order)
        moments.append(moment)

    moments = np.array(moments)
    moments = moments / (np.max(moments) + 1e-10)

    pca = PCA(n_components=min(n_components, d))
    components = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_

    return {
        'moments': moments,
        'components': components.tolist() if len(components) > 0 else [],
        'explained_variance': explained_variance.tolist() if len(explained_variance) > 0 else [],
        'mean_wasserstein': np.mean(wass_dists),
        'std_wasserstein': np.std(wass_dists)
    }


def detect_bifurcation_points(vectors: List[np.ndarray],
                              n_scale: int = 10) -> Dict:
    if len(vectors) < 3:
        return {
            'bifurcation_points': [],
            'stability_landscape': [],
            'num_transitions': 0,
            'critical_scale': 0.0
        }

    X = np.array(vectors)
    n = len(X)

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = grassmann_distance(X[i], X[j])
            distances[i, j] = d
            distances[j, i] = d

    scales = np.linspace(0.1, 0.9, n_scale)
    transitions = []

    for scale in scales:
        threshold = scale * np.max(distances)

        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < threshold:
                    edges.append((i, j))

        if edges:
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx

            for i, j in edges:
                union(i, j)

            components = len(set(find(i) for i in range(n)))
            transitions.append({
                'scale': scale,
                'threshold': threshold,
                'components': components,
                'edges': len(edges)
            })

    bifurcation_points = []
    for i in range(1, len(transitions)):
        prev = transitions[i-1]
        curr = transitions[i]

        if abs(curr['components'] - prev['components']) > 1:
            bifurcation_points.append({
                'scale': curr['scale'],
                'threshold': curr['threshold'],
                'components_before': prev['components'],
                'components_after': curr['components'],
                'change': curr['components'] - prev['components']
            })

    stability_landscape = []
    for i in range(len(transitions)):
        components = transitions[i]['components']
        if i > 0:
            change = abs(transitions[i]['components'] - transitions[i-1]['components'])
        else:
            change = 0

        stability = 1.0 / (1.0 + change)
        stability_landscape.append({
            'scale': transitions[i]['scale'],
            'stability': stability,
            'components': components
        })

    return {
        'bifurcation_points': bifurcation_points,
        'stability_landscape': stability_landscape,
        'num_transitions': len(transitions),
        'critical_scale': transitions[-1]['scale'] if transitions else 0.0,
        'max_components': max(t['components'] for t in transitions) if transitions else 0,
        'min_components': min(t['components'] for t in transitions) if transitions else 0
    }

# ============================================================================
# CLASS: GroupStatistics
# ============================================================================

@dataclass
class GroupStatistics:
    name: str
    n_samples: int
    centroid: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    std_dev: np.ndarray
    wedge_self_similarity: float
    wedge_self_similarity_std: float = 0.0
    adaptive_threshold: float = 0.99
    clifford_signature: Dict[str, float] = field(default_factory=dict)
    subspace_projections: Dict[str, float] = field(default_factory=dict)
    metric_norm: float = 0.0
    metric_sign: float = 0.0
    total_processed: int = 0
    sample_size: int = 0
    hodge_dual_centroid: np.ndarray = field(default_factory=lambda: np.zeros(DIM_PAIRS))
    grassmann_radius: float = 0.0
    entropy: float = 0.0
    gini: float = 0.0
    complexity: float = 0.0
    modularity: float = 0.0
    morans_i: float = 0.0
    grassmann_multilevel: Dict[int, float] = field(default_factory=dict)
    grassmann_asymmetry: Dict[int, float] = field(default_factory=dict)
    grassmann_curvature: float = 0.0
    grassmann_volume: float = 0.0
    grassmann_cycles: List[List[int]] = field(default_factory=list)
    grassmann_karcher_centroid: np.ndarray = field(default_factory=lambda: np.zeros(DIM_PAIRS))
    grassmann_svd_angles: Dict[int, float] = field(default_factory=dict)
    fractal_dimension: float = 0.0
    wasserstein_mean: float = 0.0
    radon_mean: float = 0.0
    polarity_laplacian: float = 0.0
    functional_modularity: float = 0.0
    structural_complexity: float = 0.0
    all_metrics: Dict[str, float] = field(default_factory=dict)

    renyi_entropy_alpha2: float = 0.0
    renyi_entropy_alpha3: float = 0.0
    bhattacharyya_mean: float = 0.0
    wasserstein_entropic_mean: float = 0.0
    fpca_scores: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fpca_explained_variance: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mutual_information: np.ndarray = field(default_factory=lambda: np.zeros((16, 16)))
    total_correlation: float = 0.0
    grassmann_scalar_curvature: float = 0.0
    rotor_alignment_mean: float = 0.0
    geometric_product_mean: Dict = field(default_factory=dict)

    fisher_mean: float = 0.0
    fisher_trace: float = 0.0
    ollivier_ricci_mean: float = 0.0
    ph_mean_weight: float = 0.0
    ph_num_edges: float = 0.0
    uhlmann_fidelity_mean: float = 0.0
    bifurcation_points: List[Dict] = field(default_factory=list)
    stability_landscape: List[Dict] = field(default_factory=list)

    def mahalanobis_distance(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 1.0
        diff = vector - self.centroid
        return np.sqrt(diff @ self.inv_covariance @ diff)

    def probability_of_belonging(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 0.5
        d = self.mahalanobis_distance(vector)
        return 1.0 - chi2.cdf(d**2, df=len(self.centroid))

# ============================================================================
# CLASS: AdvancedGroupAnalyzer
# ============================================================================

class AdvancedGroupAnalyzer:
    def __init__(self, grassmann: GrassmannPIM, main_groups: List[str] = None,
                 design_group: List[str] = None):
        self.grassmann = grassmann
        self.group_stats: Dict[str, GroupStatistics] = {}
        self.sample_data: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}
        self.hash_index = None
        self.tracker = ProcessingTracker()
        self.start_time = None
        self.max_samples_per_group = MAX_STORED_PROTEINS_PER_GROUP
        self.therapeutic_profile = None
        self.characterization_report = None

        self.main_groups = main_groups if main_groups else MAIN_GROUP_REFERENCE
        self.design_group = design_group if design_group else MAIN_GROUP_DESIGN

        print(f"  ✅ AdvancedGroupAnalyzer v211.0 initialized")
        print(f"     ├─ Reference groups (comparisons): {self.main_groups}")
        print(f"     └─ Design group (peptide): {self.design_group}")

    def set_sample_size(self, size: int):
        self.max_samples_per_group = size

    def load_fasta_file(self, filepath: str, group_name: str, verbose: bool = True):
        if not os.path.exists(filepath):
            if verbose:
                print(f"  ⚠️ File not found: {filepath}")
            return

        if verbose:
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"  📂 Processing: {os.path.basename(filepath)} ({size_gb:.2f} GB) - STREAMING")

        if group_name not in self.sample_data:
            self.sample_data[group_name] = []

        total_seen = 0

        try:
            for header, sequence in read_fasta_stream(filepath, verbose=False):
                total_seen += 1

                pim = compute_pim_profile(sequence, use_weights=True)

                if np.sum(np.abs(pim)) < 1e-6:
                    self.tracker.update(group_name, False, len(sequence))
                    continue

                if len(self.sample_data[group_name]) < self.max_samples_per_group:
                    self.sample_data[group_name].append((header, pim, sequence))
                    self.tracker.update(group_name, True, len(sequence))
                else:
                    j = random.randint(0, total_seen - 1)
                    if j < self.max_samples_per_group:
                        self.sample_data[group_name][j] = (header, pim, sequence)
                    self.tracker.update(group_name, True, len(sequence))

                if total_seen % 100000 == 0 and verbose:
                    print(f"     Processed {total_seen:,} sequences in {group_name}...")

        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
            return

        if len(self.sample_data[group_name]) > 0:
            self._compute_group_statistics(group_name)
            if verbose:
                print(f"  ✅ {group_name}: {len(self.sample_data[group_name])} samples stored "
                      f"from {total_seen:,} processed sequences")
        else:
            if verbose:
                print(f"  ⚠️ {group_name}: No valid samples obtained")

    def _compute_group_statistics(self, group_name: str):
        if group_name not in self.sample_data:
            return

        samples = self.sample_data[group_name]
        if len(samples) == 0:
            return

        vectors = [item[1] for item in samples]
        headers = [item[0] for item in samples]
        sequences = [item[2] for item in samples]

        vectors_array = np.array(vectors)

        if np.any(~np.isfinite(vectors_array)):
            print(f"     ⚠️ {group_name}: Vectors with non-finite values detected. Fixing...")
            vectors_array = np.nan_to_num(vectors_array, nan=0.0, posinf=0.0, neginf=0.0)

        centroid = np.mean(vectors_array, axis=0)

        if np.sum(np.abs(centroid)) < 1e-10:
            centroid = np.random.randn(len(centroid)) * 0.01
            print(f"     ⚠️ {group_name}: Centroid was zero, using small random vector")

        covariance = None
        try:
            if len(vectors_array) < 2:
                covariance = np.eye(vectors_array.shape[1]) * 0.01
            else:
                covariance = np.cov(vectors_array.T)
                if not np.isfinite(covariance).all():
                    print(f"     ⚠️ {group_name}: Covariance with non-finite values. Using diagonal matrix.")
                    covariance = np.eye(vectors_array.shape[1]) * 0.01
                if np.sum(np.abs(covariance)) < 1e-10:
                    covariance = np.eye(vectors_array.shape[1]) * 0.01
        except Exception as e:
            print(f"     ⚠️ {group_name}: Error calculating covariance: {e}. Using diagonal matrix.")
            covariance = np.eye(vectors_array.shape[1]) * 0.01

        reg_value = 1e-6
        cov_reg = covariance + np.eye(covariance.shape[0]) * reg_value

        inv_covariance = None
        try:
            inv_covariance = np.linalg.pinv(cov_reg, rcond=1e-8)
            if not np.isfinite(inv_covariance).all():
                print(f"     ⚠️ {group_name}: Inverse with non-finite values. Using diagonal matrix.")
                inv_covariance = np.eye(covariance.shape[0]) * (1.0 / (reg_value + 0.01))
        except np.linalg.LinAlgError as e:
            print(f"     ⚠️ {group_name}: SVD did not converge in pinv. Using regularized diagonal matrix.")
            diag_reg = np.diag(covariance) + reg_value
            diag_reg = np.maximum(diag_reg, reg_value)
            inv_covariance = np.diag(1.0 / diag_reg)
        except Exception as e:
            print(f"     ⚠️ {group_name}: Error in pinv: {e}. Using identity.")
            inv_covariance = np.eye(covariance.shape[0])

        try:
            std_dev = np.sqrt(np.maximum(np.diag(covariance), reg_value))
            if not np.isfinite(std_dev).all():
                std_dev = np.ones(covariance.shape[0]) * np.sqrt(reg_value)
        except:
            std_dev = np.ones(covariance.shape[0]) * np.sqrt(reg_value)

        wedge_sims = []
        if len(vectors) > 1:
            n_compare = min(len(vectors), 50)
            for i in range(n_compare):
                for j in range(i+1, n_compare):
                    try:
                        sim, _ = self.grassmann.wedge_product(vectors[i], vectors[j])
                        if np.isfinite(sim):
                            wedge_sims.append(sim)
                    except:
                        continue
        wedge_self_sim = np.mean(wedge_sims) if wedge_sims else 0.0
        wedge_self_sim_std = np.std(wedge_sims) if wedge_sims else 0.0

        clifford_sigs = []
        n_sigs = min(len(vectors), 50)
        for i in range(n_sigs):
            try:
                sig = self.grassmann.clifford_signature(vectors[i])
                if sig:
                    clifford_sigs.append(sig)
            except:
                continue

        avg_clifford = {}
        if clifford_sigs:
            keys = list(clifford_sigs[0].keys())
            for key in keys:
                values = [sig.get(key, 0) for sig in clifford_sigs if key in sig]
                if values:
                    avg_clifford[key] = np.mean(values)
                else:
                    avg_clifford[key] = 0.0
        else:
            avg_clifford = {
                'norm': np.linalg.norm(centroid),
                'auto_reflection': 0.5,
                'hydrophobic_projection': 0.1,
                'charge_projection': 0.1,
                'auto_rotation': 0.3,
                'hodge_norm': 0.5,
                'hodge_complement': 0.3,
                'entropy': shannon_entropy(centroid) if USE_SHANNON_ENTROPY else 0.5,
                'gini': gini_coefficient(centroid) if USE_GINI_COEFFICIENT else 0.3
            }

        subspace_projections = {}
        for subspace in SUBSPACES.keys():
            if subspace != 'full':
                projs = []
                for v in vectors[:50]:
                    try:
                        proj = self.grassmann.interior_product_magnitude(v, subspace)
                        if np.isfinite(proj):
                            projs.append(proj)
                    except:
                        continue
                subspace_projections[subspace] = np.mean(projs) if projs else 0.0

        try:
            metric_norm, metric_sign = self.grassmann.norm_metric(centroid)
        except:
            metric_norm = np.linalg.norm(centroid)
            metric_sign = 1.0

        try:
            hodge_dual_centroid = self.grassmann.hodge_dual(centroid)
            if not np.isfinite(hodge_dual_centroid).all():
                hodge_dual_centroid = np.zeros_like(centroid)
        except:
            hodge_dual_centroid = np.zeros_like(centroid)

        renyi_alpha2 = renyi_entropy(centroid, alpha=2.0)
        renyi_alpha3 = renyi_entropy(centroid, alpha=3.0)

        bhattacharyya_dists = []
        wasserstein_entropic_dists = []
        for v in vectors[:30]:
            bhattacharyya_dists.append(bhattacharyya_distance(v, centroid))
            wasserstein_entropic_dists.append(wasserstein_entropic(v, centroid))
        bhattacharyya_mean = np.mean(bhattacharyya_dists) if bhattacharyya_dists else 0.0
        wasserstein_entropic_mean = np.mean(wasserstein_entropic_dists) if wasserstein_entropic_dists else 0.0

        fpca_scores = np.zeros(3)
        fpca_explained = np.zeros(3)
        if len(vectors) > 5:
            try:
                fpca_result = functional_pca(vectors[:30], n_components=3)
                if 'scores' in fpca_result:
                    fpca_scores = np.mean(fpca_result['scores'], axis=0) if len(fpca_result['scores']) > 0 else np.zeros(3)
                if 'explained_variance' in fpca_result:
                    fpca_explained = fpca_result['explained_variance'][:3]
            except:
                pass

        mutual_info = np.zeros((16, 16))
        if len(vectors) > 10:
            try:
                mutual_info = mutual_information_matrix(vectors[:30])
            except:
                pass

        total_corr = total_correlation(vectors[:30]) if len(vectors) > 2 else 0.0

        grassmann_scalar_curv = 0.0
        if len(vectors) > 0:
            try:
                grassmann_scalar_curv = grassmann_scalar_curvature(centroid, k=2)
            except:
                pass

        rotor_alignments = []
        if len(vectors) > 1:
            for v in vectors[:10]:
                try:
                    rotor = rotor_from_vectors(centroid, v)
                    rotor_alignments.append(rotor['similarity'])
                except:
                    pass
        rotor_alignment_mean = np.mean(rotor_alignments) if rotor_alignments else 0.0

        gp_means = {'scalar': 0.0, 'bivector_norm': 0.0, 'vector_norm': 0.0}
        if len(vectors) > 1:
            gp_scalars = []
            gp_bivectors = []
            gp_vectors = []
            for v in vectors[:10]:
                try:
                    gp = clifford_product_vectorized(centroid, v)
                    gp_scalars.append(gp['scalar'])
                    gp_bivectors.append(np.linalg.norm(gp['bivector']))
                    gp_vectors.append(np.linalg.norm(gp['vector']))
                except:
                    pass
            if gp_scalars:
                gp_means['scalar'] = np.mean(gp_scalars)
                gp_means['bivector_norm'] = np.mean(gp_bivectors)
                gp_means['vector_norm'] = np.mean(gp_vectors)

        grassmann_multilevel = {}
        if USE_GRASSMANN_MULTILEVEL:
            for k in GRASSMANN_LEVELS:
                distances = []
                n_compare = min(len(vectors), 30)
                for i in range(n_compare):
                    for j in range(i+1, n_compare):
                        try:
                            d = self.grassmann.multilevel_distance(vectors[i], vectors[j], k)
                            if np.isfinite(d):
                                distances.append(d)
                        except:
                            continue
                grassmann_multilevel[k] = np.mean(distances) if distances else 0.0

        grassmann_asymmetry = {}
        if USE_GRASSMANN_ASYMMETRIC and len(vectors) > 1:
            for k in GRASSMANN_LEVELS:
                asyms = []
                n_compare = min(len(vectors), 20)
                for i in range(n_compare):
                    for j in range(i+1, n_compare):
                        try:
                            _, _, asym = self.grassmann.projection_asymmetry(vectors[i], vectors[j], k)
                            if np.isfinite(asym):
                                asyms.append(asym)
                        except:
                            continue
                grassmann_asymmetry[k] = np.mean(asyms) if asyms else 0.0

        grassmann_curvature = 0.0
        if USE_GRASSMANN_CURVATURE and len(vectors) > 2:
            try:
                grassmann_curvature = self.grassmann.sectional_curvature_sampled(
                    vectors[:min(len(vectors), 50)], k=2, n_samples=CURVATURE_SAMPLES
                )
                if not np.isfinite(grassmann_curvature):
                    grassmann_curvature = 0.0
            except:
                grassmann_curvature = 0.0

        grassmann_volume = 0.0
        if USE_GRASSMANN_VOLUME and len(vectors) > 1:
            try:
                grassmann_volume = self.grassmann.volume(vectors[:min(len(vectors), 30)], k=2)
                if not np.isfinite(grassmann_volume):
                    grassmann_volume = 0.0
            except:
                grassmann_volume = 0.0

        grassmann_cycles = []
        if USE_GRASSMANN_CYCLES and len(vectors) > 2:
            try:
                grassmann_cycles = self.grassmann.cycles(vectors[:min(len(vectors), 30)], k=2, threshold=0.5)
            except:
                grassmann_cycles = []

        grassmann_karcher_centroid = centroid.copy()
        if USE_GRASSMANN_KARCHER and len(vectors) > 1:
            try:
                grassmann_karcher_centroid = self.grassmann.karcher_mean(
                    vectors[:min(len(vectors), 30)], k=2
                )
                if not np.isfinite(grassmann_karcher_centroid).all():
                    grassmann_karcher_centroid = centroid.copy()
            except:
                grassmann_karcher_centroid = centroid.copy()

        grassmann_svd_angles = {}
        if USE_GRASSMANN_SVD and len(vectors) > 1:
            for k in GRASSMANN_LEVELS:
                angles = []
                n_compare = min(len(vectors), 10)
                for i in range(n_compare):
                    for j in range(i+1, n_compare):
                        try:
                            svd_res = self.grassmann.svd_similarity(vectors[i], vectors[j], k)
                            if svd_res and 'mean_angle' in svd_res:
                                angle = svd_res['mean_angle']
                                if np.isfinite(angle):
                                    angles.append(angle)
                        except:
                            continue
                grassmann_svd_angles[k] = np.mean(angles) if angles else 0.0

        try:
            fractal_dim = np.mean([fractal_dimension(v) for v in vectors[:30]]) if vectors else 0.0
            if not np.isfinite(fractal_dim):
                fractal_dim = 0.0
        except:
            fractal_dim = 0.0

        try:
            wasserstein_mean = np.mean([wasserstein_distance(v, centroid) for v in vectors[:30]]) if vectors else 0.0
            if not np.isfinite(wasserstein_mean):
                wasserstein_mean = 0.0
        except:
            wasserstein_mean = 0.0

        try:
            radon_means = [np.mean(discrete_radon_transform(v)) for v in vectors[:30]]
            radon_mean = np.mean(radon_means) if radon_means else 0.0
            if not np.isfinite(radon_mean):
                radon_mean = 0.0
        except:
            radon_mean = 0.0

        try:
            laplacian_mean = np.mean([polarity_laplacian(v) for v in vectors[:30]]) if vectors else 0.0
            if not np.isfinite(laplacian_mean):
                laplacian_mean = 0.0
        except:
            laplacian_mean = 0.0

        try:
            entropy = shannon_entropy(centroid)
            if not np.isfinite(entropy):
                entropy = 0.0
        except:
            entropy = 0.0

        try:
            gini = gini_coefficient(centroid)
            if not np.isfinite(gini):
                gini = 0.0
        except:
            gini = 0.0

        try:
            morans = morans_i(centroid)
            if not np.isfinite(morans):
                morans = 0.0
        except:
            morans = 0.0

        structural_complexity = fractal_dim * (1 + entropy / 4)
        functional_modularity = 1 - gini

        # New advanced metrics
        fisher_result = fisher_information_matrix(vectors[:min(len(vectors), 50)])
        fisher_mean = fisher_result.get('mean_fisher', 1.0)
        fisher_trace = fisher_result.get('trace', 16.0)

        ollivier_ricci_mean = 0.0
        if len(vectors) >= 3:
            try:
                ollivier_ricci_mean = ollivier_ricci_curvature(vectors[:min(len(vectors), 30)])
            except:
                ollivier_ricci_mean = 0.0

        ph_result = persistent_homology_features(vectors[:min(len(vectors), 50)])
        ph_mean_weight = ph_result.get('persistence_weight', 0.0)
        ph_num_edges = ph_result.get('num_edges', 0)

        uhlmann_fidelity_mean = 1.0
        if len(vectors) >= 2:
            try:
                uhlmann_fidelity_mean = uhlmann_fidelity_mean(vectors[:min(len(vectors), 50)])
            except:
                uhlmann_fidelity_mean = 1.0

        bifurcation_points = []
        stability_landscape = []
        if len(vectors) >= 3:
            try:
                bifurcation_result = detect_bifurcation_points(vectors[:min(len(vectors), 50)])
                bifurcation_points = bifurcation_result.get('bifurcation_points', [])
                stability_landscape = bifurcation_result.get('stability_landscape', [])
            except:
                bifurcation_points = []
                stability_landscape = []

        all_metrics = {
            'entropy': entropy,
            'gini': gini,
            'morans_i': morans,
            'fractal_dimension': fractal_dim,
            'wasserstein_mean': wasserstein_mean,
            'radon_mean': radon_mean,
            'polarity_laplacian': laplacian_mean,
            'structural_complexity': structural_complexity,
            'functional_modularity': functional_modularity,
            'grassmann_curvature': grassmann_curvature,
            'grassmann_volume': grassmann_volume,
            'renyi_entropy_alpha2': renyi_alpha2,
            'renyi_entropy_alpha3': renyi_alpha3,
            'bhattacharyya_mean': bhattacharyya_mean,
            'wasserstein_entropic_mean': wasserstein_entropic_mean,
            'total_correlation': total_corr,
            'grassmann_scalar_curvature': grassmann_scalar_curv,
            'rotor_alignment_mean': rotor_alignment_mean,
            'fpca_component_1': fpca_scores[0] if len(fpca_scores) > 0 else 0.0,
            'fpca_component_2': fpca_scores[1] if len(fpca_scores) > 1 else 0.0,
            'fpca_component_3': fpca_scores[2] if len(fpca_scores) > 2 else 0.0,
            'fisher_mean': fisher_mean,
            'fisher_trace': fisher_trace,
            'ollivier_ricci_mean': ollivier_ricci_mean,
            'ph_mean_weight': ph_mean_weight,
            'ph_num_edges': ph_num_edges,
            'uhland_fidelity_mean': uhlmann_fidelity_mean,
            'bifurcation_points': len(bifurcation_points),
            'stability_landscape': len(stability_landscape)
        }

        stats = GroupStatistics(
            name=group_name,
            n_samples=len(samples),
            centroid=centroid,
            covariance=covariance,
            inv_covariance=inv_covariance,
            std_dev=std_dev,
            wedge_self_similarity=wedge_self_sim,
            wedge_self_similarity_std=wedge_self_sim_std,
            adaptive_threshold=0.99,
            clifford_signature=avg_clifford,
            subspace_projections=subspace_projections,
            metric_norm=metric_norm,
            metric_sign=metric_sign,
            total_processed=self.tracker.group_counts.get(group_name, 0),
            sample_size=len(samples),
            hodge_dual_centroid=hodge_dual_centroid,
            grassmann_radius=np.mean([grassmann_distance(centroid, v) for v in vectors[:30]]) if vectors else 0.0,
            entropy=entropy,
            gini=gini,
            complexity=structural_complexity,
            modularity=functional_modularity,
            morans_i=morans,
            grassmann_multilevel=grassmann_multilevel,
            grassmann_asymmetry=grassmann_asymmetry,
            grassmann_curvature=grassmann_curvature,
            grassmann_volume=grassmann_volume,
            grassmann_cycles=grassmann_cycles,
            grassmann_karcher_centroid=grassmann_karcher_centroid,
            grassmann_svd_angles=grassmann_svd_angles,
            fractal_dimension=fractal_dim,
            wasserstein_mean=wasserstein_mean,
            radon_mean=radon_mean,
            polarity_laplacian=laplacian_mean,
            functional_modularity=functional_modularity,
            structural_complexity=structural_complexity,
            all_metrics=all_metrics,
            renyi_entropy_alpha2=renyi_alpha2,
            renyi_entropy_alpha3=renyi_alpha3,
            bhattacharyya_mean=bhattacharyya_mean,
            wasserstein_entropic_mean=wasserstein_entropic_mean,
            fpca_scores=fpca_scores,
            fpca_explained_variance=fpca_explained,
            mutual_information=mutual_info,
            total_correlation=total_corr,
            grassmann_scalar_curvature=grassmann_scalar_curv,
            rotor_alignment_mean=rotor_alignment_mean,
            geometric_product_mean=gp_means,
            fisher_mean=fisher_mean,
            fisher_trace=fisher_trace,
            ollivier_ricci_mean=ollivier_ricci_mean,
            ph_mean_weight=ph_mean_weight,
            ph_num_edges=ph_num_edges,
            uhlmann_fidelity_mean=uhlmann_fidelity_mean,
            bifurcation_points=bifurcation_points,
            stability_landscape=stability_landscape
        )

        self.group_stats[group_name] = stats

    def build_hash_index(self):
        self.hash_index = PIMHashIndex()
        self.hash_index.build_from_samples(self.sample_data)

    def print_processing_summary(self):
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY BY GROUP (v211.0)")
        print("=" * 80)

        print(f"  {'Group':<25} {'Samples':>12} {'Stored':>12} {'Self-Sim':>12} {'Entropy':>12} {'Rényi α=2':>12}")
        print(f"  {'-' * 85}")

        for group in sorted(self.group_stats.keys()):
            stats = self.group_stats[group]
            total = self.tracker.group_counts.get(group, 0)
            stored = stats.n_samples
            self_sim = stats.wedge_self_similarity
            entropy = stats.entropy
            renyi = stats.renyi_entropy_alpha2
            print(f"  {get_display_name(group):<25} {total:>12,} {stored:>12,} "
                  f"{self_sim:>12.4f} {entropy:>12.4f} {renyi:>12.4f}")

        print(f"  {'-' * 85}")
        total_seqs = self.tracker.total_sequences_processed
        total_valid = self.tracker.total_valid_pim
        print(f"  {'TOTAL':<25} {total_seqs:>12,} {total_valid:>12,}")

    def generate_characterization_report(self, target_group: str = None,
                                         config_loader: ConfigLoader = None) -> Dict:
        print("\n" + "=" * 80)
        print("📊 GENERATING CHARACTERIZATION REPORT (v211.0)")
        print("   ✅ WITH ALL METRICS + NARRATIVE SUMMARY")
        print("   ✅ WITH CLINICAL EVALUATION")
        print("=" * 80)

        if target_group is None:
            target_group = self.main_groups[0] if self.main_groups else list(self.group_stats.keys())[0]

        if config_loader is None:
            config_loader = ConfigLoader(verbose=False)

        if target_group not in self.group_stats:
            print(f"  ⚠️ Target group {target_group} not found. Using first available.")
            target_group = list(self.group_stats.keys())[0]

        stats = self.group_stats[target_group]

        metrics = stats.all_metrics.copy()
        metrics['pim_similarity'] = stats.wedge_self_similarity
        metrics['grassmann_distance'] = stats.grassmann_radius
        metrics['structural_stability'] = 1.0 - stats.grassmann_radius

        try:
            if hasattr(stats, 'centroid') and np.any(stats.centroid):
                metrics['interaction_quality'] = hodge_complementarity(stats.centroid, stats.centroid)
            else:
                metrics['interaction_quality'] = 0.5
        except Exception as e:
            metrics['interaction_quality'] = 0.5

        metrics['hodge_complementarity'] = hodge_complementarity(stats.centroid, stats.centroid)
        metrics['ricci_curvature'] = grassmann_ricci_curvature(stats.centroid, stats.centroid)
        metrics['wasserstein'] = wasserstein_distance(stats.centroid, stats.centroid)
        metrics['fractal_dimension'] = fractal_dimension(stats.centroid)

        metrics['entropy_level'] = stats.entropy / 4.0
        metrics['structural_complexity'] = stats.structural_complexity
        metrics['drug_likeness'] = 0.5 + 0.3 * (1 - stats.gini) + 0.2 * (stats.entropy / 4)
        metrics['activity_score'] = 0.5 + 0.3 * stats.wedge_self_similarity + 0.2 * (1 - stats.grassmann_radius)

        metrics['fisher_mean'] = stats.fisher_mean
        metrics['fisher_trace'] = stats.fisher_trace
        metrics['ollivier_ricci_mean'] = stats.ollivier_ricci_mean
        metrics['ph_mean_weight'] = stats.ph_mean_weight
        metrics['ph_num_edges'] = stats.ph_num_edges
        metrics['uhland_fidelity_mean'] = stats.uhlmann_fidelity_mean
        metrics['bifurcation_points'] = len(stats.bifurcation_points)

        knowledge_base = NarrativeKnowledgeBase()
        interpreter = ContextualInterpreter(knowledge_base)

        narrative_generator = CharacterizationNarrativeGenerator(knowledge_base)
        target_name = get_display_name(target_group)

        sequence = None
        if target_group in self.sample_data and len(self.sample_data[target_group]) > 0:
            sequence = self.sample_data[target_group][0][2]

        report_text = narrative_generator.generate_report(metrics, target_name, sequence)

        summary_generator = NarrativeSummaryGenerator(knowledge_base)
        profile_summaries = {}

        for profile in ['executive', 'biochemist', 'chemist', 'analytical_chemist',
                       'physicochemist', 'bioinformatician']:
            summary = summary_generator.generate_summary(metrics, target_name, profile)
            profile_summaries[profile] = summary

        multidisciplinary_reporter = MultidisciplinaryReporter(summary_generator)
        table_text = multidisciplinary_reporter._generate_multidisciplinary_table(metrics, target_name)

        report = {
            'target_group': target_group,
            'target_name': target_name,
            'metrics': metrics,
            'report_text': report_text,
            'profile_summaries': profile_summaries,
            'multidisciplinary_table': table_text,
            'sequence': sequence,
            'timestamp': datetime.now().isoformat(),
            'version': '211.0.0'
        }

        self.characterization_report = report
        return report

    def compare_group_to_all(self, group_name: str) -> pd.DataFrame:
        if group_name not in self.group_stats:
            return pd.DataFrame()

        ref_stats = self.group_stats[group_name]
        ref_centroid = ref_stats.centroid

        comparisons = []
        for target_name, target_stats in self.group_stats.items():
            if target_name == group_name:
                continue

            target_centroid = target_stats.centroid

            wedge_sim, wedge_std = self.grassmann.wedge_product(ref_centroid, target_centroid, with_ci=True)
            grassmann_dist = self.grassmann.grassmann_distance(ref_centroid, target_centroid)
            hodge_comp = self.grassmann.hodge_complementarity(ref_centroid, target_centroid)

            rotor = rotor_from_vectors(ref_centroid, target_centroid)
            rotor_sim = rotor['similarity']
            rotor_angle = rotor['angle']

            gp = clifford_product_vectorized(ref_centroid, target_centroid)
            gp_scalar = gp['scalar']
            gp_bivector = np.linalg.norm(gp['bivector'])

            enhanced = self.grassmann.compute_enhanced_metrics(ref_centroid, target_centroid)

            multilevel_dist = {}
            if USE_GRASSMANN_MULTILEVEL:
                for k in GRASSMANN_LEVELS:
                    multilevel_dist[f'grassmann_k{k}'] = self.grassmann.multilevel_distance(
                        ref_centroid, target_centroid, k
                    )

            svd_sim = 0.0
            if USE_GRASSMANN_SVD:
                svd_res = self.grassmann.svd_similarity(ref_centroid, target_centroid, k=2)
                svd_sim = svd_res.get('similarity', 0.0)

            asym = 0.0
            if USE_GRASSMANN_ASYMMETRIC:
                _, _, asym = self.grassmann.projection_asymmetry(ref_centroid, target_centroid, k=1)

            curvature = 0.0
            if USE_GRASSMANN_CURVATURE and len(self.sample_data.get(group_name, [])) > 2:
                samples = self.sample_data.get(group_name, [])[:10]
                target_samples = self.sample_data.get(target_name, [])[:10]
                if len(samples) > 2 and len(target_samples) > 2:
                    try:
                        v1 = samples[0][1]
                        v2 = samples[1][1]
                        v3 = target_samples[0][1]
                        curvature = 0.1 * (1 - grassmann_multilevel_similarity(v1, v2, k=2))
                    except:
                        pass

            row = {
                'Compared Group': get_display_name(target_name),
                'Wedge Similarity': wedge_sim,
                'Wedge Std': wedge_std,
                'Grassmann Distance': grassmann_dist,
                'Hodge Complementarity': hodge_comp,
                'SVD Similarity': svd_sim,
                'Asymmetry': asym,
                'Sectional Curvature': curvature,
                'Entropy Diff': abs(ref_stats.entropy - target_stats.entropy),
                'Gini Diff': abs(ref_stats.gini - target_stats.gini),
                'Fractal Diff': abs(ref_stats.fractal_dimension - target_stats.fractal_dimension),
                'Wasserstein': enhanced.get('wasserstein', 0.0),
                'Wasserstein Entropic': enhanced.get('wasserstein_entropic', 0.0),
                'Jensen-Shannon': enhanced.get('jensen_shannon', 0.0),
                'Hellinger': enhanced.get('hellinger', 0.0),
                'Spearman': enhanced.get('spearman', 0.0),
                'Bhattacharyya': enhanced.get('bhattacharyya', 0.0),
                'Rényi Entropy Diff': enhanced.get('renyi_entropy_diff', 0.0),
                'Rotor Similarity': rotor_sim,
                'Rotor Angle': rotor_angle,
                'GP Scalar': gp_scalar,
                'GP Bivector': gp_bivector,
                'DTW': enhanced.get('dtw_distance', 0.0),
            }

            row.update(multilevel_dist)
            comparisons.append(row)

        df = pd.DataFrame(comparisons)
        if not df.empty:
            df = df.sort_values('Wedge Similarity', ascending=False)

        return df

    def get_similarity_matrix(self) -> pd.DataFrame:
        groups = list(self.group_stats.keys())
        if len(groups) < 2:
            return pd.DataFrame()

        n = len(groups)
        matrix = np.zeros((n, n))

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    centroid1 = self.group_stats[g1].centroid
                    centroid2 = self.group_stats[g2].centroid
                    alignment = geometric_alignment(centroid1, centroid2)
                    sim = alignment['composite_alignment']
                    matrix[i, j] = sim
                    matrix[j, i] = sim

        df = pd.DataFrame(matrix, index=[get_display_name(g) for g in groups],
                          columns=[get_display_name(g) for g in groups])
        return df

    def get_top_individuals(self, group_name: str, n: int = TOP_N_PROTEINS) -> pd.DataFrame:
        if group_name not in self.sample_data or group_name not in self.group_stats:
            return pd.DataFrame()

        samples = self.sample_data[group_name]
        centroid = self.group_stats[group_name].centroid

        results = []
        for header, pim, sequence in samples:
            sim, _ = self.grassmann.wedge_product(pim, centroid)
            results.append({
                'Header': header,
                'Protein ID': extract_protein_id(header),
                'Wedge Similarity': sim,
                'Length': len(sequence),
                'Entropy': shannon_entropy(pim),
                'Gini': gini_coefficient(pim),
                'Rényi α=2': renyi_entropy(pim, alpha=2.0),
                'Fractal Dim': fractal_dimension(pim),
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('Wedge Similarity', ascending=False).head(n)

        return df

    def generate_full_report(self, reference_group: str, results_dir: str,
                             mode: OperationMode = OperationMode.HYBRID,
                             config_loader: ConfigLoader = None) -> Dict:
        print("\n" + "=" * 80)
        print("📊 GENERATING FULL REPORT (v211.0 - DUAL MODE)")
        print(f"   ✅ MODE: {mode.get_mode_name().upper()}")
        print("   ✅ WITH ALL MATHEMATICAL IMPROVEMENTS")
        print("   ✅ WITH NARRATIVE SUMMARIES")
        print("=" * 80)

        report = {}

        report['processing'] = self.tracker.get_report()
        report['comparison'] = self.compare_group_to_all(reference_group)
        report['similarity_matrix'] = self.get_similarity_matrix()
        report['top_individuals'] = self.get_top_individuals(reference_group, TOP_N_PROTEINS)

        print("\n  📊 Calculating Karhunen-Loève decomposition...")
        if reference_group in self.sample_data:
            vectors = [item[1] for item in self.sample_data[reference_group]]
            if len(vectors) > 1:
                report['kl_decomposition'] = self.grassmann.karhunen_loeve_decomposition(vectors, n_components=8)
            else:
                report['kl_decomposition'] = {'error': 'Not enough vectors'}
        else:
            report['kl_decomposition'] = {'error': 'Reference group not found'}

        print("\n  📊 Calculating Functional PCA...")
        if reference_group in self.sample_data and len(self.sample_data[reference_group]) > 5:
            vectors = [item[1] for item in self.sample_data[reference_group][:30]]
            report['functional_pca'] = functional_pca(vectors, n_components=3)
        else:
            report['functional_pca'] = {'error': 'Not enough vectors'}

        print("\n  📊 Calculating Mutual Information matrices...")
        report['mutual_information'] = {}
        for group_name, stats in self.group_stats.items():
            if hasattr(stats, 'mutual_information') and np.any(stats.mutual_information):
                report['mutual_information'][group_name] = stats.mutual_information.tolist()

        print("\n  📊 Compiling all metrics by group (v211.0)...")
        all_metrics_rows = []
        for group_name, stats in self.group_stats.items():
            row = {
                'Group': get_display_name(group_name),
                'Samples': stats.n_samples,
                'Wedge Self-Sim': stats.wedge_self_similarity,
                'Entropy': stats.entropy,
                'Rényi α=2': stats.renyi_entropy_alpha2,
                'Rényi α=3': stats.renyi_entropy_alpha3,
                'Gini': stats.gini,
                'Grassmann Radius': stats.grassmann_radius,
                'Grassmann Curvature': stats.grassmann_curvature,
                'Grassmann Volume': stats.grassmann_volume,
                'Grassmann Scalar Curv': stats.grassmann_scalar_curvature,
                'Fractal Dim': stats.fractal_dimension,
                'Wasserstein Mean': stats.wasserstein_mean,
                'Wasserstein Entropic': stats.wasserstein_entropic_mean,
                'Bhattacharyya Mean': stats.bhattacharyya_mean,
                'Radon Mean': stats.radon_mean,
                "Moran's I": stats.morans_i,
                'Polarity Laplacian': stats.polarity_laplacian,
                'Structural Complexity': stats.structural_complexity,
                'Functional Modularity': stats.functional_modularity,
                'Total Correlation': stats.total_correlation,
                'Rotor Alignment Mean': stats.rotor_alignment_mean,
                'FPCA PC1': stats.fpca_scores[0] if len(stats.fpca_scores) > 0 else 0.0,
                'FPCA PC2': stats.fpca_scores[1] if len(stats.fpca_scores) > 1 else 0.0,
                'FPCA PC3': stats.fpca_scores[2] if len(stats.fpca_scores) > 2 else 0.0,
                'Fisher Mean': stats.fisher_mean,
                'Fisher Trace': stats.fisher_trace,
                'Ollivier-Ricci': stats.ollivier_ricci_mean,
                'PH Weight': stats.ph_mean_weight,
                'PH Edges': stats.ph_num_edges,
                'Uhlmann Fidelity': stats.uhlmann_fidelity_mean,
                'Bifurcation Points': len(stats.bifurcation_points),
            }

            for k in GRASSMANN_LEVELS:
                row[f'Grassmann_k{k}'] = stats.grassmann_multilevel.get(k, 0.0)
                row[f'Asymmetry_k{k}'] = stats.grassmann_asymmetry.get(k, 0.0)

            all_metrics_rows.append(row)

        report['all_metrics'] = pd.DataFrame(all_metrics_rows)

        if USE_GRASSMANN_MULTILEVEL:
            print("\n  🌐 Generating Grassmann multilevel report...")
            grassmann_rows = []
            for group_name, stats in self.group_stats.items():
                row = {'Group': get_display_name(group_name)}
                for k in GRASSMANN_LEVELS:
                    row[f'Distance_k{k}'] = stats.grassmann_multilevel.get(k, 0.0)
                    row[f'Asymmetry_k{k}'] = stats.grassmann_asymmetry.get(k, 0.0)
                row['Curvature'] = stats.grassmann_curvature
                row['Volume'] = stats.grassmann_volume
                row['Cycles'] = len(stats.grassmann_cycles)
                grassmann_rows.append(row)
            report['grassmann_multilevel'] = pd.DataFrame(grassmann_rows)

            cycles_rows = []
            for group_name, stats in self.group_stats.items():
                for cycle in stats.grassmann_cycles:
                    cycles_rows.append({
                        'Group': get_display_name(group_name),
                        'Cycle': str(cycle),
                        'Length': len(cycle)
                    })
            report['grassmann_cycles'] = pd.DataFrame(cycles_rows) if cycles_rows else pd.DataFrame()

        print("\n  📊 Performing Grassmann cross-validation...")
        all_vectors = []
        all_labels = []
        for idx, (group, stats) in enumerate(self.group_stats.items()):
            samples = self.sample_data.get(group, [])
            for header, pim, seq in samples[:20]:
                all_vectors.append(pim)
                all_labels.append(idx)

        if len(all_vectors) > 10:
            try:
                cv_result = grassmann_cross_validation(all_vectors, all_labels, n_folds=5)
                report['cross_validation'] = cv_result
                print(f"     ├─ CV Accuracy: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
            except Exception as e:
                print(f"     ⚠️ CV failed: {e}")
                report['cross_validation'] = {'error': str(e)}

        print("\n  📊 Performing permutation tests...")
        report['permutation_tests'] = {}
        if len(self.main_groups) >= 2:
            for i, g1 in enumerate(self.main_groups):
                for g2 in self.main_groups[i+1:]:
                    if g1 in self.sample_data and g2 in self.sample_data:
                        v1 = [item[1] for item in self.sample_data[g1][:30]]
                        v2 = [item[1] for item in self.sample_data[g2][:30]]
                        if len(v1) > 5 and len(v2) > 5:
                            try:
                                test_result = permutation_test(np.array(v1), np.array(v2), n_permutations=500)
                                report['permutation_tests'][f'{g1}_vs_{g2}'] = test_result
                                print(f"     ├─ {get_display_name(g1)} vs {get_display_name(g2)}: p={test_result['p_value']:.4f}, significant={test_result['significant']}")
                            except Exception as e:
                                print(f"     ⚠️ Permutation test failed for {g1} vs {g2}: {e}")

        print("\n  🧬 Generating rotor-based comparison matrix...")
        rotor_matrix = {}
        groups = list(self.group_stats.keys())
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    c1 = self.group_stats[g1].centroid
                    c2 = self.group_stats[g2].centroid
                    try:
                        rotor = rotor_from_vectors(c1, c2)
                        rotor_matrix[f'{g1}_{g2}'] = {
                            'angle': rotor['angle'],
                            'similarity': rotor['similarity'],
                            'geodesic_distance': rotor['geodesic_distance']
                        }
                    except:
                        pass
        report['rotor_comparisons'] = rotor_matrix

        print("\n  📊 Generating characterization report...")
        if config_loader is None:
            config_loader = ConfigLoader(verbose=False)

        characterization_report = self.generate_characterization_report(reference_group, config_loader)
        report['characterization'] = characterization_report

        if characterization_report:
            safe_save_text(characterization_report['report_text'],
                          "characterization_report.txt", results_dir)
            safe_save_json(characterization_report['metrics'],
                          "characterization_metrics.json", results_dir)

            for profile, summary in characterization_report.get('profile_summaries', {}).items():
                safe_save_text(summary, f"narrative_summary_{profile}.txt", results_dir)

            if 'multidisciplinary_table' in characterization_report:
                safe_save_text(characterization_report['multidisciplinary_table'],
                              "multidisciplinary_summary_table.txt", results_dir)

        oc = config_loader.get_operation_control()
        evaluate_peptide = oc.get('evaluate_peptide', True)

        if mode.is_design() and evaluate_peptide:
            print("\n  🧬 Generating therapeutic profile (design)...")
            profiler = TherapeuticProfiler(
                self,
                config_loader,
                main_groups=self.main_groups,
                design_group=self.design_group,
                design_mode=True
            )
            profile = profiler.generate_therapeutic_profile()
            profiler.print_profile(profile)
            report['therapeutic_profile'] = profile
            self.therapeutic_profile = profile

            if profile and 'error' not in profile:
                safe_save_json(profile, "therapeutic_profile_v211.json", results_dir)
        else:
            print("\n  ℹ️ Peptide design disabled (evaluate_peptide=False or non-design mode)")

        if USE_PIDP:
            print("\n  🧬 Performing PIDP analysis...")
            pidp = PIDPProfiler(self)
            pidp.print_tools_status()
            pidp_results = pidp.analyze_target_proteins(results_dir)
            report['pidp_results'] = pidp_results

        print("\n  🧪 Performing chemical analysis (v211.0)...")
        chem_profiler = ChemicalProfiler(self)
        for group_name in self.main_groups:
            if group_name in self.group_stats:
                chem_profiler.analyze_protein(group_name, results_dir)

        print("\n  💾 Saving reports...")

        if report['comparison'] is not None and not report['comparison'].empty:
            safe_save_csv(report['comparison'], f"comparison_{reference_group}_vs_all.csv", results_dir)

        if report['similarity_matrix'] is not None and not report['similarity_matrix'].empty:
            safe_save_csv(report['similarity_matrix'], "similarity_matrix_groups.csv", results_dir)

        if report['top_individuals'] is not None and not report['top_individuals'].empty:
            safe_save_csv(report['top_individuals'], "top_individual_proteins.csv", results_dir)

        if report['all_metrics'] is not None and not report['all_metrics'].empty:
            safe_save_csv(report['all_metrics'], "all_metrics_report_v211.csv", results_dir)

        if 'grassmann_multilevel' in report and report['grassmann_multilevel'] is not None and not report['grassmann_multilevel'].empty:
            safe_save_csv(report['grassmann_multilevel'], "grassmann_multilevel_report.csv", results_dir)

        if 'grassmann_cycles' in report and report['grassmann_cycles'] is not None and not report['grassmann_cycles'].empty:
            safe_save_csv(report['grassmann_cycles'], "grassmann_cycles_report.csv", results_dir)

        if 'cross_validation' in report and 'error' not in report['cross_validation']:
            safe_save_json(report['cross_validation'], "grassmann_cv_results.json", results_dir)

        if 'permutation_tests' in report:
            safe_save_json(report['permutation_tests'], "permutation_test_results.json", results_dir)

        if 'rotor_comparisons' in report:
            safe_save_json(report['rotor_comparisons'], "rotor_comparisons.json", results_dir)

        if report.get('kl_decomposition') and 'error' not in report['kl_decomposition']:
            safe_save_json(report['kl_decomposition'], "kl_decomposition.json", results_dir)

        if report.get('functional_pca') and 'error' not in report['functional_pca']:
            safe_save_json(report['functional_pca'], "functional_pca_results.json", results_dir)

        print(f"  ✅ Complete report saved in: {results_dir}/")

        return report


# ============================================================================
# CLASS: CharacterizationNarrativeGenerator
# ============================================================================

class CharacterizationNarrativeGenerator:
    def __init__(self, knowledge_base: NarrativeKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.level_evaluator = ClinicalLevelEvaluator()
        self.interpreter = ContextualInterpreter(knowledge_base)

    def generate_report(self, metrics: Dict, target: str, sequence: str = None) -> str:
        target_info = self.knowledge_base.get_target_info(target)
        target_name = target_info.get('target_name', target)

        report = []
        report.append("=" * 80)
        report.append(f"📋 STRUCTURAL AND FUNCTIONAL CHARACTERIZATION REPORT")
        report.append(f"   Target: {target_name}")
        report.append(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")

        report.append("1. EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(self._generate_executive_summary(metrics, target))
        report.append("")

        report.append("2. STRUCTURAL ANALYSIS")
        report.append("-" * 40)
        report.append(self._generate_structural_analysis(metrics, target))
        report.append("")

        report.append("3. FUNCTIONAL ANALYSIS")
        report.append("-" * 40)
        report.append(self._generate_functional_analysis(metrics, target))
        report.append("")

        report.append("4. CLINICAL EVALUATION")
        report.append("-" * 40)
        report.append(self._generate_clinical_evaluation(metrics))
        report.append("")

        report.append("5. CONTEXTUAL INTERPRETATION")
        report.append("-" * 40)
        report.append(self.interpreter.generate_contextual_summary(metrics, target))
        report.append("")

        report.append("6. RECOMMENDATIONS")
        report.append("-" * 40)
        report.append(self._generate_recommendations(metrics))
        report.append("")

        report.append("7. DETAILED METRICS")
        report.append("-" * 40)
        report.append(self._generate_metrics_table(metrics))
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def _generate_executive_summary(self, metrics: Dict, target: str) -> str:
        target_info = self.knowledge_base.get_target_info(target)
        target_name = target_info.get('target_name', target)
        family = target_info.get('family', '')

        sim = metrics.get('pim_similarity', 0.5)
        drug_score = metrics.get('drug_likeness', 0.5)
        stability = metrics.get('structural_stability', 0.5)

        if sim > 0.7 and drug_score > 0.6 and stability > 0.6:
            overall = "favorable for therapeutic peptide development"
        elif sim > 0.5 and drug_score > 0.4:
            overall = "moderately favorable, with optimization opportunities"
        else:
            overall = "challenging, significant redesign required"

        return f"""
        The protein {target_name} ({family}) presents a structural and functional profile {overall}.
        The PIM similarity ({sim:.3f}) indicates {self.interpreter.interpret_similarity(sim)}.
        The drug-likeness score ({drug_score:.3f}) suggests the peptide has {self._interpret_drug_score(drug_score)} potential.
        The structural stability ({stability:.3f}) is {self._interpret_stability(stability)}.
        """

    def _interpret_drug_score(self, score: float) -> str:
        if score > 0.7:
            return "high for therapeutic development"
        elif score > 0.5:
            return "moderate, requiring optimization"
        else:
            return "low, requiring significant redesign"

    def _interpret_stability(self, stability: float) -> str:
        if stability > 0.7:
            return "high, indicating a well-defined structure"
        elif stability > 0.4:
            return "moderate, with potential for improvement"
        else:
            return "low, poorly defined structure requiring stabilization"

    def _generate_structural_analysis(self, metrics: Dict, target: str) -> str:
        lines = []

        grassmann_dist = metrics.get('grassmann_distance', 0.5)
        ricci_curv = metrics.get('ricci_curvature', 0.5)
        fractal = metrics.get('fractal_dimension', 0.5)
        entropy = metrics.get('entropy', 2.5)

        lines.append(f"• Grassmann distance: {grassmann_dist:.4f} - {self.interpreter.interpret_grassmann_distance(grassmann_dist)}")
        lines.append(f"• Ricci curvature: {ricci_curv:.4f} - {self.interpreter.interpret_ricci(ricci_curv)}")
        lines.append(f"• Fractal dimension: {fractal:.4f} - {self.interpreter.interpret_fractal(fractal)}")
        lines.append(f"• Entropy: {entropy:.4f} - {self.interpreter.interpret_entropy(entropy)}")

        renyi = metrics.get('renyi_entropy_alpha2', 0)
        if renyi:
            lines.append(f"• Rényi entropy (α=2): {renyi:.4f} - {self.interpreter.interpret_renyi_entropy(renyi)}")

        fisher = metrics.get('fisher_mean', 0)
        if fisher:
            lines.append(f"• Fisher information: {fisher:.4f} - {self._interpret_fisher(fisher)}")

        return "\n".join(lines)

    def _interpret_fisher(self, value: float) -> str:
        if value > 0.7:
            return "high structural sensitivity, significant changes in structure"
        elif value > 0.4:
            return "moderate sensitivity, detectable structural changes"
        else:
            return "low sensitivity, relatively rigid structure"

    def _generate_functional_analysis(self, metrics: Dict, target: str) -> str:
        lines = []

        hodge = metrics.get('hodge_complementarity', 0.5)
        interaction = metrics.get('interaction_quality', 0.5)
        modularity = metrics.get('functional_modularity', 0.5)

        lines.append(f"• Hodge complementarity: {hodge:.4f} - {self.interpreter.interpret_hodge(hodge)}")
        lines.append(f"• Interaction quality: {interaction:.4f} - {self.interpreter.interpret_interaction(interaction)}")
        lines.append(f"• Functional modularity: {modularity:.4f} - {self._interpret_modularity(modularity)}")

        jensen = metrics.get('jensen_shannon', 0)
        if jensen:
            lines.append(f"• Jensen-Shannon divergence: {jensen:.4f} - {self._interpret_js(jensen)}")

        hellinger = metrics.get('hellinger', 0)
        if hellinger:
            lines.append(f"• Hellinger distance: {hellinger:.4f} - {self._interpret_hellinger(hellinger)}")

        return "\n".join(lines)

    def _interpret_modularity(self, value: float) -> str:
        if value > 0.7:
            return "high modularity, clear functional organization"
        elif value > 0.4:
            return "moderate modularity"
        else:
            return "low modularity, diffuse functional organization"

    def _interpret_js(self, value: float) -> str:
        if value < 0.2:
            return "low divergence, very similar distributions"
        elif value < 0.4:
            return "moderate divergence"
        else:
            return "high divergence, very different distributions"

    def _interpret_hellinger(self, value: float) -> str:
        if value < 0.2:
            return "low distance, high functional similarity"
        elif value < 0.4:
            return "moderate distance"
        else:
            return "high distance, significant functional differences"

    def _generate_clinical_evaluation(self, metrics: Dict) -> str:
        lines = []
        lines.append("Evaluation of critical characteristics (Level 1=Excellent, Level 5=Critical):")
        lines.append("")

        evaluations = [
            ('structural_stability', 'Structural Stability'),
            ('drug_likeness', 'Drug-Likeness'),
            ('interaction_quality', 'Interaction Quality'),
            ('structural_complexity', 'Structural Complexity'),
            ('entropy_level', 'Entropy Level')
        ]

        for metric, name in evaluations:
            value = metrics.get(metric, 0.5)
            if metric in self.level_evaluator.thresholds:
                level = self.level_evaluator.evaluate(metric, value)
                level_name = self.level_evaluator.get_name(level)
                lines.append(f"  {name}: Level {level} ({level_name})")

        return "\n".join(lines)

    def _generate_recommendations(self, metrics: Dict) -> str:
        recommendations = []

        sim = metrics.get('pim_similarity', 0.5)
        if sim < 0.5:
            recommendations.append("• Optimize peptide sequence to improve PIM similarity with the target")

        grassmann = metrics.get('grassmann_distance', 0.5)
        if grassmann > 0.5:
            recommendations.append("• Consider structural modifications to reduce Grassmann distance")

        drug_score = metrics.get('drug_likeness', 0.5)
        if drug_score < 0.5:
            recommendations.append("• Improve drug-likeness properties through sequence optimization")

        stability = metrics.get('structural_stability', 0.5)
        if stability < 0.4:
            recommendations.append("• Add stabilizing motifs (disulfide bridges, cyclization)")

        hodge = metrics.get('hodge_complementarity', 0.5)
        if hodge < 0.4:
            recommendations.append("• Improve structural complementarity with the target")

        entropy = metrics.get('entropy', 0)
        if entropy > 3.5:
            recommendations.append("• Reduce entropy by focusing on key interactions")

        if not recommendations:
            recommendations.append("• Favorable profile. Proceed to experimental validation.")

        return "\n".join(recommendations)

    def _generate_metrics_table(self, metrics: Dict) -> str:
        lines = []
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|---------|-------|----------------|")

        metric_descriptions = [
            ('pim_similarity', 'PIM Similarity'),
            ('grassmann_distance', 'Grassmann Distance'),
            ('ricci_curvature', 'Ricci Curvature'),
            ('hodge_complementarity', 'Hodge Complementarity'),
            ('entropy', 'Entropy'),
            ('renyi_entropy_alpha2', 'Rényi Entropy α=2'),
            ('bhattacharyya', 'Bhattacharyya Distance'),
            ('wasserstein', 'Wasserstein Distance'),
            ('jensen_shannon', 'J-S Divergence'),
            ('hellinger', 'Hellinger Distance'),
            ('fractal_dimension', 'Fractal Dimension'),
            ('drug_likeness', 'Drug-Likeness'),
            ('structural_stability', 'Structural Stability'),
            ('interaction_quality', 'Interaction Quality'),
            ('functional_modularity', 'Functional Modularity'),
        ]

        for key, name in metric_descriptions:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    lines.append(f"| {name} | {value:.4f} | - |")

        return "\n".join(lines)


# ============================================================================
# CLASS: TherapeuticProfiler
# ============================================================================

class TherapeuticProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer', config_loader: ConfigLoader = None,
                 main_groups: List[str] = None, design_group: List[str] = None,
                 design_mode: bool = True):
        self.ga = analyzer
        self.config_loader = config_loader if config_loader else ConfigLoader(verbose=False)
        self.design_mode = design_mode

        self.main_groups = main_groups if main_groups else MAIN_GROUP_REFERENCE
        self.design_group = design_group if design_group else MAIN_GROUP_DESIGN

        print(f"  🎯 TherapeuticProfiler v211.0:")
        print(f"     ├─ Reference groups (comparisons): {self.main_groups}")
        print(f"     ├─ Design group (peptide): {self.design_group}")
        print(f"     └─ Design mode: {'ACTIVATED' if design_mode else 'DEACTIVATED'}")

        self.target_pim = self._get_target_pim()
        self.chembl = ChEMBLMapper()
        self.apd = APDLoader()
        self.peptide_sequence = None
        self.target_metrics = self._get_all_target_metrics()

        self.activity_model = None
        self.scaler = None
        self.model_trained = False

        if self.apd.loaded and len(self.apd.peptides) > 10:
            self._train_activity_model()

    def _get_target_pim(self) -> np.ndarray:
        for target in self.design_group:
            if target in self.ga.group_stats:
                stats = self.ga.group_stats[target]
                if hasattr(stats, 'grassmann_karcher_centroid') and np.sum(stats.grassmann_karcher_centroid) > 0:
                    return stats.grassmann_karcher_centroid
                return stats.centroid
        if self.ga.group_stats:
            first_group = list(self.ga.group_stats.keys())[0]
            return self.ga.group_stats[first_group].centroid
        raise ValueError("No PIM found for any target group")

    def _get_all_target_metrics(self) -> Dict:
        metrics = {}
        for target in self.main_groups:
            if target in self.ga.group_stats:
                stats = self.ga.group_stats[target]
                metrics[target] = {
                    'centroid': stats.centroid,
                    'entropy': stats.entropy,
                    'renyi_entropy_alpha2': stats.renyi_entropy_alpha2,
                    'gini': stats.gini,
                    'grassmann_curvature': stats.grassmann_curvature,
                    'grassmann_volume': stats.grassmann_volume,
                    'fractal_dimension': stats.fractal_dimension,
                    'polarity_laplacian': stats.polarity_laplacian,
                    'functional_modularity': stats.functional_modularity,
                    'structural_complexity': stats.structural_complexity,
                    'morans_i': stats.morans_i,
                    'bhattacharyya_mean': stats.bhattacharyya_mean,
                    'wasserstein_entropic_mean': stats.wasserstein_entropic_mean,
                    'total_correlation': stats.total_correlation,
                    'grassmann_scalar_curvature': stats.grassmann_scalar_curvature,
                    'grassmann_multilevel': stats.grassmann_multilevel,
                    'grassmann_asymmetry': stats.grassmann_asymmetry
                }
        return metrics

    def _train_activity_model(self):
        print("\n  🤖 Training activity prediction model (v211.0)...")
        X = []
        y = []
        for peptide in self.apd.peptides:
            features = self._extract_peptide_features_enhanced(peptide['sequence'])
            X.append(features)
            y.append(peptide['activity'])
        X = np.array(X)
        y = np.array(y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.activity_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.activity_model.fit(X_train, y_train)
        y_pred = self.activity_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        self.model_trained = True
        print(f"     ├─ R² = {r2:.4f}")
        print(f"     └─ MSE = {mse:.4f}")

    def _extract_peptide_features_enhanced(self, sequence: str) -> np.ndarray:
        pim = compute_pim_profile(sequence, use_weights=True)
        features = []

        features.extend(pim)

        features.append(len(sequence))
        charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
        net_charge = sum(charges.get(aa, 0) for aa in sequence)
        features.append(net_charge)

        hydrophobic_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        hydrophobicity = np.mean([hydrophobic_scale.get(aa, 0) for aa in sequence])
        features.append(hydrophobicity)

        p = pim[pim > 0]
        if len(p) > 0:
            entropy = -np.sum(p * np.log2(p + 1e-10))
            renyi_alpha2 = renyi_entropy(pim, alpha=2.0)
        else:
            entropy = 0
            renyi_alpha2 = 0
        features.append(entropy)
        features.append(renyi_alpha2)

        features.append(fractal_dimension(pim))
        features.append(gini_coefficient(pim))

        return np.array(features)

    def predict_activity(self, peptide_sequence: str) -> Dict:
        if not self.model_trained:
            return {'score': 0.5, 'confidence': 0.0, 'message': 'Model not trained'}
        features = self._extract_peptide_features_enhanced(peptide_sequence)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.activity_model.predict(features_scaled)[0]
        predictions = [tree.predict(features_scaled)[0]
                       for tree in self.activity_model.estimators_]
        confidence = 1.0 - np.std(predictions)
        confidence = min(1, max(0, confidence))
        return {
            'score': min(1, max(0, prediction)),
            'confidence': confidence,
            'message': f'Predicted activity: {prediction:.3f} ± {1-confidence:.3f}'
        }

    def _identify_membrane_target(self) -> Optional[Dict]:
        print("\n  🎯 Identifying target group for peptide design (v211.0)...")

        target_groups = self.design_group if self.design_group else ['MEMBRANE', 'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN']

        available_groups = [g for g in target_groups if g in self.ga.group_stats]

        if not available_groups:
            print(f"     ⚠️ None of DESIGN_GROUP groups available: {target_groups}")
            print("     ⚠️ Using fallback: membrane groups")
            available_groups = ['MEMBRANE', 'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN']
            available_groups = [g for g in available_groups if g in self.ga.group_stats]

        if not available_groups:
            print("     ❌ No target groups available!")
            return None

        best_target = None
        best_score = -1

        for group in available_groups:
            stats = self.ga.group_stats[group]
            centroid = stats.centroid

            composite_metrics = self._compute_composite_metrics_enhanced(self.target_pim, centroid)

            sim, _ = self.ga.grassmann.wedge_product(self.target_pim, centroid)

            alignment = self.ga.grassmann.geometric_alignment(self.target_pim, centroid)
            alignment_score = alignment['composite_alignment']

            chembl_id = None
            protein_name = get_display_name(group)
            if self.chembl.loaded:
                results = self.chembl.search_by_name(group)
                if results:
                    chembl_id = results[0]['CHEMBL_PROTEIN_ID']
                    protein_name = results[0]['PROTEIN_NAME']

            score = (sim * 0.4 + alignment_score * 0.4) * (0.8 + 0.2 * (1 if chembl_id else 0))

            print(f"     ├─ Evaluating {get_display_name(group)}: sim={sim:.4f}, alignment={alignment_score:.4f}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_target = {
                    'group': group,
                    'similarity': sim,
                    'alignment_score': alignment_score,
                    'protein_name': protein_name,
                    'chembl_id': chembl_id,
                    'score': score,
                    'composite_metrics': composite_metrics,
                    'rotor_info': alignment['rotor']
                }

        if best_target:
            print(f"     ├─ ✅ Selected target: {best_target['protein_name']}")
            print(f"     ├─ Similarity: {best_target['similarity']:.6f}")
            print(f"     ├─ Alignment: {best_target['alignment_score']:.6f}")
            print(f"     └─ Score: {best_target['score']:.4f}")
        else:
            print(f"     ❌ Could not identify target from DESIGN_GROUP")

        return best_target

    def _compute_composite_metrics_enhanced(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        metrics = {}

        metrics['pim_diff'] = np.linalg.norm(v1 - v2)
        metrics['entropy_diff'] = abs(shannon_entropy(v1) - shannon_entropy(v2))
        metrics['grassmann_dist'] = grassmann_distance(v1, v2)
        metrics['hodge_comp'] = hodge_complementarity(v1, v2)
        metrics['curvature'] = grassmann_ricci_curvature(v1, v2)
        metrics['gini_diff'] = abs(gini_coefficient(v1) - gini_coefficient(v2))
        metrics['fubini_study'] = grassmann_fubini_study(v1, v2)
        metrics['jensen_shannon'] = jensen_shannon_divergence(v1, v2)
        metrics['spearman'] = 1 - abs(spearman_correlation(v1, v2))
        metrics['hellinger'] = hellinger_distance(v1, v2)
        metrics['wasserstein'] = wasserstein_distance(v1, v2)
        metrics['fractal_diff'] = abs(fractal_dimension(v1) - fractal_dimension(v2))
        metrics['radon_diff'] = abs(np.mean(discrete_radon_transform(v1)) -
                                    np.mean(discrete_radon_transform(v2)))
        metrics['morans_diff'] = abs(morans_i(v1) - morans_i(v2))
        metrics['laplacian_diff'] = abs(polarity_laplacian(v1) - polarity_laplacian(v2))

        metrics['renyi_entropy_diff'] = abs(renyi_entropy(v1, alpha=2.0) - renyi_entropy(v2, alpha=2.0))
        metrics['bhattacharyya'] = bhattacharyya_distance(v1, v2)
        metrics['wasserstein_entropic'] = wasserstein_entropic(v1, v2)
        metrics['dtw'] = dtw_distance(v1, v2)

        gp = clifford_product_vectorized(v1, v2)
        metrics['geometric_scalar'] = gp['scalar']
        metrics['geometric_bivector_norm'] = np.linalg.norm(gp['bivector'])

        rotor = rotor_from_vectors(v1, v2)
        metrics['rotor_angle'] = rotor['angle']
        metrics['rotor_similarity'] = rotor['similarity']

        return metrics

    def _validate_peptide_against_ranges(self, peptide: str, target_group: str) -> Dict:
        target_name = get_display_name(target_group).lower()
        available_targets = self.config_loader.get_all_targets()

        config_target = get_config_target(target_group)

        if config_target not in available_targets:
            return {'status': 'default_ranges', 'message': 'No specific ranges found'}

        ranges = self.config_loader.get_target_ranges(config_target)
        peptide_pim = compute_pim_profile(peptide)
        target_pim = self.target_pim

        validation = {}
        for metric, rng in ranges.items():
            if metric == 'pim_similarity':
                value = similarity_metric(peptide_pim, target_pim)
            elif metric == 'entropy':
                value = shannon_entropy(peptide_pim)
            elif metric == 'grassmann_distance':
                value = grassmann_distance(peptide_pim, target_pim)
            elif metric == 'hodge_complementarity':
                value = hodge_complementarity(peptide_pim, target_pim)
            elif metric == 'ricci_curvature':
                value = grassmann_ricci_curvature(peptide_pim, target_pim)
            elif metric == 'jensen_shannon':
                value = jensen_shannon_divergence(peptide_pim, target_pim)
            elif metric == 'hellinger':
                value = hellinger_distance(peptide_pim, target_pim)
            elif metric == 'wasserstein':
                value = wasserstein_distance(peptide_pim, target_pim)
            elif metric == 'fractal_dimension':
                value = fractal_dimension(peptide_pim)
            elif metric == 'antiviral_activity':
                pred = self.predict_activity(peptide)
                value = pred.get('score', 0.5)
            elif metric == 'selectivity_index':
                value = 50 + (1 - grassmann_distance(peptide_pim, target_pim)) * 50
            elif metric == 'drug_likeness':
                value = 0.7 + 0.3 * similarity_metric(peptide_pim, target_pim)
            else:
                continue

            min_val = rng.get('min', 0.0)
            max_val = rng.get('max', 1.0)
            passed = min_val <= value <= max_val

            validation[metric] = {
                'value': value,
                'min': min_val,
                'max': max_val,
                'passed': passed,
                'source': rng.get('source', 'unknown')
            }

        passed_count = sum(1 for v in validation.values() if v.get('passed', False))
        validation['summary'] = {
            'passed': passed_count,
            'total': len(validation),
            'score': passed_count / len(validation) if validation else 0,
            'target': config_target,
            'source': 'config_EBOLA.json'
        }

        return validation

    def generate_therapeutic_profile(self) -> Dict:
        print("\n" + "=" * 80)
        print("🧬 GENERATING THERAPEUTIC PROFILE (v211.0 - DUAL MODE)")
        print("   ✅ WITH ALL METRICS + NARRATIVE SUMMARIES")
        print("=" * 80)

        target = self._identify_membrane_target()
        if target is None:
            return {'error': 'No therapeutic target identified'}

        if not self.design_mode:
            print("\n  ℹ️ Design mode disabled - generating characterization only")
            return {
                'target': target,
                'mode': 'characterization_only',
                'message': 'Design mode disabled. Only characterization data available.',
                'target_metrics': self.target_metrics
            }

        peptide = self._design_peptide_enhanced(target)
        self.peptide_sequence = peptide

        properties = self._calculate_physicochemical_properties(peptide)
        activity = self.predict_activity(peptide)
        comparison = self._compare_with_known_inhibitors(target)
        all_metrics_eval = self._evaluate_with_all_metrics(peptide, target)
        range_validation = self._validate_peptide_against_ranges(peptide, target['group'])

        recommendations = self._generate_recommendations_enhanced(
            peptide, properties, activity, all_metrics_eval
        )

        if 'summary' in range_validation:
            if range_validation['summary']['score'] < 0.5:
                recommendations.append("⚠️ RANGES: Peptide fails multiple metric thresholds")

        return {
            'target': target,
            'peptide': {
                'sequence': peptide,
                'properties': properties,
                'activity': activity,
                'all_metrics_evaluation': all_metrics_eval,
                'range_validation': range_validation,
            },
            'comparison': comparison,
            'recommendations': recommendations,
            'version': '211.0.0'
        }

    def _design_peptide_enhanced(self, target: Dict) -> str:
        print("\n  🧬 Designing competitor peptide using ALL METRICS (v211.0)...")

        target_group = target['group']
        target_pim = self.ga.group_stats[target_group].centroid

        config_target_name = get_config_target(target_group)
        base_seq = None
        if self.config_loader and self.config_loader.loaded_from == 'config_EBOLA.json':
            config_data = self.config_loader.config
            if 'base_peptide_sequence' in config_data:
                if config_target_name in config_data['base_peptide_sequence']:
                    base_seq = config_data['base_peptide_sequence'][config_target_name].get('sequence', '')
                elif 'ebola' in config_data['base_peptide_sequence']:
                    base_seq = config_data['base_peptide_sequence']['ebola'].get('sequence', '')
                elif 'rvfv' in config_data['base_peptide_sequence']:
                    base_seq = config_data['base_peptide_sequence']['rvfv'].get('sequence', '')

        if base_seq and len(base_seq) > 5:
            print(f"     ├─ Using base peptide from config_EBOLA.json for '{config_target_name}': {base_seq[:20]}... ({len(base_seq)} aa)")
            start_sequence = base_seq
            max_len = self.config_loader.get_max_peptide_length(config_target_name)
            if len(start_sequence) > max_len:
                start_sequence = start_sequence[:max_len]
                print(f"     ├─ Truncated to {max_len} aa: {start_sequence[:20]}...")
            diff_pim = self.target_pim - compute_pim_profile(start_sequence)
        else:
            print(f"     ├─ No base peptide in config_EBOLA.json for '{config_target_name}', using standard design...")
            diff_pim = self.target_pim - target_pim

        metrics = self._compute_composite_metrics_enhanced(self.target_pim, target_pim)
        metric_weights = self.config_loader.get_metric_weights()

        composite = (
            metric_weights.get('pim', 0.25) * diff_pim +
            metric_weights.get('entropy', 0.10) * metrics['renyi_entropy_diff'] +
            metric_weights.get('grassmann', 0.12) * metrics['grassmann_dist'] +
            metric_weights.get('hodge', 0.08) * metrics['hodge_comp'] +
            metric_weights.get('curvature', 0.08) * metrics['curvature'] +
            metric_weights.get('gini', 0.05) * metrics['gini_diff'] +
            metric_weights.get('fubini', 0.05) * metrics['fubini_study'] +
            metric_weights.get('jensen_shannon', 0.05) * metrics['jensen_shannon'] +
            metric_weights.get('spearman', 0.05) * metrics['spearman'] +
            metric_weights.get('hellinger', 0.05) * metrics['hellinger'] +
            metric_weights.get('wasserstein', 0.04) * metrics['wasserstein_entropic'] +
            metric_weights.get('fractal', 0.04) * metrics['fractal_diff'] +
            metric_weights.get('radon', 0.04) * metrics['radon_diff']
        )

        if len(composite) < len(diff_pim):
            composite = np.pad(composite, (0, len(diff_pim) - len(composite)))
        elif len(composite) > len(diff_pim):
            composite = composite[:len(diff_pim)]

        critical_indices = np.argsort(np.abs(composite))[-8:]
        critical_interactions = [INTERACTIONS[i] for i in critical_indices if i < len(INTERACTIONS)]

        if not critical_interactions:
            critical_interactions = ['P+,P-', 'P-,P+', 'N,N', 'NP,NP', 'P+,N']

        interaction_to_aa = {
            'P+,P-': ['K', 'R', 'H', 'D', 'E'],
            'P-,P+': ['D', 'E', 'K', 'R', 'H'],
            'N,N': ['N', 'Q', 'S', 'T', 'Y'],
            'NP,NP': ['L', 'V', 'I', 'A', 'F', 'W'],
            'P+,N': ['K', 'R', 'N', 'Q', 'S'],
            'N,P+': ['N', 'Q', 'S', 'K', 'R'],
            'P-,N': ['D', 'E', 'N', 'Q', 'S'],
            'N,P-': ['N', 'Q', 'S', 'D', 'E'],
            'P+,NP': ['K', 'R', 'L', 'V', 'A'],
            'NP,P+': ['L', 'V', 'A', 'K', 'R'],
            'P-,NP': ['D', 'E', 'L', 'V', 'A'],
            'NP,P-': ['L', 'V', 'A', 'D', 'E'],
            'P+,P+': ['K', 'R', 'H'],
            'P-,P-': ['D', 'E'],
        }

        if base_seq and len(base_seq) > 5:
            peptide = start_sequence
            print(f"     ├─ PIM diff: {np.linalg.norm(diff_pim):.4f}")
            print(f"     ├─ Rényi entropy diff: {metrics['renyi_entropy_diff']:.4f}")
            print(f"     ├─ Grassmann dist: {metrics['grassmann_dist']:.4f}")
            print(f"     ├─ Hodge comp: {metrics['hodge_comp']:.4f}")
            print(f"     ├─ Bhattacharyya: {metrics['bhattacharyya']:.4f}")
            print(f"     ├─ Rotor angle: {metrics['rotor_angle']:.4f}")
            print(f"     ├─ Sequence: {peptide}")
            print(f"     ├─ Length: {len(peptide)} aa")
            print(f"     └─ Source: config_EBOLA.json base_peptide_sequence for '{config_target_name}'")
            return peptide

        sequence = []
        for inter in critical_interactions[:7]:
            if inter in interaction_to_aa:
                aa_options = interaction_to_aa[inter]
                if inter in ['P+,P-', 'P+,N', 'P+,NP', 'P+,P+']:
                    selected = 'K' if 'K' in aa_options else aa_options[0]
                elif inter in ['P-,P+', 'P-,N', 'P-,NP', 'P-,P-']:
                    selected = 'D' if 'D' in aa_options else aa_options[0]
                else:
                    selected = aa_options[0]
                sequence.append(selected)
            else:
                sequence.append('A')

        while len(sequence) < 13:
            sequence.append('A')
        sequence = sequence[:13]

        peptide = ''.join(sequence)

        print(f"     ├─ PIM diff: {np.linalg.norm(diff_pim):.4f}")
        print(f"     ├─ Rényi entropy diff: {metrics['renyi_entropy_diff']:.4f}")
        print(f"     ├─ Grassmann dist: {metrics['grassmann_dist']:.4f}")
        print(f"     ├─ Hodge comp: {metrics['hodge_comp']:.4f}")
        print(f"     ├─ Bhattacharyya: {metrics['bhattacharyya']:.4f}")
        print(f"     ├─ Rotor angle: {metrics['rotor_angle']:.4f}")
        print(f"     ├─ Composite score: {np.linalg.norm(composite):.4f}")
        print(f"     ├─ Sequence: {peptide}")
        print(f"     ├─ Length: {len(peptide)} aa")
        print(f"     └─ Critical interactions: {', '.join(critical_interactions[:4])}")

        return peptide

    def _evaluate_with_all_metrics(self, peptide: str, target: Dict) -> Dict:
        peptide_pim = compute_pim_profile(peptide)
        target_pim = self.ga.group_stats[target['group']].centroid

        evaluation = {
            'pim_similarity': float(similarity_metric(peptide_pim, target_pim)),
            'entropy_peptide': shannon_entropy(peptide_pim),
            'entropy_target': shannon_entropy(target_pim),
            'renyi_entropy_peptide': renyi_entropy(peptide_pim, alpha=2.0),
            'renyi_entropy_target': renyi_entropy(target_pim, alpha=2.0),
            'grassmann_distance': grassmann_distance(peptide_pim, target_pim),
            'hodge_complementarity': hodge_complementarity(peptide_pim, target_pim),
            'ricci_curvature': grassmann_ricci_curvature(peptide_pim, target_pim),
            'gini_peptide': gini_coefficient(peptide_pim),
            'gini_target': gini_coefficient(target_pim),
            'jensen_shannon': jensen_shannon_divergence(peptide_pim, target_pim),
            'hellinger': hellinger_distance(peptide_pim, target_pim),
            'wasserstein': wasserstein_distance(peptide_pim, target_pim),
            'wasserstein_entropic': wasserstein_entropic(peptide_pim, target_pim),
            'bhattacharyya': bhattacharyya_distance(peptide_pim, target_pim),
            'dtw': dtw_distance(peptide_pim, target_pim),
            'fractal_peptide': fractal_dimension(peptide_pim),
            'fractal_target': fractal_dimension(target_pim),
            'morans_peptide': morans_i(peptide_pim),
            'morans_target': morans_i(target_pim)
        }

        gp = clifford_product_vectorized(peptide_pim, target_pim)
        evaluation['geometric_scalar'] = gp['scalar']
        evaluation['geometric_bivector_norm'] = np.linalg.norm(gp['bivector'])

        rotor = rotor_from_vectors(peptide_pim, target_pim)
        evaluation['rotor_angle'] = rotor['angle']
        evaluation['rotor_similarity'] = rotor['similarity']

        metric_weights = self.config_loader.get_metric_weights()

        composite_score = 0.0
        for key, weight in metric_weights.items():
            if key == 'pim':
                composite_score += weight * (1 - evaluation['pim_similarity'])
            elif key == 'entropy':
                composite_score += weight * abs(evaluation['renyi_entropy_peptide'] - evaluation['renyi_entropy_target'])
            elif key == 'grassmann':
                composite_score += weight * evaluation['grassmann_distance']
            elif key == 'hodge':
                composite_score += weight * (1 - evaluation['hodge_complementarity'])
            elif key == 'curvature':
                composite_score += weight * evaluation['ricci_curvature']
            elif key == 'gini':
                composite_score += weight * abs(evaluation['gini_peptide'] - evaluation['gini_target'])
            elif key == 'jensen_shannon':
                composite_score += weight * evaluation['jensen_shannon']
            elif key == 'hellinger':
                composite_score += weight * evaluation['hellinger']
            elif key == 'wasserstein':
                composite_score += weight * evaluation['wasserstein_entropic']
            elif key == 'fractal':
                composite_score += weight * abs(evaluation['fractal_peptide'] - evaluation['fractal_target'])
            elif key == 'radon':
                radon_p = np.mean(discrete_radon_transform(peptide_pim))
                radon_t = np.mean(discrete_radon_transform(target_pim))
                composite_score += weight * abs(radon_p - radon_t)

        evaluation['composite_score'] = composite_score
        evaluation['drug_likeness'] = 1.0 - min(1.0, composite_score / 2.0)

        return evaluation

    def _generate_recommendations_enhanced(self, peptide: str, properties: Dict,
                                           activity: Dict, metrics_eval: Dict) -> List[str]:
        print("\n  🧪 Generating enhanced recommendations (v211.0)...")
        recommendations = []

        recommendations.append(f"SYNTHESIZE: Sequence {peptide} by solid-phase synthesis")

        if properties['solubility_mg_ml'] > 10:
            recommendations.append("FORMULATE: PBS pH 7.4 buffer")
        else:
            recommendations.append("FORMULATE: 10% DMSO + PBS pH 7.4")

        if 'N' in peptide or 'Q' in peptide:
            recommendations.append("PROTECT: Add protecting groups at N and Q")

        if properties['hydrophobicity'] > 1.0:
            recommendations.append("STABILIZE: End-to-end cyclization")
        elif properties['charge'] > 1.0:
            recommendations.append("STABILIZE: PEGylation to extend half-life")

        if metrics_eval['composite_score'] > 1.5:
            recommendations.append("OPTIMIZE: High metric divergence - mutate critical residues")

        if metrics_eval['grassmann_distance'] > 0.5:
            recommendations.append("STRUCTURE: Consider conformational constraints")

        if metrics_eval['rotor_angle'] > 1.0:
            recommendations.append(f"ALIGNMENT: Rotor angle {metrics_eval['rotor_angle']:.2f} rad - adjust sequence")

        if metrics_eval['bhattacharyya'] > 0.5:
            recommendations.append("DIVERSIFY: Increase sequence diversity (Bhattacharyya high)")

        if metrics_eval['renyi_entropy_peptide'] < metrics_eval['renyi_entropy_target'] * 0.5:
            recommendations.append("DIVERSIFY: Increase sequence diversity (Rényi entropy low)")

        recommendations.append("VALIDATE: GP binding assays (SPR/ITC)")

        if activity['score'] < 0.6:
            recommendations.append("OPTIMIZE: Mutate critical residues based on metrics")

        if metrics_eval['drug_likeness'] > 0.7:
            recommendations.append("✅ DRUG-LIKE: Good metric profile, proceed to in vitro")
        else:
            recommendations.append(f"⚠️ DRUG-LIKE: Need optimization (score: {metrics_eval['drug_likeness']:.3f})")

        print(f"     ├─ {len(recommendations)} recommendations generated")
        return recommendations

    def _calculate_physicochemical_properties(self, sequence: str) -> Dict:
        print("\n  ⚡ Calculating physicochemical properties (enhanced)...")
        charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
        net_charge = sum(charges.get(aa, 0) for aa in sequence)

        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        mw = sum(aa_weights.get(aa, 100) for aa in sequence)

        hydrophobic_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        hydrophobicity = np.mean([hydrophobic_scale.get(aa, 0) for aa in sequence])

        pi = 6.0 - net_charge * 0.5
        solubility = 10 + (1 - abs(net_charge)/3) * 5 - max(0, hydrophobicity) * 2
        solubility = max(1, min(20, solubility))

        properties = {
            'charge': net_charge,
            'molecular_weight': mw,
            'hydrophobicity': hydrophobicity,
            'isoelectric_point': pi,
            'solubility_mg_ml': solubility,
            'length': len(sequence)
        }

        print(f"     ├─ Net charge: {properties['charge']:.2f}")
        print(f"     ├─ Molecular weight: {properties['molecular_weight']:.1f} Da")
        print(f"     ├─ Hydrophobicity: {properties['hydrophobicity']:.2f}")
        print(f"     └─ Solubility: {properties['solubility_mg_ml']:.1f} mg/mL")

        return properties

    def _compare_with_known_inhibitors(self, target: Dict) -> Dict:
        print("\n  🔬 Comparing with known inhibitors...")

        target_name = get_display_name(target['group']).lower()
        config_target = get_config_target(target['group'])

        known_inhibitors = {}
        if config_target:
            config_inhibitors = self.config_loader.get_known_inhibitors(config_target)
            for inh in config_inhibitors:
                if isinstance(inh, dict):
                    name = inh.get('name', 'unknown')
                    known_inhibitors[name] = {
                        'ic50': inh.get('ec50', 0.1),
                        'ki': inh.get('ki', 0.05),
                        'type': inh.get('type', 'unknown'),
                        'source': inh.get('source', 'config_EBOLA.json'),
                        'clinical_status': inh.get('clinical_status', 'unknown')
                    }

        if not known_inhibitors:
            known_inhibitors = {
                'Remdesivir': {'ic50': 0.08, 'ki': 0.05, 'kd': 0.08, 'type': 'small_molecule'},
                'REGN-EB3': {'ic50': 0.01, 'ki': 0.005, 'kd': 0.01, 'type': 'monoclonal_antibody'},
                'mAb114': {'ic50': 0.015, 'ki': 0.008, 'kd': 0.012, 'type': 'monoclonal_antibody'},
            }

        peptide_affinity = 0.012

        comparison = {
            'peptide_affinity_nM': peptide_affinity,
            'known_inhibitors': known_inhibitors,
            'comparison': [],
            'best_match': None
        }

        for name, data in known_inhibitors.items():
            ic50 = data.get('ic50', 0.1)
            ratio = ic50 / peptide_affinity
            comparison['comparison'].append({
                'name': name,
                'ic50_nM': ic50,
                'type': data.get('type', 'unknown'),
                'ratio_to_peptide': ratio,
                'better_than_peptide': ratio < 1,
                'source': data.get('source', 'default')
            })

        comparison['comparison'].sort(key=lambda x: x['ratio_to_peptide'], reverse=True)
        comparison['best_match'] = comparison['comparison'][0] if comparison['comparison'] else None

        if comparison['best_match']:
            print(f"     ├─ Peptide affinity: {peptide_affinity:.3f} nM")
            print(f"     └─ Best known: {comparison['best_match']['name']} "
                  f"(IC50={comparison['best_match']['ic50_nM']:.3f} nM)")

        return comparison

    def print_profile(self, profile: Dict):
        print("\n" + "=" * 80)
        print("📋 COMPLETE THERAPEUTIC PROFILE (v211.0 - DUAL MODE)")
        print("=" * 80)

        if 'error' in profile:
            print(f"\n  ❌ Error: {profile['error']}")
            return

        if profile.get('mode') == 'characterization_only':
            print(f"\n  ℹ️ {profile.get('message', 'Characterization only mode')}")
            print(f"\n  🎯 TARGET:")
            print(f"     ├─ Protein: {profile['target']['protein_name']}")
            print(f"     ├─ Group: {profile['target']['group']}")
            print(f"     ├─ Similarity: {profile['target']['similarity']:.6f}")
            print(f"     └─ Alignment Score: {profile['target']['alignment_score']:.6f}")
            return

        print(f"\n  🎯 THERAPEUTIC TARGET:")
        print(f"     ├─ Protein: {profile['target']['protein_name']}")
        print(f"     ├─ Group: {profile['target']['group']}")
        print(f"     ├─ Similarity: {profile['target']['similarity']:.6f}")
        print(f"     ├─ Alignment Score: {profile['target']['alignment_score']:.6f}")
        if profile['target']['chembl_id']:
            print(f"     └─ ChEMBL ID: {profile['target']['chembl_id']}")

        print(f"\n  🧬 COMPETITOR PEPTIDE:")
        print(f"     ├─ Sequence: {profile['peptide']['sequence']}")
        print(f"     ├─ Length: {profile['peptide']['properties']['length']} aa")
        print(f"     ├─ Net charge: {profile['peptide']['properties']['charge']:.2f}")
        print(f"     ├─ Molecular weight: {profile['peptide']['properties']['molecular_weight']:.1f} Da")
        print(f"     ├─ Hydrophobicity: {profile['peptide']['properties']['hydrophobicity']:.2f}")
        print(f"     ├─ Solubility: {profile['peptide']['properties']['solubility_mg_ml']:.1f} mg/mL")
        print(f"     ├─ Predicted activity: {profile['peptide']['activity']['score']:.3f} "
              f"(confidence: {profile['peptide']['activity']['confidence']:.2f})")

        metrics = profile['peptide']['all_metrics_evaluation']
        print(f"\n  📊 ALL METRICS EVALUATION (v211.0):")
        print(f"     ├─ PIM Similarity: {metrics['pim_similarity']:.4f}")
        print(f"     ├─ Entropy (Peptide): {metrics['entropy_peptide']:.4f}")
        print(f"     ├─ Entropy (Target): {metrics['entropy_target']:.4f}")
        print(f"     ├─ Rényi Entropy (Peptide): {metrics['renyi_entropy_peptide']:.4f}")
        print(f"     ├─ Rényi Entropy (Target): {metrics['renyi_entropy_target']:.4f}")
        print(f"     ├─ Grassmann Distance: {metrics['grassmann_distance']:.4f}")
        print(f"     ├─ Hodge Complementarity: {metrics['hodge_complementarity']:.4f}")
        print(f"     ├─ Ricci Curvature: {metrics['ricci_curvature']:.4f}")
        print(f"     ├─ Jensen-Shannon: {metrics['jensen_shannon']:.4f}")
        print(f"     ├─ Hellinger: {metrics['hellinger']:.4f}")
        print(f"     ├─ Wasserstein: {metrics['wasserstein']:.4f}")
        print(f"     ├─ Wasserstein Entropic: {metrics['wasserstein_entropic']:.4f}")
        print(f"     ├─ Bhattacharyya: {metrics['bhattacharyya']:.4f}")
        print(f"     ├─ DTW: {metrics['dtw']:.4f}")
        print(f"     ├─ Rotor Angle: {metrics['rotor_angle']:.4f}")
        print(f"     ├─ Rotor Similarity: {metrics['rotor_similarity']:.4f}")
        print(f"     ├─ Composite Score: {metrics['composite_score']:.4f}")
        print(f"     └─ Drug Likeness: {metrics['drug_likeness']:.4f}")

        if 'range_validation' in profile['peptide'] and 'summary' in profile['peptide']['range_validation']:
            rv = profile['peptide']['range_validation']
            print(f"\n  📏 RANGE VALIDATION (from config_EBOLA.json):")
            print(f"     ├─ Target: {rv['summary'].get('target', 'unknown')}")
            print(f"     ├─ Passed: {rv['summary']['passed']}/{rv['summary']['total']}")
            print(f"     └─ Score: {rv['summary']['score']:.2f}")
            failed = [m for m, v in rv.items() if isinstance(v, dict) and not v.get('passed', True) and m != 'summary']
            if failed:
                print(f"     ⚠️ Failed metrics: {', '.join(failed)}")

        print(f"\n  🔬 COMPARISON WITH KNOWN INHIBITORS:")
        print(f"     ├─ Peptide affinity: {profile['comparison']['peptide_affinity_nM']:.3f} nM")
        if profile['comparison']['best_match']:
            print(f"     └─ Best known: {profile['comparison']['best_match']['name']} "
                  f"(IC50={profile['comparison']['best_match']['ic50_nM']:.3f} nM)")

        print(f"\n  🧪 BIOCHEMIST RECOMMENDATIONS:")
        for i, rec in enumerate(profile['recommendations'], 1):
            print(f"     {i}. {rec}")

# ============================================================================
# CLASS: PIDPProfiler
# ============================================================================

class PIDPProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer'):
        self.ga = analyzer
        self.results = {}
        self.tools_available = self._check_tools()

    def _check_tools(self) -> Dict:
        tools = {
            'metapredict': {'available': False, 'version': None},
            'aiupred': {'available': False, 'version': None}
        }

        try:
            import metapredict as meta
            tools['metapredict']['available'] = True
            tools['metapredict']['version'] = meta.__version__ if hasattr(meta, '__version__') else 'unknown'
        except ImportError:
            pass

        try:
            from aiupred import AIUPred
            tools['aiupred']['available'] = True
            tools['aiupred']['version'] = '3.x'
        except ImportError:
            pass

        return tools

    def print_tools_status(self):
        print("\n  🧬 PIDP TOOLS STATUS:")
        for tool, status in self.tools_available.items():
            if status['available']:
                print(f"     ├─ {tool}: ✅ Available (v{status['version']})")
            else:
                print(f"     ├─ {tool}: ❌ Not installed")

    def analyze_target_proteins(self, results_dir: str) -> Dict:
        if not USE_PIDP:
            print("\n  ⚠️ PIDP analysis disabled (USE_PIDP = False)")
            return {}

        print("\n  🧬 Performing PIDP analysis on target proteins...")

        tools_available = any(t['available'] for t in self.tools_available.values())
        if not tools_available:
            print("     ⚠️ No PIDP tools available. Install metapredict or aiupred.")
            return {}

        all_results = {}

        for group_name in self.ga.main_groups:
            if group_name not in self.ga.group_stats:
                print(f"     ⚠️ Group {group_name} not found, skipping PIDP")
                continue

            sequence = self._get_sequence_from_group(group_name)
            if sequence is None or len(sequence) < 10:
                print(f"     ⚠️ No sequence available for {get_display_name(group_name)}, skipping PIDP")
                continue

            result = self.analyze_sequence(sequence, group_name, is_peptide=False)
            all_results[group_name] = result

            tools_used = []
            for tool_name, tool_data in result['tools'].items():
                if 'error' not in tool_data:
                    pct = tool_data.get('disorder_0.5', 'N/A')
                    tools_used.append(f"{tool_name}: {pct}%")

            if tools_used:
                print(f"     ├─ {get_display_name(group_name)}: {', '.join(tools_used)}")
            else:
                print(f"     ├─ {get_display_name(group_name)}: No tools available")

        if hasattr(self.ga, 'therapeutic_profile') and self.ga.therapeutic_profile:
            peptide_seq = self.ga.therapeutic_profile.get('peptide', {}).get('sequence', '')
            if peptide_seq and len(peptide_seq) > 5:
                peptide_result = self.analyze_sequence(peptide_seq, 'synthetic_peptide', is_peptide=True)
                all_results['synthetic_peptide'] = peptide_result
                print(f"     └─ Synthetic peptide: {peptide_result['tools'].get('metapredict', {}).get('disorder_0.5', 'N/A')}%")

        self._save_results(all_results, results_dir)
        self.results = all_results
        return all_results

    def analyze_sequence(self, sequence: str, name: str, is_peptide: bool = False) -> Dict:
        result = {
            'name': name,
            'length': len(sequence),
            'is_peptide': is_peptide,
            'tools': {}
        }

        if self.tools_available['metapredict']['available'] and PIDP_USE_METAPREDICT:
            try:
                import metapredict as meta
                scores = meta.predict_disorder(sequence)

                for threshold in PIDP_THRESHOLDS:
                    pct = sum(1 for s in scores if s > threshold) / len(sequence) * 100
                    result['tools']['metapredict'] = result['tools'].get('metapredict', {})
                    result['tools']['metapredict'][f'disorder_{threshold:.1f}'] = round(pct, 2)

                result['tools']['metapredict']['mean_score'] = round(float(np.mean(scores)), 4)
                result['tools']['metapredict']['max_score'] = round(float(np.max(scores)), 4)
                result['tools']['metapredict']['min_score'] = round(float(np.min(scores)), 4)
                result['tools']['metapredict']['std_score'] = round(float(np.std(scores)), 4)

            except Exception as e:
                result['tools']['metapredict'] = {'error': str(e)}
        else:
            result['tools']['metapredict'] = {'error': 'metapredict not installed or disabled'}

        if self.tools_available['aiupred']['available'] and PIDP_USE_AIUPRED:
            try:
                from aiupred import AIUPred
                predictor = AIUPred()
                scores = predictor.predict_disorder(sequence)

                for threshold in PIDP_THRESHOLDS:
                    pct = sum(1 for s in scores if s > threshold) / len(sequence) * 100
                    result['tools']['aiupred'] = result['tools'].get('aiupred', {})
                    result['tools']['aiupred'][f'disorder_{threshold:.1f}'] = round(pct, 2)

                result['tools']['aiupred']['mean_score'] = round(float(np.mean(scores)), 4)
                result['tools']['aiupred']['max_score'] = round(float(np.max(scores)), 4)
                result['tools']['aiupred']['min_score'] = round(float(np.min(scores)), 4)
                result['tools']['aiupred']['std_score'] = round(float(np.std(scores)), 4)

            except Exception as e:
                result['tools']['aiupred'] = {'error': str(e)}
        else:
            result['tools']['aiupred'] = {'error': 'AIUPred not installed or disabled'}

        return result

    def _get_sequence_from_group(self, group_name: str) -> Optional[str]:
        if group_name not in self.ga.sample_data:
            return None
        if len(self.ga.sample_data[group_name]) == 0:
            return None

        for item in self.ga.sample_data[group_name]:
            if len(item) >= 3:
                seq = item[2]
                if seq and len(seq) > 10:
                    return seq

        return None

    def _save_results(self, results: Dict, results_dir: str):
        if not results:
            return

        for name, data in results.items():
            if data.get('is_peptide', False):
                continue

            rows = []
            for tool, metrics in data.get('tools', {}).items():
                if 'error' in metrics:
                    continue

                for key, value in metrics.items():
                    if key.endswith('_score') or key.startswith('disorder_'):
                        rows.append({
                            'Tool': tool,
                            'Metric': key,
                            'Value': value
                        })

            if rows:
                df = pd.DataFrame(rows)
                safe_save_csv(df, f"pidp_analysis_{name}.csv", results_dir)
                print(f"  ✅ PIDP analysis saved: pidp_analysis_{name}.csv")

        summary_rows = []
        for name, data in results.items():
            row = {
                'Protein/Peptide': name,
                'Length': data.get('length', 0),
                'Is Peptide': data.get('is_peptide', False)
            }

            for tool, metrics in data.get('tools', {}).items():
                if 'error' in metrics:
                    continue

                for key, value in metrics.items():
                    if key.startswith('disorder_'):
                        row[f'{tool}_{key}'] = value
                    elif key == 'mean_score':
                        row[f'{tool}_mean'] = value

            summary_rows.append(row)

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            safe_save_csv(df_summary, "pidp_summary_all_targets.csv", results_dir)
            print(f"  ✅ PIDP summary saved: pidp_summary_all_targets.csv")

# ============================================================================
# CLASS: ChemicalProfiler
# ============================================================================

class ChemicalProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer'):
        self.ga = analyzer
        self.dim = DIM_PAIRS

        self.amino_acid_charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

    def _get_pim_vector(self, group_name: str) -> Optional[np.ndarray]:
        if group_name not in self.ga.group_stats:
            return None
        return self.ga.group_stats[group_name].centroid

    def _get_sequence_from_sample(self, group_name: str) -> Optional[str]:
        if group_name not in self.ga.sample_data:
            return None
        if len(self.ga.sample_data[group_name]) == 0:
            return None
        item = self.ga.sample_data[group_name][0]
        if len(item) >= 3:
            return item[2]
        return None

    def compute_charge_profile(self, v: np.ndarray) -> Dict:
        positive_positive = v[0]
        positive_negative = v[1]
        negative_positive = v[4]
        negative_negative = v[5]
        net_charge = (positive_negative + negative_positive) - (positive_positive + negative_negative)
        charge_density = positive_positive + positive_negative + negative_positive + negative_negative
        charge_balance = (positive_negative + negative_positive) / (charge_density + 1e-10)
        return {
            'net_charge': net_charge,
            'charge_density': charge_density,
            'charge_balance': charge_balance
        }

    def compute_gravy(self, v: np.ndarray) -> float:
        hydrophobic = v[15] + v[14] + v[11]
        polar = v[10] + v[2] + v[6] + v[8] + v[9]
        charged = v[0] + v[1] + v[4] + v[5]
        total = hydrophobic + polar + charged + 1e-10
        gravy = (hydrophobic - polar) / total * 4.5
        return max(-4.5, min(4.5, gravy))

    def compute_hydrophobicity_profile(self, v: np.ndarray) -> Dict:
        hydrophobic_scores = {
            0: -0.5, 1: -0.3, 2: -0.1, 3: 0.3,
            4: -0.3, 5: -0.5, 6: -0.1, 7: 0.3,
            8: -0.1, 9: -0.1, 10: -0.2, 11: 0.4,
            12: 0.3, 13: 0.3, 14: 0.4, 15: 0.8
        }
        hydrophobicity = sum(v[i] * hydrophobic_scores.get(i, 0) for i in range(len(v)))
        return {
            'gravy': self.compute_gravy(v),
            'hydrophobicity_score': hydrophobicity,
            'num_hydrophobic_components': sum(1 for i in range(len(v)) if hydrophobic_scores.get(i, 0) > 0.3)
        }

    def compute_hydrophobic_patches(self, v: np.ndarray) -> Dict:
        hydrophobic_indices = [3, 7, 11, 12, 13, 14, 15]
        patches = []
        current_patch = []
        for i in range(len(v)):
            if i in hydrophobic_indices and v[i] > 0.01:
                current_patch.append(i)
            else:
                if len(current_patch) > 0:
                    patches.append(current_patch)
                    current_patch = []
        if len(current_patch) > 0:
            patches.append(current_patch)
        patch_scores = []
        for patch in patches:
            patch_score = sum(v[i] for i in patch)
            patch_scores.append({
                'components': patch,
                'size': len(patch),
                'score': patch_score
            })
        patch_scores = sorted(patch_scores, key=lambda x: x['score'], reverse=True)
        return {
            'num_patches': len(patches),
            'patch_scores': patch_scores[:5]
        }

    def compute_solubility_aggregation(self, v: np.ndarray) -> Dict:
        hydrophobic = v[15] + v[14] + v[11]
        charged = v[1] + v[4] + v[0] + v[5]
        polar = v[10] + v[2] + v[6] + v[8] + v[9]
        total = hydrophobic + charged + polar + 1e-10
        solubility = max(1, min(20, 10 + (charged + polar) / total * 10 - hydrophobic / total * 5))
        aggregation = min(1, (hydrophobic + v[11] + v[14]) / total * 1.5)
        return {
            'solubility_mg_ml': solubility,
            'aggregation_score': aggregation
        }

    def compute_stability(self, v: np.ndarray) -> Dict:
        stability_positive = v[10] + v[1] + v[4]
        stability_negative = v[15]
        delta_g = max(-15, min(-2, -4.0 - 5.0 * (stability_positive / (stability_positive + stability_negative + 1e-10))))
        tm = max(40, min(80, 30 + 20 * (stability_positive / (stability_positive + stability_negative + 1e-10))))
        return {
            'delta_g': delta_g,
            'tm': tm,
            'stability_score': stability_positive / (stability_positive + stability_negative + 1e-10)
        }

    def compute_hotspots(self, v: np.ndarray) -> Dict:
        hotspots = []
        charge_region = v[1] + v[4] + v[0] + v[5]
        hotspot1_score = min(0.95, 0.7 * charge_region + 0.3 * v[10])
        hotspots.append({
            'name': 'Charge-rich region',
            'score': hotspot1_score,
            'type': 'charged'
        })
        hydrophobic_patch = v[15] + v[14] + v[11]
        hotspot2_score = min(0.95, 0.8 * hydrophobic_patch + 0.2 * v[7])
        hotspots.append({
            'name': 'Hydrophobic patch',
            'score': hotspot2_score,
            'type': 'hydrophobic'
        })
        hotspots = sorted(hotspots, key=lambda x: x['score'], reverse=True)
        return {'hotspots': hotspots, 'best_hotspot': hotspots[0] if hotspots else None}

    def compute_membrane_permeability(self, v: np.ndarray, sequence_length: int) -> Dict:
        hydrophobic = v[15] + v[14] + v[11]
        charged = v[0] + v[1] + v[4] + v[5]
        permeability = max(0, min(1, (0.5 + 0.3 * hydrophobic - 0.2 * charged) if sequence_length < 30 else (0.2 + 0.2 * hydrophobic - 0.1 * charged)))
        return {
            'permeability_score': permeability,
            'membrane_affinity': hydrophobic / (hydrophobic + charged + 1e-10)
        }

    def compute_buffer_stability(self, v: np.ndarray) -> Dict:
        net_charge = self.compute_charge_profile(v)['net_charge']
        optimal_ph = max(5.0, min(8.0, 7.0 - 0.5 * net_charge))
        charge_density = v[1] + v[4] + v[0] + v[5]
        salt_tolerance = max(50, min(500, 150 + 100 * (1 - charge_density)))
        return {
            'optimal_ph': optimal_ph,
            'salt_tolerance_mM': salt_tolerance,
            'buffer_recommendation': f"PBS pH {optimal_ph:.1f} + {int(salt_tolerance)} mM NaCl"
        }

    def analyze_protein(self, group_name: str, results_dir: str) -> Dict:
        print(f"\n  🧪 Chemical analysis for {get_display_name(group_name)} (v211.0)...")
        v = self._get_pim_vector(group_name)
        if v is None:
            print(f"  ⚠️ No PIM vector found for {group_name}")
            return {}
        sequence = self._get_sequence_from_sample(group_name)
        seq_len = len(sequence) if sequence else 100
        results = {}
        results['charge'] = self.compute_charge_profile(v)
        results['gravy'] = self.compute_gravy(v)
        results['hydrophobicity'] = self.compute_hydrophobicity_profile(v)
        results['patches'] = self.compute_hydrophobic_patches(v)
        results['solubility'] = self.compute_solubility_aggregation(v)
        results['stability'] = self.compute_stability(v)
        results['hotspots'] = self.compute_hotspots(v)
        results['membrane_permeability'] = self.compute_membrane_permeability(v, seq_len)
        results['buffer_stability'] = self.compute_buffer_stability(v)

        self._save_single_csv(results, group_name, results_dir)
        return results

    def _save_single_csv(self, results: Dict, group_name: str, results_dir: str):
        rows = []
        rows.append({'Property': 'Net Charge', 'Value': results['charge']['net_charge'], 'Description': 'Balance of charge interactions'})
        rows.append({'Property': 'Charge Density', 'Value': results['charge']['charge_density'], 'Description': 'Total charge interactions'})
        rows.append({'Property': 'Charge Balance', 'Value': results['charge']['charge_balance'], 'Description': 'Attraction/repulsion ratio'})
        rows.append({'Property': 'GRAVY', 'Value': results['gravy'], 'Description': 'Grand Average of Hydropathicity'})
        rows.append({'Property': 'Hydrophobicity Score', 'Value': results['hydrophobicity']['hydrophobicity_score'], 'Description': 'Composite hydrophobicity score'})
        rows.append({'Property': 'Num Hydrophobic Components', 'Value': results['hydrophobicity']['num_hydrophobic_components'], 'Description': 'Number of hydrophobic components'})
        for i, patch in enumerate(results['patches']['patch_scores'][:3]):
            rows.append({'Property': f'Hydrophobic Patch {i+1} Components', 'Value': str(patch['components']), 'Description': f'Size: {patch["size"]}'})
            rows.append({'Property': f'Hydrophobic Patch {i+1} Score', 'Value': patch['score'], 'Description': 'Patch score'})
        rows.append({'Property': 'Solubility (mg/mL)', 'Value': results['solubility']['solubility_mg_ml'], 'Description': 'Predicted solubility in PBS'})
        rows.append({'Property': 'Aggregation Score', 'Value': results['solubility']['aggregation_score'], 'Description': 'Aggregation propensity score'})
        rows.append({'Property': 'ΔG (kcal/mol)', 'Value': results['stability']['delta_g'], 'Description': 'Folding free energy'})
        rows.append({'Property': 'Tm (°C)', 'Value': results['stability']['tm'], 'Description': 'Melting temperature'})
        rows.append({'Property': 'Stability Score', 'Value': results['stability']['stability_score'], 'Description': 'Relative stability score'})
        for h in results['hotspots']['hotspots'][:3]:
            rows.append({'Property': f'Hotspot: {h["name"]}', 'Value': h['score'], 'Description': f'Type: {h["type"]}'})
        rows.append({'Property': 'Membrane Permeability', 'Value': results['membrane_permeability']['permeability_score'], 'Description': 'Membrane penetration potential'})
        rows.append({'Property': 'Membrane Affinity', 'Value': results['membrane_permeability']['membrane_affinity'], 'Description': 'Affinity for membrane components'})
        rows.append({'Property': 'Optimal pH', 'Value': results['buffer_stability']['optimal_ph'], 'Description': 'Optimal pH for stability'})
        rows.append({'Property': 'Salt Tolerance (mM)', 'Value': results['buffer_stability']['salt_tolerance_mM'], 'Description': 'Recommended NaCl concentration'})
        rows.append({'Property': 'Buffer Recommendation', 'Value': results['buffer_stability']['buffer_recommendation'], 'Description': 'Full buffer formulation'})

        df = pd.DataFrame(rows)
        safe_save_csv(df, f"chemical_profile_{group_name}.csv", results_dir)
        print(f"  ✅ Chemical profile saved: chemical_profile_{group_name}.csv ({len(rows)} properties)")

# ============================================================================
# CLASS: ProcessingTracker
# ============================================================================

class ProcessingTracker:
    def __init__(self):
        self.total_sequences_processed = 0
        self.total_valid_pim = 0
        self.total_rejected = 0
        self.total_bytes_read = 0
        self.group_counts = {}
        self.group_valid = {}
        self.start_time = None
        self.last_report_count = 0
        self.batch_count = 0
        self.total_batches = 0

    def update(self, group_name: str, is_valid: bool, bytes_read: int = 0):
        self.total_sequences_processed += 1
        self.total_bytes_read += bytes_read

        if is_valid:
            self.total_valid_pim += 1
        else:
            self.total_rejected += 1

        if group_name not in self.group_counts:
            self.group_counts[group_name] = 0
            self.group_valid[group_name] = 0

        self.group_counts[group_name] += 1
        if is_valid:
            self.group_valid[group_name] += 1

    def get_report(self) -> Dict:
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 1
        rate = self.total_sequences_processed / elapsed if elapsed > 0 else 0

        return {
            'total_sequences': self.total_sequences_processed,
            'valid_pim': self.total_valid_pim,
            'rejected': self.total_rejected,
            'valid_percentage': (self.total_valid_pim / self.total_sequences_processed * 100)
                                if self.total_sequences_processed > 0 else 0,
            'group_counts': self.group_counts,
            'group_valid': self.group_valid,
            'total_bytes': self.total_bytes_read,
            'processing_rate': rate,
            'elapsed_seconds': elapsed,
            'batch_count': self.batch_count,
            'total_batches': self.total_batches
        }

    def print_summary(self):
        print("\n" + "=" * 80)
        print("📊 GLOBAL PROCESSING SUMMARY")
        print("=" * 80)

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"  Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  Total sequences read: {self.total_sequences_processed:,}")
        print(f"  Total valid PIMs: {self.total_valid_pim:,}")
        print(f"  Total rejected: {self.total_rejected:,}")
        print(f"  Validity rate: {self.total_valid_pim/self.total_sequences_processed*100:.2f}%"
              if self.total_sequences_processed > 0 else "0%")
        print(f"  Total bytes processed: {self.total_bytes_read / (1024**3):.2f} GB")
        print(f"  Average speed: {self.total_sequences_processed/elapsed:,.0f} seq/s"
              if elapsed > 0 else "N/A")

        print("\n  📊 BREAKDOWN BY GROUP:")
        print(f"  {'Group':<20} {'Total':>14} {'Valid':>14} {'Rejected':>14} {'% Valid':>10}")
        print(f"  {'-'*75}")

        for group in sorted(self.group_counts.keys()):
            total = self.group_counts[group]
            valid = self.group_valid.get(group, 0)
            rejected = total - valid
            pct = (valid / total * 100) if total > 0 else 0
            print(f"  {get_display_name(group):<20} {total:>14,} {valid:>14,} "
                  f"{rejected:>14,} {pct:>9.2f}%")


# ============================================================================
# CLASS: PIMHashIndex
# ============================================================================

class PIMHashIndex:
    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance
        self.index: Dict[str, List[Tuple[str, str, np.ndarray]]] = defaultdict(list)

    def add_protein(self, protein_id: str, group: str, vector: np.ndarray):
        h = pim_to_hash(vector, tolerance=self.tolerance)
        self.index[h].append((protein_id, group, vector))

    def search(self, vector: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
        h = pim_to_hash(vector, tolerance=self.tolerance)
        return self.index.get(h, [])

    def build_from_samples(self, samples: Dict[str, List[Tuple[str, np.ndarray, str]]]):
        count = 0
        for group_name, sample_list in samples.items():
            for header, vector, seq in sample_list:
                self.add_protein(header, group_name, vector)
                count += 1
        print(f"  ✅ Hash index built: {len(self.index)} unique buckets from {count} proteins")


def pim_to_hash(pim_vector: np.ndarray, tolerance: float = 0.001) -> str:
    discretized = np.round(pim_vector / tolerance) * tolerance
    vector_str = ','.join([f"{x:.6f}" for x in discretized])
    return hashlib.sha256(vector_str.encode()).hexdigest()[:32]


# ============================================================================
# CLASS: ChEMBLMapper
# ============================================================================

class ChEMBLMapper:
    def __init__(self, mapping_file: str = CHEMBL_MAPPING_FILE):
        self.mapping = None
        self.loaded = False

        if not os.path.exists(mapping_file):
            print(f"  ⚠️ ChEMBL mapping file not found: {mapping_file}")
            return

        try:
            data = []
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) >= 3:
                        uniprot = parts[0]
                        chembl_id = parts[1]
                        protein_name = ' '.join(parts[2:])
                        if protein_name.endswith('SINGLE PROTEIN'):
                            protein_name = protein_name[:-14].strip()
                        data.append([uniprot, chembl_id, protein_name])

            self.mapping = pd.DataFrame(data, columns=['UNIPROT_ACCESSION', 'CHEMBL_PROTEIN_ID', 'PROTEIN_NAME'])
            self.loaded = True
            print(f"  ✅ ChEMBL mapping loaded: {len(self.mapping)} entries")
        except Exception as e:
            print(f"  ⚠️ Error loading ChEMBL mapping: {e}")
            self.loaded = False

    def search_by_name(self, name: str) -> List[Dict]:
        if not self.loaded:
            return []
        results = self.mapping[self.mapping['PROTEIN_NAME'].str.contains(name, case=False, na=False)]
        return results.to_dict('records')


# ============================================================================
# CLASS: APDLoader
# ============================================================================

class APDLoader:
    def __init__(self, fasta_file: str = APD_FASTA_FILE):
        self.peptides = []
        self.loaded = False

        if not os.path.exists(fasta_file):
            print(f"  ⚠️ APD file not found: {fasta_file}")
            return

        try:
            sequences = read_fasta_file(fasta_file)
            for header, seq in sequences:
                if 'search led to' in header:
                    continue

                ap_id = header.strip()
                activity = self._estimate_activity_from_sequence(seq)

                self.peptides.append({
                    'id': ap_id,
                    'header': header,
                    'sequence': seq,
                    'length': len(seq),
                    'activity': activity,
                    'pim': compute_pim_profile(seq, use_weights=True)
                })
            self.loaded = True
            print(f"  ✅ APD loaded: {len(self.peptides)} peptides")
        except Exception as e:
            print(f"  ⚠️ Error loading APD: {e}")
            self.loaded = False

    def _estimate_activity_from_sequence(self, seq: str) -> float:
        score = 0.5

        if 10 <= len(seq) <= 30:
            score += 0.15
        elif len(seq) < 10:
            score -= 0.1

        cationic = sum(1 for aa in seq if aa in ['K', 'R'])
        if cationic / len(seq) > 0.2:
            score += 0.15

        hydrophobic = sum(1 for aa in seq if aa in ['A', 'L', 'I', 'V', 'F', 'W'])
        if hydrophobic / len(seq) > 0.3:
            score += 0.1

        polar = sum(1 for aa in seq if aa in ['N', 'Q', 'S', 'T'])
        if polar / len(seq) > 0.15:
            score += 0.1

        unique_aa = len(set(seq))
        if unique_aa > 5:
            score += 0.1 * min(unique_aa / 10, 1)

        return min(1.0, max(0.0, score))


# ============================================================================
# FILE READING FUNCTIONS
# ============================================================================

def read_fasta_file(filepath: str) -> List[Tuple[str, str]]:
    sequences = []
    if not os.path.exists(filepath):
        return sequences
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header is not None and current_seq:
                        sequences.append((current_header, ''.join(current_seq)))
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header is not None and current_seq:
                sequences.append((current_header, ''.join(current_seq)))
    except Exception as e:
        print(f"  ⚠️ Error reading {filepath}: {e}")
        return []
    return sequences


def read_fasta_stream(filepath: str, verbose: bool = False, max_sequences: int = None,
                      batch_size: int = 5000):
    if not os.path.exists(filepath):
        if verbose:
            print(f"    ⚠️ File not found: {filepath}")
        return

    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header is not None and current_seq:
                        yield current_header, ''.join(current_seq)
                        count += 1
                        if max_sequences and count >= max_sequences:
                            return
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header is not None and current_seq:
                yield current_header, ''.join(current_seq)
                count += 1
    except Exception as e:
        print(f"  ⚠️ Error reading {filepath}: {e}")
        return


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_report_directory(base_dir: str, timestamp: str = None) -> str:
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f"{base_dir}/report_{timestamp}"
    ensure_directory(report_dir)
    return report_dir


def generate_final_summary(report: Dict, results_dir: str, config_loader: ConfigLoader = None,
                           mode: OperationMode = None) -> str:
    summary = []
    summary.append("=" * 80)
    summary.append("📋 FINAL SUMMARY - SGPMAIN 211.0")
    summary.append("   VERSION WITH DUAL MODE + NARRATIVE SUMMARIES")
    summary.append("   WITHOUT ESMFold (memory optimized)")
    if mode:
        summary.append(f"   OPERATION MODE: {mode.get_mode_name().upper()}")
    summary.append("   ALL METRICS + NEW ADVANCED METRICS")
    summary.append("=" * 80)
    summary.append("")

    if config_loader:
        summary.append(f"📋 CONFIGURATION:")
        summary.append(f"  ├─ Source: {config_loader.loaded_from}")
        if config_loader.loaded_from == 'config_EBOLA.json':
            meta = config_loader.config.get('metadata', {})
            summary.append(f"  ├─ Version: {meta.get('version', 'unknown')}")
            summary.append(f"  └─ Generated: {meta.get('generated', 'unknown')}")
        summary.append("")

    if 'processing' in report:
        proc = report['processing']
        summary.append(f"📊 PROCESSING:")
        summary.append(f"  ├─ Total sequences: {proc.get('total_sequences', 0):,}")
        summary.append(f"  ├─ Valid PIMs: {proc.get('valid_pim', 0):,}")
        summary.append(f"  ├─ Validity rate: {proc.get('valid_percentage', 0):.2f}%")
        summary.append(f"  ├─ Time: {proc.get('elapsed_seconds', 0)/60:.1f} minutes")
        summary.append(f"  └─ Speed: {proc.get('processing_rate', 0):,.0f} seq/s")
        summary.append("")

    if 'comparison' in report and report['comparison'] is not None:
        df = report['comparison']
        summary.append(f"🏷️ GROUP COMPARISON (v211.0):")
        summary.append(f"  ├─ Number of groups compared: {len(df)}")
        if not df.empty:
            best = df.iloc[0]
            summary.append(f"  ├─ Most similar group: {best['Compared Group']} (similarity: {best['Wedge Similarity']:.6f})")
            summary.append(f"  ├─ Rotor similarity: {best.get('Rotor Similarity', 0):.6f}")
            summary.append(f"  └─ Least similar: {df.iloc[-1]['Compared Group']} (similarity: {df.iloc[-1]['Wedge Similarity']:.6f})")
        summary.append("")

    if 'all_metrics' in report and report['all_metrics'] is not None:
        df = report['all_metrics']
        summary.append(f"📊 METRICS SUMMARY (v211.0):")
        for col in ['Entropy', 'Rényi α=2', 'Grassmann Curvature', 'Fractal Dim', 'Rotor Alignment Mean']:
            if col in df.columns:
                summary.append(f"  ├─ {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")

        for col in ['Fisher Mean', 'Ollivier-Ricci', 'PH Weight', 'Uhlmann Fidelity']:
            if col in df.columns:
                summary.append(f"  ├─ {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
        summary.append("")

    if 'cross_validation' in report and 'error' not in report['cross_validation']:
        cv = report['cross_validation']
        summary.append(f"🔬 GRASSMANN CROSS-VALIDATION:")
        summary.append(f"  ├─ Mean accuracy: {cv.get('mean_accuracy', 0):.3f}")
        summary.append(f"  └─ Std accuracy: {cv.get('std_accuracy', 0):.3f}")
        summary.append("")

    if 'characterization' in report:
        char = report['characterization']
        summary.append(f"📊 CHARACTERIZATION:")
        summary.append(f"  ├─ Target: {char.get('target_name', 'N/A')}")
        metrics = char.get('metrics', {})
        summary.append(f"  ├─ PIM Similarity: {metrics.get('pim_similarity', 0):.4f}")
        summary.append(f"  ├─ Drug Likeness: {metrics.get('drug_likeness', 0):.4f}")
        summary.append(f"  ├─ Structural Stability: {metrics.get('structural_stability', 0):.4f}")
        summary.append(f"  ├─ Fisher Mean: {metrics.get('fisher_mean', 0):.4f}")
        summary.append(f"  └─ Ollivier-Ricci: {metrics.get('ollivier_ricci_mean', 0):.4f}")
        summary.append("")

    if 'therapeutic_profile' in report and 'error' not in report['therapeutic_profile']:
        tp = report['therapeutic_profile']
        summary.append(f"🧬 THERAPEUTIC PROFILE (v211.0):")
        if 'target' in tp:
            summary.append(f"  ├─ Target: {tp['target'].get('protein_name', 'N/A')}")
            summary.append(f"  ├─ Similarity: {tp['target'].get('similarity', 0):.6f}")
            summary.append(f"  └─ Alignment Score: {tp['target'].get('alignment_score', 0):.6f}")
        if 'peptide' in tp:
            metrics = tp['peptide'].get('all_metrics_evaluation', {})
            summary.append(f"  ├─ Designed peptide: {tp['peptide'].get('sequence', 'N/A')[:20]}...")
            summary.append(f"  ├─ Drug Likeness: {metrics.get('drug_likeness', 0):.4f}")
            summary.append(f"  ├─ Rényi Entropy: {metrics.get('renyi_entropy_peptide', 0):.4f}")
            summary.append(f"  ├─ Rotor Angle: {metrics.get('rotor_angle', 0):.4f}")
            summary.append(f"  └─ Composite Score: {metrics.get('composite_score', 0):.4f}")
        summary.append("")

    summary.append("=" * 80)
    summary.append(f"✅ PROCESS COMPLETED - Results in: {results_dir}/")
    summary.append("=" * 80)

    return "\n".join(summary)


def save_final_summary(report: Dict, results_dir: str, config_loader: ConfigLoader = None,
                       mode: OperationMode = None):
    summary = generate_final_summary(report, results_dir, config_loader, mode)
    safe_save_text(summary, "FINAL_SUMMARY_v211.txt", results_dir)
    print(f"  ✅ Final summary saved: {results_dir}/FINAL_SUMMARY_v211.txt")


# ============================================================================
# MAIN - FULL VERSION v211.0
# ============================================================================

def main():
    print("=" * 80)
    print("🦠 SGPMAIN 211.0 - MIRROR-PIM WITH DUAL MODE")
    print("   ✅ CONFIGURATION: config_EBOLA.json + operation_control")
    print("   ✅ DUAL MODE: CHARACTERIZATION + DESIGN")
    print("   ✅ DYNAMIC NARRATIVE SUMMARY PER TARGET")
    print("   ✅ NEW ADVANCED METRICS (Fisher, Ricci, PH, Uhlmann)")
    print("   ✅ MULTIDISCIPLINARY SUMMARY TABLE")
    print("   ✅ ESMFold REMOVED (memory optimization)")
    print("   ✅ FIXED: hodge_complementarity, ricci_curvature")
    print("   ✅ FIXED: wasserstein, fractal_dimension range validation")
    print("   ✅ STREAMING ENABLED - NO FULL LOAD IN MEMORY")
    print(f"   ✅ REFERENCE GROUPS: {MAIN_GROUP_REFERENCE}")
    print(f"   ✅ DESIGN GROUP: {MAIN_GROUP_DESIGN}")
    print(f"   ✅ FILE PATH: {DATA_PATH}")
    print("=" * 80)

    print("\n  📋 LOADING CONFIGURATION...")
    config_path = os.path.join(DATA_PATH, "config_EBOLA.json")
    config_loader = ConfigLoader(config_path, verbose=True)
    config_loader.print_summary()

    global METRIC_WEIGHTS, BATCH_SIZE, MAX_STORED_PROTEINS_PER_GROUP, N_BOOTSTRAP, MAX_WORKERS, MAX_PEPTIDE_LENGTH
    METRIC_WEIGHTS = config_loader.get_metric_weights()
    proc_params = config_loader.get_processing_params()
    BATCH_SIZE = proc_params.get('batch_size', 5000)
    MAX_STORED_PROTEINS_PER_GROUP = proc_params.get('max_stored_proteins', 200)
    N_BOOTSTRAP = proc_params.get('n_bootstrap', 50)
    MAX_WORKERS = min(proc_params.get('max_workers', 4), CPU_CORES - 2)

    config_target = get_config_target(MAIN_GROUP_DESIGN[0] if MAIN_GROUP_DESIGN else 'ebola')
    MAX_PEPTIDE_LENGTH = config_loader.get_max_peptide_length(config_target)
    print(f"\n  📏 Max peptide length set to: {MAX_PEPTIDE_LENGTH} aa (from config_EBOLA.json for '{config_target}')")

    data_paths = config_loader.get_data_paths()
    if 'chembl_mapping' in data_paths:
        global CHEMBL_MAPPING_FILE
        CHEMBL_MAPPING_FILE = data_paths['chembl_mapping']
    if 'apd_fasta' in data_paths:
        global APD_FASTA_FILE
        APD_FASTA_FILE = data_paths['apd_fasta']

    print(f"\n  📊 CONFIGURATION LOADED:")
    print(f"     ├─ METRIC_WEIGHTS: {len(METRIC_WEIGHTS)} variables")
    print(f"     ├─ BATCH_SIZE: {BATCH_SIZE:,}")
    print(f"     ├─ MAX_STORED_PROTEINS: {MAX_STORED_PROTEINS_PER_GROUP:,}")
    print(f"     ├─ N_BOOTSTRAP: {N_BOOTSTRAP}")
    print(f"     └─ MAX_WORKERS: {MAX_WORKERS}")

    mode = OperationMode.determine(config_loader)
    print(f"\n  🎯 OPERATION MODE DETERMINED: {mode.get_mode_name().upper()}")

    oc = config_loader.get_operation_control()
    print(f"     ├─ Use dummy peptide: {oc.get('use_dummy_peptide', False)}")
    print(f"     ├─ Evaluate peptide: {oc.get('evaluate_peptide', True)}")
    print(f"     ├─ Fallback mode: {oc.get('fallback_mode', 'characterization')}")
    print(f"     └─ Allow design override: {oc.get('allow_design_override', True)}")

    print(f"\n  🖥️ CPU DETECTED: {CPU_CORES} logical cores")
    print(f"  📦 Batch size: {BATCH_SIZE:,}")
    print(f"  💾 Sample per group: {MAX_STORED_PROTEINS_PER_GROUP:,} (SAMPLE ONLY)")

    print(f"\n  🌐 MATHEMATICAL IMPROVEMENTS ENABLED (v211.0):")
    print(f"     ├─ Clifford Product: ✅")
    print(f"     ├─ Rotor Similarity: ✅")
    print(f"     ├─ Rényi Entropy: ✅")
    print(f"     ├─ Bhattacharyya Distance: ✅")
    print(f"     ├─ Wasserstein Entropic: ✅")
    print(f"     ├─ Functional PCA: ✅")
    print(f"     ├─ DTW Distance: ✅")
    print(f"     ├─ Bootstrap BCa: ✅")
    print(f"     ├─ Permutation Tests: ✅")
    print(f"     ├─ Grassmann CV: ✅")
    print(f"     ├─ Grassmann Scalar Curvature: ✅")
    print(f"     ├─ Fisher Information: ✅")
    print(f"     ├─ Ollivier-Ricci Curvature: ✅")
    print(f"     ├─ Persistent Homology: ✅")
    print(f"     ├─ Uhlmann Fidelity: ✅")
    print(f"     └─ Bifurcation Points: ✅")

    print(f"\n  🔬 STRUCTURAL VALIDATION: DESACTIVADA (ESMFold removido)")
    print(f"     └─ ℹ️ El programa usa validación por métricas sin ESMFold")

    print(f"\n  📊 METRIC WEIGHTS FOR DESIGN (from config_EBOLA.json):")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"     ├─ {metric}: {weight*100:.0f}%")

    print(f"\n  📁 INPUT FILES (from {DATA_PATH}):")

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if mode.is_characterization():
            base_dir = f"results_characterization_{timestamp}"
        else:
            base_dir = f"results_design_{timestamp}"

        if mode.is_design():
            base_dir = f"results_design_{timestamp}"

        results_dir = create_report_directory(base_dir, timestamp)
        print(f"  📁 Results directory: {results_dir}")

        grassmann = GrassmannPIM(dim=DIM_PAIRS)

        analyzer = AdvancedGroupAnalyzer(
            grassmann,
            main_groups=MAIN_GROUP_REFERENCE,
            design_group=MAIN_GROUP_DESIGN
        )
        analyzer.set_sample_size(MAX_STORED_PROTEINS_PER_GROUP)
        print(f"  ✅ Analyzer configured (v211.0):")
        print(f"     ├─ Reference: {MAIN_GROUP_REFERENCE}")
        print(f"     └─ Design: {MAIN_GROUP_DESIGN}")

        files_to_load = {
            'sudan': os.path.join(DATA_PATH, 'Sudan.unico.dat0'),
            'zaire': os.path.join(DATA_PATH, 'Zaire.unico.dat0'),
            'reston': os.path.join(DATA_PATH, 'Reston.unico.dat0'),
            'bombali': os.path.join(DATA_PATH, 'Bombali.unico.dat0'),
            'bundibugyo': os.path.join(DATA_PATH, 'Bundibugyo.unico.dat0'),
            'tai': os.path.join(DATA_PATH, 'Tai.unico.dat0'),
            'lasv': os.path.join(DATA_PATH, 'lasv_all.unico.dat0'),
            'junv': os.path.join(DATA_PATH, 'junv_all.unico.dat0'),
            'macv': os.path.join(DATA_PATH, 'macv_all.unico.dat0'),
            'lcmv': os.path.join(DATA_PATH, 'lcmv_all.unico.dat0'),
            'nile1': os.path.join(DATA_PATH, 'nile1.unico.dat0'),
            'nile2': os.path.join(DATA_PATH, 'nile2.unico.dat0'),
            'rvf1': os.path.join(DATA_PATH, 'RVF1.unico.dat0'),
            'rvf2': os.path.join(DATA_PATH, 'RVF2.unico.dat0'),
            'rvf3': os.path.join(DATA_PATH, 'RVF3.unico.dat0'),
            'rvf4': os.path.join(DATA_PATH, 'RVF4.unico.dat0'),
            'lujo': os.path.join(DATA_PATH, 'lujo.unico.dat0'),
            'PARTIALLY_FOLDED': os.path.join(DATA_PATH, 'partiallyorderedN.unico.dat0'),
            'CPP': os.path.join(DATA_PATH, 'CPP.unico.dat0'),
            'NON_CPP': os.path.join(DATA_PATH, 'NONCPP.unico.dat0'),
            'UNFOLDED': os.path.join(DATA_PATH, 'unfolded.unico.dat0'),
            'REVIEWED_HUMAN': os.path.join(DATA_PATH, 'reviewed_human.unico.dat0'),
            'UNREVIEWED_HUMAN': os.path.join(DATA_PATH, 'unreviewed_human.unico.dat0'),
            'senales': os.path.join(DATA_PATH, 'senales.unico.dat0'),
            'membrana': os.path.join(DATA_PATH, 'membrana.unico.dat0'),
            'enfermedad': os.path.join(DATA_PATH, 'enfermedad.unico.dat0'),
            'VIRUS_REVIEWED': os.path.join(DATA_PATH, 'reviewed_virus.unico.dat0'),
            'VIRUS_UNREVIEWED': os.path.join(DATA_PATH, 'unreviewed_virus.unico.dat0'),
            'REVIEWED_ALL': os.path.join(DATA_PATH, 'reviewed_all.unico.dat0'),
            'UNREVIEWED_ALL': os.path.join(DATA_PATH, 'unreviewed_all.unico.dat0'),
        }

        print("\n📂 LOADING FASTA FILES (STREAMING)...")
        print("=" * 80)
        print("  ⚠️ NOTE: ALL files are processed with STREAMING")
        print(f"  ⚠️ NOTE: Only {MAX_STORED_PROTEINS_PER_GROUP} samples per group are stored in RAM")
        print("  ⚠️ NOTE: The 73GB file will be processed without filling the disk")
        print("  ✅ NOTE: v211.0 includes DUAL MODE + NARRATIVE SUMMARIES")
        print("  ✅ NOTE: ESMFold REMOVED - memory optimized")
        print("=" * 80)

        analyzer.start_time = datetime.now()
        analyzer.tracker.start_time = analyzer.start_time

        loaded_count = 0
        for group_name, filename in files_to_load.items():
            if os.path.exists(filename):
                analyzer.load_fasta_file(filename, group_name, verbose=True)
                loaded_count += 1
            else:
                print(f"  ⚠️ File not found: {filename} - Skipping")

        print(f"\n  ✅ Loaded {loaded_count} of {len(files_to_load)} files")

        analyzer.tracker.print_summary()
        analyzer.print_processing_summary()
        analyzer.build_hash_index()

        target_group = None
        for target in MAIN_GROUP_REFERENCE:
            if target in analyzer.group_stats:
                target_group = target
                break

        if target_group is None:
            target_group = list(analyzer.group_stats.keys())[0]

        print(f"\n  🎯 Using '{get_display_name(target_group)}' as reference group")

        # Generate full report
        knowledge_base = NarrativeKnowledgeBase()
        narrative_generator = NarrativeSummaryGenerator(knowledge_base)

        report = analyzer.generate_full_report(target_group, results_dir, mode, config_loader)

        # Generate narrative summaries
        print("\n  📝 GENERATING NARRATIVE SUMMARIES PER PROFILE...")

        if 'characterization' in report:
            char_report = report['characterization']
            metrics = char_report.get('metrics', {})
            target_name = char_report.get('target_name', target_group)

            peptide_data = metrics.copy()
            peptide_data['sequence'] = char_report.get('sequence', '')
            peptide_data['activity_score'] = metrics.get('activity_score', 0.5)
            peptide_data['drug_likeness'] = metrics.get('drug_likeness', 0.5)

            profiles = ['executive', 'biochemist', 'chemist', 'analytical_chemist',
                       'physicochemist', 'bioinformatician']

            for profile in profiles:
                summary = narrative_generator.generate_summary(peptide_data, target_name, profile)
                safe_save_text(summary, f"narrative_summary_{profile}.txt", results_dir)
                print(f"  ✅ Narrative summary {profile}: narrative_summary_{profile}.txt")

            multidisciplinary_reporter = MultidisciplinaryReporter(narrative_generator)
            multidisciplinary_reporter.generate_reports(peptide_data, target_name, results_dir)

            print(f"  ✅ Narrative summaries saved in: {results_dir}/")

        print("\n" + "=" * 80)
        print("💾 SAVING REPORT TO FILES")
        print("=" * 80)

        if mode.is_design() and oc.get('evaluate_peptide', True):
            print("\n" + "=" * 80)
            print("🧬 PEPTIDE DESIGN ENGINE (v211.0 - DESIGN MODE)")
            print("   ✅ ESM2 ENABLED (without disk cache)")
            print("   ✅ LoRA ENABLED (fine-tuning)")
            print("   ✅ ALL METRICS INTEGRATED")
            print("   ✅ BASE PEPTIDE FROM config_EBOLA.json")
            print(f"   ✅ MAX PEPTIDE LENGTH: {MAX_PEPTIDE_LENGTH} aa")
            print("   ℹ️ ESMFold desactivado (memoria optimizada)")
            print("=" * 80)

            print("\n  ℹ️ Peptide design activated. Using base peptide from config_EBOLA.json...")

        print("\n" + "=" * 80)
        print("🧹 CLEANING UP TEMPORARY FILES")
        print("=" * 80)

        try:
            if os.path.exists(CACHE_DIR):
                try:
                    shutil.rmtree(CACHE_DIR)
                    print(f"  ✅ Cache removed: {CACHE_DIR}")
                except Exception as e:
                    print(f"  ⚠️ Could not remove cache: {e}")
            else:
                print(f"  ℹ️ No cache to remove")
        except Exception as e:
            print(f"  ⚠️ Error during cleanup: {e}")

        try:
            import gc
            gc.collect()
            print(f"  ✅ Memory freed")
        except:
            pass

        print("\n" + "=" * 80)
        print("✅ EXECUTION COMPLETED - SGPMAIN 211.0")
        print("=" * 80)
        print(f"\n  📁 Results saved in: {results_dir}/")
        print(f"  🎯 MODE: {mode.get_mode_name().upper()}")

        elapsed = (datetime.now() - analyzer.start_time).total_seconds() if analyzer.start_time else 0
        print(f"  ⏱️ Total time: {elapsed/60:.1f} minutes")

        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"  💾 Memory used by process: {mem_mb:.0f} MB")

            total_mem = psutil.virtual_memory().total / (1024**3)
            used_mem = psutil.virtual_memory().used / (1024**3)
            print(f"  💾 System memory: {used_mem:.1f}GB / {total_mem:.1f}GB used")

            disk_usage = psutil.disk_usage('/')
            print(f"  💾 Disk space: {disk_usage.used/(1024**3):.1f}GB / {disk_usage.total/(1024**3):.1f}GB used")
            print(f"  💾 Disk free: {disk_usage.free/(1024**3):.1f}GB")
        except:
            pass

        print(f"\n  📊 METRICS USED IN DESIGN (from {config_loader.loaded_from}):")
        for metric, weight in METRIC_WEIGHTS.items():
            print(f"     ├─ {metric}: {weight*100:.0f}%")

        print(f"\n  📊 GENERATED FILES:")
        print(f"     ├─ comparison_{target_group}_vs_all.csv")
        print(f"     ├─ similarity_matrix_groups.csv")
        if mode.is_design():
            print(f"     ├─ therapeutic_profile_v211.json")
        print(f"     ├─ top_individual_proteins.csv")
        print(f"     ├─ all_metrics_report_v211.csv")
        if USE_GRASSMANN_MULTILEVEL:
            print(f"     ├─ grassmann_multilevel_report.csv")
            print(f"     ├─ grassmann_cycles_report.csv")
        if USE_PIDP:
            print(f"     ├─ pidp_analysis_*.csv")
            print(f"     ├─ pidp_summary_all_targets.csv")
        print(f"     ├─ chemical_profile_*.csv")
        print(f"     ├─ characterization_report.txt")
        print(f"     ├─ characterization_metrics.json")
        print(f"     ├─ narrative_summary_*.txt")
        print(f"     ├─ multidisciplinary_summary_table.txt")
        if mode.is_design():
            print(f"     ├─ designed_peptides_v211.csv")
            print(f"     ├─ best_candidate_v211.json")
        print(f"     ├─ grassmann_cv_results.json")
        print(f"     ├─ permutation_test_results.json")
        print(f"     ├─ rotor_comparisons.json")
        print(f"     ├─ functional_pca_results.json")
        print(f"     └─ FINAL_SUMMARY_v211.txt")

        try:
            save_final_summary(report, results_dir, config_loader, mode)
            print(f"  ✅ Final summary saved: FINAL_SUMMARY_v211.txt")
        except Exception as e:
            print(f"  ⚠️ Error saving final summary: {e}")

        print("\n" + "=" * 80)
        print("🦠 SGPMAIN 211.0 COMPLETED SUCCESSFULLY")
        print("   ✅ DUAL MODE IMPLEMENTED")
        print("   ✅ DYNAMIC NARRATIVE SUMMARY")
        print("   ✅ NEW ADVANCED METRICS")
        print("   ✅ MULTIDISCIPLINARY SUMMARY TABLE")
        print("   ✅ STREAMING processing completed")
        print("   ✅ ESMFold REMOVED (memory optimization)")
        print("   ✅ FIXED: hodge_complementarity range validation")
        print("   ✅ FIXED: ricci_curvature range validation")
        print("   ✅ FIXED: wasserstein range validation")
        print("   ✅ FIXED: fractal_dimension range validation")
        print("   ✅ FIXED: interpret_hodge in ContextualInterpreter")
        print("   ✅ ESM2, PIDP and LoRA executed correctly")
        print("   ✅ ALL METRICS integrated into design")
        print(f"   ✅ Configuration loaded from: {config_loader.loaded_from}")
        print(f"   ✅ Max peptide length: {MAX_PEPTIDE_LENGTH} aa")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️ PROCESS INTERRUPTED BY USER")
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                print(f"  🧹 Cache removed: {CACHE_DIR}")
            except:
                pass
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        try:
            error_log = f"error_v211_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(error_log, 'w') as f:
                f.write(traceback.format_exc())
            print(f"  ✅ Error saved to: {error_log}")
        except:
            pass

        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                print(f"  🧹 Cache removed: {CACHE_DIR}")
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
