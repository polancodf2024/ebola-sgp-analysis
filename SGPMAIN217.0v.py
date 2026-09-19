#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SGPMAIN 217.0 - MIRROR-PIM CON BIOPYTHON + WILKINSON-HARRISON + CONFIG COMPLETO
================================================================================
CAMBIOS RESPECTO A v216.0:
  1. ✅ CORREGIDO: get_config_target normaliza a minúsculas (zaire → ebola)
  2. ✅ CORREGIDO: DATA_PATH se actualiza desde data_paths['base_dir'] en main()
  3. ✅ CORREGIDO: _analytical usa fallback 'N/A' si no hay secuencia
  4. ✅ CORREGIDO: _metrics_table salta bool (no los formatea como float)
  5. ✅ CORREGIDO: _log_map_fallback elimina B (código muerto)
  6. ✅ CORREGIDO: bootstrap_metric valida None y tipos no numéricos
  7. ✅ CORREGIDO: get_metric_weights solo añade defaults si config está vacío
  8. ✅ AÑADIDO: all_metrics_report incluye columna 'Ranges Passed'
  9. ✅ CORREGIDO: get_top_individuals marca NaN si centroide o PIM son cero
 10. ✅ CORREGIDO: therapeutic_profile usa nombre versionado coherente
 11. ✅ AÑADIDO: _validate_peptide_against_ranges normaliza min/max numéricos
 12. ✅ AÑADIDO: get_target_ranges filtra claves _comment
 13. ✅ AÑADIDO (v217.0): _design_peptide_enhanced devuelve lista (base + optimized)
 14. ✅ AÑADIDO (v217.0): generate_therapeutic_profile itera sobre lista de péptidos
 15. ✅ AÑADIDO (v217.0): _compare_base_vs_optimized genera tabla comparativa
 16. ✅ AÑADIDO (v217.0): print_profile imprime comparación base vs optimizado
 17. ✅ AÑADIDO (v217.0): generate_full_report guarda base_vs_optimized_comparison.csv
 18. ✅ AÑADIDO (v217.0): generate_full_report guarda ml_cv_metrics.csv
 19. ✅ AÑADIDO (v217.0): generate_final_summary imprime comparación base vs optimized
 20. ✅ CORREGIDO (v217.0): indentación de _design_peptide_enhanced al nivel de clase

NÚCLEO DE 10 MÉTRICAS (todas verificadas con 17 tests unitarios):
  GRUPO A: GRASSMANN (5)
     1. principal_angles      - Ángulos principales vía SVD
     2. geodesic_distance     - Distancia geodésica ‖θ‖₂
     3. scalar_curvature      - Curvatura escalar k(n-k) [constante estructural]
     4. exponential_map       - Mapa exponencial (Alg. 3.2)
     5. logarithmic_map       - Mapa logarítmico (Alg. 3.1, robusto)
  GRUPO B: INFORMACIÓN (3)
     6. shannon_entropy       - Entropía de Shannon
     7. jensen_shannon        - Divergencia Jensen-Shannon
     8. hellinger_distance    - Distancia de Hellinger
  GRUPO C: ESTADÍSTICA (2)
     9. gini_coefficient      - Coeficiente de Gini
    10. spearman_correlation  - Correlación de Spearman
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
from itertools import combinations
from enum import Enum

if sys.version_info < (3, 8):
    print("❌ ERROR: Python 3.8 or higher is required")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2, pearsonr, linregress, spearmanr, norm
from scipy.linalg import eigh, solve, svd, pinv, qr
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             roc_auc_score, matthews_corrcoef, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA

# ============================================================================
# BIOPYTHON (opcional)
# ============================================================================
BIOPYTHON_AVAILABLE = False
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis as _BioProteinAnalysis
    BIOPYTHON_AVAILABLE = True
    print("  🧬 Biopython available (real instability index, Guruprasad 1990)")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("  ⚠️ Biopython not available (pip install biopython)")
    print("     → instability will use simplified heuristic (marked as such)")

# ============================================================================
# XGBoost (opcional)
# ============================================================================
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("  🚀 XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  ⚠️ XGBoost not available (pip install xgboost)")

# ============================================================================
# ESMFOLD - DESACTIVADO
# ============================================================================
ESMFOLD_AVAILABLE = False
TORCH_AVAILABLE = False
GPU_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    try:
        import torch.cuda as cuda
        GPU_AVAILABLE = cuda.is_available()
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False

print("  ℹ️ ESMFold DESACTIVADO (memory optimization)")

TRANSFORMERS_AVAILABLE = False
PEFT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# RUTAS Y SEMILLAS
# ============================================================================
DATA_PATH = os.environ.get("SGPM_DATA", "/home/cpolanco/POLANCO/ARCHIVOMAESTRO")
np.random.seed(42)
random.seed(42)

CPU_CORES = mp.cpu_count()
MAX_WORKERS = min(CPU_CORES - 2, 4)
BATCH_SIZE = 5000
MAX_STORED_PROTEINS_PER_GROUP = 200
COHESION_CALC_SAMPLE_SIZE = 100

os.environ['OMP_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['MKL_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['NUMEXPR_NUM_THREADS'] = str(MAX_WORKERS)
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SIMILARITY_THRESHOLD = None
CONFIDENCE_LEVEL = 0.95
TOP_N_PROTEINS = 10
TOLERANCE = 0.001
USE_BOOTSTRAP = True
N_BOOTSTRAP = 50
USE_WEIGHTS = True
COHESION_SAMPLE_SIZE = COHESION_CALC_SAMPLE_SIZE
GENERATE_PLOTS = False

APPLY_METRIC_WEIGHTS = True
APPLY_CLASSIFICATION = True
VALIDATE_TARGET_RANGES_IN_CHARACTERIZATION = True
PROCESS_SECONDARY_TARGETS = False
MIN_SAMPLES_PER_GROUP = 2
BOOTSTRAP_CI = 0.95

GRASSMANN_RANK_TOL = 1e-10
GRASSMANN_PERP_TOL = 1e-12
GRASSMANN_ORTHO_TOL = 1e-6
GRASSMANN_IDENTITY_TOL = 1e-9

GRASSMANN_SUBSPACE_DIM = 2
AMBIENT_DIM = 16

METRIC_WEIGHTS = {
    'geodesic_distance': 0.25,
    'principal_angles': 0.10,
    'scalar_curvature': 0.05,
    'shannon_entropy': 0.15,
    'jensen_shannon': 0.15,
    'hellinger': 0.10,
    'gini': 0.10,
    'spearman': 0.10,
}

MAX_PEPTIDE_LENGTH = 99999

USE_PIDP = True
PIDP_USE_METAPREDICT = True
PIDP_USE_AIUPRED = True
PIDP_THRESHOLDS = [0.3, 0.4, 0.5]

MAIN_GROUP_REFERENCE = ['zaire', 'sudan', 'reston', 'bundibugyo', 'tai', 'bombali']
MAIN_GROUP_DESIGN = ['zaire']
MAIN_GROUP = MAIN_GROUP_REFERENCE

CONFIG_GROUP_MAP = {
    'west_nile': ['nile1', 'nile2', 'NILE1', 'NILE2'],
    'rvfv': ['rvf1', 'rvf2', 'rvf3', 'rvf4', 'RVF1', 'RVF2', 'RVF3', 'RVF4'],
    'ebola': ['EBOLA_ZAIRE', 'EBOLA_SUDAN', 'EBOLA_RESTON',
              'EBOLA_BOMBALI', 'EBOLA_BUNDIBUGYO', 'EBOLA_TAI_FOREST',
              'zaire', 'sudan', 'reston', 'bombali', 'bundibugyo', 'tai'],
    'lasv': ['LASV', 'lasv'],
    'junv': ['JUNV', 'junv'],
    'macv': ['MACV', 'macv'],
    'lcmv': ['LCMV', 'lcmv'],
    'lujo': ['LUJO', 'lujo'],
}

def get_config_target(group_name: str) -> str:
    """
    v217.0: normaliza a minúsculas para mapear 'zaire' -> 'ebola'.
    """
    if group_name is None:
        return 'unknown'
    group_lower = group_name.lower()
    for config_key, groups in CONFIG_GROUP_MAP.items():
        groups_lower = [g.lower() for g in groups]
        if group_lower in groups_lower:
            return config_key
    return group_lower

CHEMBL_MAPPING_FILE = os.path.join(DATA_PATH, "chembl_uniprot.txt")
DRAMP_TSV_FILE = os.path.join(DATA_PATH, "general_amps.txt")

GROUP_NAME_MAP = {
    'enfermedad': 'DISEASE', 'membrana': 'MEMBRANE', 'senales': 'SIGNALS',
    'sudan': 'EBOLA_SUDAN', 'zaire': 'EBOLA_ZAIRE', 'reston': 'EBOLA_RESTON',
    'bombali': 'EBOLA_BOMBALI', 'bundibugyo': 'EBOLA_BUNDIBUGYO', 'tai': 'EBOLA_TAI_FOREST',
    'lasv': 'LASV', 'junv': 'JUNV', 'macv': 'MACV', 'lcmv': 'LCMV',
    'nile1': 'NILE1', 'nile2': 'NILE2',
    'rvf1': 'RVF1 (Gn)', 'rvf2': 'RVF2 (Gc)', 'rvf3': 'RVF3 (Gn-strain)', 'rvf4': 'RVF4 (Gc-strain)',
    'lujo': 'LUJO',
}

def get_display_name(group_name: str) -> str:
    if group_name is None:
        return "UNKNOWN"
    return GROUP_NAME_MAP.get(group_name, group_name)

def extract_protein_id(header: str) -> str:
    if not header:
        return ""
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    if header.startswith('>'):
        header = header[1:]
    return header.split()[0] if header.split() else header[:20]

DIM_PAIRS = 16

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

BIOLOGICAL_WEIGHTS = {
    'P+,P-': 2.0, 'P-,P+': 2.0, 'N,N': 1.5,
    'N,P+': 1.3, 'P+,N': 1.3, 'N,P-': 1.3, 'P-,N': 1.3,
    'NP,NP': 1.0, 'NP,N': 0.9, 'N,NP': 0.9,
    'NP,P+': 0.7, 'P+,NP': 0.7, 'NP,P-': 0.7, 'P-,NP': 0.7,
    'P+,P+': 0.4, 'P-,P-': 0.4,
}

PKA_VALUES = {
    'C_term': 3.55, 'N_term': 7.50,
    'D': 4.05, 'E': 4.45, 'C': 9.00, 'Y': 10.00,
    'H': 5.98, 'K': 10.00, 'R': 12.00,
}

WW_OCTANOL = {
    'A': 0.17, 'R': -1.81, 'N': -0.42, 'D': -1.23, 'C': 0.24,
    'Q': -0.58, 'E': -1.12, 'G': 0.01, 'H': -0.14, 'I': 0.88,
    'L': 0.79, 'K': -0.99, 'M': 0.47, 'F': 0.76, 'P': -0.04,
    'S': -0.13, 'T': -0.29, 'W': 1.08, 'Y': 0.47, 'V': 0.78
}

AA_CHARGES = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}

AA_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

AA_WEIGHTS = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

AA_HELIX_PROPENSITY = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'



# ============================================================================
# HELPERS
# ============================================================================

def expand_env_vars(path: str) -> str:
    if not path or not isinstance(path, str):
        return path
    pattern = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        value = os.environ.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return default
        return match.group(0)
    return pattern.sub(replacer, path)


def ensure_directory(path: str) -> str:
    if not path:
        path = "."
    path = expand_env_vars(path)
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


def read_fasta_stream(filepath: str, verbose: bool = False, max_sequences: int = None):
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
    except Exception as e:
        print(f"  ⚠️ Error reading {filepath}: {e}")
        return



# ============================================================================
# CÁLCULOS FISICOQUÍMICOS REALES
# ============================================================================

def compute_pI_real(sequence: str) -> float:
    if not sequence:
        return 7.0
    seq = sequence.upper()
    def charge_at_ph(ph: float) -> float:
        q = 0.0
        q -= 1.0 / (1.0 + 10**(PKA_VALUES['C_term'] - ph))
        q += 1.0 / (1.0 + 10**(ph - PKA_VALUES['N_term']))
        for aa in seq:
            if aa in 'KRH':
                q += 1.0 / (1.0 + 10**(ph - PKA_VALUES[aa]))
            elif aa in 'DECY':
                q -= 1.0 / (1.0 + 10**(PKA_VALUES[aa] - ph))
        return q
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_mw_real(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(AA_WEIGHTS.get(aa, 110.0) for aa in sequence.upper()) - 18.015 * (len(sequence) - 1)


def compute_gravy_real(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(AA_HYDROPHOBICITY.get(aa, 0) for aa in sequence.upper()) / len(sequence)


def compute_net_charge_real(sequence: str, ph: float = 7.4) -> float:
    if not sequence:
        return 0.0
    seq = sequence.upper()
    q = 0.0
    q -= 1.0 / (1.0 + 10**(PKA_VALUES['C_term'] - ph))
    q += 1.0 / (1.0 + 10**(ph - PKA_VALUES['N_term']))
    for aa in seq:
        if aa in 'KRH':
            q += 1.0 / (1.0 + 10**(ph - PKA_VALUES[aa]))
        elif aa in 'DECY':
            q -= 1.0 / (1.0 + 10**(PKA_VALUES[aa] - ph))
    return q


def compute_solubility_real(sequence: str) -> Dict:
    if not sequence:
        return {
            'probability': None, 'cv': 0.0, 'applicable': False,
            'source': 'Wilkinson-Harrison 1991', 'confidence': 0.0,
            'note': 'empty sequence'
        }
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return {
            'probability': None, 'cv': 0.0, 'applicable': False,
            'source': 'Wilkinson-Harrison 1991', 'confidence': 0.0,
            'note': 'empty sequence'
        }
    N = seq.count('N') + seq.count('G') + seq.count('P') + seq.count('S')
    charge = (seq.count('R') + seq.count('K')) - (seq.count('D') + seq.count('E'))
    cv = 15.43 * (N / n) - 29.56 * abs(charge / n - 0.03)
    cv_prime = 1.71
    delta = cv - cv_prime
    applicable = n > 100
    if delta < 0:
        prob = 0.4934 + 0.276 * abs(delta) - 0.0392 * (delta ** 2)
        prob = float(max(0.0, min(1.0, prob)))
    else:
        prob = 0.0
    if not applicable:
        return {
            'probability': None, 'cv': float(cv), 'applicable': False,
            'source': 'Wilkinson-Harrison 1991', 'confidence': 0.0,
            'note': f'Modelo para proteínas recombinantes (n>100); secuencia n={n} fuera de dominio.'
        }
    return {
        'probability': prob, 'cv': float(cv), 'applicable': True,
        'source': 'Wilkinson-Harrison 1991', 'confidence': 0.6,
        'note': 'Modelo para proteínas recombinantes en E. coli'
    }


def compute_membrane_permeability_real(sequence: str) -> float:
    if not sequence:
        return 0.0
    delta_g = sum(WW_OCTANOL.get(aa, 0) for aa in sequence.upper()) / len(sequence)
    return float(delta_g)


def compute_instability_index_real(sequence: str) -> float:
    if not sequence or len(sequence) < 2:
        return 0.0
    if BIOPYTHON_AVAILABLE:
        try:
            analysis = _BioProteinAnalysis(sequence)
            return float(analysis.instability_index())
        except Exception:
            pass
    seq = sequence.upper()
    n = len(seq)
    cmnq_frac = sum(1 for aa in seq if aa in ['C', 'M', 'N', 'Q']) / (n + 1e-10)
    return float(cmnq_frac * 100.0)



# ============================================================================
# NÚCLEO REAL DE 10 MÉTRICAS
# ============================================================================

def _validate_subspace_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    if X is None or X.size == 0:
        raise ValueError(f"{name} no puede ser vacío")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} debe ser 2D, recibido {X.ndim}D")
    Q, R = np.linalg.qr(X)
    if np.abs(np.linalg.det(R[:Q.shape[1], :])) < 1e-10:
        raise ValueError(f"{name} tiene columnas linealmente dependientes")
    return Q


def _orthonormalize(U: np.ndarray, tol: float = GRASSMANN_ORTHO_TOL) -> np.ndarray:
    n, k = U.shape
    if n == 0 or k == 0:
        return U
    Q, R = np.linalg.qr(U)
    diag_R = np.abs(np.diag(R))
    rank_deficient = np.any(diag_R < tol)
    if not rank_deficient:
        return Q
    Q_full = Q.copy()
    for j in range(k):
        if diag_R[j] >= tol:
            continue
        np.random.seed(42 + j)
        v = np.random.randn(n)
        for i in range(j):
            v -= (Q_full[:, i] @ v) * Q_full[:, i]
        for i in range(j + 1, k):
            v -= (Q_full[:, i] @ v) * Q_full[:, i]
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            Q_full[:, j] = v / nv
    return Q_full


def principal_angles(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = _validate_subspace_matrix(X, "X")
    Y = _validate_subspace_matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Dimensiones incompatibles: X{X.shape} vs Y{Y.shape}")
    _, S, _ = svd(X.T @ Y, full_matrices=False)
    S = np.clip(S, -1.0, 1.0)
    return np.sort(np.arccos(S))


def geodesic_distance(X: np.ndarray, Y: np.ndarray) -> float:
    theta = principal_angles(X, Y)
    return float(np.linalg.norm(theta))


def scalar_curvature(k: int, n: int) -> float:
    if k < 0 or n < 0:
        raise ValueError("k y n deben ser no negativos")
    if k > n:
        raise ValueError(f"k={k} no puede ser mayor que n={n}")
    return float(k * (n - k))


def exponential_map(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    X = _validate_subspace_matrix(X, "X")
    if V.shape[0] != X.shape[0] or V.shape[1] != X.shape[1]:
        raise ValueError(f"V{V.shape} y X{X.shape} deben tener la misma forma (n,k)")
    if np.linalg.norm(V) < GRASSMANN_IDENTITY_TOL:
        return X.copy()
    U, S, Vt = svd(V, full_matrices=False)
    cos_S = np.diag(np.cos(S))
    sin_S = np.diag(np.sin(S))
    V_right = Vt.T
    Y = X @ V_right @ cos_S @ Vt + U @ sin_S @ Vt
    Y, _ = np.linalg.qr(Y)
    return Y[:, :X.shape[1]]


def logarithmic_map(X: np.ndarray, Y: np.ndarray,
                    verify: bool = False,
                    allow_fallback: bool = True) -> np.ndarray:
    X = _validate_subspace_matrix(X, "X")
    Y = _validate_subspace_matrix(Y, "Y")
    if X.shape != Y.shape:
        raise ValueError(f"X{X.shape} y Y{Y.shape} deben tener la misma forma")
    n, k = X.shape
    if geodesic_distance(X, Y) < GRASSMANN_IDENTITY_TOL:
        return np.zeros((n, k))
    U_H, S, Vt = svd(X.T @ Y, full_matrices=False)
    S = np.clip(S, -1.0, 1.0)
    Theta = np.arccos(S)
    S_perp_sq = np.maximum(1.0 - S * S, 0.0)
    S_perp = np.sqrt(S_perp_sq)
    V = Vt.T
    S_perp_inv = np.where(
        S_perp > GRASSMANN_PERP_TOL,
        1.0 / np.maximum(S_perp, GRASSMANN_PERP_TOL),
        0.0
    )
    Theta = np.where(S_perp > GRASSMANN_PERP_TOL, Theta, 0.0)
    YV = Y @ V
    XU = X @ U_H @ np.diag(S)
    U_perp = (YV - XU) @ np.diag(S_perp_inv)
    Delta = U_perp @ np.diag(Theta) @ Vt
    if verify:
        try:
            Y_rec = exponential_map(X, Delta)
            d_orig = geodesic_distance(X, Y)
            d_rec = geodesic_distance(X, Y_rec)
            err = abs(d_orig - d_rec)
            if err > 1e-3 and allow_fallback:
                Delta_fb = _log_map_fallback(X, Y)
                if Delta_fb is not None:
                    Y_rec_fb = exponential_map(X, Delta_fb)
                    d_rec_fb = geodesic_distance(X, Y_rec_fb)
                    err_fb = abs(d_orig - d_rec_fb)
                    if err_fb < err:
                        return Delta_fb
        except Exception:
            if allow_fallback:
                Delta_fb = _log_map_fallback(X, Y)
                if Delta_fb is not None:
                    return Delta_fb
    return Delta


def _log_map_fallback(X: np.ndarray, Y: np.ndarray) -> Optional[np.ndarray]:
    """
    v217.0: eliminado B = X_orth.T @ Y (código muerto).
    """
    try:
        n, k = X.shape
        M = np.hstack([X, np.eye(n)])
        Q, _ = np.linalg.qr(M)
        X_orth = Q[:, k:]
        A = X.T @ Y
        U_A, S_A, Vt_A = svd(A, full_matrices=False)
        S_A = np.clip(S_A, -1.0, 1.0)
        Theta_A = np.arccos(S_A)
        S_perp = np.sqrt(np.maximum(1.0 - S_A * S_A, 0.0))
        S_perp_inv = np.where(
            S_perp > GRASSMANN_PERP_TOL,
            1.0 / np.maximum(S_perp, GRASSMANN_PERP_TOL),
            0.0
        )
        Theta_A = np.where(S_perp > GRASSMANN_PERP_TOL, Theta_A, 0.0)
        V_A = Vt_A.T
        YV = Y @ V_A
        XU = X @ U_A @ np.diag(S_A)
        U_perp = (YV - XU) @ np.diag(S_perp_inv)
        U_perp = _orthonormalize(U_perp)
        Delta = U_perp @ np.diag(Theta_A) @ Vt_A
        return Delta
    except Exception:
        return None



# ---------- GRUPO B: INFORMACIÓN ----------

def _to_probability(v: np.ndarray) -> np.ndarray:
    v_abs = np.abs(v)
    total = np.sum(v_abs)
    if total < 1e-10:
        return np.ones(len(v)) / len(v)
    return v_abs / total


def shannon_entropy(v: np.ndarray, normalize: bool = False) -> float:
    p = _to_probability(v)
    p_nonzero = p[p > 1e-10]
    if len(p_nonzero) <= 1:
        return 0.0
    H = -np.sum(p_nonzero * np.log2(p_nonzero))
    if normalize:
        max_H = np.log2(len(v))
        if max_H > 0:
            H = H / max_H
    return float(H)


def jensen_shannon_divergence(v1: np.ndarray, v2: np.ndarray) -> float:
    if len(v1) != len(v2):
        raise ValueError(f"Dimensiones incompatibles: {len(v1)} vs {len(v2)}")
    p = _to_probability(v1)
    q = _to_probability(v2)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 1e-10
        if not np.any(mask):
            return 0.0
        a_m = a[mask]
        b_m = np.maximum(b[mask], 1e-10)
        return float(np.sum(a_m * np.log2(a_m / b_m)))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.clip(jsd, 0.0, 1.0))


def hellinger_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    if len(v1) != len(v2):
        raise ValueError(f"Dimensiones incompatibles: {len(v1)} vs {len(v2)}")
    p = _to_probability(v1)
    q = _to_probability(v2)
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0.0, 1.0)
    return float(np.sqrt(1.0 - bc))


def gini_coefficient(v: np.ndarray) -> float:
    v_abs = np.abs(v)
    n = len(v_abs)
    if n < 2:
        return 0.0
    total = np.sum(v_abs)
    if total < 1e-10:
        return 0.0
    sorted_v = np.sort(v_abs)
    cumsum = np.cumsum(sorted_v)
    gini = 1.0 - 2.0 * np.sum(cumsum) / (n * total)
    return float(np.clip(gini, 0.0, 1.0))


def spearman_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    if len(v1) != len(v2):
        raise ValueError(f"Dimensiones incompatibles: {len(v1)} vs {len(v2)}")
    if len(v1) < 3:
        return 0.0
    result = spearmanr(v1, v2)
    rho = result.correlation
    if np.isnan(rho):
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def pim_to_subspace(pim_vector: np.ndarray, k: int = GRASSMANN_SUBSPACE_DIM) -> np.ndarray:
    pim_vector = np.asarray(pim_vector).flatten()
    if len(pim_vector) != 16:
        raise ValueError(f"PIM debe tener 16 dimensiones, recibido {len(pim_vector)}")
    norm = np.linalg.norm(pim_vector)
    if norm < 1e-10:
        return np.eye(16)[:, :k]
    pim_norm = pim_vector / norm
    if k == 1:
        return pim_norm.reshape(-1, 1)
    M = np.column_stack([pim_norm, np.eye(16)])
    Q, _ = np.linalg.qr(M)
    return Q[:, :k]


def compare_pim_vectors(pim1: np.ndarray, pim2: np.ndarray,
                        k: int = GRASSMANN_SUBSPACE_DIM) -> Dict:
    if len(pim1) != 16 or len(pim2) != 16:
        raise ValueError("Ambos PIM deben tener 16 dimensiones")
    X = pim_to_subspace(pim1, k=k)
    Y = pim_to_subspace(pim2, k=k)
    theta = principal_angles(X, Y)
    results = {
        'geodesic_distance': geodesic_distance(X, Y),
        'principal_angles_max': float(np.max(theta)) if len(theta) > 0 else 0.0,
        'principal_angles_min': float(np.min(theta)) if len(theta) > 0 else 0.0,
        'principal_angles_mean': float(np.mean(theta)) if len(theta) > 0 else 0.0,
        'scalar_curvature': scalar_curvature(k, 16),
        'shannon_entropy_1': shannon_entropy(pim1, normalize=True),
        'shannon_entropy_2': shannon_entropy(pim2, normalize=True),
        'jensen_shannon': jensen_shannon_divergence(pim1, pim2),
        'hellinger': hellinger_distance(pim1, pim2),
        'gini_1': gini_coefficient(pim1),
        'gini_2': gini_coefficient(pim2),
        'spearman': spearman_correlation(pim1, pim2),
    }
    return results



def run_validation_tests() -> bool:
    print("=" * 70)
    print("TESTS DE VALIDACIÓN - NÚCLEO DE 10 MÉTRICAS")
    print("=" * 70)
    all_passed = True

    print("\n📐 GRUPO A: MÉTRICAS DE GRASSMANN")
    X = np.eye(3)[:, :1]
    Y = np.eye(3)[:, 1:2]
    theta = principal_angles(X, Y)
    p = np.isclose(theta[0], np.pi/2, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Ángulos ortogonales: {theta[0]:.6f}")
    all_passed &= p
    p = np.allclose(principal_angles(X, X), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Ángulos idénticos")
    all_passed &= p
    p = np.isclose(geodesic_distance(X, X), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Geodesic identidad")
    all_passed &= p
    p = np.isclose(geodesic_distance(X, Y), np.pi/2, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Geodesic ortogonal")
    all_passed &= p
    p = np.isclose(scalar_curvature(1, 16), 15.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Scalar curvature Gr(1,16)=15")
    all_passed &= p
    np.random.seed(42)
    X_r = np.random.randn(4, 2)
    X_r, _ = np.linalg.qr(X_r)
    Y_r = X_r + 0.1 * np.random.randn(4, 2)
    Y_r, _ = np.linalg.qr(Y_r)
    try:
        d_original = geodesic_distance(X_r, Y_r)
        V = logarithmic_map(X_r, Y_r, verify=True, allow_fallback=True)
        Y_rec = exponential_map(X_r, V)
        d_recovered = geodesic_distance(X_r, Y_rec)
        p = np.isclose(d_original, d_recovered, atol=1e-3)
        print(f"  {'✅' if p else '❌'} Exp/Log preserves geodesic: "
              f"{d_original:.6f} → {d_recovered:.6f}")
        all_passed &= p
    except Exception as e:
        print(f"  ❌ Exp/Log roundtrip ERROR: {e}")
        all_passed = False

    print("\n📊 GRUPO B: MÉTRICAS DE INFORMACIÓN")
    p = np.isclose(shannon_entropy([1,0,0,0]), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Shannon determinista=0")
    all_passed &= p
    p = np.isclose(shannon_entropy([1,1,1,1]), 2.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Shannon uniforme=2")
    all_passed &= p
    p = np.isclose(jensen_shannon_divergence([1,0,0], [1,0,0]), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} JS idénticos=0")
    all_passed &= p
    p = np.isclose(jensen_shannon_divergence([1,0], [0,1]), 1.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} JS disjuntos=1")
    all_passed &= p
    p = np.isclose(hellinger_distance([1,0,0], [1,0,0]), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Hellinger idénticos=0")
    all_passed &= p
    p = np.isclose(hellinger_distance([1,0], [0,1]), 1.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Hellinger disjuntos=1")
    all_passed &= p

    print("\n📈 GRUPO C: MÉTRICAS ESTADÍSTICAS")
    p = np.isclose(gini_coefficient([1,1,1,1]), 0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Gini uniforme=0")
    all_passed &= p
    p = np.isclose(gini_coefficient([0,0,0,1]), 0.5, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Gini [0,0,0,1]=0.5 (relative formula)")
    all_passed &= p
    p = np.isclose(spearman_correlation([1,2,3,4,5], [1,2,3,4,5]), 1.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Spearman idénticos=1")
    all_passed &= p
    p = np.isclose(spearman_correlation([1,2,3,4,5], [5,4,3,2,1]), -1.0, atol=1e-6)
    print(f"  {'✅' if p else '❌'} Spearman inversos=-1")
    all_passed &= p

    print("\n🔗 TEST DE INTEGRACIÓN")
    np.random.seed(42)
    pim1 = np.random.rand(16)
    pim2 = np.random.rand(16)
    try:
        results = compare_pim_vectors(pim1, pim2)
        print(f"  ✅ Comparación completa ejecutada")
        print(f"     - Geodesic distance: {results['geodesic_distance']:.6f}")
        print(f"     - Jensen-Shannon: {results['jensen_shannon']:.6f}")
        print(f"     - Hellinger: {results['hellinger']:.6f}")
        print(f"     - Spearman: {results['spearman']:.6f}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    print("✅ TODOS LOS TESTS PASARON" if all_passed else "❌ ALGUNOS TESTS FALLARON")
    print("=" * 70)
    return all_passed


def compute_pim_profile(sequence: str, use_weights: bool = True) -> np.ndarray:
    seq = ''.join([c for c in sequence.strip() if c.isalpha() and c.upper() in POLARITY_MAP])
    if len(seq) < 2:
        return np.zeros(DIM_PAIRS)
    polarities = [POLARITY_MAP[aa.upper()] for aa in seq if aa.upper() in POLARITY_MAP]
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
        weighted = np.zeros(DIM_PAIRS)
        for i, inter in enumerate(INTERACTIONS):
            weighted[i] = counts[i] * BIOLOGICAL_WEIGHTS.get(inter, 1.0)
        tw = np.sum(weighted)
        if tw > 0:
            weighted = weighted / tw
        return weighted
    return counts


def pim_to_hash(pim_vector: np.ndarray, tolerance: float = TOLERANCE) -> str:
    discretized = np.round(pim_vector / tolerance) * tolerance
    vector_str = ','.join([f"{x:.6f}" for x in discretized])
    return hashlib.sha256(vector_str.encode()).hexdigest()[:32]



# ============================================================================
# CLASES DE INFRAESTRUCTURA
# ============================================================================

class OnlineStatistics:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

    def get_covariance(self) -> np.ndarray:
        if self.n < 2:
            return np.eye(self.dim) * 0.01
        return self.M2 / (self.n - 1)

    def get_mean(self) -> np.ndarray:
        return self.mean

    def get_std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim) * 0.01
        return np.sqrt(np.diag(self.get_covariance()))


class ProgressiveSampler:
    def __init__(self, max_samples: int = MAX_STORED_PROTEINS_PER_GROUP):
        self.max_samples = max_samples
        self.samples = []
        self.headers = []
        self.total_seen = 0

    def add(self, vector: np.ndarray, header: str):
        self.total_seen += 1
        if len(self.samples) < self.max_samples:
            self.samples.append(vector)
            self.headers.append(header)
        else:
            j = random.randint(0, self.total_seen - 1)
            if j < self.max_samples:
                self.samples[j] = vector
                self.headers[j] = header

    def get_samples(self) -> List[np.ndarray]:
        return self.samples

    def get_headers(self) -> List[str]:
        return self.headers

    def size(self) -> int:
        return len(self.samples)


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
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = (datetime.now() - self.start_time).total_seconds()
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
        if self.start_time is None:
            print("  ⚠️ start_time no definido; el resumen puede estar incompleto")
            elapsed = 0.0
        else:
            elapsed = (datetime.now() - self.start_time).total_seconds()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"  Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  Total sequences read: {self.total_sequences_processed:,}")
        print(f"  Total valid PIMs: {self.total_valid_pim:,}")
        print(f"  Total rejected: {self.total_rejected:,}")
        if self.total_sequences_processed > 0:
            print(f"  Validity rate: {self.total_valid_pim/self.total_sequences_processed*100:.2f}%")
        print(f"  Total bytes processed: {self.total_bytes_read / (1024**3):.2f} GB")
        if elapsed > 0:
            print(f"  Average speed: {self.total_sequences_processed/elapsed:,.0f} seq/s")
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


class PIMHashIndex:
    def __init__(self, tolerance: float = TOLERANCE):
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
    total_processed: int = 0
    sample_size: int = 0
    grassmann_radius: float = 0.0
    entropy: float = 0.0
    gini: float = 0.0
    geodesic_spread: float = 0.0
    scalar_curvature_value: float = 0.0
    jensen_shannon_mean: float = 0.0
    hellinger_mean: float = 0.0
    spearman_mean: float = 0.0
    all_metrics: Dict[str, float] = field(default_factory=dict)
    bootstrap_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    composite_score: Optional[float] = None
    classification: Optional[str] = None
    insufficient_n: bool = False

    def mahalanobis_distance(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 1.0
        diff = vector - self.centroid
        return float(np.sqrt(diff @ self.inv_covariance @ diff))

    def probability_of_belonging(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 0.5
        d = self.mahalanobis_distance(vector)
        return float(1.0 - chi2.cdf(d**2, df=len(self.centroid)))



class ChEMBLMapper:
    def __init__(self, mapping_file: str = CHEMBL_MAPPING_FILE):
        self.mapping = None
        self.loaded = False
        mapping_file = expand_env_vars(mapping_file)
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
            self.mapping = pd.DataFrame(data, columns=['UNIPROT_ACCESSION',
                                                       'CHEMBL_PROTEIN_ID',
                                                       'PROTEIN_NAME'])
            self.loaded = True
            print(f"  ✅ ChEMBL mapping loaded: {len(self.mapping)} entries")
        except Exception as e:
            print(f"  ⚠️ Error loading ChEMBL mapping: {e}")
            self.loaded = False

    def get_chembl_id(self, uniprot_id: str) -> Optional[str]:
        if not self.loaded:
            return None
        result = self.mapping[self.mapping['UNIPROT_ACCESSION'] == uniprot_id]
        if len(result) > 0:
            return result.iloc[0]['CHEMBL_PROTEIN_ID']
        return None

    def get_uniprot_id(self, chembl_id: str) -> Optional[str]:
        if not self.loaded:
            return None
        result = self.mapping[self.mapping['CHEMBL_PROTEIN_ID'] == chembl_id]
        if len(result) > 0:
            return result.iloc[0]['UNIPROT_ACCESSION']
        return None

    def search_by_name(self, name: str) -> List[Dict]:
        if not self.loaded:
            return []
        results = self.mapping[self.mapping['PROTEIN_NAME'].str.contains(name, case=False, na=False)]
        return results.to_dict('records')


class DRAMPLoader:
    UNIT_TO_UG_ML = {
        'μg/ml': 1.0, 'µg/ml': 1.0, 'ug/ml': 1.0, 'ug/mL': 1.0,
        'μg/mL': 1.0, 'µg/mL': 1.0,
        'mg/ml': 1000.0, 'mg/mL': 1000.0,
        'ng/ml': 0.001, 'ng/mL': 0.001,
        'pg/ml': 1e-6,
    }
    MOLAR_TO_UM = {
        'μM': 1.0, 'µM': 1.0, 'uM': 1.0, 'UM': 1.0,
        'nM': 0.001, 'mM': 1000.0, 'pM': 1e-6,
    }
    ACTIVITY_PATTERN = re.compile(
        r'(MIC|IC50|EC50|MBC|MFC|MIC50|MIC90)\s*[=<>≤≥]?\s*'
        r'([\d]+(?:[.,]\d+)?)\s*'
        r'(μM|µM|uM|UM|nM|mM|pM|'
        r'μg/ml|µg/ml|ug/ml|ug/mL|μg/mL|µg/mL|'
        r'mg/ml|mg/mL|ng/ml|ng/mL|pg/ml)',
        re.IGNORECASE
    )

    def __init__(self, tsv_file: str, verbose: bool = True):
        self.peptides: List[Dict] = []
        self.loaded = False
        self.tsv_file = expand_env_vars(tsv_file)
        self.verbose = verbose
        self.stats = {
            'total_rows': 0, 'rows_with_activity': 0, 'rows_parsed': 0,
            'rows_failed_parse': 0, 'rows_no_target': 0, 'rows_invalid_sequence': 0,
        }
        if not os.path.exists(self.tsv_file):
            if verbose:
                print(f"  ⚠️ DRAMP file not found: {self.tsv_file}")
            return
        try:
            self._load()
            self.loaded = len(self.peptides) > 0
            if verbose:
                print(f"  ✅ DRAMP loaded: {len(self.peptides)} peptides "
                      f"with REAL activity values")
                print(f"     ├─ Total rows: {self.stats['total_rows']}")
                print(f"     ├─ Rows with numeric activity: {self.stats['rows_with_activity']}")
                print(f"     ├─ Successfully parsed: {self.stats['rows_parsed']}")
                print(f"     ├─ Failed parse: {self.stats['rows_failed_parse']}")
                print(f"     ├─ No target text: {self.stats['rows_no_target']}")
                print(f"     └─ Invalid sequence: {self.stats['rows_invalid_sequence']}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Error loading DRAMP: {e}")
                import traceback
                traceback.print_exc()
            self.loaded = False

    def _load(self):
        df = pd.read_csv(self.tsv_file, sep='\t', dtype=str,
                         keep_default_na=False, low_memory=False)
        required_cols = ['Sequence', 'Target_Organism']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Columna requerida no encontrada: {col}. "
                    f"Columnas disponibles: {list(df.columns)}"
                )
        self.stats['total_rows'] = len(df)
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')

        for idx, row in df.iterrows():
            seq = str(row.get('Sequence', '')).strip().upper()
            target_text = str(row.get('Target_Organism', '')).strip()
            if not seq or len(seq) < 3:
                self.stats['rows_invalid_sequence'] += 1
                continue
            if not all(c in valid_aas for c in seq):
                self.stats['rows_invalid_sequence'] += 1
                continue
            if (not target_text
                    or target_text.lower() in (
                        'not found',
                        'no mics found in dramp database',
                        'no mic found in dramp database',
                        'no mics found in the dramp database',
                        ''
                    )):
                self.stats['rows_no_target'] += 1
                continue
            self.stats['rows_with_activity'] += 1
            activity_um = self._parse_activity(target_text, seq)
            if activity_um is None:
                self.stats['rows_failed_parse'] += 1
                continue
            pim = compute_pim_profile(seq, use_weights=True)
            if np.sum(np.abs(pim)) < 1e-6:
                self.stats['rows_failed_parse'] += 1
                continue
            name = str(row.get('Name', '')).strip()
            dramp_id = str(row.get('DRAMP_ID', '')).strip()
            self.peptides.append({
                'id': dramp_id if dramp_id else f"DRAMP_{idx}",
                'header': f"{dramp_id} {name}".strip(),
                'sequence': seq,
                'length': len(seq),
                'activity': activity_um,
                'activity_unit': 'μM',
                'raw_target': target_text,
                'name': name,
                'pim': pim,
            })
            self.stats['rows_parsed'] += 1

    def _parse_activity(self, target_text: str, sequence: str) -> Optional[float]:
        matches = self.ACTIVITY_PATTERN.findall(target_text)
        if not matches:
            return None
        by_type: Dict[str, List[float]] = defaultdict(list)
        mw = self._molecular_weight(sequence)
        for act_type, value_str, unit in matches:
            act_type = act_type.upper()
            try:
                value = float(value_str.replace(',', '.'))
            except ValueError:
                continue
            if value <= 0:
                continue
            value_um = None
            unit_norm = unit.replace('μ', 'µ')
            if unit in self.MOLAR_TO_UM:
                value_um = value * self.MOLAR_TO_UM[unit]
            elif unit_norm in self.MOLAR_TO_UM:
                value_um = value * self.MOLAR_TO_UM[unit_norm]
            elif unit in self.UNIT_TO_UG_ML:
                value_ug_ml = value * self.UNIT_TO_UG_ML[unit]
                if mw > 0:
                    value_um = value_ug_ml * 1000.0 / mw
            elif unit_norm in self.UNIT_TO_UG_ML:
                value_ug_ml = value * self.UNIT_TO_UG_ML[unit_norm]
                if mw > 0:
                    value_um = value_ug_ml * 1000.0 / mw
            if value_um is not None and value_um > 0:
                by_type[act_type].append(value_um)
        if not by_type:
            return None
        for preferred in ['IC50', 'MIC', 'MIC50', 'MIC90', 'EC50', 'MBC', 'MFC']:
            if preferred in by_type and by_type[preferred]:
                return float(min(by_type[preferred]))
        all_values = [v for vals in by_type.values() for v in vals]
        return float(min(all_values)) if all_values else None

    def _molecular_weight(self, sequence: str) -> float:
        return compute_mw_real(sequence)

    def get_all_peptides(self) -> List[Dict]:
        return self.peptides

    def get_activity_array(self) -> np.ndarray:
        return np.array([p['activity'] for p in self.peptides])

    def get_sequence_array(self) -> List[str]:
        return [p['sequence'] for p in self.peptides]

    def get_activity_distribution(self) -> Dict:
        if not self.peptides:
            return {}
        acts = self.get_activity_array()
        return {
            'n': len(acts),
            'min_uM': float(np.min(acts)),
            'max_uM': float(np.max(acts)),
            'mean_uM': float(np.mean(acts)),
            'median_uM': float(np.median(acts)),
            'std_uM': float(np.std(acts)),
            'log10_min': float(np.log10(np.min(acts) + 1e-10)),
            'log10_max': float(np.log10(np.max(acts) + 1e-10)),
            'log10_mean': float(np.mean(np.log10(acts + 1e-10))),
        }



class OperationMode(Enum):
    CHARACTERIZATION = "characterization"
    DESIGN = "design"
    HYBRID = "hybrid"
    AUTO = "auto"

    @classmethod
    def determine(cls, config_loader) -> 'OperationMode':
        oc = config_loader.get_operation_control()
        mode_str = oc.get('mode', 'auto')
        if mode_str == 'characterization':
            return cls.CHARACTERIZATION
        elif mode_str == 'design':
            return cls.DESIGN
        elif mode_str == 'hybrid':
            return cls.HYBRID
        base_seq = config_loader.get_base_peptide_sequence_string()
        is_dummy = config_loader.is_dummy_peptide()
        if base_seq and len(base_seq) > 5 and not is_dummy and oc.get('evaluate_peptide', True):
            return cls.HYBRID
        return cls.CHARACTERIZATION

    def is_characterization(self) -> bool:
        return self in [OperationMode.CHARACTERIZATION, OperationMode.HYBRID, OperationMode.AUTO]

    def is_design(self) -> bool:
        return self in [OperationMode.DESIGN, OperationMode.HYBRID]

    def is_hybrid(self) -> bool:
        return self == OperationMode.HYBRID

    def get_mode_name(self) -> str:
        return self.value


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
                    'Gn': {'name': 'Glycoprotein Gn', 'role': 'Attachment protein',
                           'length': 537, 'uniprot': 'P03518',
                           'source': 'UniProt P03518'},
                    'Gc': {'name': 'Glycoprotein Gc', 'role': 'Fusion protein',
                           'length': 516, 'uniprot': 'P03518',
                           'fusion_loop': 'GSSRFTNWGSVSLSLDAEGISGSNSFSFIES',
                           'domains': ['Domain I (691-850)', 'Domain II (851-1000)', 'Domain III (1001-1130)'],
                           'pdb_structures': ['7UU8', '7UU9', '8DZ7'],
                           'source': 'PDB 7UU8, 7UU9, 8DZ7'}
                },
                'clinical': {
                    'mortality_rate': '10-20% in hospitalized patients',
                    'cfr': '1-2% general, up to 50% in severe hemorrhagic cases',
                    'high_risk_groups': ['Immunocompromised', 'Elderly', 'Pregnant women'],
                    'transmission': 'Mosquito-borne (Aedes, Culex)',
                    'treatment': 'No specific antiviral treatment.',
                    'vaccine': 'Live-attenuated veterinary vaccines. No licensed human vaccine.',
                    'source': 'WHO RVFV Fact Sheet 2023'
                },
                'narrative_templates': {
                    'structural_summary': "The RVFV Gc glycoprotein ({length} aa) is a class II fusion protein with a conserved fusion loop ({fusion_loop}).",
                    'clinical_relevance': "RVFV is a zoonotic pathogen with mortality of {mortality_rate}.",
                    'peptide_recommendation': "The designed peptide shows {similarity} similarity with the Gc target."
                }
            },
            'ebola': {
                'target_name': 'Ebola Virus (EBOV)',
                'family': 'Filoviridae',
                'genome': 'ssRNA(-)-sense',
                'glycoproteins': {
                    'GP': {'name': 'Glycoprotein GP', 'role': 'Attachment and fusion protein',
                           'length': 676, 'uniprot': 'Q05320',
                           'fusion_loop': 'GAAIGLAWIPYFGPAAEGI',
                           'domains': ['Fusion Loop (511-553)', 'HR1 (554-598)'],
                           'pdb_structures': ['6VKM', '8Y3U', '2LCZ'],
                           'source': 'UniProt Q05320; PDB 6VKM, 8Y3U, 2LCZ'},
                    'VP40': {'name': 'Matrix protein VP40', 'role': 'Viral assembly and budding',
                             'length': 326, 'uniprot': 'Q05320',
                             'source': 'UniProt Q05320'},
                    'VP30': {'name': 'Transcription activator VP30', 'role': 'Viral transcription',
                             'length': 288, 'uniprot': 'Q05320',
                             'source': 'UniProt Q05320'}
                },
                'clinical': {
                    'mortality_rate': '25-90% depending on species',
                    'cfr': 'Zaire ~88%, Sudan ~53%, Bundibugyo ~25%, Tai 0%, Reston 0%',
                    'high_risk_groups': ['Healthcare workers', 'Family contacts', 'Immunocompromised'],
                    'transmission': 'Direct contact with bodily fluids',
                    'treatment': 'Monoclonal antibodies (REGN-EB3, mAb114), remdesivir',
                    'vaccine': 'ERVEBO (rVSV-ZEBOV) for Zaire ebolavirus',
                    'source': 'CDC Ebola Fact Sheet; WHO 2023; PALM trial (NEJM 2019)'
                },
                'narrative_templates': {
                    'structural_summary': "The Ebola GP ({length} aa) is a class I fusion protein with a conserved fusion loop ({fusion_loop}).",
                    'clinical_relevance': "Ebola mortality: {mortality_rate}. ERVEBO vaccine available for Zaire.",
                    'peptide_recommendation': "Peptide shows {similarity} similarity with the GP fusion loop."
                }
            },
            'lasv': {
                'target_name': 'Lassa Virus (LASV)',
                'family': 'Arenaviridae',
                'genome': 'ssRNA(-)-sense',
                'glycoproteins': {
                    'GP': {'name': 'Glycoprotein GP', 'role': 'Attachment and fusion',
                           'length': 491, 'uniprot': 'P08669',
                           'source': 'UniProt P08669'}
                },
                'clinical': {
                    'mortality_rate': '1-15% in hospitalized patients',
                    'transmission': 'Contact with rodent excreta',
                    'treatment': 'Ribavirin',
                    'vaccine': 'No licensed vaccine',
                    'source': 'WHO Lassa Fever Fact Sheet 2023'
                }
            }
        }

    def get_knowledge(self, target: str) -> Dict:
        if target is None:
            return self.knowledge.get('ebola', {})
        target_lower = target.lower()
        if any(k in target_lower for k in ['ebola', 'zaire', 'sudan', 'reston',
                                            'bombali', 'bundibugyo', 'tai']):
            return self.knowledge['ebola']
        for key in self.knowledge:
            if key in target_lower or target_lower in key:
                return self.knowledge[key]
        return self.knowledge.get('ebola', {})

    def get_narrative_template(self, target: str, template_name: str) -> str:
        knowledge = self.get_knowledge(target)
        return knowledge.get('narrative_templates', {}).get(template_name, "")

    def get_target_info(self, target: str) -> Dict:
        return self.get_knowledge(target)



class ClinicalLevelEvaluator:
    LEVELS = {
        1: {'name': 'Excellent', 'color': '#2ecc71', 'description': 'Optimal characteristics'},
        2: {'name': 'Good', 'color': '#27ae60', 'description': 'Favorable characteristics'},
        3: {'name': 'Moderate', 'color': '#f1c40f', 'description': 'Acceptable, requires optimization'},
        4: {'name': 'Poor', 'color': '#e67e22', 'description': 'Unfavorable, requires redesign'},
        5: {'name': 'Critical', 'color': '#e74c3c', 'description': 'Critical, not recommended'}
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
        self.source = 'heuristic; not from clinical guidelines'
        self.confidence = 0.3

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
        return self.get_level_info(level).get('color', '#808080')

    def get_name(self, level: int) -> str:
        return self.get_level_info(level).get('name', 'Moderate')

    def get_description(self, level: int) -> str:
        return self.get_level_info(level).get('description', '')


class ContextualInterpreter:
    def __init__(self, knowledge_base: NarrativeKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.level_evaluator = ClinicalLevelEvaluator()

    def interpret_similarity(self, value: float) -> str:
        if value >= 0.85:
            return "high structural similarity (low geodesic distance)"
        elif value >= 0.70:
            return "good similarity, key structural elements retained"
        elif value >= 0.50:
            return "moderate similarity, requires optimization"
        elif value >= 0.30:
            return "low similarity, significant divergence"
        return "very low similarity"

    def interpret_geodesic(self, value: float) -> str:
        if value < 0.2:
            return "very similar subspaces (small principal angles)"
        elif value < 0.5:
            return "moderate distance between subspaces"
        elif value < 1.0:
            return "significant structural divergence"
        return "large subspace distance"

    def interpret_entropy(self, value: float) -> str:
        if value < 2.0:
            return "low entropy, concentrated interaction distribution"
        elif value < 3.0:
            return "moderate entropy, balanced distribution"
        elif value < 3.8:
            return "high entropy, diverse interactions"
        return "very high entropy"

    def interpret_hellinger(self, value: float) -> str:
        if value < 0.2:
            return "high functional similarity"
        elif value < 0.4:
            return "moderate distance"
        return "significant functional differences"

    def interpret_jensen_shannon(self, value: float) -> str:
        if value < 0.2:
            return "low divergence, very similar distributions"
        elif value < 0.4:
            return "moderate divergence"
        return "high divergence"

    def interpret_gini(self, value: float) -> str:
        if value < 0.2:
            return "highly uniform distribution"
        elif value < 0.5:
            return "moderate inequality"
        return "high inequality"

    def interpret_spearman(self, value: float) -> str:
        if value > 0.7:
            return "strong monotonic correlation"
        elif value > 0.3:
            return "moderate monotonic correlation"
        elif value > -0.3:
            return "no monotonic correlation"
        return "negative monotonic correlation"

    def generate_contextual_summary(self, metrics: Dict, target: str) -> str:
        summaries = []
        if 'geodesic_distance' in metrics:
            summaries.append(f"Geodesic distance: {metrics['geodesic_distance']:.4f} - "
                            f"{self.interpret_geodesic(metrics['geodesic_distance'])}")
        if 'shannon_entropy_1' in metrics:
            summaries.append(f"Entropy: {metrics['shannon_entropy_1']:.4f} - "
                            f"{self.interpret_entropy(metrics['shannon_entropy_1'])}")
        if 'jensen_shannon' in metrics:
            summaries.append(f"Jensen-Shannon: {metrics['jensen_shannon']:.4f} - "
                            f"{self.interpret_jensen_shannon(metrics['jensen_shannon'])}")
        if 'hellinger' in metrics:
            summaries.append(f"Hellinger: {metrics['hellinger']:.4f} - "
                            f"{self.interpret_hellinger(metrics['hellinger'])}")
        if 'gini_1' in metrics:
            summaries.append(f"Gini: {metrics['gini_1']:.4f} - "
                            f"{self.interpret_gini(metrics['gini_1'])}")
        if 'spearman' in metrics:
            summaries.append(f"Spearman: {metrics['spearman']:.4f} - "
                            f"{self.interpret_spearman(metrics['spearman'])}")
        if 'principal_angles_max' in metrics:
            summaries.append(f"Max principal angle: {metrics['principal_angles_max']:.4f} rad")
        target_info = self.knowledge_base.get_target_info(target)
        target_name = target_info.get('target_name', 'the target')
        context = f"In the context of {target_name}:\n"
        for s in summaries:
            context += f"  • {s}\n"
        return context



class NarrativeSummaryGenerator:
    def __init__(self, knowledge_base: NarrativeKnowledgeBase = None):
        self.knowledge_base = knowledge_base if knowledge_base else NarrativeKnowledgeBase()
        self.level_evaluator = ClinicalLevelEvaluator()
        self.interpreter = ContextualInterpreter(self.knowledge_base)

    def generate_summary(self, peptide_data: Dict, target_name: str, profile: str) -> str:
        target_info = self.knowledge_base.get_target_info(target_name)
        if profile == 'executive':
            return self._executive(peptide_data, target_info)
        elif profile == 'biochemist':
            return self._biochemist(peptide_data, target_info)
        elif profile == 'chemist':
            return self._chemist(peptide_data, target_info)
        elif profile == 'analytical_chemist':
            return self._analytical(peptide_data, target_info)
        elif profile == 'physicochemist':
            return self._physicochemist(peptide_data, target_info)
        elif profile == 'bioinformatician':
            return self._bioinformatician(peptide_data, target_info)
        return self._general(peptide_data, target_info)

    def _executive(self, data: Dict, target_info: Dict) -> str:
        seq = data.get('sequence', '')
        geodesic = data.get('geodesic_distance', 0.5)
        activity = data.get('ic50_um', None)
        target_name = target_info.get('target_name', 'Target')
        if geodesic < 0.5:
            status, rec = "FAVORABLE", "Proceed to experimental validation"
        elif geodesic < 1.0:
            status, rec = "MODERATE", "Optimize before validation"
        else:
            status, rec = "UNFAVORABLE", "Redesign completely"
        activity_str = f"{activity:.4f} μM" if activity is not None else "NOT AVAILABLE"
        return f"""
        📊 EXECUTIVE SUMMARY - {target_name}
        ================================================
        Peptide: {seq[:20]}... ({len(seq)} aa)
        Geodesic Distance: {geodesic:.4f}
        Predicted IC50: {activity_str}
        Status: {status}
        Recommendation: {rec}
        ================================================
        """

    def _biochemist(self, data: Dict, target_info: Dict) -> str:
        seq = data.get('sequence', '')
        geodesic = data.get('geodesic_distance', 0.5)
        hellinger = data.get('hellinger', 0.5)
        js = data.get('jensen_shannon', 0.5)
        entropy = data.get('shannon_entropy_1', 2.5)
        target_name = target_info.get('target_name', 'Target')
        return f"""
        🧬 BIOCHEMICAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        LENGTH: {len(seq)} aa

        GRASSMANN METRICS:
        • Geodesic distance: {geodesic:.4f}
        • Interpretation: {self.interpreter.interpret_geodesic(geodesic)}

        INFORMATION METRICS:
        • Shannon entropy: {entropy:.4f}
        • Jensen-Shannon: {js:.4f} ({self.interpreter.interpret_jensen_shannon(js)})
        • Hellinger: {hellinger:.4f} ({self.interpreter.interpret_hellinger(hellinger)})

        RECOMMENDATIONS:
        • Evaluate binding via SPR or ITC
        • Membrane fusion assays
        • Mutagenesis of key residues
        ================================================
        """

    def _chemist(self, data: Dict, target_info: Dict) -> str:
        seq = data.get('sequence', '')
        mw = compute_mw_real(seq) if seq else 0
        charge = compute_net_charge_real(seq) if seq else 0
        hydro = compute_gravy_real(seq) if seq else 0
        sol = compute_solubility_real(seq) if seq else {'probability': None}
        sol_prob = sol.get('probability') if isinstance(sol, dict) else sol
        sol_str = f"{sol_prob:.3f}" if sol_prob is not None else "N/A (fuera de dominio)"
        target_name = target_info.get('target_name', 'Target')
        return f"""
        🧪 CHEMICAL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        MOLECULAR WEIGHT: {mw:.1f} Da
        NET CHARGE (pH 7.4): {charge:.2f}
        GRAVY: {hydro:.2f}
        SOLUBILITY PROB: {sol_str} (Wilkinson-Harrison 1991)

        SYNTHESIS:
        • Fmoc solid-phase peptide synthesis (SPPS)
        • Preparative HPLC purification
        • {"Cyclization recommended" if hydro > 1.0 else "Linear synthesis"}
        ================================================
        """

    def _analytical(self, data: Dict, target_info: Dict) -> str:
        """
        v217.0: usa fallback 'N/A' si no hay secuencia.
        """
        seq = data.get('sequence', '') or 'N/A'
        mw = compute_mw_real(seq) if seq and seq != 'N/A' else 0
        target_name = target_info.get('target_name', 'Target')
        return f"""
        🔬 QUALITY CONTROL SUMMARY - {target_name}
        ================================================
        PEPTIDE: {seq}
        THEORETICAL MASS: {mw:.1f} Da

        RECOMMENDED TECHNIQUES:
        • HPLC-MS/MS: mass confirmation (>95% purity)
        • SEC: aggregation state
        • DLS: particle size
        • CD: secondary structure
        • SPR: binding kinetics

        VALIDATION PROTOCOL:
        • Antiviral activity in plaque assay
        • Cytotoxicity (CC50)
        • Selectivity Index (SI = CC50/IC50)
        ================================================
        """

    def _physicochemist(self, data: Dict, target_info: Dict) -> str:
        entropy = data.get('shannon_entropy_1', 2.5)
        js = data.get('jensen_shannon', 0.5)
        geo = data.get('geodesic_distance', 0.5)
        target_name = target_info.get('target_name', 'Target')
        return f"""
        ⚡ PHYSICOCHEMICAL SUMMARY - {target_name}
        ================================================
        INFORMATION METRICS:
        • Shannon entropy: {entropy:.4f}
        • Jensen-Shannon: {js:.4f}
        • Geodesic distance: {geo:.4f}

        DYNAMICS:
        • Flexibility: {"High" if entropy > 3.0 else "Moderate" if entropy > 2.0 else "Low"}

        INTERPRETATION:
        • {self.interpreter.interpret_geodesic(geo)}
        ================================================
        """

    def _bioinformatician(self, data: Dict, target_info: Dict) -> str:
        metrics = [(k, v) for k, v in data.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in ['length', 'molecular_weight']]
        metrics_sorted = sorted(metrics, key=lambda x: x[1], reverse=True)
        target_name = target_info.get('target_name', 'Target')
        lines = "\n".join([f"    • {k}: {v:.4f}" for k, v in metrics_sorted[:15]])
        return f"""
        💻 BIOINFORMATICS SUMMARY - {target_name}
        ================================================
        DESIGN METRICS:
        {lines}

        METHODS:
        • Grassmann manifold: principal angles + geodesic distance
        • Information theory: Shannon, Jensen-Shannon, Hellinger
        • Statistics: Gini, Spearman
        • ML: Random Forest / XGBoost with 44 physicochemical features (DRAMP labels)

        OPTIMIZATION:
        • Verify Grassmann subspace embedding
        • Bootstrap confidence intervals
        ================================================
        """

    def _general(self, data: Dict, target_info: Dict) -> str:
        target_name = target_info.get('target_name', 'Target')
        lines = "\n".join([f"    • {k}: {v:.4f}" for k, v in data.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)][:10])
        return f"""
        📋 GENERAL SUMMARY - {target_name}
        ================================================
        MAIN METRICS:
        {lines}
        ================================================
        """


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
        table = self._generate_table(peptide_data, target_name)
        safe_save_text(table, "multidisciplinary_summary_table.txt", output_dir)
        return reports

    def _generate_table(self, peptide_data: Dict, target_name: str) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 MULTIDISCIPLINARY SUMMARY TABLE - {target_name}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("TABLE 1: EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Peptide: {peptide_data.get('sequence', '')}")
        lines.append(f"Length: {len(peptide_data.get('sequence', ''))} aa")
        lines.append(f"Geodesic distance: {peptide_data.get('geodesic_distance', 0):.4f}")
        lines.append("")
        lines.append("TABLE 2: CORE METRICS BY GROUP")
        lines.append("-" * 40)
        lines.append("| Group | Metric | Value |")
        lines.append("|-------|--------|-------|")

        grassmann_metrics = [
            ('geodesic_distance', 'Geodesic Distance'),
            ('principal_angles_max', 'Max Principal Angle'),
            ('principal_angles_mean', 'Mean Principal Angle'),
            ('scalar_curvature', 'Scalar Curvature (constante estructural)'),
        ]
        info_metrics = [
            ('shannon_entropy_1', 'Shannon Entropy'),
            ('jensen_shannon', 'Jensen-Shannon'),
            ('hellinger', 'Hellinger'),
        ]
        stat_metrics = [
            ('gini_1', 'Gini Coefficient'),
            ('spearman', 'Spearman Correlation'),
        ]

        for key, name in grassmann_metrics:
            if key in peptide_data:
                val = peptide_data[key]
                lines.append(f"| Grassmann | {name} | {val:.4f} |")
        for key, name in info_metrics:
            if key in peptide_data:
                val = peptide_data[key]
                lines.append(f"| Information | {name} | {val:.4f} |")
        for key, name in stat_metrics:
            if key in peptide_data:
                val = peptide_data[key]
                lines.append(f"| Statistics | {name} | {val:.4f} |")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)



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
        report.append(self._executive(metrics, target))
        report.append("")
        report.append("2. GRASSMANN ANALYSIS")
        report.append("-" * 40)
        report.append(self._grassmann(metrics))
        report.append("")
        report.append("3. INFORMATION ANALYSIS")
        report.append("-" * 40)
        report.append(self._information(metrics))
        report.append("")
        report.append("4. STATISTICAL ANALYSIS")
        report.append("-" * 40)
        report.append(self._statistics(metrics))
        report.append("")
        report.append("5. CONTEXTUAL INTERPRETATION")
        report.append("-" * 40)
        report.append(self.interpreter.generate_contextual_summary(metrics, target))
        report.append("")
        report.append("6. METRICS TABLE")
        report.append("-" * 40)
        report.append(self._metrics_table(metrics))
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)

    def _executive(self, metrics: Dict, target: str) -> str:
        geodesic = metrics.get('geodesic_distance', 0.5)
        entropy = metrics.get('shannon_entropy_1', 2.5)
        return f"""
        The target shows a geodesic distance of {geodesic:.4f} and normalized entropy of {entropy:.4f}.
        {self.interpreter.interpret_geodesic(geodesic)}.
        """

    def _grassmann(self, metrics: Dict) -> str:
        lines = []
        if 'geodesic_distance' in metrics:
            lines.append(f"• Geodesic distance: {metrics['geodesic_distance']:.4f}")
        if 'principal_angles_max' in metrics:
            lines.append(f"• Max principal angle: {metrics['principal_angles_max']:.4f}")
        if 'principal_angles_min' in metrics:
            lines.append(f"• Min principal angle: {metrics['principal_angles_min']:.4f}")
        if 'principal_angles_mean' in metrics:
            lines.append(f"• Mean principal angle: {metrics['principal_angles_mean']:.4f}")
        if 'scalar_curvature' in metrics:
            lines.append(f"• Scalar curvature: {metrics['scalar_curvature']:.2f} "
                        f"(constante estructural de Gr(2,16), no varía por grupo)")
        return "\n".join(lines) if lines else "No Grassmann metrics available"

    def _information(self, metrics: Dict) -> str:
        lines = []
        if 'shannon_entropy_1' in metrics:
            lines.append(f"• Shannon entropy: {metrics['shannon_entropy_1']:.4f}")
        if 'jensen_shannon' in metrics:
            lines.append(f"• Jensen-Shannon: {metrics['jensen_shannon']:.4f}")
        if 'hellinger' in metrics:
            lines.append(f"• Hellinger: {metrics['hellinger']:.4f}")
        return "\n".join(lines) if lines else "No information metrics available"

    def _statistics(self, metrics: Dict) -> str:
        lines = []
        if 'gini_1' in metrics:
            lines.append(f"• Gini coefficient: {metrics['gini_1']:.4f}")
        if 'spearman' in metrics:
            lines.append(f"• Spearman correlation: {metrics['spearman']:.4f}")
        return "\n".join(lines) if lines else "No statistical metrics available"

    def _metrics_table(self, metrics: Dict) -> str:
        """
        v217.0: salta bool (no los formatea como float).
        """
        lines = ["| Metric | Value |", "|--------|-------|"]
        for k, v in sorted(metrics.items()):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                lines.append(f"| {k} | {v:.4f} |")
            elif v is not None and isinstance(v, str):
                lines.append(f"| {k} | {v} |")
        return "\n".join(lines)


class GrassmannPIM:
    def __init__(self, dim: int = DIM_PAIRS):
        self.dim = dim
        self.k = GRASSMANN_SUBSPACE_DIM
        print(f"  ✅ GrassmannPIM v217.0 initialized (REAL 10-metric core, robust)")

    def principal_angles(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        X = pim_to_subspace(v1, k=self.k)
        Y = pim_to_subspace(v2, k=self.k)
        return principal_angles(X, Y)

    def geodesic_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        X = pim_to_subspace(v1, k=self.k)
        Y = pim_to_subspace(v2, k=self.k)
        return geodesic_distance(X, Y)

    def geodesic_distance_with_ci(self, v1: np.ndarray, v2: np.ndarray,
                                   n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
        d = self.geodesic_distance(v1, v2)
        if not USE_BOOTSTRAP:
            return d, 0.0
        dim = len(v1)
        bootstrapped = []
        for _ in range(min(n_bootstrap, 100)):
            idx = np.random.choice(dim, dim, replace=True)
            try:
                d_boot = self.geodesic_distance(v1[idx], v2[idx])
                if np.isfinite(d_boot):
                    bootstrapped.append(d_boot)
            except Exception:
                continue
        if bootstrapped:
            return float(np.mean(bootstrapped)), float(np.std(bootstrapped))
        return d, 0.0

    def scalar_curvature(self) -> float:
        return scalar_curvature(self.k, self.dim)

    def exponential_map(self, v_base: np.ndarray, v_tangent: np.ndarray) -> np.ndarray:
        X = pim_to_subspace(v_base, k=self.k)
        V = pim_to_subspace(v_tangent, k=self.k)
        Y = exponential_map(X, V)
        return Y.flatten()[:self.dim]

    def logarithmic_map(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        X = pim_to_subspace(v1, k=self.k)
        Y = pim_to_subspace(v2, k=self.k)
        try:
            T = logarithmic_map(X, Y, verify=False, allow_fallback=True)
            return T.flatten()[:self.dim]
        except Exception:
            return np.zeros(self.dim)

    def shannon_entropy(self, v: np.ndarray, normalize: bool = False) -> float:
        return shannon_entropy(v, normalize=normalize)

    def jensen_shannon(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return jensen_shannon_divergence(v1, v2)

    def hellinger(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return hellinger_distance(v1, v2)

    def gini(self, v: np.ndarray) -> float:
        return gini_coefficient(v)

    def spearman(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return spearman_correlation(v1, v2)

    def compare(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        return compare_pim_vectors(v1, v2, k=self.k)

    def compute_enhanced_metrics(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        return self.compare(v1, v2)

    def wedge_product(self, v1: np.ndarray, v2: np.ndarray,
                      with_ci: bool = False) -> Tuple[float, float]:
        d = self.geodesic_distance(v1, v2)
        max_d = np.pi / 2 * np.sqrt(self.k)
        sim = max(0.0, 1.0 - d / max_d) if max_d > 0 else 0.0
        if with_ci:
            d_mean, d_std = self.geodesic_distance_with_ci(v1, v2)
            sim_std = d_std / max_d if max_d > 0 else 0.0
            return sim, sim_std
        return sim, 0.0

    def grassmann_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return self.geodesic_distance(v1, v2)

    def hodge_complementarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return 1.0 - self.jensen_shannon(v1, v2)

    def norm_metric(self, v: np.ndarray) -> Tuple[float, float]:
        n = float(np.linalg.norm(v))
        sign = 1.0 if n > 0 else 0.0
        return n, sign

    def similarity_metric(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.abs(np.dot(v1, v2)) / (n1 * n2))

    def metric_signature_info(self) -> Dict:
        return {
            'total_components': self.dim,
            'is_biological': USE_WEIGHTS,
            'description': 'PIM vector of 16 interaction types'
        }

    def fractal_dimension(self, v: np.ndarray) -> float:
        n = len(v)
        if n < 2:
            return 0.0
        v_abs = np.abs(v)
        cumsum = np.cumsum(v_abs)
        rng = np.max(cumsum) - np.min(cumsum)
        if rng < 1e-10:
            return 0.0
        cumsum = (cumsum - np.min(cumsum)) / rng
        counts = []
        for scale in range(1, 10):
            box_size = max(1, n // (2**scale))
            boxes = set()
            for i in range(0, n - box_size, box_size):
                box_value = np.mean(cumsum[i:i+box_size])
                boxes.add(int(box_value * 100))
            counts.append(len(boxes))
        if len(counts) > 2:
            log_scales = np.log(np.array([max(1, n // (2**s)) for s in range(1, 10)]))
            log_counts = np.log(np.array(counts) + 1)
            slope, _, _, _, _ = linregress(log_scales, log_counts)
            return float(-slope)
        return 0.0

    def discrete_radon_transform(self, v: np.ndarray, n_angles: int = 8) -> np.ndarray:
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

    def wasserstein_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        p1 = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
        p2 = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
        cdf1 = np.cumsum(p1)
        cdf2 = np.cumsum(p2)
        return float(np.sum(np.abs(cdf1 - cdf2)) / len(v1))

    def polarity_interaction_laplacian(self, v: np.ndarray) -> Dict:
        n = len(v)
        v_norm = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = min(v_norm[i], v_norm[j]) / (max(v_norm[i], v_norm[j]) + 1e-10)
                if sim > 0.3:
                    adj[i, j] = adj[j, i] = sim
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-10))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        try:
            eigvals = np.linalg.eigvalsh(L_norm)
        except Exception:
            eigvals = np.zeros(n)
        return {
            'eigenvalues': eigvals,
            'spectral_gap': float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0,
            'connectivity': float(np.sum(adj > 0) / (n * (n - 1))) if n > 1 else 0.0
        }

    def karhunen_loeve_decomposition(self, vectors: List[np.ndarray],
                                     n_components: int = 8) -> Dict:
        if len(vectors) < 2:
            return {
                'eigenvalues': np.array([]),
                'eigenvectors': np.array([]),
                'components': np.array([]),
                'explained_variance': np.array([]),
                'mean': np.zeros(self.dim) if not vectors else np.zeros(len(vectors[0]))
            }
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
        total_var = np.sum(eigvals) + 1e-10
        return {
            'eigenvalues': eigvals[:n_components],
            'eigenvectors': eigvecs[:, :n_components],
            'components': components,
            'explained_variance': eigvals[:n_components] / total_var,
            'mean': mean
        }



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
            tools['metapredict']['version'] = getattr(meta, '__version__', 'unknown')
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

    def analyze_sequence(self, sequence: str, name: str, is_peptide: bool = False) -> Dict:
        result = {'name': name, 'length': len(sequence),
                 'is_peptide': is_peptide, 'tools': {}}

        if self.tools_available['metapredict']['available'] and PIDP_USE_METAPREDICT:
            try:
                import metapredict as meta
                scores = meta.predict_disorder(sequence)
                md = {}
                for threshold in PIDP_THRESHOLDS:
                    pct = sum(1 for s in scores if s > threshold) / len(sequence) * 100
                    md[f'disorder_{threshold:.1f}'] = round(pct, 2)
                md['mean_score'] = round(float(np.mean(scores)), 4)
                md['max_score'] = round(float(np.max(scores)), 4)
                md['min_score'] = round(float(np.min(scores)), 4)
                md['std_score'] = round(float(np.std(scores)), 4)
                result['tools']['metapredict'] = md
            except Exception as e:
                result['tools']['metapredict'] = {'error': str(e)}

        if self.tools_available['aiupred']['available'] and PIDP_USE_AIUPRED:
            try:
                from aiupred import AIUPred
                predictor = AIUPred()
                scores = predictor.predict_disorder(sequence)
                ad = {}
                for threshold in PIDP_THRESHOLDS:
                    pct = sum(1 for s in scores if s > threshold) / len(sequence) * 100
                    ad[f'disorder_{threshold:.1f}'] = round(pct, 2)
                ad['mean_score'] = round(float(np.mean(scores)), 4)
                ad['max_score'] = round(float(np.max(scores)), 4)
                ad['min_score'] = round(float(np.min(scores)), 4)
                ad['std_score'] = round(float(np.std(scores)), 4)
                try:
                    redox_plus, redox_minus = predictor.predict_redox_profiles(sequence)
                    ad['redox_sensitivity'] = round(float(np.mean(redox_plus - redox_minus)), 4)
                    ad['redox_plus_mean'] = round(float(np.mean(redox_plus)), 4)
                    ad['redox_minus_mean'] = round(float(np.mean(redox_minus)), 4)
                except Exception as e:
                    ad['redox_error'] = str(e)
                result['tools']['aiupred'] = ad
            except Exception as e:
                result['tools']['aiupred'] = {'error': str(e)}

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

    def analyze_target_proteins(self, results_dir: str) -> Dict:
        if not USE_PIDP:
            print("\n  ⚠️ PIDP analysis disabled")
            return {}
        print("\n  🧬 Performing PIDP analysis on target proteins...")
        all_results = {}
        for group_name in self.ga.main_groups:
            if group_name not in self.ga.group_stats:
                print(f"     ⚠️ Group {group_name} not found")
                continue
            sequence = self._get_sequence_from_group(group_name)
            if sequence is None or len(sequence) < 10:
                print(f"     ⚠️ No sequence for {get_display_name(group_name)}")
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

        if hasattr(self.ga, 'therapeutic_profile') and self.ga.therapeutic_profile:
            peptide_seq = self.ga.therapeutic_profile.get('peptide', {}).get('sequence', '')
            if peptide_seq and len(peptide_seq) > 5:
                peptide_result = self.analyze_sequence(peptide_seq, 'synthetic_peptide', is_peptide=True)
                all_results['synthetic_peptide'] = peptide_result
                print(f"     └─ Synthetic peptide analyzed")

        self._save_results(all_results, results_dir)
        self.results = all_results
        return all_results

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
                        rows.append({'Tool': tool, 'Metric': key, 'Value': value})
            if rows:
                df = pd.DataFrame(rows)
                safe_save_csv(df, f"pidp_analysis_{name}.csv", results_dir)
                print(f"  ✅ PIDP analysis saved: pidp_analysis_{name}.csv")

        summary_rows = []
        for name, data in results.items():
            row = {'Protein/Peptide': name, 'Length': data.get('length', 0),
                   'Is Peptide': data.get('is_peptide', False)}
            for tool, metrics in data.get('tools', {}).items():
                if 'error' in metrics:
                    continue
                for key, value in metrics.items():
                    if key.startswith('disorder_'):
                        row[f'{tool}_{key}'] = value
                    elif key == 'mean_score':
                        row[f'{tool}_mean'] = value
                    elif key == 'redox_sensitivity':
                        row[f'{tool}_redox'] = value
            summary_rows.append(row)
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            safe_save_csv(df_summary, "pidp_summary_all_targets.csv", results_dir)
            print(f"  ✅ PIDP summary saved: pidp_summary_all_targets.csv")

        peptide_data = results.get('synthetic_peptide')
        if peptide_data:
            rows = []
            for tool, metrics in peptide_data.get('tools', {}).items():
                if 'error' in metrics:
                    continue
                for key, value in metrics.items():
                    if key.endswith('_score') or key.startswith('disorder_'):
                        rows.append({'Tool': tool, 'Metric': key, 'Value': value})
            if rows:
                df = pd.DataFrame(rows)
                safe_save_csv(df, "pidp_peptide_analysis.csv", results_dir)
                print(f"  ✅ PIDP peptide analysis saved")


class ChemicalProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer'):
        self.ga = analyzer
        self.dim = DIM_PAIRS
        self.n_glycosylation_motif = re.compile(r'N[^P][ST]')
        self.o_glycosylation_motif = re.compile(r'[ST]')
        self.phosphorylation_motifs = {
            'PKA': re.compile(r'[RK][ST][^P]'),
            'PKC': re.compile(r'[ST][^P][KR]'),
            'CK2': re.compile(r'[ST][^P][DE]'),
            'Tyr': re.compile(r'Y[^P]')
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
        pp, pn, np_, nn = v[0], v[1], v[4], v[5]
        net_charge = (pn + np_) - (pp + nn)
        charge_density = pp + pn + np_ + nn
        charge_balance = (pn + np_) / (charge_density + 1e-10)
        return {
            'net_charge': net_charge, 'charge_density': charge_density,
            'charge_balance': charge_balance, 'positive_positive': pp,
            'positive_negative': pn, 'negative_positive': np_, 'negative_negative': nn,
            'source': 'PIM charge interactions'
        }

    def compute_pka_profile(self, sequence: str) -> Dict:
        if not sequence:
            return {'pI': 7.0, 'net_charge_at_pH_7': 0.0,
                    'source': 'Henderson-Hasselbalch', 'confidence': 0.9}
        pI = compute_pI_real(sequence)
        net_charge = compute_net_charge_real(sequence, ph=7.4)
        return {
            'pI': pI,
            'net_charge_at_pH_7': net_charge,
            'source': 'Henderson-Hasselbalch (EMBOSS pKa values)',
            'confidence': 0.9,
            'pka_values_used': PKA_VALUES
        }

    def compute_electrostatic_map(self, v: np.ndarray) -> Dict:
        pos = v[0] + v[2] + v[8]
        neg = v[5] + v[6] + v[9]
        neutral = v[10] + v[14] + v[15]
        total = pos + neg + neutral + 1e-10
        return {
            'positive_charge_fraction': pos / total,
            'negative_charge_fraction': neg / total,
            'surface_charge_density': (pos + neg) / total,
            'electrostatic_potential': (pos - neg) / total,
            'source': 'PIM', 'confidence': 0.5
        }

    def compute_gravy(self, sequence: str) -> float:
        return compute_gravy_real(sequence)

    def compute_hydrophobicity_profile(self, v: np.ndarray) -> Dict:
        weighted_sum = sum(
            v[i] * BIOLOGICAL_WEIGHTS.get(INTERACTIONS[i], 1.0)
            for i in range(len(v))
        )
        peaks = [
            {'component': i, 'interaction': INTERACTIONS[i],
             'weight': float(v[i]),
             'biological_weight': BIOLOGICAL_WEIGHTS.get(INTERACTIONS[i], 1.0)}
            for i in range(len(v))
            if BIOLOGICAL_WEIGHTS.get(INTERACTIONS[i], 1.0) >= 1.3
        ]
        peaks = sorted(peaks, key=lambda x: x['weight'], reverse=True)
        return {
            'hydrophobicity_score': float(weighted_sum),
            'hydrophobic_peaks': peaks[:5],
            'num_hydrophobic_components': len(peaks),
            'source': 'BIOLOGICAL_WEIGHTS (literature-based)',
            'confidence': 0.5
        }

    def compute_hydrophobic_patches(self, v: np.ndarray) -> Dict:
        indices = [3, 7, 11, 12, 13, 14, 15]
        patches = []
        current = []
        for i in range(len(v)):
            if i in indices and v[i] > 0.01:
                current.append(i)
            else:
                if current:
                    patches.append(current)
                    current = []
        if current:
            patches.append(current)
        patch_scores = [{'components': p, 'size': len(p), 'score': float(sum(v[i] for i in p))}
                        for p in patches]
        patch_scores = sorted(patch_scores, key=lambda x: x['score'], reverse=True)
        return {
            'num_patches': len(patches),
            'patch_scores': patch_scores[:5],
            'hydrophobic_component_fraction': float(sum(v[i] for i in indices) / (sum(v) + 1e-10)),
            'source': 'PIM', 'confidence': 0.5
        }

    def compute_glycosylation_sites(self, sequence: str) -> Dict:
        if sequence is None or len(sequence) < 3:
            return {'n_sites': [], 'o_sites': [], 'count_n': 0, 'count_o': 0,
                    'source': 'regex', 'confidence': 0.7}
        n_sites = []
        for match in self.n_glycosylation_motif.finditer(sequence):
            pos = match.start()
            if pos + 2 < len(sequence) and sequence[pos + 1] != 'P':
                n_sites.append({
                    'position': pos,
                    'motif': sequence[pos:pos+3],
                    'sequence': sequence[max(0, pos-3):min(len(sequence), pos+6)]
                })
        o_sites = []
        for match in self.o_glycosylation_motif.finditer(sequence):
            pos = match.start()
            o_sites.append({
                'position': pos,
                'residue': sequence[pos],
                'sequence': sequence[max(0, pos-3):min(len(sequence), pos+4)]
            })
        return {
            'n_sites': n_sites, 'o_sites': o_sites,
            'count_n': len(n_sites), 'count_o': len(o_sites),
            'density_n': len(n_sites) / (len(sequence) + 1),
            'density_o': len(o_sites) / (len(sequence) + 1),
            'source': 'regex motif scan (N[^P][ST])',
            'confidence': 0.7
        }

    def compute_ptm_sites(self, sequence: str) -> Dict:
        if sequence is None or len(sequence) < 3:
            return {'phosphorylation': [], 'acetylation': [], 'ubiquitination': [],
                    'source': 'regex', 'confidence': 0.6}
        phospho = []
        for kinase, pattern in self.phosphorylation_motifs.items():
            for match in pattern.finditer(sequence):
                pos = match.start()
                for offset, residue in enumerate(match.group()):
                    if residue in ['S', 'T', 'Y']:
                        phospho.append({
                            'position': pos + offset,
                            'residue': residue,
                            'kinase': kinase,
                            'motif': match.group()
                        })
        acetyl = [{'position': pos, 'residue': 'K'}
                  for pos, r in enumerate(sequence) if r == 'K']
        ubiq = [{'position': pos, 'residue': 'K'}
                for pos, r in enumerate(sequence) if r == 'K']
        return {
            'phosphorylation': phospho,
            'acetylation': acetyl,
            'ubiquitination': ubiq,
            'source': 'regex motif scan',
            'confidence': 0.6
        }

    def compute_solubility_aggregation(self, sequence: str) -> Dict:
        if not sequence:
            return {'solubility_prob': None, 'applicable': False,
                    'aggregation_score': 0.0,
                    'source': 'Wilkinson-Harrison 1991', 'confidence': 0.0,
                    'note': 'empty sequence'}
        sol = compute_solubility_real(sequence)
        hydro_frac = sum(1 for aa in sequence.upper() if aa in 'AILMFWYV') / len(sequence)
        aggregation = float(min(1.0, hydro_frac * 1.5))
        return {
            'solubility_prob': sol['probability'],
            'applicable': sol['applicable'],
            'cv': sol['cv'],
            'aggregation_score': aggregation,
            'source': 'Wilkinson-Harrison 1991 (solubility) + heuristic (aggregation)',
            'confidence': sol['confidence'],
            'note': sol.get('note', '')
        }

    def compute_stability(self, sequence: str) -> Dict:
        if not sequence:
            return {'delta_g': None, 'tm': None, 'stability_score': None,
                    'source': 'not_implemented', 'confidence': 0.0,
                    'mutations': [], 'note': 'Requires FoldX/Rosetta',
                    'informative': False}
        pos = sum(1 for aa in sequence if aa in 'KRH')
        neg = sum(1 for aa in sequence if aa in 'DE')
        delta_g_raw = -4.0 - 0.1 * (pos + neg)
        tm_raw = 40.0 + 1.5 * (pos + neg)
        return {
            'delta_g': float(delta_g_raw),
            'tm': float(tm_raw),
            'stability_score': float(min(1.0, (pos + neg) / len(sequence) * 3)),
            'mutations': [],
            'source': 'heuristic; NOT calibrated against experimental data',
            'confidence': 0.0,
            'note': 'Valores heurísticos sin calibración. NO usar como ΔG/Tm reales. Para ΔΔG usar FoldX/Rosetta.',
            'informative': False
        }

    def compute_hotspots(self, v: np.ndarray) -> Dict:
        hotspots = []
        cr = v[1] + v[4] + v[0] + v[5]
        hotspots.append({
            'name': 'Charge-rich region',
            'score': float(min(0.95, 0.7 * cr + 0.3 * v[10])),
            'type': 'charged',
            'key_residues': [],
            'source': 'PIM-based heuristic; no experimental validation',
            'confidence': 0.2
        })
        hp = v[15] + v[14] + v[11]
        hotspots.append({
            'name': 'Hydrophobic patch',
            'score': float(min(0.95, 0.8 * hp + 0.2 * v[7])),
            'type': 'hydrophobic',
            'key_residues': [],
            'source': 'PIM-based heuristic',
            'confidence': 0.2
        })
        mr = v[11] + v[14] + v[2] + v[6]
        hotspots.append({
            'name': 'Mixed polar/hydrophobic',
            'score': float(min(0.95, 0.6 * mr + 0.4 * v[10])),
            'type': 'mixed',
            'key_residues': [],
            'source': 'PIM-based heuristic',
            'confidence': 0.2
        })
        hotspots = sorted(hotspots, key=lambda x: x['score'], reverse=True)
        return {'hotspots': hotspots, 'best_hotspot': hotspots[0] if hotspots else None}

    def compute_reactivity_profile(self, v: np.ndarray) -> Dict:
        cys = min(1, (v[0] + v[1] + v[2] + v[8] + v[9]) * 2)
        lys = min(1, (v[4] + v[5] + v[6] + v[8] + v[9]) * 2)
        tyr = min(1, (v[2] + v[6] + v[8] + v[9] + v[10]) * 2)
        return {
            'cysteine_accessibility': float(cys),
            'lysine_accessibility': float(lys),
            'tyrosine_accessibility': float(tyr),
            'reactive_residues': {
                'cysteine_sites': [],
                'lysine_sites': [],
                'tyrosine_sites': []
            },
            'source': 'PIM-based heuristic',
            'confidence': 0.3
        }

    def compute_metal_binding(self, v: np.ndarray) -> Dict:
        ca = min(1, (v[1] + v[4] + v[6] + v[7] + v[9] + v[13]) * 1.5)
        zn = min(1, (v[0] + v[2] + v[3] + v[8] + v[10] + v[12]) * 1.2)
        return {
            'calcium_binding_score': float(ca),
            'zinc_binding_score': float(zn),
            'metal_sites': {'calcium': [], 'zinc': []},
            'source': 'PIM-based heuristic',
            'confidence': 0.3
        }

    def compute_membrane_permeability(self, sequence: str) -> Dict:
        delta_g = compute_membrane_permeability_real(sequence) if sequence else 0.0
        return {
            'delta_g_octanol_kcal_mol': float(delta_g),
            'source': 'Wimley-White 1996 (kcal/mol)',
            'confidence': 0.8,
            'is_peptide': len(sequence) < 30 if sequence else False,
            'interpretation': 'negative=hydrophobic, positive=hydrophilic'
        }

    def compute_lipid_binding(self, v: np.ndarray) -> Dict:
        hydro = v[15] + v[14] + v[11]
        charged = v[1] + v[4]
        ps = min(1, (0.6 * hydro + 0.4 * charged) * 1.5)
        chol = min(1, (0.8 * hydro + 0.2 * v[15]) * 1.5)
        gang = min(1, (0.3 * hydro + 0.7 * v[10]) * 1.5)
        types = {'phosphatidylserine': ps, 'cholesterol': chol, 'ganglioside': gang}
        return {
            'phosphatidylserine_score': float(ps),
            'cholesterol_score': float(chol),
            'ganglioside_score': float(gang),
            'best_lipid': max(types, key=types.get),
            'source': 'PIM-based heuristic; no experimental validation',
            'confidence': 0.2
        }

    def compute_buffer_stability(self, sequence: str) -> Dict:
        if not sequence:
            return {'optimal_ph': 7.0, 'salt_tolerance_mM': 150,
                    'glycerol_percent': 5.0,
                    'buffer_recommendation': 'PBS pH 7.0',
                    'storage_temperature': -20,
                    'source': 'default', 'confidence': 0.2}
        pi = compute_pI_real(sequence)
        optimal_ph = max(5.0, min(8.0, pi))
        charge = abs(compute_net_charge_real(sequence))
        salt_tolerance = 150 + 50 * charge
        hydro = compute_gravy_real(sequence)
        glycerol = max(0, min(20, 5 + 10 * max(0, hydro)))
        return {
            'optimal_ph': float(optimal_ph),
            'salt_tolerance_mM': float(min(1000, salt_tolerance)),
            'glycerol_percent': float(glycerol),
            'buffer_recommendation': f"PBS pH {optimal_ph:.1f} + {int(salt_tolerance)} mM NaCl + {glycerol:.1f}% glycerol",
            'storage_temperature': -80 if salt_tolerance > 300 else -20,
            'source': 'heuristic; pI from Henderson-Hasselbalch (real)',
            'confidence': 0.3
        }

    def analyze_protein(self, group_name: str, results_dir: str) -> Dict:
        print(f"\n  🧪 Chemical analysis for {get_display_name(group_name)}...")
        v = self._get_pim_vector(group_name)
        if v is None:
            print(f"  ⚠️ No PIM vector found")
            return {}
        sequence = self._get_sequence_from_sample(group_name)
        results = {}
        results['charge'] = self.compute_charge_profile(v)
        results['pka'] = self.compute_pka_profile(sequence)
        results['electrostatic'] = self.compute_electrostatic_map(v)
        results['gravy'] = self.compute_gravy(sequence) if sequence else 0.0
        results['hydrophobicity'] = self.compute_hydrophobicity_profile(v)
        results['patches'] = self.compute_hydrophobic_patches(v)
        results['glycosylation'] = self.compute_glycosylation_sites(sequence) if sequence else {'n_sites': [], 'o_sites': [], 'count_n': 0, 'count_o': 0}
        results['ptms'] = self.compute_ptm_sites(sequence) if sequence else {'phosphorylation': [], 'acetylation': [], 'ubiquitination': []}
        results['solubility'] = self.compute_solubility_aggregation(sequence)
        results['stability'] = self.compute_stability(sequence)
        results['hotspots'] = self.compute_hotspots(v)
        results['reactivity'] = self.compute_reactivity_profile(v)
        results['metal_binding'] = self.compute_metal_binding(v)
        results['membrane_permeability'] = self.compute_membrane_permeability(sequence)
        results['lipid_binding'] = self.compute_lipid_binding(v)
        results['buffer_stability'] = self.compute_buffer_stability(sequence)
        self._save_single_csv(results, group_name, results_dir)
        return results

    def _save_single_csv(self, results: Dict, group_name: str, results_dir: str):
        rows = []
        rows.append({'Property': 'Net Charge (pH 7.4)', 'Value': results['charge']['net_charge'], 'Source': results['charge'].get('source', 'PIM')})
        rows.append({'Property': 'Charge Density', 'Value': results['charge']['charge_density'], 'Source': 'PIM'})
        rows.append({'Property': 'Charge Balance', 'Value': results['charge']['charge_balance'], 'Source': 'PIM'})
        rows.append({'Property': 'pI (Henderson-Hasselbalch)', 'Value': results['pka']['pI'], 'Source': results['pka'].get('source', 'N/A')})
        rows.append({'Property': 'Net Charge at pH 7.4 (real)', 'Value': results['pka']['net_charge_at_pH_7'], 'Source': 'Henderson-Hasselbalch'})
        rows.append({'Property': 'Positive Charge Fraction', 'Value': results['electrostatic']['positive_charge_fraction'], 'Source': 'PIM'})
        rows.append({'Property': 'Negative Charge Fraction', 'Value': results['electrostatic']['negative_charge_fraction'], 'Source': 'PIM'})
        rows.append({'Property': 'Surface Charge Density', 'Value': results['electrostatic']['surface_charge_density'], 'Source': 'PIM'})
        rows.append({'Property': 'Electrostatic Potential', 'Value': results['electrostatic']['electrostatic_potential'], 'Source': 'PIM'})
        rows.append({'Property': 'GRAVY (Kyte-Doolittle)', 'Value': results['gravy'], 'Source': 'Kyte-Doolittle'})
        rows.append({'Property': 'Hydrophobicity Score', 'Value': results['hydrophobicity']['hydrophobicity_score'], 'Source': results['hydrophobicity'].get('source', 'N/A')})
        rows.append({'Property': 'Num Hydrophobic Components', 'Value': results['hydrophobicity']['num_hydrophobic_components'], 'Source': 'BIOLOGICAL_WEIGHTS'})
        for i, patch in enumerate(results['patches']['patch_scores'][:3]):
            rows.append({'Property': f'Hydrophobic Patch {i+1} Components', 'Value': str(patch['components']), 'Source': f'Size: {patch["size"]}'})
            rows.append({'Property': f'Hydrophobic Patch {i+1} Score', 'Value': patch['score'], 'Source': 'PIM'})
        rows.append({'Property': 'N-Glycosylation Sites', 'Value': results['glycosylation']['count_n'], 'Source': 'regex N[^P][ST]'})
        rows.append({'Property': 'O-Glycosylation Sites', 'Value': results['glycosylation']['count_o'], 'Source': 'regex [ST]'})
        rows.append({'Property': 'Phosphorylation Sites', 'Value': len(results['ptms']['phosphorylation']), 'Source': 'regex'})
        rows.append({'Property': 'Acetylation Sites', 'Value': len(results['ptms']['acetylation']), 'Source': 'regex K'})
        rows.append({'Property': 'Ubiquitination Sites', 'Value': len(results['ptms']['ubiquitination']), 'Source': 'regex K'})
        sol_prob = results['solubility']['solubility_prob']
        sol_str = f"{sol_prob:.4f}" if sol_prob is not None else "N/A (fuera de dominio)"
        rows.append({'Property': 'Solubility Probability', 'Value': sol_str, 'Source': results['solubility'].get('source', 'Wilkinson-Harrison 1991')})
        rows.append({'Property': 'Aggregation Score', 'Value': results['solubility']['aggregation_score'], 'Source': 'empirical'})
        rows.append({'Property': 'ΔG (kcal/mol)', 'Value': results['stability']['delta_g'], 'Source': results['stability'].get('source', 'N/A') + ' [NO INFORMATIVO]'})
        rows.append({'Property': 'Tm (°C)', 'Value': results['stability']['tm'], 'Source': results['stability'].get('source', 'N/A') + ' [NO INFORMATIVO]'})
        rows.append({'Property': 'Stability Score', 'Value': results['stability']['stability_score'], 'Source': results['stability'].get('source', 'N/A')})
        for h in results['hotspots']['hotspots'][:3]:
            rows.append({'Property': f'Hotspot: {h["name"]}', 'Value': h['score'], 'Source': h.get('source', 'PIM')})
        rows.append({'Property': 'Cysteine Accessibility', 'Value': results['reactivity']['cysteine_accessibility'], 'Source': results['reactivity'].get('source', 'PIM')})
        rows.append({'Property': 'Lysine Accessibility', 'Value': results['reactivity']['lysine_accessibility'], 'Source': 'PIM'})
        rows.append({'Property': 'Tyrosine Accessibility', 'Value': results['reactivity']['tyrosine_accessibility'], 'Source': 'PIM'})
        rows.append({'Property': 'Calcium Binding Score', 'Value': results['metal_binding']['calcium_binding_score'], 'Source': results['metal_binding'].get('source', 'PIM')})
        rows.append({'Property': 'Zinc Binding Score', 'Value': results['metal_binding']['zinc_binding_score'], 'Source': 'PIM'})
        rows.append({'Property': 'ΔG_octanol (kcal/mol)', 'Value': results['membrane_permeability']['delta_g_octanol_kcal_mol'], 'Source': results['membrane_permeability'].get('source', 'Wimley-White 1996')})
        rows.append({'Property': 'Phosphatidylserine Score', 'Value': results['lipid_binding']['phosphatidylserine_score'], 'Source': results['lipid_binding'].get('source', 'PIM')})
        rows.append({'Property': 'Cholesterol Score', 'Value': results['lipid_binding']['cholesterol_score'], 'Source': 'PIM'})
        rows.append({'Property': 'Ganglioside Score', 'Value': results['lipid_binding']['ganglioside_score'], 'Source': 'PIM'})
        rows.append({'Property': 'Best Lipid Binding', 'Value': results['lipid_binding']['best_lipid'], 'Source': 'PIM'})
        rows.append({'Property': 'Optimal pH', 'Value': results['buffer_stability']['optimal_ph'], 'Source': results['buffer_stability'].get('source', 'N/A')})
        rows.append({'Property': 'Salt Tolerance (mM)', 'Value': results['buffer_stability']['salt_tolerance_mM'], 'Source': 'heuristic'})
        rows.append({'Property': 'Glycerol (%)', 'Value': results['buffer_stability']['glycerol_percent'], 'Source': 'heuristic'})
        rows.append({'Property': 'Buffer Recommendation', 'Value': results['buffer_stability']['buffer_recommendation'], 'Source': 'heuristic'})
        rows.append({'Property': 'Storage Temperature (°C)', 'Value': results['buffer_stability']['storage_temperature'], 'Source': 'heuristic'})
        df = pd.DataFrame(rows)
        safe_save_csv(df, f"chemical_profile_{group_name}.csv", results_dir)
        print(f"  ✅ Chemical profile saved: chemical_profile_{group_name}.csv ({len(rows)} properties)")



# ============================================================================
# v217.0: COMPOSITE SCORE Y CLASIFICACIÓN
# ============================================================================

class CompositeScoreCalculator:
    """
    Calcula un score compuesto por grupo aplicando metric_weights del config.

    v217.0-fix1:
      - Si stats.insufficient_n=True, se EXCLUYEN todas las métricas intra-grupo
        (entropy, gini, jensen_shannon, hellinger, spearman, principal_angles,
        geodesic_distance) porque no son confiables con n < min_samples_per_group.
      - scalar_curvature sigue excluido siempre (constante estructural).
      - Si tras las exclusiones no queda ninguna métrica, composite_score = None.

    Métricas usadas (si están disponibles y el grupo tiene n suficiente):
      - geodesic_distance (menor es mejor → invertir)
      - principal_angles (menor es mejor → invertir)
      - shannon_entropy (mayor es mejor, normalizada)
      - jensen_shannon (mayor divergencia es mejor → directo)
      - hellinger (mayor es mejor → directo)
      - gini (mayor es mejor → directo)
      - spearman (mayor es mejor → directo)
      - pim (similaridad PIM → mayor es mejor)
      - scalar_curvature (CONSTANTE ESTRUCTURAL = 28 → excluir siempre)

    El score final es una suma ponderada normalizada a [0, 1].
    """
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.metric_mapping = {
            'pim': 'wedge_self_similarity',
            'entropy': 'entropy',
            'grassmann': 'geodesic_distance',
            'gini': 'gini',
            'jensen_shannon': 'jensen_shannon_mean',
            'spearman': 'spearman_mean',
            'hellinger': 'hellinger_mean',
            'principal_angles': 'principal_angles_mean',
        }
        self.invert_metrics = {
            'geodesic_distance', 'principal_angles_mean'
        }
        self.excluded_metrics = {'curvature', 'scalar_curvature', 'scalar_curvature_value'}

    def compute(self, stats: 'GroupStatistics') -> Dict:
        components = {}
        weighted_sum = 0.0
        total_weight = 0.0
        missing = []
        excluded = []

        # Fix 1: si insufficient_n, excluir todas las métricas intra-grupo
        insufficient = bool(getattr(stats, 'insufficient_n', False))

        for metric_key, weight in self.weights.items():
            # Excluir siempre scalar_curvature (constante estructural)
            if metric_key in self.excluded_metrics:
                excluded.append(metric_key)
                continue

            # Excluir métricas intra-grupo si n insuficiente
            if insufficient:
                excluded.append(metric_key)
                continue

            attr = self.metric_mapping.get(metric_key)
            if attr is None:
                missing.append(metric_key)
                continue

            value = getattr(stats, attr, None)
            if value is None or not np.isfinite(value):
                missing.append(metric_key)
                continue

            # Invertir si menor es mejor
            if attr in self.invert_metrics:
                if attr == 'geodesic_distance':
                    max_val = np.pi / 2 * np.sqrt(GRASSMANN_SUBSPACE_DIM)
                elif attr == 'principal_angles_mean':
                    max_val = np.pi / 2
                else:
                    max_val = 1.0
                normalized = max(0.0, min(1.0, 1.0 - value / max_val)) if max_val > 0 else 0.0
            else:
                normalized = max(0.0, min(1.0, value))

            components[metric_key] = {
                'raw': float(value),
                'normalized': float(normalized),
                'weight': float(weight),
                'contribution': float(normalized * weight),
            }
            weighted_sum += normalized * weight
            total_weight += weight

        # Si insufficient_n o no hay métricas válidas, composite = None
        if insufficient or total_weight <= 0:
            composite = None
        else:
            composite = weighted_sum / total_weight

        return {
            'composite_score': composite,
            'components': components,
            'total_weight_used': float(total_weight),
            'missing_metrics': missing,
            'excluded_metrics': excluded,
            'insufficient_n': insufficient,
            'weights_source': 'config.metric_weights',
        }


class GroupClassifier:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds or {}
        self.ordered = sorted(
            [(k, v) for k, v in self.thresholds.items()
             if not k.startswith('_') and isinstance(v, (int, float)) and not isinstance(v, bool)],
            key=lambda x: x[1], reverse=True
        )

    def classify(self, score: float) -> Dict:
        if score is None or not np.isfinite(score):
            return {'label': 'unknown', 'score': None, 'threshold': None}
        if not self.ordered:
            return {'label': 'unclassified', 'score': float(score), 'threshold': None}
        for label, threshold in self.ordered:
            if score >= threshold:
                return {'label': label, 'score': float(score), 'threshold': float(threshold)}
        return {'label': 'poor', 'score': float(score), 'threshold': 0.0}


def bootstrap_metric(values: List[float], n_bootstrap: int = 50,
                     ci: float = 0.95) -> Tuple[float, float, float]:
    """
    v217.0: valida None y tipos no numéricos.
    """
    if n_bootstrap is None or not isinstance(n_bootstrap, (int, float)):
        n_bootstrap = 50
    if ci is None or not isinstance(ci, (int, float)):
        ci = 0.95
    if not values or len(values) < 2:
        if values:
            m = float(np.mean(values))
            return m, m, m
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        m = float(np.mean(arr)) if len(arr) > 0 else 0.0
        return m, m, m
    n_bootstrap = max(1, int(n_bootstrap))
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(float(np.mean(arr[idx])))
    boot_means = np.array(boot_means)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(np.mean(boot_means)), lower, upper


class TherapeuticProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer', config_loader=None,
                 main_groups: List[str] = None, design_group: List[str] = None,
                 design_mode: bool = True):
        self.ga = analyzer
        self.config_loader = config_loader
        self.design_mode = design_mode
        self.main_groups = main_groups if main_groups else MAIN_GROUP_REFERENCE
        self.design_group = design_group if design_group else MAIN_GROUP_DESIGN
        print(f"  🎯 TherapeuticProfiler v217.0:")
        print(f"     ├─ Reference: {self.main_groups}")
        print(f"     ├─ Design: {self.design_group}")
        print(f"     ├─ Design mode: {'ACTIVATED' if design_mode else 'DEACTIVATED'}")
        print(f"     └─ Biopython: {'✅' if BIOPYTHON_AVAILABLE else '❌ (fallback heuristic)'}")

        self.target_pim = self._get_target_pim()
        self.chembl = ChEMBLMapper()

        self.data_source = None
        self.dramp = None

        dramp_file = DRAMP_TSV_FILE
        if not os.path.exists(expand_env_vars(dramp_file)):
            print(f"\n  ❌ DRAMP file not found: {dramp_file}")
            print(f"  ❌ ML will NOT be trained.")
        else:
            print(f"\n  📊 Loading DRAMP (real IC50/MIC data)...")
            self.dramp = DRAMPLoader(dramp_file, verbose=True)
            if self.dramp.loaded and len(self.dramp.peptides) >= 50:
                self.data_source = 'DRAMP'
                dist = self.dramp.get_activity_distribution()
                print(f"  📈 DRAMP activity distribution (μM):")
                print(f"     ├─ n = {dist['n']}")
                print(f"     ├─ range = [{dist['min_uM']:.4f}, {dist['max_uM']:.2f}]")
                print(f"     ├─ median = {dist['median_uM']:.4f}")
                print(f"     ├─ mean = {dist['mean_uM']:.4f}")
                print(f"     └─ log10 range = [{dist['log10_min']:.2f}, {dist['log10_max']:.2f}]")
            else:
                print(f"  ❌ DRAMP loaded but insufficient data "
                      f"({len(self.dramp.peptides) if self.dramp else 0} peptides).")

        self.peptide_sequence = None
        self.activity_model = None
        self.activity_model_type = None
        self.scaler = None
        self.model_trained = False
        self.cv_results = None
        self.activity_normalization = 'log10'

        if self.data_source == 'DRAMP':
            self._train_activity_model()
        else:
            print(f"\n  ⚠️ Model NOT trained. Predictions will return neutral values.")

    def _get_target_pim(self) -> np.ndarray:
        for target in self.design_group:
            if target in self.ga.group_stats:
                return self.ga.group_stats[target].centroid
        if self.ga.group_stats:
            return self.ga.group_stats[list(self.ga.group_stats.keys())[0]].centroid
        raise ValueError("No PIM found")

    def _extract_physicochemical_features(self, sequence: str) -> np.ndarray:
        features = []
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return np.zeros(8, dtype=np.float32)
        net_charge = compute_net_charge_real(seq, ph=7.4)
        features.append(net_charge)
        gravy = compute_gravy_real(seq)
        features.append(gravy)
        mw = compute_mw_real(seq) / 1000.0
        features.append(mw)
        pi = compute_pI_real(seq)
        features.append(pi)
        hydrophobic_count = sum(1 for aa in seq if AA_HYDROPHOBICITY.get(aa, 0) > 1.0)
        amphiphilicity = (net_charge * hydrophobic_count) / (n + 1e-10)
        features.append(amphiphilicity)
        helix = sum(AA_HELIX_PROPENSITY.get(aa, 1.0) for aa in seq) / (n + 1e-10)
        features.append(helix)
        aromatic = sum(1 for aa in seq if aa in ['F', 'W', 'Y']) / (n + 1e-10)
        features.append(aromatic)
        instability = compute_instability_index_real(seq)
        features.append(instability)
        return np.array(features, dtype=np.float32)

    def _extract_full_features(self, sequence: str) -> np.ndarray:
        physico = self._extract_physicochemical_features(sequence)
        aac = np.zeros(20)
        seq = sequence.upper()
        for aa in seq:
            if aa in AA_LIST:
                aac[AA_LIST.index(aa)] += 1
        aac = aac / (len(seq) + 1e-10)
        pim = compute_pim_profile(sequence, use_weights=True)
        return np.concatenate([physico, aac, pim])

    def _train_activity_model(self):
        print(f"\n  🤖 Training activity prediction model (v217.0)...")
        print(f"     ├─ Data source: DRAMP (REAL IC50/MIC)")
        print(f"     ├─ Features: 8 physicochemical (pI+instability REAL) + 20 AAC + 16 PIM")
        X, y_raw = [], []
        for peptide in self.dramp.peptides:
            features = self._extract_full_features(peptide['sequence'])
            X.append(features)
            y_raw.append(peptide['activity'])
        X = np.array(X)
        y_raw = np.array(y_raw)
        print(f"     ├─ Training samples: {len(X)}")
        print(f"     ├─ Feature dimension: {X.shape[1]}")
        y = np.log10(y_raw + 1e-6)
        self.activity_normalization = 'log10'
        print(f"     ├─ Label transform: log10(IC50_μM)")
        print(f"     ├─ Label range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"     ├─ Label mean: {y.mean():.2f}, std: {y.std():.2f}")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_binary = (y > np.median(y)).astype(int)
        print(f"\n     ├─ Random Forest ({n_splits}-fold CV):")
        rf_cv_results = self._cross_validate_model(
            RandomForestRegressor(n_estimators=200, max_depth=15,
                                 min_samples_split=5, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1),
            X_scaled, y, y_binary, skf, "Random Forest"
        )
        xgb_cv_results = None
        if XGBOOST_AVAILABLE:
            print(f"\n     ├─ XGBoost ({n_splits}-fold CV):")
            try:
                xgb_cv_results = self._cross_validate_model(
                    xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42,
                                    n_jobs=-1, verbosity=0),
                    X_scaled, y, y_binary, skf, "XGBoost"
                )
            except Exception as e:
                print(f"     ⚠️ XGBoost failed: {e}")
                xgb_cv_results = None
        if xgb_cv_results and xgb_cv_results['r2_mean'] > rf_cv_results['r2_mean']:
            print(f"\n     ├─ ✅ Selected: XGBoost (R²={xgb_cv_results['r2_mean']:.4f})")
            self.activity_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            )
            self.activity_model_type = 'XGBoost'
            self.cv_results = xgb_cv_results
        else:
            print(f"\n     ├─ ✅ Selected: Random Forest (R²={rf_cv_results['r2_mean']:.4f})")
            self.activity_model = RandomForestRegressor(
                n_estimators=200, max_depth=15,
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )
            self.activity_model_type = 'Random Forest'
            self.cv_results = rf_cv_results
        self.activity_model.fit(X_scaled, y)
        y_pred_train = self.activity_model.predict(X_scaled)
        r2_train = r2_score(y, y_pred_train)
        mse_train = mean_squared_error(y, y_pred_train)
        print(f"\n     ├─ Final model: {self.activity_model_type}")
        print(f"     ├─ Train R²: {r2_train:.4f}")
        print(f"     ├─ Train MSE: {mse_train:.4f}")
        print(f"     ├─ CV R²: {self.cv_results['r2_mean']:.4f} ± {self.cv_results['r2_std']:.4f}")
        print(f"     ├─ CV AUC: {self.cv_results['auc_mean']:.4f} ± {self.cv_results['auc_std']:.4f}")
        print(f"     └─ CV MCC: {self.cv_results['mcc_mean']:.4f} ± {self.cv_results['mcc_std']:.4f}")
        self.model_trained = True

    def _cross_validate_model(self, model, X_scaled, y, y_binary, skf, name):
        r2_scores, mse_scores, mae_scores = [], [], []
        auc_scores, mcc_scores = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_binary)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_test_binary = y_binary[test_idx]
            from sklearn.base import clone
            model_fold = clone(model)
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            y_pred_binary = (y_pred > np.median(y)).astype(int)
            if len(np.unique(y_test_binary)) > 1:
                try:
                    auc_scores.append(roc_auc_score(y_test_binary, y_pred))
                except Exception:
                    auc_scores.append(0.5)
            else:
                auc_scores.append(0.5)
            try:
                mcc_scores.append(matthews_corrcoef(y_test_binary, y_pred_binary))
            except Exception:
                mcc_scores.append(0.0)
            print(f"     │   Fold {fold_idx+1}: R²={r2_scores[-1]:.4f}, "
                  f"AUC={auc_scores[-1]:.4f}, MCC={mcc_scores[-1]:.4f}")
        return {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'mse_mean': float(np.mean(mse_scores)),
            'mse_std': float(np.std(mse_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'mae_std': float(np.std(mae_scores)),
            'auc_mean': float(np.mean(auc_scores)),
            'auc_std': float(np.std(auc_scores)),
            'mcc_mean': float(np.mean(mcc_scores)),
            'mcc_std': float(np.std(mcc_scores)),
            'per_fold': {
                'r2': r2_scores, 'mse': mse_scores, 'mae': mae_scores,
                'auc': auc_scores, 'mcc': mcc_scores
            },
            'model_name': name,
            'n_splits': len(r2_scores),
            'data_source': 'DRAMP',
        }

    def predict_activity(self, peptide_sequence: str) -> Dict:
        if not self.model_trained:
            return {
                'score': None, 'ic50_um': None, 'confidence': 0.0,
                'message': 'Model not trained (no DRAMP data available)',
                'data_source': None, 'normalization': None,
            }
        features = self._extract_full_features(peptide_sequence)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.activity_model.predict(features_scaled)[0]
        try:
            if self.activity_model_type == 'Random Forest':
                predictions = np.array([tree.predict(features_scaled)[0]
                                       for tree in self.activity_model.estimators_])
            elif self.activity_model_type == 'XGBoost':
                predictions = np.array([
                    self.activity_model.predict(features_scaled,
                                                iteration_range=(0, i + 1))[0]
                    for i in range(min(50, self.activity_model.n_estimators))
                ])
            else:
                predictions = np.array([prediction])
            std_pred = float(np.std(predictions))
            confidence = float(max(0.0, min(1.0, 1.0 - std_pred * 2)))
        except Exception:
            confidence = 0.5
        ic50_um = float(10 ** prediction)
        return {
            'score': float(prediction),
            'ic50_um': ic50_um,
            'confidence': confidence,
            'model': self.activity_model_type,
            'data_source': 'DRAMP',
            'normalization': 'log10',
            'message': f'Predicted IC50 = {ic50_um:.4f} μM '
                       f'(log10={prediction:.3f}, confidence={confidence:.2f})',
        }

    def _identify_membrane_target(self) -> Optional[Dict]:
        print("\n  🎯 Identifying target group...")
        available = [g for g in self.design_group if g in self.ga.group_stats]
        if not available:
            print(f"     ⚠️ No design groups available")
            return None
        best_target, best_score = None, -1
        for group in available:
            stats = self.ga.group_stats[group]
            sim, _ = self.ga.grassmann.wedge_product(self.target_pim, stats.centroid)
            chembl_id = None
            protein_name = get_display_name(group)
            if self.chembl.loaded:
                results = self.chembl.search_by_name(group)
                if results:
                    chembl_id = results[0]['CHEMBL_PROTEIN_ID']
                    protein_name = results[0]['PROTEIN_NAME']
            score = sim * (0.8 + 0.2 * (1 if chembl_id else 0))
            if score > best_score:
                best_score = score
                best_target = {'group': group, 'similarity': sim,
                              'protein_name': protein_name,
                              'chembl_id': chembl_id, 'score': score}
        if best_target:
            print(f"     ├─ Target: {best_target['protein_name']}")
            print(f"     └─ Similarity: {best_target['similarity']:.6f}")
        return best_target

    def _design_peptide_enhanced(self, target: Dict) -> List[Dict]:
        """
        v217.0: devuelve una lista de dicts con 'label', 'sequence' y 'role'
        para evaluar base + optimizado en la misma corrida.
        """
        print("\n  🧬 Designing peptide(s)...")
        config_target_name = get_config_target(target['group'])
        base_seq = None
        optimized_seq = None

        if self.config_loader and self.config_loader.loaded_from == 'config_EBOLA.json':
            config_data = self.config_loader.config
            if 'base_peptide_sequence' in config_data:
                entry = None
                if config_target_name in config_data['base_peptide_sequence']:
                    entry = config_data['base_peptide_sequence'][config_target_name]
                elif 'ebola' in config_data['base_peptide_sequence']:
                    entry = config_data['base_peptide_sequence']['ebola']
                elif 'rvfv' in config_data['base_peptide_sequence']:
                    entry = config_data['base_peptide_sequence']['rvfv']

                if entry:
                    base_seq = entry.get('sequence', '')
                    optimized_seq = entry.get('optimized_sequence', None)

        if not base_seq or len(base_seq) < 5:
            raise ValueError(
                f"No base peptide sequence found in config for target '{config_target_name}'. "
                f"Please add 'base_peptide_sequence.{config_target_name}.sequence' to config_EBOLA.json."
            )

        max_len = self.config_loader.get_max_peptide_length(config_target_name) if self.config_loader else 25
        if len(base_seq) > max_len:
            base_seq = base_seq[:max_len]

        peptides = [{'label': 'base', 'sequence': base_seq, 'role': 'fusion loop lead'}]
        print(f"     ├─ Base peptide: {base_seq} ({len(base_seq)} aa)")

        if optimized_seq and len(optimized_seq) >= 5:
            if len(optimized_seq) > max_len:
                optimized_seq = optimized_seq[:max_len]
            peptides.append({
                'label': 'optimized',
                'sequence': optimized_seq,
                'role': 'optimized variant'
            })
            print(f"     ├─ Optimized peptide: {optimized_seq} ({len(optimized_seq)} aa)")
        else:
            print(f"     ├─ No optimized_sequence found; only base will be evaluated")

        return peptides

    def _calculate_physicochemical_properties(self, sequence: str) -> Dict:
        print("\n  ⚡ Calculating physicochemical properties (real)...")
        seq = sequence.upper()
        n = len(seq)
        net_charge = compute_net_charge_real(seq, ph=7.4)
        mw = compute_mw_real(seq)
        gravy = compute_gravy_real(seq)
        pi = compute_pI_real(seq)
        sol = compute_solubility_real(seq)
        instability = compute_instability_index_real(seq)
        properties = {
            'charge': net_charge,
            'molecular_weight': mw,
            'gravy': gravy,
            'isoelectric_point': pi,
            'solubility_prob': sol.get('probability'),
            'solubility_applicable': sol.get('applicable', False),
            'solubility_note': sol.get('note', ''),
            'instability_index': instability,
            'length': n,
            'source': 'Henderson-Hasselbalch + Kyte-Doolittle + Guruprasad 1990 + Wilkinson-Harrison 1991'
        }
        print(f"     ├─ Net charge (pH 7.4, real): {properties['charge']:.2f}")
        print(f"     ├─ MW (real): {properties['molecular_weight']:.1f} Da")
        print(f"     ├─ GRAVY (real): {properties['gravy']:.2f}")
        print(f"     ├─ pI (real): {properties['isoelectric_point']:.2f}")
        print(f"     ├─ Instability (real): {properties['instability_index']:.2f}")
        sol_str = f"{properties['solubility_prob']:.3f}" if properties['solubility_prob'] is not None else "N/A"
        print(f"     └─ Solubility prob (Wilkinson-Harrison): {sol_str}")
        return properties

    def _compare_with_known_inhibitors(self, target: Dict, predicted_ic50_um: Optional[float]) -> Dict:
        print("\n  🔬 Comparing with known inhibitors (from config)...")
        config_target = get_config_target(target['group'])
        known_inhibitors = {}
        if self.config_loader:
            config_inh = self.config_loader.get_known_inhibitors(config_target)
            for inh in config_inh:
                if isinstance(inh, dict):
                    name = inh.get('name', 'unknown')
                    ic50 = inh.get('ic50', None)
                    ic50_units = inh.get('ic50_units', 'µM')
                    ki = inh.get('kd', inh.get('ki', None))
                    known_inhibitors[name] = {
                        'ic50': ic50, 'ic50_units': ic50_units, 'ki': ki,
                        'type': inh.get('type', 'unknown'),
                        'source': inh.get('source', 'config_EBOLA.json'),
                        'clinical_status': inh.get('clinical_status', 'unknown'),
                        'mechanism': inh.get('mechanism', ''),
                        'target': inh.get('target', ''),
                    }
        if not known_inhibitors:
            print(f"     ⚠️ No known inhibitors in config for target '{config_target}'")
            return {'peptide_affinity_nM': None, 'known_inhibitors': {},
                    'comparison': [], 'best_match': None,
                    'message': 'No known inhibitors in config'}
        if predicted_ic50_um is not None:
            peptide_affinity_nM = predicted_ic50_um * 1000.0
            print(f"     ├─ Peptide affinity (from ML): {peptide_affinity_nM:.3f} nM")
        else:
            peptide_affinity_nM = None
            print(f"     ├─ Peptide affinity: NOT AVAILABLE (model not trained)")
        comparison = {
            'peptide_affinity_nM': peptide_affinity_nM,
            'known_inhibitors': known_inhibitors,
            'comparison': [], 'best_match': None
        }
        for name, data in known_inhibitors.items():
            ic50 = data.get('ic50')
            if ic50 is None or peptide_affinity_nM is None:
                comparison['comparison'].append({
                    'name': name, 'ic50_uM': ic50,
                    'type': data.get('type', 'unknown'),
                    'ratio_to_peptide': None, 'better_than_peptide': None,
                    'source': data.get('source', 'config_EBOLA.json'),
                    'clinical_status': data.get('clinical_status', 'unknown'),
                })
                continue
            ic50_uM = ic50 if data.get('ic50_units') == 'µM' else ic50 * 1000.0 if data.get('ic50_units') == 'nM' else ic50
            ratio = ic50_uM / predicted_ic50_um if predicted_ic50_um and predicted_ic50_um > 0 else None
            comparison['comparison'].append({
                'name': name, 'ic50_uM': ic50_uM,
                'type': data.get('type', 'unknown'),
                'ratio_to_peptide': ratio,
                'better_than_peptide': ratio < 1 if ratio else None,
                'source': data.get('source', 'config_EBOLA.json'),
                'clinical_status': data.get('clinical_status', 'unknown'),
            })
        comparison['comparison'].sort(key=lambda x: (x['ic50_uM'] is None, x['ic50_uM'] or float('inf')))
        comparison['best_match'] = comparison['comparison'][0] if comparison['comparison'] else None
        if comparison['best_match']:
            bm = comparison['best_match']
            print(f"     └─ Best known: {bm['name']} (IC50={bm['ic50_uM']} µM, {bm['type']})")
        return comparison

    def _validate_peptide_against_ranges(self, peptide: str, target_group: str) -> Dict:
        config_target = get_config_target(target_group)
        available = self.config_loader.get_all_targets() if self.config_loader else []
        if config_target not in available:
            return {'status': 'default_ranges', 'message': 'No specific ranges'}
        ranges = self.config_loader.get_target_ranges(config_target)
        peptide_pim = compute_pim_profile(peptide)
        target_pim = self.target_pim
        validation = {}
        for metric, rng in ranges.items():
            if not isinstance(rng, dict):
                continue
            min_val = rng.get('min', None)
            max_val = rng.get('max', None)
            if min_val is None or max_val is None:
                continue
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                continue
            if metric == 'geodesic_distance':
                value = self.ga.grassmann.geodesic_distance(peptide_pim, target_pim)
            elif metric == 'entropy':
                value = shannon_entropy(peptide_pim, normalize=True)
            elif metric == 'jensen_shannon':
                value = jensen_shannon_divergence(peptide_pim, target_pim)
            elif metric == 'hellinger':
                value = hellinger_distance(peptide_pim, target_pim)
            elif metric == 'gini':
                value = gini_coefficient(peptide_pim)
            elif metric == 'spearman':
                value = spearman_correlation(peptide_pim, target_pim)
            else:
                continue
            passed = min_val <= value <= max_val
            validation[metric] = {'value': value, 'min': float(min_val), 'max': float(max_val),
                                  'passed': passed, 'source': rng.get('source', 'unknown')}
        passed_count = sum(1 for v in validation.values()
                          if isinstance(v, dict) and v.get('passed', False))
        total = sum(1 for v in validation.values() if isinstance(v, dict))
        validation['summary'] = {
            'passed': passed_count, 'total': total,
            'score': passed_count / total if total > 0 else 0,
            'target': config_target, 'source': 'config_EBOLA.json'
        }
        return validation

    def _evaluate_with_all_metrics(self, peptide: str, target: Dict) -> Dict:
        peptide_pim = compute_pim_profile(peptide)
        target_pim = self.ga.group_stats[target['group']].centroid
        metrics = compare_pim_vectors(peptide_pim, target_pim)
        metrics['activity_data_source'] = self.data_source
        metrics['activity_normalization'] = self.activity_normalization
        return metrics

    def _generate_recommendations_enhanced(self, peptide: str, properties: Dict,
                                           activity: Dict, metrics_eval: Dict) -> List[str]:
        print("\n  🧪 Generating recommendations...")
        recommendations = []
        recommendations.append(f"SYNTHESIZE: Sequence {peptide} by SPPS")
        sol_prob = properties.get('solubility_prob')
        if sol_prob is not None and sol_prob > 0.5:
            recommendations.append("FORMULATE: PBS pH 7.4 buffer (predicted soluble)")
        elif sol_prob is not None:
            recommendations.append("FORMULATE: 10% DMSO + PBS pH 7.4 (may aggregate)")
        else:
            recommendations.append("FORMULATE: Solubility model not applicable (peptide); "
                                   "consider empirical solubility assay")
        if 'N' in peptide or 'Q' in peptide:
            recommendations.append("PROTECT: Add protecting groups at N and Q")
        if properties['gravy'] > 1.0:
            recommendations.append("STABILIZE: End-to-end cyclization")
        elif properties['charge'] > 1.0:
            recommendations.append("STABILIZE: PEGylation to extend half-life")
        if metrics_eval['geodesic_distance'] > 1.0:
            recommendations.append("STRUCTURE: Consider conformational constraints")
        if metrics_eval['jensen_shannon'] > 0.5:
            recommendations.append("DIVERSIFY: High JS divergence")
        if metrics_eval['hellinger'] > 0.5:
            recommendations.append("DIVERSIFY: High Hellinger distance")
        recommendations.append("VALIDATE: Binding assays (SPR/ITC)")
        if activity.get('ic50_um') is not None:
            if activity['ic50_um'] > 10:
                recommendations.append(f"OPTIMIZE: Predicted IC50 = {activity['ic50_um']:.2f} μM > 10 μM")
            elif activity['ic50_um'] < 1:
                recommendations.append(f"✅ POTENT: Predicted IC50 = {activity['ic50_um']:.4f} μM")
            else:
                recommendations.append(f"MODERATE: Predicted IC50 = {activity['ic50_um']:.4f} μM")
        else:
            recommendations.append("⚠️ ACTIVITY: Model not trained, cannot predict IC50")
        print(f"     ├─ {len(recommendations)} recommendations generated")
        return recommendations

    def _compare_base_vs_optimized(self, peptide_results: List[Dict]) -> Dict:
        """
        v217.0: construye una tabla comparativa base vs optimizado
        con deltas absolutos y relativos para las métricas clave.
        """
        if len(peptide_results) < 2:
            return {'available': False, 'message': 'Only one peptide evaluated'}

        base = next((p for p in peptide_results if p['label'] == 'base'), None)
        optimized = next((p for p in peptide_results if p['label'] == 'optimized'), None)
        if base is None or optimized is None:
            return {'available': False, 'message': 'base or optimized not found'}

        def _delta(b, o):
            if b is None or o is None:
                return None, None
            abs_d = o - b
            rel_d = (abs_d / b * 100.0) if b != 0 else None
            return abs_d, rel_d

        metrics_to_compare = [
            ('activity_ic50_um',        base['activity'].get('ic50_um'),
                                        optimized['activity'].get('ic50_um')),
            ('activity_score',          base['activity'].get('score'),
                                        optimized['activity'].get('score')),
            ('geodesic_distance',       base['all_metrics_evaluation'].get('geodesic_distance'),
                                        optimized['all_metrics_evaluation'].get('geodesic_distance')),
            ('jensen_shannon',          base['all_metrics_evaluation'].get('jensen_shannon'),
                                        optimized['all_metrics_evaluation'].get('jensen_shannon')),
            ('hellinger',               base['all_metrics_evaluation'].get('hellinger'),
                                        optimized['all_metrics_evaluation'].get('hellinger')),
            ('shannon_entropy_1',       base['all_metrics_evaluation'].get('shannon_entropy_1'),
                                        optimized['all_metrics_evaluation'].get('shannon_entropy_1')),
            ('gini_1',                  base['all_metrics_evaluation'].get('gini_1'),
                                        optimized['all_metrics_evaluation'].get('gini_1')),
            ('spearman',                base['all_metrics_evaluation'].get('spearman'),
                                        optimized['all_metrics_evaluation'].get('spearman')),
            ('principal_angles_mean',   base['all_metrics_evaluation'].get('principal_angles_mean'),
                                        optimized['all_metrics_evaluation'].get('principal_angles_mean')),
            ('range_passed',            base['range_validation'].get('summary', {}).get('passed'),
                                        optimized['range_validation'].get('summary', {}).get('passed')),
        ]

        rows = []
        for name, b, o in metrics_to_compare:
            abs_d, rel_d = _delta(b, o)
            rows.append({
                'metric': name,
                'base': b,
                'optimized': o,
                'delta_abs': abs_d,
                'delta_rel_pct': rel_d,
            })

        return {
            'available': True,
            'base_sequence': base['sequence'],
            'optimized_sequence': optimized['sequence'],
            'base_length': base['length'],
            'optimized_length': optimized['length'],
            'rows': rows,
        }

    def generate_therapeutic_profile(self) -> Dict:
        print("\n" + "=" * 80)
        print("🧬 GENERATING THERAPEUTIC PROFILE (v217.0)")
        print("=" * 80)
        target = self._identify_membrane_target()
        if target is None:
            return {'error': 'No therapeutic target identified'}
        if not self.design_mode:
            return {'target': target, 'mode': 'characterization_only',
                    'message': 'Design mode disabled',
                    'target_metrics': self.target_pim.tolist()}

        try:
            peptides = self._design_peptide_enhanced(target)
        except ValueError as e:
            print(f"\n  ❌ {e}")
            return {'error': str(e)}

        # v217.0: evaluar cada péptido (base y optimizado)
        peptide_results = []
        for pep in peptides:
            seq = pep['sequence']
            print(f"\n  ── Evaluating {pep['label']}: {seq} ──")
            properties = self._calculate_physicochemical_properties(seq)
            activity = self.predict_activity(seq)
            metrics_eval = self._evaluate_with_all_metrics(seq, target)
            range_validation = self._validate_peptide_against_ranges(seq, target['group'])
            recommendations = self._generate_recommendations_enhanced(
                seq, properties, activity, metrics_eval)
            if 'summary' in range_validation and range_validation['summary']['score'] < 0.5:
                recommendations.append("⚠️ RANGES: Peptide fails multiple metric thresholds")

            peptide_results.append({
                'label': pep['label'],
                'role': pep['role'],
                'sequence': seq,
                'length': len(seq),
                'properties': properties,
                'activity': activity,
                'all_metrics_evaluation': metrics_eval,
                'range_validation': range_validation,
                'recommendations': recommendations,
            })

        # Comparación con inhibidores conocidos (usando el mejor péptido: menor IC50)
        valid_activities = [p for p in peptide_results if p['activity'].get('ic50_um') is not None]
        if valid_activities:
            best = min(valid_activities, key=lambda p: p['activity']['ic50_um'])
            comparison = self._compare_with_known_inhibitors(target, best['activity']['ic50_um'])
        else:
            comparison = self._compare_with_known_inhibitors(target, None)

        # v217.0: comparación base vs optimizado
        base_vs_optimized = self._compare_base_vs_optimized(peptide_results)

        # Mantener compatibilidad: 'peptide' apunta al base
        base_result = next((p for p in peptide_results if p['label'] == 'base'), peptide_results[0])

        return {
            'target': target,
            'peptide': {
                'sequence': base_result['sequence'],
                'properties': base_result['properties'],
                'activity': base_result['activity'],
                'all_metrics_evaluation': base_result['all_metrics_evaluation'],
                'range_validation': base_result['range_validation'],
            },
            'peptides': peptide_results,               # lista completa
            'base_vs_optimized': base_vs_optimized,    # tabla comparativa
            'comparison': comparison,
            'recommendations': base_result['recommendations'],
            'cv_results': self.cv_results,
            'model_type': self.activity_model_type,
            'data_source': self.data_source,
            'version': '217.0',
        }

    def print_profile(self, profile: Dict):
        print("\n" + "=" * 80)
        print("📋 COMPLETE THERAPEUTIC PROFILE (v217.0)")
        print("=" * 80)
        if 'error' in profile:
            print(f"\n  ❌ Error: {profile['error']}")
            return
        if profile.get('mode') == 'characterization_only':
            print(f"\n  ℹ️ {profile.get('message', '')}")
            return
        print(f"\n  🎯 TARGET:")
        print(f"     ├─ Protein: {profile['target']['protein_name']}")
        print(f"     ├─ Group: {profile['target']['group']}")
        print(f"     └─ Similarity: {profile['target']['similarity']:.6f}")
        if profile['target'].get('chembl_id'):
            print(f"     └─ ChEMBL ID: {profile['target']['chembl_id']}")
        print(f"\n  🧬 PEPTIDE:")
        print(f"     ├─ Sequence: {profile['peptide']['sequence']}")
        print(f"     ├─ Length: {profile['peptide']['properties']['length']} aa")
        print(f"     ├─ Net charge (pH 7.4): {profile['peptide']['properties']['charge']:.2f}")
        print(f"     ├─ MW: {profile['peptide']['properties']['molecular_weight']:.1f} Da")
        print(f"     ├─ GRAVY: {profile['peptide']['properties']['gravy']:.2f}")
        print(f"     ├─ pI: {profile['peptide']['properties']['isoelectric_point']:.2f}")
        print(f"     ├─ Instability Index: {profile['peptide']['properties']['instability_index']:.2f}")
        sol_prob = profile['peptide']['properties'].get('solubility_prob')
        sol_str = f"{sol_prob:.3f}" if sol_prob is not None else "N/A"
        print(f"     └─ Solubility prob: {sol_str}")
        act = profile['peptide']['activity']
        print(f"\n  💊 ACTIVITY PREDICTION:")
        print(f"     ├─ Data source: {act.get('data_source', 'unknown')}")
        if act.get('ic50_um') is not None:
            print(f"     ├─ Predicted IC50: {act['ic50_um']:.4f} μM")
            print(f"     ├─ log10(IC50): {act['score']:.4f}")
        else:
            print(f"     ├─ Predicted IC50: NOT AVAILABLE (model not trained)")
        print(f"     ├─ Confidence: {act.get('confidence', 0.0):.4f}")
        print(f"     └─ Message: {act.get('message', '')}")
        if 'cv_results' in profile and profile['cv_results']:
            cv = profile['cv_results']
            print(f"\n  📊 MODEL PERFORMANCE (5-fold CV):")
            print(f"     ├─ Model: {profile.get('model_type', 'Unknown')}")
            print(f"     ├─ Data source: {profile.get('data_source', 'Unknown')}")
            print(f"     ├─ CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            print(f"     ├─ CV MSE: {cv['mse_mean']:.4f} ± {cv['mse_std']:.4f}")
            print(f"     ├─ CV AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
            print(f"     └─ CV MCC: {cv['mcc_mean']:.4f} ± {cv['mcc_std']:.4f}")
        metrics = profile['peptide']['all_metrics_evaluation']
        print(f"\n  📊 ALL METRICS EVALUATION (raw, no composite):")
        print(f"     ├─ Geodesic Distance: {metrics['geodesic_distance']:.4f}")
        print(f"     ├─ Max Principal Angle: {metrics['principal_angles_max']:.4f}")
        print(f"     ├─ Shannon Entropy: {metrics['shannon_entropy_1']:.4f}")
        print(f"     ├─ Jensen-Shannon: {metrics['jensen_shannon']:.4f}")
        print(f"     ├─ Hellinger: {metrics['hellinger']:.4f}")
        print(f"     ├─ Gini: {metrics['gini_1']:.4f}")
        print(f"     └─ Spearman: {metrics['spearman']:.4f}")
        if 'range_validation' in profile['peptide'] and 'summary' in profile['peptide']['range_validation']:
            rv = profile['peptide']['range_validation']
            print(f"\n  📏 RANGE VALIDATION:")
            print(f"     ├─ Target: {rv['summary'].get('target', 'unknown')}")
            print(f"     ├─ Passed: {rv['summary']['passed']}/{rv['summary']['total']}")
            print(f"     └─ Score: {rv['summary']['score']:.2f}")
        print(f"\n  🔬 COMPARISON WITH KNOWN INHIBITORS:")
        comp = profile.get('comparison', {})
        if comp.get('peptide_affinity_nM') is not None:
            print(f"     ├─ Peptide affinity (ML): {comp['peptide_affinity_nM']:.3f} nM")
        else:
            print(f"     ├─ Peptide affinity: N/A (no ML prediction)")
        if comp.get('best_match'):
            bm = comp['best_match']
            print(f"     └─ Best known: {bm['name']} (IC50={bm['ic50_uM']} µM, {bm['type']})")

        # v217.0: imprimir comparación base vs optimizado
        bvo = profile.get('base_vs_optimized', {})
        if bvo.get('available'):
            print(f"\n  🔬 BASE vs OPTIMIZED:")
            print(f"     ├─ Base:      {bvo['base_sequence']} ({bvo['base_length']} aa)")
            print(f"     ├─ Optimized: {bvo['optimized_sequence']} ({bvo['optimized_length']} aa)")
            print(f"     └─ {'Metric':<28} {'Base':>12} {'Optimized':>12} {'Δabs':>12} {'Δrel %':>10}")
            for row in bvo['rows']:
                b = row['base']
                o = row['optimized']
                da = row['delta_abs']
                dr = row['delta_rel_pct']
                b_str = f"{b:.4f}" if isinstance(b, (int, float)) else "N/A"
                o_str = f"{o:.4f}" if isinstance(o, (int, float)) else "N/A"
                da_str = f"{da:+.4f}" if isinstance(da, (int, float)) else "N/A"
                dr_str = f"{dr:+.2f}" if isinstance(dr, (int, float)) else "N/A"
                print(f"        {row['metric']:<28} {b_str:>12} {o_str:>12} {da_str:>12} {dr_str:>10}")

        print(f"\n  🧪 RECOMMENDATIONS:")
        for i, rec in enumerate(profile['recommendations'], 1):
            print(f"     {i}. {rec}")



class AdvancedGroupAnalyzer:
    def __init__(self, grassmann: GrassmannPIM, main_groups: List[str] = None,
                 design_group: List[str] = None,
                 min_samples_per_group: int = 2):
        self.grassmann = grassmann
        self.group_stats: Dict[str, GroupStatistics] = {}
        self.sample_data: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}
        self.hash_index = None
        self.tracker = ProcessingTracker()
        self.start_time = None
        self.max_samples_per_group = MAX_STORED_PROTEINS_PER_GROUP
        self.min_samples_per_group = min_samples_per_group
        self.therapeutic_profile = None
        self.characterization_report = None
        self.main_groups = main_groups if main_groups else MAIN_GROUP_REFERENCE
        self.design_group = design_group if design_group else MAIN_GROUP_DESIGN
        self.composite_calculator: Optional[CompositeScoreCalculator] = None
        self.group_classifier: Optional[GroupClassifier] = None
        self.target_ranges_validation: Dict[str, Dict] = {}
        self.primary_uniprot_reconciliation: Dict[str, Any] = {}
        print(f"  ✅ AdvancedGroupAnalyzer v217.0 initialized")
        print(f"     ├─ Reference groups: {self.main_groups}")
        print(f"     ├─ Design group: {self.design_group}")
        print(f"     └─ Min samples per group: {self.min_samples_per_group}")

    def set_sample_size(self, size: int):
        self.max_samples_per_group = size

    def set_composite_calculator(self, calculator: CompositeScoreCalculator):
        self.composite_calculator = calculator

    def set_group_classifier(self, classifier: GroupClassifier):
        self.group_classifier = classifier

    def load_fasta_file(self, filepath: str, group_name: str, verbose: bool = True):
        filepath = expand_env_vars(filepath)
        if not os.path.exists(filepath):
            if verbose:
                print(f"  ⚠️ File not found: {filepath}")
            return
        if verbose:
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"  📂 Processing: {os.path.basename(filepath)} ({size_gb:.2f} GB)")
        if group_name not in self.sample_data:
            self.sample_data[group_name] = []
        stats = OnlineStatistics(DIM_PAIRS)
        sampler = ProgressiveSampler(self.max_samples_per_group)
        total_seen = 0
        try:
            for header, sequence in read_fasta_stream(filepath, verbose=False):
                total_seen += 1
                pim = compute_pim_profile(sequence, use_weights=True)
                if np.sum(np.abs(pim)) < 1e-6:
                    self.tracker.update(group_name, False, len(sequence))
                    continue
                stats.update(pim)
                sampler.add(pim, header[:100])
                if len(self.sample_data[group_name]) < self.max_samples_per_group:
                    self.sample_data[group_name].append((header, pim, sequence))
                else:
                    j = random.randint(0, total_seen - 1)
                    if j < self.max_samples_per_group:
                        self.sample_data[group_name][j] = (header, pim, sequence)
                self.tracker.update(group_name, True, len(sequence))
                if total_seen % 100000 == 0 and verbose:
                    print(f"     Processed {total_seen:,} sequences...")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return
        if len(self.sample_data[group_name]) > 0:
            self._compute_group_statistics(group_name, stats, sampler)
            if verbose:
                print(f"  ✅ {group_name}: {len(self.sample_data[group_name])} samples stored "
                      f"from {total_seen:,} processed")

    def _compute_group_statistics(self, group_name: str, stats: OnlineStatistics,
                                   sampler: ProgressiveSampler):
        samples = self.sample_data[group_name]
        if not samples:
            return
        vectors = [item[1] for item in samples]
        vectors_array = np.array(vectors)
        vectors_array = np.nan_to_num(vectors_array, nan=0.0, posinf=0.0, neginf=0.0)
        centroid = stats.get_mean()
        if np.sum(np.abs(centroid)) < 1e-10:
            centroid = np.random.randn(DIM_PAIRS) * 0.01
        covariance = stats.get_covariance()
        reg = 1e-6
        cov_reg = covariance + np.eye(DIM_PAIRS) * reg
        try:
            inv_cov = np.linalg.pinv(cov_reg, rcond=1e-8)
        except Exception:
            inv_cov = np.eye(DIM_PAIRS) * 1.0
        std_dev = stats.get_std()
        wedge_sims = []
        n_compare = min(len(vectors), 50)
        for i in range(n_compare):
            for j in range(i + 1, n_compare):
                try:
                    sim, _ = self.grassmann.wedge_product(vectors[i], vectors[j])
                    if np.isfinite(sim):
                        wedge_sims.append(sim)
                except Exception:
                    continue
        wedge_self_sim = float(np.mean(wedge_sims)) if wedge_sims else 0.0
        wedge_self_sim_std = float(np.std(wedge_sims)) if wedge_sims else 0.0
        grassmann_radius = 0.0
        if len(vectors) > 1:
            dists = []
            for v in vectors[:min(100, len(vectors))]:
                try:
                    d = self.grassmann.geodesic_distance(centroid, v)
                    if np.isfinite(d):
                        dists.append(d)
                except Exception:
                    continue
            grassmann_radius = float(np.mean(dists)) if dists else 0.0
        js_vals, hell_vals, spear_vals, pa_vals = [], [], [], []
        for i in range(min(len(vectors), 30)):
            for j in range(i + 1, min(len(vectors), 30)):
                try:
                    js_vals.append(jensen_shannon_divergence(vectors[i], vectors[j]))
                    hell_vals.append(hellinger_distance(vectors[i], vectors[j]))
                    spear_vals.append(spearman_correlation(vectors[i], vectors[j]))
                    pa = principal_angles(pim_to_subspace(vectors[i]),
                                          pim_to_subspace(vectors[j]))
                    pa_vals.append(float(np.mean(pa)))
                except Exception:
                    continue
        entropy = shannon_entropy(centroid, normalize=True)
        gini = gini_coefficient(centroid)
        sc = scalar_curvature(self.grassmann.k, DIM_PAIRS)
        bootstrap_ci = {}
        if USE_BOOTSTRAP and len(vectors) >= 2:
            for metric_name, values in [
                ('geodesic_spread', [self.grassmann.geodesic_distance(centroid, v) for v in vectors[:min(100, len(vectors))]]),
                ('jensen_shannon_mean', js_vals),
                ('hellinger_mean', hell_vals),
                ('spearman_mean', spear_vals),
                ('principal_angles_mean', pa_vals),
            ]:
                if values:
                    mean, lo, hi = bootstrap_metric(values, N_BOOTSTRAP, BOOTSTRAP_CI)
                    bootstrap_ci[metric_name] = (lo, hi)
        all_metrics = {
            'geodesic_distance_spread': grassmann_radius,
            'shannon_entropy': entropy,
            'gini': gini,
            'scalar_curvature': sc,
            'jensen_shannon_mean': float(np.mean(js_vals)) if js_vals else 0.0,
            'hellinger_mean': float(np.mean(hell_vals)) if hell_vals else 0.0,
            'spearman_mean': float(np.mean(spear_vals)) if spear_vals else 0.0,
            'principal_angles_mean': float(np.mean(pa_vals)) if pa_vals else 0.0,
        }
        self.group_stats[group_name] = GroupStatistics(
            name=group_name, n_samples=len(samples), centroid=centroid,
            covariance=covariance, inv_covariance=inv_cov, std_dev=std_dev,
            wedge_self_similarity=wedge_self_sim,
            wedge_self_similarity_std=wedge_self_sim_std,
            total_processed=self.tracker.group_counts.get(group_name, 0),
            sample_size=len(samples), grassmann_radius=grassmann_radius,
            entropy=entropy, gini=gini, geodesic_spread=grassmann_radius,
            scalar_curvature_value=sc,
            jensen_shannon_mean=all_metrics['jensen_shannon_mean'],
            hellinger_mean=all_metrics['hellinger_mean'],
            spearman_mean=all_metrics['spearman_mean'],
            all_metrics=all_metrics,
            bootstrap_ci=bootstrap_ci,
            insufficient_n=(len(samples) < self.min_samples_per_group),
        )

    def build_hash_index(self):
        self.hash_index = PIMHashIndex()
        self.hash_index.build_from_samples(self.sample_data)

    def print_processing_summary(self):
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY BY GROUP (v217.0)")
        print("=" * 80)
        print(f"  {'Group':<25} {'Samples':>12} {'Stored':>12} {'Self-Sim':>12} {'Entropy':>12} {'Status':>15}")
        print(f"  {'-' * 90}")
        for group in sorted(self.group_stats.keys()):
            stats = self.group_stats[group]
            total = self.tracker.group_counts.get(group, 0)
            status = "insufficient_n" if stats.insufficient_n else "ok"
            print(f"  {get_display_name(group):<25} {total:>12,} {stats.n_samples:>12,} "
                  f"{stats.wedge_self_similarity:>12.4f} {stats.entropy:>12.4f} {status:>15}")



    def compare_group_to_all(self, group_name: str) -> pd.DataFrame:
        if group_name not in self.group_stats:
            return pd.DataFrame()
        ref = self.group_stats[group_name]
        ref_centroid = ref.centroid
        comparisons = []
        for target_name, target_stats in self.group_stats.items():
            if target_name == group_name:
                continue
            tc = target_stats.centroid
            try:
                metrics = compare_pim_vectors(ref_centroid, tc)
            except Exception:
                continue
            row = {
                'Compared Group': get_display_name(target_name),
                'Geodesic Distance': metrics['geodesic_distance'],
                'Max Principal Angle': metrics['principal_angles_max'],
                'Mean Principal Angle': metrics['principal_angles_mean'],
                'Jensen-Shannon': metrics['jensen_shannon'],
                'Hellinger': metrics['hellinger'],
                'Spearman': metrics['spearman'],
                'Entropy Diff': abs(metrics['shannon_entropy_1'] - metrics['shannon_entropy_2']),
                'Gini Diff': abs(metrics['gini_1'] - metrics['gini_2']),
                'Target Insufficient N': target_stats.insufficient_n,
            }
            comparisons.append(row)
        df = pd.DataFrame(comparisons)
        if not df.empty:
            df = df.sort_values('Geodesic Distance')
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
                    try:
                        d = self.grassmann.geodesic_distance(
                            self.group_stats[g1].centroid, self.group_stats[g2].centroid)
                        sim = max(0.0, 1.0 - d / (np.pi * np.sqrt(self.grassmann.k)))
                    except Exception:
                        sim = 0.0
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        return pd.DataFrame(matrix, index=[get_display_name(g) for g in groups],
                            columns=[get_display_name(g) for g in groups])

    def get_top_individuals(self, group_name: str, n: int = TOP_N_PROTEINS) -> pd.DataFrame:
        """
        v217.0: marca NaN si centroide o PIM son cero.
        """
        if group_name not in self.sample_data or group_name not in self.group_stats:
            return pd.DataFrame()
        samples = self.sample_data[group_name]
        centroid = self.group_stats[group_name].centroid
        centroid_norm = np.linalg.norm(centroid)
        results = []
        for header, pim, sequence in samples:
            pim_norm = np.linalg.norm(pim)
            if centroid_norm < 1e-10 or pim_norm < 1e-10:
                sim = float('nan')
                geo = float('nan')
            else:
                try:
                    sim, _ = self.grassmann.wedge_product(pim, centroid)
                    geo = self.grassmann.geodesic_distance(pim, centroid)
                except Exception:
                    sim = float('nan')
                    geo = float('nan')
            results.append({
                'Header': header, 'Protein ID': extract_protein_id(header),
                'Wedge Similarity': sim, 'Length': len(sequence),
                'Entropy': shannon_entropy(pim, normalize=True),
                'Gini': gini_coefficient(pim),
                'Geodesic Distance': geo
            })
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('Wedge Similarity', ascending=False, na_position='last').head(n)
        return df

    def reconcile_primary_uniprot(self, config_loader) -> Dict:
        result = {
            'primary_target': None,
            'primary_uniprot_ids': [],
            'observed_ids': {},
            'mismatches': [],
            'status': 'unknown',
        }
        if config_loader is None:
            result['status'] = 'no_config'
            return result
        char_targets = config_loader.config.get('characterization_targets', {})
        result['primary_target'] = char_targets.get('primary', None)
        result['primary_uniprot_ids'] = char_targets.get('primary_uniprot_ids', [])
        expected = set(result['primary_uniprot_ids'])
        for group_name, samples in self.sample_data.items():
            observed = set()
            for header, _, _ in samples[:min(10, len(samples))]:
                pid = extract_protein_id(header)
                if pid:
                    observed.add(pid)
            result['observed_ids'][group_name] = list(observed)
        all_observed = set()
        for ids in result['observed_ids'].values():
            all_observed.update(ids)
        if expected and all_observed:
            intersection = expected & all_observed
            if intersection:
                result['status'] = 'matched'
            else:
                result['status'] = 'mismatch'
                result['mismatches'] = list(all_observed - expected)
        elif not expected:
            result['status'] = 'no_expected_ids'
        else:
            result['status'] = 'no_observed_ids'
        return result

    def validate_target_ranges(self, config_loader) -> Dict:
        result = {}
        if config_loader is None:
            return result
        config_target = get_config_target(self.main_groups[0]) if self.main_groups else 'ebola'
        ranges = config_loader.get_target_ranges(config_target)
        if not ranges:
            return result
        for group_name, stats in self.group_stats.items():
            group_result = {}
            for metric_name, rng in ranges.items():
                if not isinstance(rng, dict):
                    continue
                min_val = rng.get('min', None)
                max_val = rng.get('max', None)
                if min_val is None or max_val is None:
                    continue
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    continue
                value = None
                if metric_name == 'entropy':
                    value = stats.entropy
                elif metric_name == 'gini':
                    value = stats.gini
                elif metric_name == 'grassmann_distance':
                    value = stats.grassmann_radius
                elif metric_name == 'jensen_shannon':
                    value = stats.jensen_shannon_mean
                elif metric_name == 'hellinger':
                    value = stats.hellinger_mean
                else:
                    continue
                if value is None:
                    continue
                passed = min_val <= value <= max_val
                group_result[metric_name] = {
                    'value': float(value),
                    'min': float(min_val),
                    'max': float(max_val),
                    'passed': bool(passed),
                    'insufficient_n': stats.insufficient_n,
                }
            result[group_name] = group_result
        return result

    def generate_characterization_report(self, target_group: str = None,
                                          config_loader=None) -> Dict:
        print("\n" + "=" * 80)
        print("📊 GENERATING CHARACTERIZATION REPORT (v217.0)")
        print("=" * 80)
        if target_group is None:
            target_group = self.main_groups[0] if self.main_groups else list(self.group_stats.keys())[0]
        if target_group not in self.group_stats:
            target_group = list(self.group_stats.keys())[0]
        stats = self.group_stats[target_group]
        metrics = {}
        if target_group in self.sample_data and len(self.sample_data[target_group]) > 1:
            v1 = self.sample_data[target_group][0][1]
            v2 = self.sample_data[target_group][1][1]
            metrics = compare_pim_vectors(v1, v2)
        metrics['geodesic_distance'] = stats.grassmann_radius
        metrics['shannon_entropy_1'] = stats.entropy
        metrics['gini_1'] = stats.gini
        metrics['scalar_curvature'] = stats.scalar_curvature_value
        metrics['jensen_shannon'] = stats.jensen_shannon_mean
        metrics['hellinger'] = stats.hellinger_mean
        metrics['spearman'] = stats.spearman_mean
        metrics['insufficient_n'] = stats.insufficient_n
        metrics['composite_score'] = stats.composite_score
        metrics['classification'] = stats.classification
        knowledge_base = NarrativeKnowledgeBase()
        narrative_gen = CharacterizationNarrativeGenerator(knowledge_base)
        target_name = get_display_name(target_group)
        sequence = None
        if target_group in self.sample_data and len(self.sample_data[target_group]) > 0:
            sequence = self.sample_data[target_group][0][2]
        report_text = narrative_gen.generate_report(metrics, target_name, sequence)
        summary_gen = NarrativeSummaryGenerator(knowledge_base)
        profile_summaries = {}
        for profile in ['executive', 'biochemist', 'chemist', 'analytical_chemist',
                       'physicochemist', 'bioinformatician']:
            profile_summaries[profile] = summary_gen.generate_summary(metrics, target_name, profile)
        mdr = MultidisciplinaryReporter(summary_gen)
        table_text = mdr._generate_table(metrics, target_name)
        report = {
            'target_group': target_group, 'target_name': target_name,
            'metrics': metrics, 'report_text': report_text,
            'profile_summaries': profile_summaries,
            'multidisciplinary_table': table_text,
            'sequence': sequence, 'timestamp': datetime.now().isoformat(),
            'version': '217.0'
        }
        self.characterization_report = report
        return report

    def generate_full_report(self, reference_group: str, results_dir: str,
                             mode: OperationMode = OperationMode.HYBRID,
                             config_loader=None) -> Dict:
        print("\n" + "=" * 80)
        print("📊 GENERATING FULL REPORT (v217.0)")
        print(f"   MODE: {mode.get_mode_name().upper()}")
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
                report['kl_decomposition'] = self.grassmann.karhunen_loeve_decomposition(vectors, 8)
            else:
                report['kl_decomposition'] = {'error': 'Not enough vectors'}
        else:
            report['kl_decomposition'] = {'error': 'Reference not found'}

        if APPLY_METRIC_WEIGHTS and self.composite_calculator is not None:
            print("\n  📊 Computing composite scores (metric_weights from config)...")
            for group_name, stats in self.group_stats.items():
                composite = self.composite_calculator.compute(stats)
                stats.composite_score = composite['composite_score']
                stats.all_metrics['composite_score'] = composite['composite_score']
                stats.all_metrics['composite_components'] = composite['components']

        if APPLY_CLASSIFICATION and self.group_classifier is not None:
            print("\n  📊 Classifying groups (classification_thresholds from config)...")
            for group_name, stats in self.group_stats.items():
                if stats.composite_score is not None:
                    cls = self.group_classifier.classify(stats.composite_score)
                    stats.classification = cls['label']
                    stats.all_metrics['classification'] = cls['label']
                    stats.all_metrics['classification_threshold'] = cls['threshold']

        if VALIDATE_TARGET_RANGES_IN_CHARACTERIZATION and config_loader is not None:
            print("\n  📏 Validating target_ranges (characterization mode)...")
            self.target_ranges_validation = self.validate_target_ranges(config_loader)
            report['target_ranges_validation'] = self.target_ranges_validation

        print("\n  📊 Compiling all metrics by group...")
        rows = []
        for group_name, stats in self.group_stats.items():
            ranges_passed = 0
            ranges_total = 0
            if self.target_ranges_validation and group_name in self.target_ranges_validation:
                validation = self.target_ranges_validation[group_name]
                ranges_passed = sum(1 for v in validation.values()
                                   if isinstance(v, dict) and v.get('passed', False))
                ranges_total = sum(1 for v in validation.values() if isinstance(v, dict))
            ranges_str = f"{ranges_passed}/{ranges_total}" if ranges_total > 0 else "N/A"
            row = {
                'Group': get_display_name(group_name),
                'Samples': stats.n_samples,
                'Self-Similarity': stats.wedge_self_similarity,
                'Entropy': stats.entropy,
                'Gini': stats.gini,
                'Geodesic Spread': stats.geodesic_spread,
                'JS Mean': stats.jensen_shannon_mean,
                'Hellinger Mean': stats.hellinger_mean,
                'Spearman Mean': stats.spearman_mean,
                'Principal Angles Mean': stats.all_metrics.get('principal_angles_mean', 0.0),
                'Composite Score': stats.composite_score,
                'Classification': stats.classification,
                'Insufficient N': stats.insufficient_n,
                'Ranges Passed': ranges_str,
                'Bootstrap CI JS': str(stats.bootstrap_ci.get('jensen_shannon_mean', '')),
                'Bootstrap CI Hellinger': str(stats.bootstrap_ci.get('hellinger_mean', '')),
            }
            rows.append(row)
        report['all_metrics'] = pd.DataFrame(rows)

        print("\n  📊 Generating characterization report...")
        if config_loader is None:
            config_loader = ConfigLoader(verbose=False)
        char_report = self.generate_characterization_report(reference_group, config_loader)
        report['characterization'] = char_report
        if char_report:
            safe_save_text(char_report['report_text'], "characterization_report.txt", results_dir)
            safe_save_json(char_report['metrics'], "characterization_metrics.json", results_dir)
            for profile, summary in char_report.get('profile_summaries', {}).items():
                safe_save_text(summary, f"narrative_summary_{profile}.txt", results_dir)
            if 'multidisciplinary_table' in char_report:
                safe_save_text(char_report['multidisciplinary_table'],
                              "multidisciplinary_summary_table.txt", results_dir)

        oc = config_loader.get_operation_control()
        if mode.is_design() and oc.get('evaluate_peptide', True):
            print("\n  🧬 Generating therapeutic profile...")
            profiler = TherapeuticProfiler(
                self, config_loader, main_groups=self.main_groups,
                design_group=self.design_group, design_mode=True
            )
            profile = profiler.generate_therapeutic_profile()
            profiler.print_profile(profile)
            report['therapeutic_profile'] = profile
            self.therapeutic_profile = profile
            if profile and 'error' not in profile:
                # v217.0: guardar perfil completo
                safe_save_json(profile, "therapeutic_profile_v2170.json", results_dir)
                # v217.0: guardar tabla base vs optimizado como CSV
                bvo = profile.get('base_vs_optimized', {})
                if bvo.get('available'):
                    df_bvo = pd.DataFrame(bvo['rows'])
                    safe_save_csv(df_bvo, "base_vs_optimized_comparison.csv", results_dir)
                    print(f"  ✅ Base vs optimized comparison saved")
                # v217.0: guardar métricas ML como CSV independiente
                if profile.get('cv_results'):
                    cv = profile['cv_results']
                    cv_rows = [
                        {'Metric': 'R2_mean',  'Value': cv.get('r2_mean')},
                        {'Metric': 'R2_std',   'Value': cv.get('r2_std')},
                        {'Metric': 'MSE_mean', 'Value': cv.get('mse_mean')},
                        {'Metric': 'MSE_std',  'Value': cv.get('mse_std')},
                        {'Metric': 'MAE_mean', 'Value': cv.get('mae_mean')},
                        {'Metric': 'MAE_std',  'Value': cv.get('mae_std')},
                        {'Metric': 'AUC_mean', 'Value': cv.get('auc_mean')},
                        {'Metric': 'AUC_std',  'Value': cv.get('auc_std')},
                        {'Metric': 'MCC_mean', 'Value': cv.get('mcc_mean')},
                        {'Metric': 'MCC_std',  'Value': cv.get('mcc_std')},
                        {'Metric': 'model_type', 'Value': profile.get('model_type')},
                        {'Metric': 'data_source', 'Value': profile.get('data_source')},
                        {'Metric': 'n_splits', 'Value': cv.get('n_splits')},
                    ]
                    df_cv = pd.DataFrame(cv_rows)
                    safe_save_csv(df_cv, "ml_cv_metrics.csv", results_dir)
                    print(f"  ✅ ML CV metrics saved: ml_cv_metrics.csv")

        if USE_PIDP:
            print("\n  🧬 Performing PIDP analysis...")
            pidp = PIDPProfiler(self)
            pidp.print_tools_status()
            report['pidp_results'] = pidp.analyze_target_proteins(results_dir)

        print("\n  🧪 Performing chemical analysis...")
        chem = ChemicalProfiler(self)
        for group_name in self.main_groups:
            if group_name in self.group_stats:
                chem.analyze_protein(group_name, results_dir)

        print("\n  💾 Saving reports...")
        if report['comparison'] is not None and not report['comparison'].empty:
            safe_save_csv(report['comparison'], f"comparison_{reference_group}_vs_all.csv", results_dir)
        if report['similarity_matrix'] is not None and not report['similarity_matrix'].empty:
            safe_save_csv(report['similarity_matrix'], "similarity_matrix_groups.csv", results_dir)
        if report['top_individuals'] is not None and not report['top_individuals'].empty:
            safe_save_csv(report['top_individuals'], "top_individual_proteins.csv", results_dir)
        if report['all_metrics'] is not None and not report['all_metrics'].empty:
            safe_save_csv(report['all_metrics'], "all_metrics_report_v2170.csv", results_dir)
        if report.get('kl_decomposition') and 'error' not in report['kl_decomposition']:
            safe_save_json(report['kl_decomposition'], "kl_decomposition.json", results_dir)
        if self.target_ranges_validation:
            safe_save_json(self.target_ranges_validation, "target_ranges_validation.json", results_dir)
        if self.primary_uniprot_reconciliation:
            safe_save_json(self.primary_uniprot_reconciliation,
                          "primary_uniprot_reconciliation.json", results_dir)
        print(f"  ✅ Complete report saved in: {results_dir}/")
        return report



class ConfigLoader:
    DEFAULT_METRIC_WEIGHTS = METRIC_WEIGHTS.copy()
    DEFAULT_CLASSIFICATION_THRESHOLDS = {
        'excellent': 0.80, 'good': 0.60, 'moderate': 0.40, 'poor': 0.00
    }
    DEFAULT_PROCESSING_PARAMS = {
        'batch_size': 5000, 'max_stored_proteins': 200,
        'n_bootstrap': 50, 'max_workers': 4,
        'use_bootstrap': True, 'bootstrap_ci': 0.95,
        'min_samples_per_group': 2,
    }
    DEFAULT_OPERATION_CONTROL = {
        'mode': 'auto', 'use_dummy_peptide': False,
        'dummy_sequence': 'AAAAAAAAAAAAA',
        'fallback_mode': 'characterization',
        'allow_design_override': True, 'evaluate_peptide': True,
        'use_bootstrap': True, 'bootstrap_ci': 0.95,
        'apply_metric_weights': True, 'apply_classification': True,
        'process_secondary_targets': False,
        'validate_target_ranges_in_characterization': True,
    }

    def __init__(self, config_path: str = "config_EBOLA.json", verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.loaded_from = None
        self.warnings = []
        self._load()

    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.config = config
                self.loaded_from = 'config_EBOLA.json'
                if self.verbose:
                    print(f"  ✅ Configuration loaded from: {self.config_path}")
                return
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"  ❌ JSON error: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Error: {e}")
        self.config = self._get_default_config()
        self.loaded_from = 'default'
        if self.verbose:
            print(f"  📋 Using default values")

    def _get_default_config(self) -> Dict:
        return {
            'metadata': {'version': '217.0', 'schema_version': '3.4.0',
                        'generated': '2026-09-12T00:00:00', 'active_target': 'default'},
            'operation_control': self.DEFAULT_OPERATION_CONTROL.copy(),
            'metric_weights': self.DEFAULT_METRIC_WEIGHTS.copy(),
            'target_ranges': {},
            'classification_thresholds': self.DEFAULT_CLASSIFICATION_THRESHOLDS.copy(),
            'known_inhibitors': {},
            'virus_specific_parameters': {},
            'peptide_design_parameters': {},
            'processing_params': self.DEFAULT_PROCESSING_PARAMS.copy(),
            'base_peptide_sequence': {},
            'characterization_targets': {
                'primary': None, 'primary_uniprot_ids': [],
                'secondary': [], 'process_secondary': False,
            },
            'characterization_parameters': {
                'depth': 'comprehensive', 'include_dynamics': True,
                'include_evolutionary': False, 'report_level': 'clinical',
                'output_formats': ['narrative', 'tabular', 'clinical']
            },
            'data_paths': {}
        }

    def get_operation_mode(self) -> str:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL).get('mode', 'auto')

    def get_operation_control(self) -> Dict:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL.copy())

    def is_dummy_peptide(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        if oc.get('use_dummy_peptide', False):
            return True
        base_seq = self.get_base_peptide_sequence_string()
        dummy = oc.get('dummy_sequence', 'AAAAAAAAAAAAA')
        return base_seq == dummy

    def get_dummy_sequence(self) -> str:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL).get('dummy_sequence', 'AAAAAAAAAAAAA')

    def get_fallback_mode(self) -> str:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL).get('fallback_mode', 'characterization')

    def get_evaluate_peptide(self) -> bool:
        return self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL).get('evaluate_peptide', True)

    def get_use_bootstrap(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        pp = self.config.get('processing_params', self.DEFAULT_PROCESSING_PARAMS)
        return oc.get('use_bootstrap', pp.get('use_bootstrap', True))

    def get_bootstrap_ci(self) -> float:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        pp = self.config.get('processing_params', self.DEFAULT_PROCESSING_PARAMS)
        return float(oc.get('bootstrap_ci', pp.get('bootstrap_ci', 0.95)))

    def get_apply_metric_weights(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('apply_metric_weights', True)

    def get_apply_classification(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('apply_classification', True)

    def get_process_secondary_targets(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        ct = self.config.get('characterization_targets', {})
        return oc.get('process_secondary_targets', ct.get('process_secondary', False))

    def get_validate_target_ranges_in_characterization(self) -> bool:
        oc = self.config.get('operation_control', self.DEFAULT_OPERATION_CONTROL)
        return oc.get('validate_target_ranges_in_characterization', True)

    def get_min_samples_per_group(self) -> int:
        pp = self.config.get('processing_params', self.DEFAULT_PROCESSING_PARAMS)
        return int(pp.get('min_samples_per_group', 2))

    def get_metric_weights(self) -> Dict:
        """
        v217.0-fix2: avisa si metric_weights no suma 1.0 antes de normalizar.
        Solo añade defaults si el config está completamente vacío.
        """
        weights = self.config.get('metric_weights', {}).copy()
        weights = {k: v for k, v in weights.items()
                   if not k.startswith('_') and isinstance(v, (int, float))
                   and not isinstance(v, bool)}

        if not weights:
            if self.verbose:
                print(f"  ⚠️ metric_weights vacío en config; usando DEFAULT_METRIC_WEIGHTS")
            weights = {k: v for k, v in self.DEFAULT_METRIC_WEIGHTS.items()
                      if not k.startswith('_')}

        total = sum(weights.values())
        if total <= 0:
            if self.verbose:
                print(f"  ❌ metric_weights suma <= 0; usando DEFAULT_METRIC_WEIGHTS")
            weights = {k: v for k, v in self.DEFAULT_METRIC_WEIGHTS.items()
                      if not k.startswith('_')}
            total = sum(weights.values())

        if abs(total - 1.0) > 1e-6:
            if self.verbose:
                print(f"  ⚠️ metric_weights suma {total:.4f}, no 1.0. "
                      f"Normalizando a 1.0...")
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_target_ranges(self, target: str) -> Dict:
        """
        v217.0-fix3: filtra claves _comment y _documentation_only.
        Solo devuelve las métricas validadas por el pipeline.
        """
        ranges = self.config.get('target_ranges', {}).get(target, {})
        return {k: v for k, v in ranges.items()
                if not k.startswith('_')}

    def get_target_ranges_documentation_only(self, target: str) -> Dict:
        """
        v217.0-fix3: devuelve el bloque _documentation_only si existe.
        No se usa en validación; solo para reportes/documentación.
        """
        ranges = self.config.get('target_ranges', {}).get(target, {})
        doc_only = ranges.get('_documentation_only', {})
        if not isinstance(doc_only, dict):
            return {}
        return {k: v for k, v in doc_only.items() if not k.startswith('_')}

    def get_classification_thresholds(self, target: str = None) -> Dict:
        thresholds = self.config.get('classification_thresholds', {})
        if target and target in thresholds:
            result = {k: v for k, v in thresholds[target].items()
                     if not k.startswith('_') and isinstance(v, (int, float))
                     and not isinstance(v, bool)}
            if result:
                return result
        default = {k: v for k, v in self.DEFAULT_CLASSIFICATION_THRESHOLDS.items()
                  if not k.startswith('_')}
        return default

    def get_known_inhibitors(self, target: str = None) -> List[Dict]:
        inhibitors = self.config.get('known_inhibitors', {})
        if target and target in inhibitors:
            return inhibitors[target]
        all_inh = []
        for target_inh in inhibitors.values():
            if isinstance(target_inh, list):
                all_inh.extend(target_inh)
        return all_inh

    def get_virus_parameters(self, target: str) -> Dict:
        return self.config.get('virus_specific_parameters', {}).get(target, {})

    def get_peptide_design_parameters(self, target: str) -> Dict:
        return self.config.get('peptide_design_parameters', {}).get(target, {})

    def get_processing_params(self) -> Dict:
        return self.config.get('processing_params', self.DEFAULT_PROCESSING_PARAMS)

    def get_data_paths(self) -> Dict:
        paths = self.config.get('data_paths', {})
        return {k: expand_env_vars(v) for k, v in paths.items()
                if not k.startswith('_') and isinstance(v, str)}

    def get_base_dir(self) -> Optional[str]:
        """
        v217.0: devuelve base_dir expandido del config, o None si no existe.
        """
        paths = self.config.get('data_paths', {})
        base_dir = paths.get('base_dir')
        if base_dir and isinstance(base_dir, str):
            return expand_env_vars(base_dir)
        return None

    def get_all_targets(self) -> List[str]:
        return list(self.config.get('target_ranges', {}).keys())

    def get_characterization_targets(self) -> Dict:
        return self.config.get('characterization_targets', {})

    def get_primary_uniprot_ids(self) -> List[str]:
        return self.config.get('characterization_targets', {}).get('primary_uniprot_ids', [])

    def get_secondary_targets(self) -> List[str]:
        return self.config.get('characterization_targets', {}).get('secondary', [])

    def get_characterization_parameters(self) -> Dict:
        return self.config.get('characterization_parameters', {
            'depth': 'comprehensive', 'include_dynamics': True,
            'include_evolutionary': False, 'report_level': 'clinical',
            'output_formats': ['narrative', 'tabular', 'clinical']
        })

    def get_base_peptide_sequence(self, target: str = None) -> Dict:
        base = self.config.get('base_peptide_sequence', {})
        if target:
            target_base = self.config.get('target_base_sequences', {})
            if target in target_base:
                return target_base[target]
        return base

    def get_base_peptide_sequence_string(self, target: str = None) -> str:
        base = self.get_base_peptide_sequence(target)
        if target and target in base:
            seq = base.get('sequence', '') if isinstance(base, dict) else ''
            if seq:
                return seq
        if 'ebola' in base:
            return base['ebola'].get('sequence', '') if isinstance(base['ebola'], dict) else ''
        if 'rvfv' in base:
            return base['rvfv'].get('sequence', '') if isinstance(base['rvfv'], dict) else ''
        if isinstance(base, dict):
            return base.get('sequence', '')
        return ''

    def get_base_peptide_length(self, target: str = None) -> int:
        seq = self.get_base_peptide_sequence_string(target)
        return len(seq) if seq else 0

    def get_max_peptide_length(self, target: str = None) -> int:
        if target:
            design = self.get_peptide_design_parameters(target)
            if 'max_peptide_length' in design:
                return design['max_peptide_length']
        base = self.get_base_peptide_sequence(target)
        if isinstance(base, dict) and 'recommended_length_range' in base:
            return base['recommended_length_range'].get('max', 25)
        return 25

    def get_peptide_length_range(self, target: str = None) -> Dict:
        if target:
            design = self.get_peptide_design_parameters(target)
            if 'length_range' in design:
                return design['length_range']
        return {'min': 15, 'max': 25, 'optimal': 20}

    def is_default(self) -> bool:
        return self.loaded_from == 'default'

    def print_summary(self):
        print("\n  📋 CONFIGURATION SUMMARY")
        print("  " + "=" * 50)
        print(f"     ├─ Source: {self.loaded_from}")
        if self.loaded_from == 'config_EBOLA.json':
            meta = self.config.get('metadata', {})
            print(f"     ├─ Version: {meta.get('version', 'unknown')}")
        oc = self.get_operation_control()
        print(f"     ├─ Operation mode: {self.get_operation_mode()}")
        print(f"     ├─ Use dummy peptide: {oc.get('use_dummy_peptide', False)}")
        print(f"     ├─ Evaluate peptide: {oc.get('evaluate_peptide', True)}")
        print(f"     ├─ Use bootstrap: {self.get_use_bootstrap()}")
        print(f"     ├─ Apply metric weights: {self.get_apply_metric_weights()}")
        print(f"     ├─ Apply classification: {self.get_apply_classification()}")
        print(f"     ├─ Process secondary targets: {self.get_process_secondary_targets()}")
        print(f"     ├─ Validate ranges in char: {self.get_validate_target_ranges_in_characterization()}")
        print(f"     ├─ Min samples per group: {self.get_min_samples_per_group()}")
        print(f"     ├─ Targets configured: {len(self.get_all_targets())}")
        base_seq = self.get_base_peptide_sequence_string()
        if base_seq:
            print(f"     ├─ Base peptide: {base_seq[:20]}... ({len(base_seq)} aa)")
        else:
            print(f"     ├─ Base peptide: NOT CONFIGURED")
        max_len = self.get_max_peptide_length()
        print(f"     ├─ Max peptide length: {max_len} aa")
        primary_ids = self.get_primary_uniprot_ids()
        if primary_ids:
            print(f"     └─ Primary UniProt IDs: {primary_ids}")



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
    summary.append("📋 FINAL SUMMARY - SGPMAIN 217.0")
    summary.append("   NÚCLEO REAL + BIOPYTHON + WILKINSON-HARRISON + CONFIG COMPLETO")
    if mode:
        summary.append(f"   MODE: {mode.get_mode_name().upper()}")
    summary.append("=" * 80)
    summary.append("")
    if config_loader:
        summary.append(f"📋 CONFIGURATION:")
        summary.append(f"  ├─ Source: {config_loader.loaded_from}")
        summary.append(f"  ├─ Use bootstrap: {config_loader.get_use_bootstrap()}")
        summary.append(f"  ├─ Apply metric weights: {config_loader.get_apply_metric_weights()}")
        summary.append(f"  ├─ Apply classification: {config_loader.get_apply_classification()}")
        summary.append(f"  ├─ Process secondary: {config_loader.get_process_secondary_targets()}")
        summary.append(f"  └─ Validate ranges: {config_loader.get_validate_target_ranges_in_characterization()}")
        summary.append("")
    if 'processing' in report:
        proc = report['processing']
        summary.append(f"📊 PROCESSING:")
        summary.append(f"  ├─ Total sequences: {proc.get('total_sequences', 0):,}")
        summary.append(f"  ├─ Valid PIMs: {proc.get('valid_pim', 0):,}")
        summary.append(f"  ├─ Validity rate: {proc.get('valid_percentage', 0):.2f}%")
        summary.append(f"  ├─ Time: {proc.get('elapsed_seconds', 0)/60:.1f} min")
        summary.append(f"  └─ Speed: {proc.get('processing_rate', 0):,.0f} seq/s")
        summary.append("")
    if 'comparison' in report and report['comparison'] is not None:
        df = report['comparison']
        summary.append(f"🏷️ GROUP COMPARISON:")
        summary.append(f"  ├─ Groups compared: {len(df)}")
        if not df.empty:
            best = df.iloc[0]
            summary.append(f"  ├─ Most similar: {best['Compared Group']} (d={best['Geodesic Distance']:.6f})")
            summary.append(f"  └─ Least similar: {df.iloc[-1]['Compared Group']}")
        summary.append("")
    if 'all_metrics' in report and report['all_metrics'] is not None:
        df = report['all_metrics']
        summary.append(f"📊 METRICS SUMMARY:")
        for col in ['Entropy', 'Gini', 'Geodesic Spread', 'JS Mean', 'Hellinger Mean']:
            if col in df.columns:
                summary.append(f"  ├─ {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
        if 'Composite Score' in df.columns:
            valid_composite = df['Composite Score'].dropna()
            if len(valid_composite) > 0:
                summary.append(f"  ├─ Composite Score: {valid_composite.mean():.4f} ± {valid_composite.std():.4f}")
        if 'Classification' in df.columns:
            valid_class = df['Classification'].dropna()
            if len(valid_class) > 0:
                counts = valid_class.value_counts().to_dict()
                summary.append(f"  └─ Classification counts: {counts}")
        summary.append("")
    if 'characterization' in report:
        char = report['characterization']
        summary.append(f"📊 CHARACTERIZATION:")
        summary.append(f"  ├─ Target: {char.get('target_name', 'N/A')}")
        m = char.get('metrics', {})
        summary.append(f"  ├─ Geodesic Distance: {m.get('geodesic_distance', 0):.4f}")
        summary.append(f"  ├─ Shannon Entropy: {m.get('shannon_entropy_1', 0):.4f}")
        summary.append(f"  ├─ Gini: {m.get('gini_1', 0):.4f}")
        summary.append(f"  ├─ Jensen-Shannon: {m.get('jensen_shannon', 0):.4f}")
        summary.append(f"  ├─ Hellinger: {m.get('hellinger', 0):.4f}")
        if m.get('insufficient_n'):
            summary.append(f"  ├─ ⚠️ INSUFFICIENT N: métricas intra-grupo no confiables")
        if m.get('composite_score') is not None:
            summary.append(f"  ├─ Composite Score: {m['composite_score']:.4f}")
        if m.get('classification') is not None:
            summary.append(f"  └─ Classification: {m['classification']}")
        summary.append("")
    if 'primary_uniprot_reconciliation' in report and report['primary_uniprot_reconciliation']:
        rec = report['primary_uniprot_reconciliation']
        summary.append(f"🔍 PRIMARY UNIPROT RECONCILIATION:")
        summary.append(f"  ├─ Primary target: {rec.get('primary_target')}")
        summary.append(f"  ├─ Expected IDs: {rec.get('primary_uniprot_ids')}")
        summary.append(f"  └─ Status: {rec.get('status')}")
        if rec.get('status') == 'mismatch':
            summary.append(f"  ⚠️ Mismatches: {rec.get('mismatches', [])[:5]}")
        summary.append("")
    if 'target_ranges_validation' in report and report['target_ranges_validation']:
        summary.append(f"📏 TARGET RANGES VALIDATION:")
        for group, validation in report['target_ranges_validation'].items():
            passed = sum(1 for v in validation.values()
                        if isinstance(v, dict) and v.get('passed', False))
            total = sum(1 for v in validation.values() if isinstance(v, dict))
            if total > 0:
                summary.append(f"  ├─ {get_display_name(group)}: {passed}/{total} passed")
        summary.append("")
    if 'therapeutic_profile' in report and isinstance(report['therapeutic_profile'], dict) and 'error' not in report['therapeutic_profile']:
        tp = report['therapeutic_profile']
        summary.append(f"🧬 THERAPEUTIC PROFILE:")
        if 'target' in tp:
            summary.append(f"  ├─ Target: {tp['target'].get('protein_name', 'N/A')}")
        if 'peptide' in tp:
            m = tp['peptide'].get('all_metrics_evaluation', {})
            summary.append(f"  ├─ Peptide: {tp['peptide'].get('sequence', '')[:20]}...")
            summary.append(f"  ├─ Geodesic: {m.get('geodesic_distance', 0):.4f}")
        act = tp['peptide'].get('activity', {}) if 'peptide' in tp else {}
        if act.get('ic50_um') is not None:
            summary.append(f"  ├─ Predicted IC50: {act['ic50_um']:.4f} μM")
        if 'cv_results' in tp and tp['cv_results']:
            cv = tp['cv_results']
            summary.append(f"  ├─ CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            summary.append(f"  ├─ CV AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
            summary.append(f"  └─ CV MCC: {cv['mcc_mean']:.4f} ± {cv['mcc_std']:.4f}")
        # v217.0: comparación base vs optimizado en el resumen
        bvo = tp.get('base_vs_optimized', {})
        if bvo.get('available'):
            summary.append(f"  ├─ BASE vs OPTIMIZED:")
            summary.append(f"  │   ├─ Base:      {bvo['base_sequence']}")
            summary.append(f"  │   └─ Optimized: {bvo['optimized_sequence']}")
            for row in bvo['rows']:
                if row['delta_rel_pct'] is not None:
                    summary.append(f"  │   ├─ {row['metric']}: "
                                   f"{row['base']} → {row['optimized']} "
                                   f"({row['delta_rel_pct']:+.2f}%)")
        summary.append("")
    summary.append("📐 NÚCLEO REAL DE 10 MÉTRICAS:")
    summary.append("  GRASSMANN (5):")
    summary.append("    1. Principal angles (SVD de X^T Y)")
    summary.append("    2. Geodesic distance ‖θ‖₂")
    summary.append("    3. Scalar curvature k(n-k) [constante estructural]")
    summary.append("    4. Exponential map (Alg. 3.2, Absil-Mahony-Sepulchre 2004)")
    summary.append("    5. Logarithmic map (Alg. 3.1, ROBUSTO, Absil-Mahony-Sepulchre 2004)")
    summary.append("  INFORMACIÓN (3):")
    summary.append("    6. Shannon entropy")
    summary.append("    7. Jensen-Shannon divergence")
    summary.append("    8. Hellinger distance")
    summary.append("  ESTADÍSTICA (2):")
    summary.append("    9. Gini coefficient")
    summary.append("   10. Spearman correlation")
    summary.append("")
    summary.append("🤖 ML DE ACTIVIDAD (sin ESM, con DRAMP real):")
    summary.append("  • 8 fisicoquímicas (pI+instability REAL) + 20 AAC + 16 PIM = 44 dim")
    summary.append("  • Etiqueta: log10(IC50 en μM) de DRAMP")
    summary.append("  • Random Forest + XGBoost (si disponible)")
    summary.append("  • StratifiedKFold 5-fold cross-validation")
    summary.append("  • Métricas: R², MSE, MAE, AUC, MCC")
    summary.append("")
    summary.append("✅ v217.0: CONFIG-DRIVEN + MATEMÁTICAMENTE CORRECTO + NUMÉRICAMENTE ROBUSTO")
    summary.append("  • Aplica metric_weights del config → composite_score")
    summary.append("  • Aplica classification_thresholds del config → clasificación")
    summary.append("  • Ejecuta bootstrap si use_bootstrap=true → IC en all_metrics")
    summary.append("  • Valida target_ranges en characterization → target_ranges_validation.json")
    summary.append("  • Reconcilia primary_uniprot_ids con FASTA → primary_uniprot_reconciliation.json")
    summary.append("  • Marca insufficient_n si n < min_samples_per_group")
    summary.append("  • Expande variables de entorno en data_paths")
    summary.append("  • DATA_PATH se actualiza desde data_paths['base_dir']")
    summary.append("  • Solubilidad Wilkinson-Harrison corregida (devuelve applicable)")
    summary.append("  • Stability heurístico sin clips artificiales (marcado no informativo)")
    summary.append("  • scalar_curvature excluido del composite (constante estructural)")
    summary.append("  • Biopython: " + ("✅ disponible" if BIOPYTHON_AVAILABLE else "❌ fallback heuristic"))
    summary.append("  • v217.0: base vs optimized en una sola corrida")
    summary.append("  • v217.0: guarda base_vs_optimized_comparison.csv")
    summary.append("  • v217.0: guarda ml_cv_metrics.csv")
    summary.append("")
    summary.append("=" * 80)
    summary.append(f"✅ COMPLETED - Results in: {results_dir}/")
    summary.append("=" * 80)
    return "\n".join(summary)


def save_final_summary(report: Dict, results_dir: str, config_loader: ConfigLoader = None,
                       mode: OperationMode = None):
    summary = generate_final_summary(report, results_dir, config_loader, mode)
    safe_save_text(summary, "FINAL_SUMMARY_v2170.txt", results_dir)
    print(f"  ✅ Final summary saved: {results_dir}/FINAL_SUMMARY_v2170.txt")


def main():
    global DATA_PATH, CHEMBL_MAPPING_FILE, DRAMP_TSV_FILE
    print("=" * 80)
    print("🦠 SGPMAIN 217.0 - MIRROR-PIM CON CONFIG COMPLETO")
    print("   ✅ 10 MÉTRICAS MATEMÁTICAMENTE VÁLIDAS")
    print("   ✅ ML CON DATOS REALES DE DRAMP (IC50/MIC en μM)")
    print("   ✅ pI/MW/GRAVY/INSTABILITY REALES")
    print("   ✅ SOLUBILIDAD: WILKINSON-HARRISON 1991 (CORREGIDA)")
    print("   ✅ PERMEABILIDAD: WIMLEY-WHITE 1996 (ΔG directo)")
    print("   ✅ EXP/LOG MAP: ABSIL-MAHONY-SEPULCHRE 2004 (Alg. 3.1/3.2 ROBUSTO)")
    print("   ✅ APLICA metric_weights → composite_score")
    print("   ✅ APLICA classification_thresholds → clasificación")
    print("   ✅ EJECUTA bootstrap si use_bootstrap=true")
    print("   ✅ VALIDA target_ranges en characterization")
    print("   ✅ RECONCILIA primary_uniprot_ids con FASTA")
    print("   ✅ MARCA insufficient_n si n < min_samples_per_group")
    print("   ✅ v217.0: BASE vs OPTIMIZED en una sola corrida")
    print(f"   ✅ BIOPYTHON: {'AVAILABLE' if BIOPYTHON_AVAILABLE else 'NOT INSTALLED'}")
    print(f"   ✅ REFERENCE GROUPS: {MAIN_GROUP_REFERENCE}")
    print(f"   ✅ DESIGN GROUP: {MAIN_GROUP_DESIGN}")
    print(f"   ✅ FILE PATH: {DATA_PATH}")
    print("=" * 80)

    print("\n  🧪 VALIDANDO NÚCLEO DE 10 MÉTRICAS...")
    if not run_validation_tests():
        print("❌ Tests fallaron. Abortando.")
        sys.exit(1)
    print("✅ Tests pasaron. Continuando...\n")

    print("\n  📋 LOADING CONFIGURATION...")
    config_path = os.path.join(DATA_PATH, "config_EBOLA.json")
    config_loader = ConfigLoader(config_path, verbose=True)

    base_dir_from_config = config_loader.get_base_dir()
    if base_dir_from_config:
        DATA_PATH = base_dir_from_config
        print(f"  📁 DATA_PATH actualizado desde config: {DATA_PATH}")
        CHEMBL_MAPPING_FILE = os.path.join(DATA_PATH, "chembl_uniprot.txt")
        DRAMP_TSV_FILE = os.path.join(DATA_PATH, "general_amps.txt")

    config_loader.print_summary()

    global METRIC_WEIGHTS, BATCH_SIZE, MAX_STORED_PROTEINS_PER_GROUP, N_BOOTSTRAP, MAX_WORKERS, MAX_PEPTIDE_LENGTH
    global APPLY_METRIC_WEIGHTS, APPLY_CLASSIFICATION, VALIDATE_TARGET_RANGES_IN_CHARACTERIZATION
    global PROCESS_SECONDARY_TARGETS, MIN_SAMPLES_PER_GROUP, BOOTSTRAP_CI, USE_BOOTSTRAP

    METRIC_WEIGHTS = config_loader.get_metric_weights()
    proc = config_loader.get_processing_params()
    BATCH_SIZE = proc.get('batch_size', 5000)
    MAX_STORED_PROTEINS_PER_GROUP = proc.get('max_stored_proteins', 200)
    N_BOOTSTRAP = proc.get('n_bootstrap', 50)
    MAX_WORKERS = min(proc.get('max_workers', 4), CPU_CORES - 2)
    USE_BOOTSTRAP = config_loader.get_use_bootstrap()
    BOOTSTRAP_CI = config_loader.get_bootstrap_ci()
    APPLY_METRIC_WEIGHTS = config_loader.get_apply_metric_weights()
    APPLY_CLASSIFICATION = config_loader.get_apply_classification()
    VALIDATE_TARGET_RANGES_IN_CHARACTERIZATION = config_loader.get_validate_target_ranges_in_characterization()
    PROCESS_SECONDARY_TARGETS = config_loader.get_process_secondary_targets()
    MIN_SAMPLES_PER_GROUP = config_loader.get_min_samples_per_group()

    config_target = get_config_target(MAIN_GROUP_DESIGN[0] if MAIN_GROUP_DESIGN else 'ebola')
    MAX_PEPTIDE_LENGTH = config_loader.get_max_peptide_length(config_target)
    print(f"\n  📏 Max peptide length: {MAX_PEPTIDE_LENGTH} aa")
    print(f"  📏 Min samples per group: {MIN_SAMPLES_PER_GROUP}")
    print(f"  🔁 Bootstrap: {USE_BOOTSTRAP} (n={N_BOOTSTRAP}, CI={BOOTSTRAP_CI})")
    print(f"  ⚖️ Apply metric weights: {APPLY_METRIC_WEIGHTS}")
    print(f"  🏷️ Apply classification: {APPLY_CLASSIFICATION}")
    print(f"  📏 Validate target ranges in char: {VALIDATE_TARGET_RANGES_IN_CHARACTERIZATION}")
    print(f"  🔬 Process secondary targets: {PROCESS_SECONDARY_TARGETS}")

    data_paths = config_loader.get_data_paths()
    if 'chembl_mapping' in data_paths:
        CHEMBL_MAPPING_FILE = data_paths['chembl_mapping']
    if 'dramp_tsv' in data_paths:
        DRAMP_TSV_FILE = data_paths['dramp_tsv']

    mode = OperationMode.determine(config_loader)
    print(f"\n  🎯 OPERATION MODE: {mode.get_mode_name().upper()}")

    oc = config_loader.get_operation_control()
    print(f"     ├─ Use dummy peptide: {oc.get('use_dummy_peptide', False)}")
    print(f"     ├─ Evaluate peptide: {oc.get('evaluate_peptide', True)}")
    print(f"     └─ Fallback mode: {oc.get('fallback_mode', 'characterization')}")

    print(f"\n  🖥️ CPU: {CPU_CORES} cores")
    print(f"  📦 Batch: {BATCH_SIZE:,}")
    print(f"  💾 Sample/group: {MAX_STORED_PROTEINS_PER_GROUP:,}")
    print(f"  🧬 Biopython: {'AVAILABLE' if BIOPYTHON_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  🚀 XGBoost: {'AVAILABLE' if XGBOOST_AVAILABLE else 'NOT INSTALLED (usando RF)'}")

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = f"results_{mode.get_mode_name()}_{timestamp}"
        results_dir = create_report_directory(base_dir, timestamp)
        print(f"  📁 Results: {results_dir}")

        grassmann = GrassmannPIM(dim=DIM_PAIRS)
        analyzer = AdvancedGroupAnalyzer(
            grassmann, main_groups=MAIN_GROUP_REFERENCE,
            design_group=MAIN_GROUP_DESIGN,
            min_samples_per_group=MIN_SAMPLES_PER_GROUP,
        )
        analyzer.set_sample_size(MAX_STORED_PROTEINS_PER_GROUP)

        if APPLY_METRIC_WEIGHTS:
            composite_calc = CompositeScoreCalculator(METRIC_WEIGHTS)
            analyzer.set_composite_calculator(composite_calc)
            print(f"  ⚖️ CompositeScoreCalculator configured with {len(METRIC_WEIGHTS)} weights")
        if APPLY_CLASSIFICATION:
            config_target_name = get_config_target(MAIN_GROUP_DESIGN[0]) if MAIN_GROUP_DESIGN else 'ebola'
            thresholds = config_loader.get_classification_thresholds(config_target_name)
            classifier = GroupClassifier(thresholds)
            analyzer.set_group_classifier(classifier)
            print(f"  🏷️ GroupClassifier configured with thresholds: {thresholds}")

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

        if PROCESS_SECONDARY_TARGETS:
            secondary_files = {
                'ebola_vp40': os.path.join(DATA_PATH, 'EBOV_VP40.unico.dat0'),
                'ebola_vp30': os.path.join(DATA_PATH, 'EBOV_VP30.unico.dat0'),
            }
            for k, v in secondary_files.items():
                if os.path.exists(v):
                    files_to_load[k] = v

        print("\n📂 LOADING FASTA FILES (STREAMING)...")
        print("=" * 80)
        print("  ⚠️ STREAMING: solo se almacenan muestras por grupo")
        print("=" * 80)

        analyzer.start_time = datetime.now()
        analyzer.tracker.start_time = analyzer.start_time

        loaded = 0
        for group_name, filename in files_to_load.items():
            if os.path.exists(filename):
                analyzer.load_fasta_file(filename, group_name, verbose=True)
                loaded += 1
            else:
                print(f"  ⚠️ File not found: {filename}")

        print(f"\n  ✅ Loaded {loaded}/{len(files_to_load)} files")
        analyzer.tracker.print_summary()
        analyzer.print_processing_summary()
        analyzer.build_hash_index()

        print("\n  🔍 Reconciling primary UniProt IDs (config vs FASTA)...")
        analyzer.primary_uniprot_reconciliation = analyzer.reconcile_primary_uniprot(config_loader)
        rec = analyzer.primary_uniprot_reconciliation
        print(f"     ├─ Primary target: {rec.get('primary_target')}")
        print(f"     ├─ Expected IDs: {rec.get('primary_uniprot_ids')}")
        print(f"     └─ Status: {rec.get('status')}")
        if rec.get('status') == 'mismatch':
            print(f"     ⚠️ Mismatch detected: observed IDs not in expected set")
            print(f"        Observed: {rec.get('mismatches', [])[:10]}")

        target_group = None
        for target in MAIN_GROUP_REFERENCE:
            if target in analyzer.group_stats:
                target_group = target
                break
        if target_group is None and analyzer.group_stats:
            target_group = list(analyzer.group_stats.keys())[0]
        if target_group is None:
            print("❌ No groups loaded. Aborting.")
            sys.exit(1)

        print(f"\n  🎯 Reference group: {get_display_name(target_group)}")

        report = analyzer.generate_full_report(target_group, results_dir, mode, config_loader)

        if 'primary_uniprot_reconciliation' not in report:
            report['primary_uniprot_reconciliation'] = analyzer.primary_uniprot_reconciliation

        if 'characterization' in report:
            char = report['characterization']
            metrics = char.get('metrics', {})
            target_name = char.get('target_name', target_group)
            peptide_data = dict(metrics)
            peptide_data['sequence'] = char.get('sequence', '')
            kb = NarrativeKnowledgeBase()
            narr = NarrativeSummaryGenerator(kb)
            mdr = MultidisciplinaryReporter(narr)
            mdr.generate_reports(peptide_data, target_name, results_dir)

        save_final_summary(report, results_dir, config_loader, mode)

        print("\n" + "=" * 80)
        print("✅ SGPMAIN 217.0 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n  📁 Results: {results_dir}/")
        print(f"  🎯 Mode: {mode.get_mode_name().upper()}")

        elapsed = (datetime.now() - analyzer.start_time).total_seconds() if analyzer.start_time else 0
        print(f"  ⏱️ Time: {elapsed/60:.1f} min")

        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / (1024 * 1024)
                print(f"  💾 Memory: {mem_mb:.0f} MB")
            except Exception:
                pass

        print(f"\n  ✅ CAMBIOS v217.0:")
        print(f"     ├─ Biopython: {'✅ disponible' if BIOPYTHON_AVAILABLE else '❌ fallback'}")
        print(f"     ├─ pI real (Henderson-Hasselbalch)")
        print(f"     ├─ Instability real (Guruprasad 1990)")
        print(f"     ├─ Solubilidad Wilkinson-Harrison 1991 (CORREGIDA)")
        print(f"     ├─ Permeabilidad Wimley-White 1996 (ΔG directo)")
        print(f"     ├─ Exponential map: Alg. 3.2 (Absil-Mahony-Sepulchre 2004)")
        print(f"     ├─ Logarithmic map: Alg. 3.1 ROBUSTO (Absil-Mahony-Sepulchre 2004)")
        print(f"     ├─ CompositeScoreCalculator (metric_weights del config)")
        print(f"     ├─ GroupClassifier (classification_thresholds del config)")
        print(f"     ├─ Bootstrap real (use_bootstrap del config)")
        print(f"     ├─ target_ranges validado en characterization")
        print(f"     ├─ primary_uniprot_ids reconciliado con FASTA")
        print(f"     ├─ insufficient_n si n < min_samples_per_group")
        print(f"     ├─ data_paths con expansión de variables de entorno")
        print(f"     ├─ DATA_PATH actualizado desde config base_dir")
        print(f"     ├─ get_config_target normalizado a minúsculas")
        print(f"     ├─ get_metric_weights solo añade defaults si config vacío")
        print(f"     ├─ all_metrics_report incluye columna 'Ranges Passed'")
        print(f"     ├─ get_top_individuals marca NaN si centroide/PIM cero")
        print(f"     ├─ _design_peptide_enhanced devuelve lista (base + optimized)")
        print(f"     ├─ generate_therapeutic_profile itera sobre lista de péptidos")
        print(f"     ├─ _compare_base_vs_optimized genera tabla comparativa")
        print(f"     ├─ print_profile imprime comparación base vs optimizado")
        print(f"     ├─ generate_full_report guarda base_vs_optimized_comparison.csv")
        print(f"     ├─ generate_full_report guarda ml_cv_metrics.csv")
        print(f"     ├─ generate_final_summary imprime comparación base vs optimized")
        print(f"     └─ Heurísticos marcados con source + confidence")

    except KeyboardInterrupt:
        print("\n\n⚠️ INTERRUPTED BY USER")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            error_log = f"error_v2170_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(error_log, 'w') as f:
                f.write(traceback.format_exc())
            print(f"  ✅ Error saved to: {error_log}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
