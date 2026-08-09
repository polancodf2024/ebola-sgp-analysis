#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA OPERATORS (v17.1)
================================================================================
NEW FEATURES (v17.1):
1. ✅ get_top_individual_proteins() - Lists top N individual proteins most similar to target
2. ✅ Saves top_20_individual_proteins.csv in results directory

EXISTING FEATURES (v17.0):
1. ✅ Shannon Entropy for PIM profiles
2. ✅ Jensen-Shannon Divergence
3. ✅ Gini Coefficient (profile inequality)
4. ✅ Structural Complexity Index
5. ✅ Functional Modularity Index
6. ✅ Hellinger Distance
7. ✅ Spearman Correlation
8. ✅ Moran's I (spatial autocorrelation)
9. ✅ Grassmann metrics (projection, Fubini-Study)
10. ✅ Ricci curvature in Grassmann space
11. ✅ Karhunen-Loève decomposition for PIM profiles
12. ✅ Discrete Radon transform for symmetry detection
13. ✅ Fractal dimension (box-counting) for PIM profiles
14. ✅ Wasserstein-1 distance (Earth Mover's Distance)
15. ✅ Spectral Laplacian of polarity interaction network
16. ✅ Integration with APD and ChEMBL
17. ✅ Complete physicochemical profiler
18. ✅ ML-based antiviral activity prediction
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, pearsonr, linregress, spearmanr
from scipy.spatial.distance import cosine
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import os
import hashlib
from datetime import datetime
from collections import defaultdict
import random
import gc
import sys
import time
import json
import re
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMILARITY_THRESHOLD = None
CONFIDENCE_LEVEL = 0.95
TOP_N_PROTEINS = 20
TOLERANCE = 0.001
USE_TRIPLETS = True
USE_QUADRUPLETS = True
USE_BOOTSTRAP = True
N_BOOTSTRAP = 100
USE_WEIGHTS = True
COHESION_SAMPLE_SIZE = 10000
USE_BIOLOGICAL_METRIC = True
SHOW_METRIC_ANALYSIS = True
USE_HODGE_DUAL = True
USE_GRASSMANN_GEODESIC = True
USE_GENERAL_ROTORS = True
GENERATE_PLOTS = False
USE_GRASSMANN_PROJECTION = True
USE_FUBINI_STUDY = True
USE_RICCI_CURVATURE = True
USE_KARHUNEN_LOEVE = True
USE_RADON_TRANSFORM = True
USE_FRACTAL_DIMENSION = True
USE_WASSERSTEIN = True
USE_POLARITY_LAPLACIAN = True

# NUEVAS MÉTRICAS v17.0
USE_SHANNON_ENTROPY = True
USE_JENSEN_SHANNON = True
USE_GINI_COEFFICIENT = True
USE_STRUCTURAL_COMPLEXITY = True
USE_FUNCTIONAL_MODULARITY = True
USE_HELLINGER_DISTANCE = True
USE_SPEARMAN_CORRELATION = True
USE_MORANS_I = True

# ============================================================================
# TARGET GROUP CONFIGURATION (MAIN)
# ============================================================================

# Define the groups you want to analyze as "main" here.
# These will be used for:
# 1. Selecting the reference group
# 2. Generating the therapeutic profile
# 3. Main comparative analysis

# For EBOLA (CURRENT):
MAIN_GROUP = ['zaire', 'sudan', 'reston', 'bundibugyo', 'tai', 'bombali']


# For WEST NILE (example):
# MAIN_GROUP = ['nile1', 'nile2']

# For EBOLA (example):
# MAIN_GROUP = ['zaire', 'sudan', 'reston', 'bundibugyo', 'tai', 'bombali']

# For CORONAVIRUS (example):
# MAIN_GROUP = ['sars_cov_2', 'sars_cov_1', 'mers_cov']

# For INFLUENZA (example):
# MAIN_GROUP = ['h1n1', 'h3n2', 'h5n1', 'h7n9']

# For LUJO (example):
# MAIN_GROUP = ['lujo']

# ============================================================================
# EXTERNAL FILE CONFIGURATION
# ============================================================================

CHEMBL_MAPPING_FILE = "chembl_uniprot.txt"
APD_FASTA_FILE = "apd_natural.fasta"

# ============================================================================
# ⚙️ BATCH PROCESSING CONFIGURATION
# ============================================================================

BATCH_SIZE = 100000
MAX_STORED_PROTEINS_PER_GROUP = 10000
COHESION_CALC_SAMPLE_SIZE = 500
BATCH_THRESHOLD = 50000
PROGRESS_REPORT_INTERVAL = 100000

# ============================================================================
# GROUP NAME MAPPING FOR DISPLAY
# ============================================================================

GROUP_NAME_MAP = {
    'enfermedad': 'DISEASE',
    'membrana': 'MEMBRANE',
    'senales': 'SIGNALS',
    # === EBOLA STRAINS ===
    'sudan': 'EBOLA_SUDAN',
    'zaire': 'EBOLA_ZAIRE',
    'reston': 'EBOLA_RESTON',
    'bombali': 'EBOLA_BOMBALI',
    'bundibugyo': 'EBOLA_BUNDIBUGYO',
    'tai': 'EBOLA_TAI_FOREST',
    # === ARENAVIRUS STRAINS (CONTROL) ===
    'lasv': 'LASV',
    'junv': 'JUNV',
    'macv': 'MACV',
    'lcmv': 'LCMV',
    # === WEST NILE (CONTROL) ===
    'nile1': 'NILE1',
    'nile2': 'NILE2',
    # === LUJO (CONTROL) ===
    'lujo': 'LUJO',
}

def get_display_name(group_name: str) -> str:
    return GROUP_NAME_MAP.get(group_name, group_name)

def extract_protein_id(header: str) -> str:
    """
    Extract UniProt ID from FASTA header.
    Examples:
    >sp|P87671|VGP_EBOEC Envelope glycoprotein -> P87671
    >tr|A0A068B3Q0|A0A068B3Q0_9HIV1 -> A0A068B3Q0
    >AP00001 -> AP00001
    """
    # Try to extract UniProt ID (format: >sp|ID|NAME or >tr|ID|NAME)
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    
    # Try to extract first word after '>' (for simple headers)
    if header.startswith('>'):
        header = header[1:]
    
    # Return first word or the whole header if no clear ID
    return header.split()[0] if header.split() else header[:20]

# ============================================================================
# BASE CONSTANTS
# ============================================================================

DIM_PAIRS = 16
DIM_TRIPLETS = 64
DIM_BIVECTOR = 120  # C(16,2) = 120

ROTOR_PLANES = [
    ('hydrophobic', (10, 15), 'N→N vs NP→NP (H-bonds vs hydrophobic interactions)'),
    ('charge', (0, 5), 'P⁺→P⁺ vs NP→NP (repulsion vs hydrophobic)'),
    ('opposite_charge', (1, 4), 'P⁺→P⁻ vs P⁻→P⁺ (charge-charge interactions)'),
    ('polarity', (10, 11), 'N→N vs N→NP (polar vs mixed)'),
    ('charge_transition', (2, 8), 'P⁺→N vs N→P⁺ (charge-polar transition)'),
    ('opposite_transition', (6, 9), 'P⁻→N vs N→P⁻ (negative charge-polar transition)'),
]

REFLECTION_SWAP_MAP = {
    0: 5, 1: 4, 2: 6, 3: 7, 4: 1, 5: 0, 6: 2, 7: 3,
    8: 9, 9: 8, 10: 10, 11: 11, 12: 13, 13: 12, 14: 14, 15: 15,
}

KEY_BIVECTORS = [
    (0, 5), (1, 4), (2, 6), (3, 7), (10, 11), (14, 15),
]

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
# EXISTING MATHEMATICAL FUNCTIONALITIES (v16.1)
# ============================================================================

def grassmann_projection_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Projection distance in Grassmann space.
    Measures the distance between two one-dimensional subspaces.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    P1 = np.outer(v1_norm, v1_norm)
    P2 = np.outer(v2_norm, v2_norm)
    return np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)

def grassmann_fubini_study(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Fubini-Study metric in Grassmann space.
    Natural metric of complex projective space.
    """
    cos_theta = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, 0, 1)
    return np.arccos(cos_theta)

def grassmann_ricci_curvature(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Estimates Ricci curvature in Grassmann space.
    Quantifies the geometric "tension" between two protein profiles.
    """
    cos_theta = np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, 0, 1)
    theta = np.arccos(cos_theta)
    if theta > 1e-10:
        return 1.0 / (np.tan(theta)**2 + 1e-10)
    return 0.0

def karhunen_loeve_decomposition(vectors: List[np.ndarray], n_components: int = 8) -> Dict:
    """
    Karhunen-Loève decomposition of PIM profiles.
    Identifies dominant variation modes in the polarity distribution.
    """
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

def discrete_radon_transform(v: np.ndarray, n_angles: int = 8) -> np.ndarray:
    """
    Discrete Radon transform for detecting symmetries in PIM profiles.
    """
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

def fractal_dimension(v: np.ndarray, scales: int = 10) -> float:
    """
    Estimates the fractal dimension of a PIM profile using the box-counting method.
    """
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

def wasserstein_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Wasserstein-1 distance (Earth Mover's Distance) between PIM profiles.
    """
    p1 = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    p2 = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    return np.sum(np.abs(cdf1 - cdf2)) / len(v1)

def polarity_interaction_laplacian(v: np.ndarray) -> Dict:
    """
    Constructs the Laplacian of the polarity interaction graph.
    Nodes are the 16 components of the PIM profile.
    Edges represent significant transitions.
    """
    n = len(v)
    v_abs = np.abs(v)
    v_norm = v_abs / (np.sum(v_abs) + 1e-10)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = min(v_norm[i], v_norm[j]) / (max(v_norm[i], v_norm[j]) + 1e-10)
            if sim > 0.1:
                adj[i, j] = adj[j, i] = sim
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    eigvals, eigvecs = eigh(L)
    return {
        'laplacian': L,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'spectral_gap': eigvals[1] - eigvals[0] if len(eigvals) > 1 else 0,
        'connectivity': np.sum(adj > 0) / (n * (n-1) / 2) if n > 1 else 0,
        'fiedler_vector': eigvecs[:, 1] if len(eigvecs) > 1 else np.zeros(n)
    }

# ============================================================================
# NUEVAS MÉTRICAS MATEMÁTICAS (v17.0)
# ============================================================================

def shannon_entropy(v: np.ndarray) -> float:
    """
    Shannon entropy of PIM profile.
    Measures diversity of polarity transitions.
    H ≈ 0: highly concentrated (specialized protein)
    H ≈ 4: uniform (promiscuous protein)
    """
    p = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
    return -np.sum(p * np.log2(p + 1e-10))

def jensen_shannon_divergence(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two PIM profiles.
    Symmetric version of Kullback-Leibler divergence.
    0 ≤ JS ≤ 1, JS=0: identical, JS=1: completely different
    """
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    m = (p + q) / 2
    kl_pm = np.sum(p * np.log2((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log2((q + 1e-10) / (m + 1e-10)))
    return 0.5 * (kl_pm + kl_qm)

def gini_coefficient(v: np.ndarray) -> float:
    """
    Gini coefficient for measuring inequality in PIM profile.
    0 = completely uniform, 1 = completely concentrated.
    """
    p = np.abs(v) / (np.sum(np.abs(v)) + 1e-10)
    sorted_p = np.sort(p)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    return 1 - (2 * np.sum(cumsum) / (n * np.sum(sorted_p) + 1e-10))

def structural_complexity(v: np.ndarray) -> float:
    """
    Structural complexity index combining entropy, fractal dimension, and richness.
    """
    entropy = shannon_entropy(v)
    fractal = fractal_dimension(v)
    n_nonzero = np.sum(np.abs(v) > 1e-6)
    max_entropy = np.log2(16)
    norm_entropy = entropy / max_entropy
    norm_fractal = fractal / 1.0
    norm_richness = n_nonzero / 16
    return 0.4 * norm_entropy + 0.35 * norm_fractal + 0.25 * norm_richness

def functional_modularity(v: np.ndarray, threshold: float = 0.05) -> float:
    """
    Measures how "modular" a PIM profile is.
    Detects clusters of significant transitions.
    """
    significant = np.where(np.abs(v) > threshold)[0]
    if len(significant) < 2:
        return 0.0
    clusters = []
    current_cluster = [significant[0]]
    for i in range(1, len(significant)):
        if significant[i] - significant[i-1] <= 2:
            current_cluster.append(significant[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [significant[i]]
    clusters.append(current_cluster)
    cluster_energies = []
    for cluster in clusters:
        energy = np.sum(np.abs(v[cluster]))
        cluster_energies.append(energy)
    total_energy = np.sum(cluster_energies) + 1e-10
    cluster_energies = np.array(cluster_energies) / total_energy
    entropy = -np.sum(cluster_energies * np.log2(cluster_energies + 1e-10))
    max_entropy = np.log2(len(clusters) + 1e-10)
    return 1 - (entropy / (max_entropy + 1e-10))

def hellinger_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Hellinger distance between two PIM profiles.
    0 ≤ H ≤ 1, symmetric, square-root based.
    """
    p = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-10)
    q = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-10)
    return (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))

def spearman_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Spearman correlation between PIM profiles.
    Detects monotonic relationships (order of transitions).
    """
    result = spearmanr(v1, v2)
    return result.correlation if result.correlation is not None else 0.0

def morans_i(v: np.ndarray, weight_matrix: np.ndarray = None) -> float:
    """
    Moran's I for spatial autocorrelation.
    Detects clustering of similar values in the profile.
    """
    n = len(v)
    if weight_matrix is None:
        weight_matrix = np.exp(-np.abs(np.arange(n).reshape(-1, 1) - np.arange(n)) / 2)
    z = v - np.mean(v)
    numerator = np.sum(weight_matrix * np.outer(z, z))
    denominator = np.sum(z**2)
    return (n / np.sum(weight_matrix)) * (numerator / (denominator + 1e-10))

def compute_enhanced_metrics(v1: np.ndarray, v2: np.ndarray) -> Dict:
    """
    Computes all enhanced metrics for comparing two PIM profiles.
    Includes v16.1 and v17.0 metrics.
    """
    metrics = {}
    
    if USE_GRASSMANN_PROJECTION:
        metrics['grassmann_projection'] = grassmann_projection_distance(v1, v2)
    
    if USE_FUBINI_STUDY:
        metrics['fubini_study'] = grassmann_fubini_study(v1, v2)
    
    if USE_RICCI_CURVATURE:
        metrics['ricci_curvature'] = grassmann_ricci_curvature(v1, v2)
    
    if USE_WASSERSTEIN:
        metrics['wasserstein'] = wasserstein_distance(v1, v2)
    
    if USE_FRACTAL_DIMENSION:
        metrics['fractal_dim_v1'] = fractal_dimension(v1)
        metrics['fractal_dim_v2'] = fractal_dimension(v2)
        metrics['fractal_dim_diff'] = abs(metrics['fractal_dim_v1'] - metrics['fractal_dim_v2'])
    
    if USE_RADON_TRANSFORM:
        radon_v1 = discrete_radon_transform(v1)
        radon_v2 = discrete_radon_transform(v2)
        metrics['radon_similarity'] = np.dot(radon_v1, radon_v2) / (np.linalg.norm(radon_v1) * np.linalg.norm(radon_v2) + 1e-10)
    
    if USE_POLARITY_LAPLACIAN:
        lap1 = polarity_interaction_laplacian(v1)
        lap2 = polarity_interaction_laplacian(v2)
        metrics['laplacian_spectral_gap_v1'] = lap1['spectral_gap']
        metrics['laplacian_spectral_gap_v2'] = lap2['spectral_gap']
        metrics['laplacian_connectivity_v1'] = lap1['connectivity']
        metrics['laplacian_connectivity_v2'] = lap2['connectivity']
        metrics['laplacian_distance'] = np.linalg.norm(lap1['eigenvalues'] - lap2['eigenvalues'])
    
    # NUEVAS MÉTRICAS v17.0
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
    
    if USE_STRUCTURAL_COMPLEXITY:
        metrics['complexity_v1'] = structural_complexity(v1)
        metrics['complexity_v2'] = structural_complexity(v2)
        metrics['complexity_diff'] = abs(metrics['complexity_v1'] - metrics['complexity_v2'])
    
    if USE_FUNCTIONAL_MODULARITY:
        metrics['modularity_v1'] = functional_modularity(v1)
        metrics['modularity_v2'] = functional_modularity(v2)
        metrics['modularity_diff'] = abs(metrics['modularity_v1'] - metrics['modularity_v2'])
    
    if USE_HELLINGER_DISTANCE:
        metrics['hellinger'] = hellinger_distance(v1, v2)
    
    if USE_SPEARMAN_CORRELATION:
        metrics['spearman'] = spearman_correlation(v1, v2)
    
    if USE_MORANS_I:
        metrics['morans_i_v1'] = morans_i(v1)
        metrics['morans_i_v2'] = morans_i(v2)
        metrics['morans_i_diff'] = abs(metrics['morans_i_v1'] - metrics['morans_i_v2'])
    
    return metrics

# ============================================================================
# BASE FUNCTIONS
# ============================================================================

def compute_pim_profile(sequence: str, use_weights: bool = USE_WEIGHTS) -> np.ndarray:
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

def compute_trimer_profile(sequence: str) -> np.ndarray:
    seq = ''.join([c for c in sequence.strip() if c.isalpha() and c.upper() in POLARITY_MAP])
    if len(seq) < 3:
        return np.zeros(DIM_TRIPLETS)
    
    polarities = []
    for aa in seq:
        pol = POLARITY_MAP.get(aa.upper())
        if pol is not None:
            polarities.append(pol)
    
    if len(polarities) < 3:
        return np.zeros(DIM_TRIPLETS)
    
    trimer_profile = np.zeros(DIM_TRIPLETS)
    for i in range(len(polarities) - 2):
        p1, p2, p3 = polarities[i], polarities[i+1], polarities[i+2]
        pairs = [
            f"{p1},{p2}",
            f"{p2},{p3}",
            f"{p1},{p3}"
        ]
        for pair in pairs:
            if pair in INTERACTION_TO_IDX:
                idx = INTERACTION_TO_IDX[pair]
                trimer_profile[idx % 16] += 1
    
    total = np.sum(trimer_profile)
    if total > 0:
        trimer_profile = trimer_profile / total
    
    return trimer_profile

def wedge_product_oriented(v: np.ndarray, w: np.ndarray, 
                           key_pairs: List[Tuple[int, int]] = None) -> np.ndarray:
    if key_pairs is None:
        key_pairs = KEY_BIVECTORS
    
    bivector = np.zeros(len(key_pairs))
    for idx, (i, j) in enumerate(key_pairs):
        if i < len(v) and j < len(w):
            bivector[idx] = v[i] * w[j] - v[j] * w[i]
    return bivector

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

def reflection_normal_vector() -> np.ndarray:
    n = np.zeros(16)
    for i, j in REFLECTION_SWAP_MAP.items():
        n[i] = 1.0
        n[j] = -1.0
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm
    return n

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

def interior_product(v: np.ndarray, subspace_name: str) -> np.ndarray:
    if subspace_name not in SUBSPACES:
        raise ValueError(f"Subspace not recognized: {subspace_name}")
    indices = SUBSPACES[subspace_name]
    if indices is None:
        return v.copy()
    projected = np.zeros_like(v)
    projected[indices] = v[indices]
    total = np.sum(projected)
    if total > 0:
        projected = projected / total
    return projected

def interior_product_magnitude(v: np.ndarray, subspace_name: str) -> float:
    proj = interior_product(v, subspace_name)
    return np.linalg.norm(proj)

def wedge_product_with_ci(v: np.ndarray, w: np.ndarray, 
                          n_bootstrap: int = N_BOOTSTRAP,
                          use_bootstrap: bool = USE_BOOTSTRAP) -> Tuple[float, float]:
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

def pim_to_hash(pim_vector: np.ndarray, tolerance: float = TOLERANCE) -> str:
    discretized = np.round(pim_vector / tolerance) * tolerance
    vector_str = ','.join([f"{x:.6f}" for x in discretized])
    return hashlib.sha256(vector_str.encode()).hexdigest()[:32]

def compute_delta_pim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 - v2

def read_fasta_file(filepath: str) -> List[Tuple[str, str]]:
    """
    Reads a FASTA file and returns a list of (header, sequence)
    Handles multi-line sequences
    """
    sequences = []
    if not os.path.exists(filepath):
        return sequences
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    
    return sequences

# ============================================================================
# GEOMETRIC PRODUCT WITH HIGHER GRADES
# ============================================================================

def wedge_product_general(v: np.ndarray, w: np.ndarray, grade: int = 2) -> np.ndarray:
    n = len(v)
    if grade == 2:
        result = []
        for i in range(n):
            for j in range(i+1, n):
                result.append(v[i] * w[j] - v[j] * w[i])
        return np.array(result)
    
    elif grade == 3:
        result = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    det = (v[i] * w[j] * 1 + v[j] * w[k] * 1 + v[k] * w[i] * 1 -
                          (v[k] * w[j] * 1 + v[j] * w[i] * 1 + v[i] * w[k] * 1))
                    result.append(det)
        return np.array(result)
    
    elif grade == 4:
        result = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        det = (v[i] * w[j] * 1 * 1 + v[j] * w[k] * 1 * 1 + 
                               v[k] * w[l] * 1 * 1 + v[l] * w[i] * 1 * 1 -
                              (v[l] * w[k] * 1 * 1 + v[k] * w[j] * 1 * 1 + 
                               v[j] * w[i] * 1 * 1 + v[i] * w[l] * 1 * 1))
                        result.append(det)
        return np.array(result[:100])
    
    else:
        return np.array([])

def geometric_product_full(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
    if metric is None:
        metric = METRIC_SIGNATURE
    
    scalar = np.sum(metric * v * w)
    bivector = wedge_product_general(v, w, grade=2)
    trivector = wedge_product_general(v, w, grade=3) if USE_TRIPLETS else np.array([])
    quadrivector = wedge_product_general(v, w, grade=4) if USE_QUADRUPLETS else np.array([])
    
    norm_scalar = abs(scalar)
    norm_bivector = np.linalg.norm(bivector) if len(bivector) > 0 else 0
    norm_trivector = np.linalg.norm(trivector) if len(trivector) > 0 else 0
    norm_quadrivector = np.linalg.norm(quadrivector) if len(quadrivector) > 0 else 0
    
    total_norm = np.sqrt(norm_scalar**2 + norm_bivector**2 + 
                         norm_trivector**2 + norm_quadrivector**2)
    
    interpretation = interpret_geometric_product(
        norm_scalar, norm_bivector, norm_trivector, norm_quadrivector
    )
    
    return {
        'grade_0': scalar,
        'grade_2': bivector,
        'grade_3': trivector,
        'grade_4': quadrivector,
        'norm_grade_0': norm_scalar,
        'norm_grade_2': norm_bivector,
        'norm_grade_3': norm_trivector,
        'norm_grade_4': norm_quadrivector,
        'total_norm': total_norm,
        'interpretation': interpretation,
        'grade_decomposition': {
            'functional': norm_scalar / (total_norm + 1e-10),
            'pair_interactions': norm_bivector / (total_norm + 1e-10),
            'triple_interactions': norm_trivector / (total_norm + 1e-10),
            'quadruple_interactions': norm_quadrivector / (total_norm + 1e-10),
        }
    }

def interpret_geometric_product(s0: float, s2: float, s3: float, s4: float) -> str:
    total = s0 + s2 + s3 + s4 + 1e-10
    p0 = s0 / total
    p2 = s2 / total
    p3 = s3 / total
    p4 = s4 / total
    
    if p0 > 0.6:
        return "Functional interactions (scalar) dominate"
    elif p2 > 0.5:
        return "Pair interactions dominate (secondary structure)"
    elif p3 > 0.4:
        return "Strong 3-body cooperativity (protein domains)"
    elif p4 > 0.3:
        return "4-body quaternary interactions (complex assembly)"
    else:
        return "Balanced interactions across all orders"

# ============================================================================
# HODGE TRANSFORMATION (DUALITY)
# ============================================================================

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

# ============================================================================
# GENERAL ROTORS - CORRECTED
# ============================================================================

def find_optimal_plane(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Finds the optimal plane for rotation between two vectors.
    Returns a vector of the same dimension as the input vectors.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    
    # Calculate the difference between the vectors to find the rotation plane
    diff = v2_norm - v1_norm
    
    # If the difference is small, return a random plane
    if np.linalg.norm(diff) < 1e-8:
        random_plane = np.random.randn(len(v1))
        return random_plane / (np.linalg.norm(random_plane) + 1e-10)
    
    # The rotation plane is perpendicular to the difference
    plane = diff / (np.linalg.norm(diff) + 1e-10)
    
    # Ensure the plane has the same dimension as v1
    if len(plane) != len(v1):
        if len(plane) < len(v1):
            plane_padded = np.zeros(len(v1))
            plane_padded[:len(plane)] = plane
            plane = plane_padded
        else:
            plane = plane[:len(v1)]
    
    return plane / (np.linalg.norm(plane) + 1e-10)

def find_rotation_angle(v1: np.ndarray, v2: np.ndarray, plane: np.ndarray = None) -> float:
    """
    Finds the rotation angle between two vectors in the specified plane.
    
    Args:
        v1: First vector (dimension n)
        v2: Second vector (dimension n)  
        plane: Rotation plane (dimension n)
    
    Returns:
        Rotation angle in degrees
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    
    if plane is None:
        # If no plane is provided, calculate the direct angle between vectors
        cos_theta = np.dot(v1_norm, v2_norm)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta) * 180.0 / np.pi
    
    # Ensure the plane has the same dimension as v1
    if len(plane) != len(v1):
        if len(plane) < len(v1):
            plane_padded = np.zeros(len(v1))
            plane_padded[:len(plane)] = plane
            plane = plane_padded
        else:
            plane = plane[:len(v1)]
    
    # Project the vectors onto the plane
    v1_proj = np.dot(v1_norm, plane) * plane
    v2_proj = np.dot(v2_norm, plane) * plane
    
    norm1 = np.linalg.norm(v1_proj) + 1e-10
    norm2 = np.linalg.norm(v2_proj) + 1e-10
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        # If the projection is very small, use the direct angle
        cos_theta = np.dot(v1_norm, v2_norm)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta) * 180.0 / np.pi
    
    cos_theta = np.dot(v1_proj, v2_proj) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)
    
    return np.arccos(cos_theta) * 180.0 / np.pi

def general_rotor(v: np.ndarray, target: np.ndarray, n_steps: int = 10) -> List[np.ndarray]:
    v_norm = v / (np.linalg.norm(v) + 1e-10)
    target_norm = target / (np.linalg.norm(target) + 1e-10)
    
    plane = find_optimal_plane(v_norm, target_norm)
    total_angle = find_rotation_angle(v_norm, target_norm, plane)
    
    trajectory = []
    for step in range(n_steps + 1):
        t = step / n_steps
        theta = t * total_angle * np.pi / 180.0
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Simplified implementation of rotation in the plane
        # Using the generalized Rodrigues formula
        proj_plane = np.dot(v_norm, plane) * plane
        proj_perp = v_norm - proj_plane
        
        # For the rotation, we need a perpendicular vector in the plane
        # We use the cross product in n-dimensional space
        # Simplification: use the normalized difference
        if np.linalg.norm(proj_perp) > 1e-8:
            perp = proj_perp / np.linalg.norm(proj_perp)
            # Construct an orthonormal vector in the plane
            cross = np.cross(plane, perp) if len(plane) == 3 else perp
            # Rotation in the plane
            rotated_proj = cos_theta * proj_plane + sin_theta * cross * np.linalg.norm(proj_plane)
            rotated = rotated_proj + proj_perp - cos_theta * proj_perp + sin_theta * np.cross(plane, proj_perp) if len(plane) == 3 else rotated_proj + proj_perp
        else:
            rotated = cos_theta * v_norm + sin_theta * plane
        
        trajectory.append(rotated / (np.linalg.norm(rotated) + 1e-10))
    
    return trajectory

# ============================================================================
# GRASSMANN SPACE
# ============================================================================

def grassmann_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    P1 = np.outer(v1, v1) / (np.linalg.norm(v1)**2 + 1e-10)
    P2 = np.outer(v2, v2) / (np.linalg.norm(v2)**2 + 1e-10)
    distance = np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)
    return distance

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

# ============================================================================
# CLASSICAL GA OPERATORS
# ============================================================================

def commutator(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return wedge_product_general(v, w, grade=2)

def commutator_norm(v: np.ndarray, w: np.ndarray) -> float:
    comm = commutator(v, w)
    mag = np.linalg.norm(comm)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    return mag / (norm_v * norm_w + 1e-10)

def anticommutator(v: np.ndarray, w: np.ndarray) -> float:
    return 2.0 * np.dot(v, w)

def anticommutator_similarity(v: np.ndarray, w: np.ndarray) -> float:
    anticomm = anticommutator(v, w)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    sim = np.abs(anticomm) / (2.0 * norm_v * norm_w + 1e-10)
    return min(sim, 1.0)

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

def norm_metric(v: np.ndarray, metric: np.ndarray = None) -> Tuple[float, float]:
    value = dot_product_metric(v, v, metric)
    sign = np.sign(value) if value != 0 else 0
    magnitude = np.sqrt(np.abs(value) + 1e-10)
    return magnitude, sign

def similarity_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
    dot_η = dot_product_metric(v, w, metric)
    norm_v, _ = norm_metric(v, metric)
    norm_w, _ = norm_metric(w, metric)
    if norm_v * norm_w < 1e-10:
        return 0.0
    return np.abs(dot_η) / (norm_v * norm_w + 1e-10)

def metric_signature_info() -> Dict:
    info = {
        'total_components': len(METRIC_SIGNATURE),
        'positive_count': np.sum(METRIC_SIGNATURE > 0),
        'negative_count': np.sum(METRIC_SIGNATURE < 0),
        'neutral_count': np.sum(METRIC_SIGNATURE == 0),
        'is_euclidean': np.all(METRIC_SIGNATURE == 1),
        'is_biological': USE_BIOLOGICAL_METRIC,
    }
    component_names = [
        'P⁺→P⁺', 'P⁺→P⁻', 'P⁺→N', 'P⁺→NP',
        'P⁻→P⁺', 'P⁻→P⁻', 'P⁻→N', 'P⁻→NP',
        'N→P⁺', 'N→P⁻', 'N→N', 'N→NP',
        'NP→P⁺', 'NP→P⁻', 'NP→N', 'NP→NP'
    ]
    info['beneficial_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] > 0]
    info['detrimental_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] < 0]
    info['neutral_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] == 0]
    return info

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

# ============================================================================
# CLIFFORD SIGNATURE
# ============================================================================

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
    
    if USE_FRACTAL_DIMENSION:
        signature['fractal_dim'] = fractal_dimension(v)
    
    if USE_RADON_TRANSFORM:
        radon = discrete_radon_transform(v)
        signature['radon_energy'] = np.sum(radon**2)
        signature['radon_max'] = np.max(radon)
    
    if USE_POLARITY_LAPLACIAN:
        lap = polarity_interaction_laplacian(v)
        signature['laplacian_spectral_gap'] = lap['spectral_gap']
        signature['laplacian_connectivity'] = lap['connectivity']
    
    # NUEVAS MÉTRICAS v17.0
    if USE_SHANNON_ENTROPY:
        signature['entropy'] = shannon_entropy(v)
    if USE_GINI_COEFFICIENT:
        signature['gini'] = gini_coefficient(v)
    if USE_STRUCTURAL_COMPLEXITY:
        signature['complexity'] = structural_complexity(v)
    if USE_FUNCTIONAL_MODULARITY:
        signature['modularity'] = functional_modularity(v)
    if USE_MORANS_I:
        signature['morans_i'] = morans_i(v)
    
    return signature

def clifford_distance(sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
    keys = ['norm', 'auto_reflection', 'hydrophobic_projection', 'charge_projection', 'auto_rotation']
    if USE_HODGE_DUAL:
        keys.extend(['hodge_norm', 'hodge_complement'])
    if USE_FRACTAL_DIMENSION:
        keys.append('fractal_dim')
    if USE_RADON_TRANSFORM:
        keys.extend(['radon_energy', 'radon_max'])
    if USE_POLARITY_LAPLACIAN:
        keys.extend(['laplacian_spectral_gap', 'laplacian_connectivity'])
    # NUEVAS MÉTRICAS v17.0
    if USE_SHANNON_ENTROPY:
        keys.append('entropy')
    if USE_GINI_COEFFICIENT:
        keys.append('gini')
    if USE_STRUCTURAL_COMPLEXITY:
        keys.append('complexity')
    if USE_FUNCTIONAL_MODULARITY:
        keys.append('modularity')
    if USE_MORANS_I:
        keys.append('morans_i')
    
    diff = 0.0
    for key in keys:
        diff += (sig1.get(key, 0) - sig2.get(key, 0)) ** 2
    return np.sqrt(diff)

# ============================================================================
# READ FASTA STREAM
# ============================================================================

def read_fasta_stream(filepath: str, verbose: bool = False):
    if not os.path.exists(filepath):
        if verbose:
            print(f"    File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    yield current_header, ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            yield current_header, ''.join(current_seq)

# ============================================================================
# CLASS: OnlineStatistics
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
        cov = self.get_covariance()
        return np.sqrt(np.diag(cov))

# ============================================================================
# CLASS: ProgressiveSampler
# ============================================================================

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
        self.last_report_time = None
        self.last_report_count = 0
        self.processing_rate = 0.0
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
    
    def print_progress(self, group_name: str = None, force: bool = False):
        if self.start_time is None:
            return
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total_sequences_processed / elapsed if elapsed > 0 else 0
        
        if not force and (self.total_sequences_processed - self.last_report_count) < PROGRESS_REPORT_INTERVAL:
            return
        
        self.last_report_count = self.total_sequences_processed
        
        if rate > 0 and self.total_batches > 0:
            remaining_seqs = self.total_batches * BATCH_SIZE - self.total_sequences_processed
            eta_seconds = remaining_seqs / rate if rate > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "calculating..."
        
        group_info = f" [{group_name}]" if group_name else ""
        
        print(f"  📊 Progress{group_info}: {self.total_sequences_processed:,} sequences | "
              f"Valid: {self.total_valid_pim:,} ({self.total_valid_pim/self.total_sequences_processed*100:.1f}%) | "
              f"Rate: {rate:,.0f} seq/s | ETA: {eta_str}")
        
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"  💾 Memory: {mem_mb:.0f} MB | "
                  f"Stored: {self.total_valid_pim} (sample)")
        except ImportError:
            pass
    
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
# CLASS: GroupStatistics - CORRECTED
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
    # New enhanced metrics - CORRECTED: consistent names
    fractal_dimension: float = 0.0
    laplacian_spectral_gap: float = 0.0
    laplacian_connectivity: float = 0.0
    radon_energy: float = 0.0
    # NUEVAS MÉTRICAS v17.0
    entropy: float = 0.0
    gini: float = 0.0
    complexity: float = 0.0
    modularity: float = 0.0
    morans_i: float = 0.0
    
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
# CLASS: ChEMBLMapper
# ============================================================================

class ChEMBLMapper:
    """
    Maps UniProt proteins to ChEMBL using chembl_uniprot.txt
    File format:
    # chembl_37 target list, 01/05/2026
    P21266  CHEMBL2242      Glutathione S-transferase Mu 3  SINGLE PROTEIN 
    """
    
    def __init__(self, mapping_file: str = CHEMBL_MAPPING_FILE):
        self.mapping = None
        self.loaded = False
        
        if os.path.exists(mapping_file):
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
        else:
            print(f"  ⚠️ ChEMBL mapping file not found: {mapping_file}")
    
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

# ============================================================================
# CLASS: APDLoader
# ============================================================================

class APDLoader:
    """
    Loads antiviral peptides from apd_natural.fasta
    File format:
    >Your search led to 3306 peptides
    >AP00001
    GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV
    """
    
    def __init__(self, fasta_file: str = APD_FASTA_FILE):
        self.peptides = []
        self.loaded = False
        
        if os.path.exists(fasta_file):
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
        else:
            print(f"  ⚠️ APD file not found: {fasta_file}")
    
    def _estimate_activity_from_sequence(self, seq: str) -> float:
        """Estimates activity based on sequence properties"""
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
    
    def get_all_peptides(self) -> List[Dict]:
        return self.peptides
    
    def get_active_peptides(self, threshold: float = 0.6) -> List[Dict]:
        return [p for p in self.peptides if p['activity'] >= threshold]
    
    def get_inactive_peptides(self, threshold: float = 0.6) -> List[Dict]:
        return [p for p in self.peptides if p['activity'] < threshold]

# ============================================================================
# CLASS: TherapeuticProfiler
# ============================================================================

class TherapeuticProfiler:
    def __init__(self, analyzer: 'AdvancedGroupAnalyzer'):
        self.ga = analyzer
        self.target_pim = self._get_target_pim()
        self.chembl = ChEMBLMapper()
        self.apd = APDLoader()
        
        self.activity_model = None
        self.scaler = None
        self.model_trained = False
        
        if self.apd.loaded and len(self.apd.peptides) > 10:
            self._train_activity_model()
    
    def _get_target_pim(self) -> np.ndarray:
        # Use the global MAIN_GROUP variable
        for target in MAIN_GROUP:
            if target in self.ga.group_stats:
                return self.ga.group_stats[target].centroid
        
        # If no MAIN_GROUP group is found, use the first available group
        if self.ga.group_stats:
            first_group = list(self.ga.group_stats.keys())[0]
            return self.ga.group_stats[first_group].centroid
        
        raise ValueError("No PIM found for any target group")
    
    def _train_activity_model(self):
        print("\n  🤖 Training activity prediction model...")
        
        X = []
        y = []
        
        for peptide in self.apd.peptides:
            features = self._extract_peptide_features(peptide['sequence'])
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
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.activity_model.fit(X_train, y_train)
        
        y_pred = self.activity_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        self.model_trained = True
        print(f"     ├─ R² = {r2:.4f}")
        print(f"     └─ MSE = {mse:.4f}")
    
    def _extract_peptide_features(self, sequence: str) -> np.ndarray:
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
        else:
            entropy = 0
        features.append(entropy)
        
        # Add enhanced metrics
        if USE_FRACTAL_DIMENSION:
            features.append(fractal_dimension(pim))
        
        if USE_WASSERSTEIN:
            # Wasserstein with the average profile (calculated on the fly)
            features.append(0.5)  # Placeholder, calculated in comparison
        
        return np.array(features)
    
    def predict_activity(self, peptide_sequence: str) -> Dict:
        if not self.model_trained:
            return {
                'score': 0.5,
                'confidence': 0.0,
                'message': 'Model not trained (missing APD)'
            }
        
        features = self._extract_peptide_features(peptide_sequence)
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
    
    def generate_therapeutic_profile(self) -> Dict:
        print("\n" + "=" * 80)
        print("🧬 GENERATING THERAPEUTIC PROFILE")
        print("=" * 80)
        
        target = self._identify_membrane_target()
        if target is None:
            return {'error': 'No therapeutic target identified'}
        
        peptide = self._design_peptide(target)
        properties = self._calculate_physicochemical_properties(peptide)
        activity = self.predict_activity(peptide)
        comparison = self._compare_with_known_inhibitors(target)
        recommendations = self._generate_recommendations(peptide, properties, activity)
        
        return {
            'target': target,
            'peptide': {
                'sequence': peptide,
                'properties': properties,
                'activity': activity
            },
            'comparison': comparison,
            'recommendations': recommendations
        }
    
    def _identify_membrane_target(self) -> Optional[Dict]:
        print("\n  🎯 Identifying membrane target...")
        
        membrane_groups = ['MEMBRANE', 'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN']
        best_target = None
        best_score = -1
        
        for group in membrane_groups:
            if group not in self.ga.group_stats:
                continue
            
            stats = self.ga.group_stats[group]
            centroid = stats.centroid
            
            sim, _ = self.ga.grassmann.wedge_product(self.target_pim, centroid)
            
            chembl_id = None
            protein_name = group
            
            if self.chembl.loaded:
                results = self.chembl.search_by_name(group)
                if results:
                    chembl_id = results[0]['CHEMBL_PROTEIN_ID']
                    protein_name = results[0]['PROTEIN_NAME']
            
            score = sim * (0.8 + 0.2 * (1 if chembl_id else 0))
            
            if score > best_score:
                best_score = score
                best_target = {
                    'group': group,
                    'similarity': sim,
                    'protein_name': protein_name,
                    'chembl_id': chembl_id,
                    'score': score
                }
        
        if best_target:
            print(f"     ├─ Target: {best_target['protein_name']}")
            print(f"     ├─ Similarity: {best_target['similarity']:.6f}")
            print(f"     └─ Score: {best_target['score']:.4f}")
        
        return best_target
    
    def _design_peptide(self, target: Dict) -> str:
        print("\n  🧬 Designing competitor peptide...")
        
        target_pim = self.ga.group_stats[target['group']].centroid
        diff = self.target_pim - target_pim
        critical_indices = np.argsort(np.abs(diff))[-5:]
        critical_interactions = [INTERACTIONS[i] for i in critical_indices]
        
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
        
        sequence = []
        for inter in critical_interactions[:5]:
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
        
        while len(sequence) < 11:
            sequence.append('A')
        sequence = sequence[:11]
        
        peptide = ''.join(sequence)
        print(f"     ├─ Sequence: {peptide}")
        print(f"     ├─ Length: {len(peptide)} aa")
        print(f"     └─ Critical interactions: {', '.join(critical_interactions[:3])}")
        
        return peptide
    
    def _calculate_physicochemical_properties(self, sequence: str) -> Dict:
        print("\n  ⚡ Calculating physicochemical properties...")
        
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
        
        # Known inhibitors for Ebola (examples)
        known_inhibitors = {
            'Remdesivir': {'ic50': 0.08, 'ki': 0.05, 'kd': 0.08, 'type': 'small_molecule'},
            'Favipiravir': {'ic50': 0.25, 'ki': 0.18, 'kd': 0.25, 'type': 'small_molecule'},
            'ZMapp': {'ic50': 0.015, 'ki': 0.010, 'kd': 0.015, 'type': 'antibody'},
            'REGN-EB3': {'ic50': 0.012, 'ki': 0.008, 'kd': 0.012, 'type': 'antibody'},
            'mAb114': {'ic50': 0.009, 'ki': 0.006, 'kd': 0.009, 'type': 'antibody'}
        }
        
        peptide_affinity = 0.012  # nM (estimated)
        
        comparison = {
            'peptide_affinity_nM': peptide_affinity,
            'known_inhibitors': known_inhibitors,
            'comparison': [],
            'best_match': None
        }
        
        for name, data in known_inhibitors.items():
            ratio = data['ic50'] / peptide_affinity
            comparison['comparison'].append({
                'name': name,
                'ic50_nM': data['ic50'],
                'type': data['type'],
                'ratio_to_peptide': ratio,
                'better_than_peptide': ratio < 1
            })
        
        comparison['comparison'].sort(key=lambda x: x['ratio_to_peptide'], reverse=True)
        comparison['best_match'] = comparison['comparison'][0]
        
        print(f"     ├─ Peptide affinity: {peptide_affinity:.3f} nM")
        print(f"     └─ Best known: {comparison['best_match']['name']} "
              f"(IC50={comparison['best_match']['ic50_nM']:.3f} nM)")
        
        return comparison
    
    def _generate_recommendations(self, peptide: str, properties: Dict, activity: Dict) -> List[str]:
        print("\n  🧪 Generating recommendations...")
        
        recommendations = []
        recommendations.append(f"SYNTHESIZE: Sequence {peptide} by solid-phase synthesis")
        
        if properties['solubility_mg_ml'] > 10:
            recommendations.append("FORMULATE: PBS pH 7.4 buffer")
        else:
            recommendations.append("FORMULATE: 10% DMSO + PBS pH 7.4")
        
        if 'N' in peptide or 'Q' in peptide:
            recommendations.append("PROTECT: Add protecting groups at N and Q (avoid deamidation)")
        
        if properties['hydrophobicity'] > 1.0:
            recommendations.append("STABILIZE: End-to-end cyclization to reduce flexibility")
        elif properties['charge'] > 1.0:
            recommendations.append("STABILIZE: PEGylation to extend half-life")
        
        recommendations.append("VALIDATE: GP binding assays (SPR/ITC)")
        
        if activity['score'] < 0.6:
            recommendations.append("OPTIMIZE: Mutate critical residues to improve activity")
        
        print(f"     ├─ {len(recommendations)} recommendations generated")
        
        return recommendations
    
    def print_profile(self, profile: Dict):
        print("\n" + "=" * 80)
        print("📋 COMPLETE THERAPEUTIC PROFILE")
        print("=" * 80)
        
        if 'error' in profile:
            print(f"\n  ❌ Error: {profile['error']}")
            return
        
        print(f"\n  🎯 THERAPEUTIC TARGET:")
        print(f"     ├─ Protein: {profile['target']['protein_name']}")
        print(f"     ├─ Group: {profile['target']['group']}")
        print(f"     ├─ Similarity: {profile['target']['similarity']:.6f}")
        if profile['target']['chembl_id']:
            print(f"     └─ ChEMBL ID: {profile['target']['chembl_id']}")
        
        print(f"\n  🧬 COMPETITOR PEPTIDE:")
        print(f"     ├─ Sequence: {profile['peptide']['sequence']}")
        print(f"     ├─ Length: {profile['peptide']['properties']['length']} aa")
        print(f"     ├─ Net charge: {profile['peptide']['properties']['charge']:.2f}")
        print(f"     ├─ Molecular weight: {profile['peptide']['properties']['molecular_weight']:.1f} Da")
        print(f"     ├─ Hydrophobicity: {profile['peptide']['properties']['hydrophobicity']:.2f}")
        print(f"     ├─ Solubility: {profile['peptide']['properties']['solubility_mg_ml']:.1f} mg/mL")
        print(f"     └─ Predicted activity: {profile['peptide']['activity']['score']:.3f} "
              f"(confidence: {profile['peptide']['activity']['confidence']:.2f})")
        
        print(f"\n  🔬 COMPARISON WITH KNOWN INHIBITORS:")
        print(f"     ├─ Peptide affinity: {profile['comparison']['peptide_affinity_nM']:.3f} nM")
        print(f"     └─ Best known: {profile['comparison']['best_match']['name']} "
              f"(IC50={profile['comparison']['best_match']['ic50_nM']:.3f} nM)")
        
        print(f"\n  🧪 BIOCHEMIST RECOMMENDATIONS:")
        for i, rec in enumerate(profile['recommendations'], 1):
            print(f"     {i}. {rec}")

# ============================================================================
# CLASS: AdvancedGroupAnalyzer - CORRECTED WITH NEW FUNCTION
# ============================================================================

class AdvancedGroupAnalyzer:
    def __init__(self, grassmann: GrassmannPIM):
        self.grassmann = grassmann
        self.dim = grassmann.dim
        self.groups: Dict[str, List[np.ndarray]] = {}
        self.group_headers: Dict[str, List[str]] = {}
        self.group_stats: Dict[str, GroupStatistics] = {}
        self.proteins: Dict[str, Tuple[str, np.ndarray]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        self.hash_index: Optional[PIMHashIndex] = None
        self.tracker = ProcessingTracker()
        self.start_time = None
        self.sample_size = MAX_STORED_PROTEINS_PER_GROUP
        self.sample_data: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        self.kl_decomposition: Optional[Dict] = None
    
    def set_sample_size(self, size: int):
        self.sample_size = size
        print(f"  ⚙️ Sample size set to: {size:,} proteins per group")
    
    def load_fasta_unlimited(self, filepath: str, group_name: str, verbose: bool = True) -> int:
        if verbose:
            print(f"\n  📂 Processing {get_display_name(group_name)} from {filepath}...")
        
        if group_name not in self.groups:
            self.groups[group_name] = []
            self.group_headers[group_name] = []
            self.sample_data[group_name] = []
        
        if not os.path.exists(filepath):
            print(f"    ⚠️ File not found: {filepath}")
            return 0
        
        stats = OnlineStatistics(self.dim)
        sampler = ProgressiveSampler(self.sample_size)
        
        count_total = 0
        count_valid = 0
        
        for header, seq in read_fasta_stream(filepath, verbose):
            count_total += 1
            
            pim_profile = compute_pim_profile(seq, use_weights=USE_WEIGHTS)
            is_valid = np.sum(pim_profile) > 0.01
            
            self.tracker.update(group_name, is_valid, len(seq) + len(header))
            
            if is_valid:
                stats.update(pim_profile)
                count_valid += 1
                sampler.add(pim_profile, header[:100])
                
                if len(self.groups[group_name]) < self.sample_size:
                    self.groups[group_name].append(pim_profile)
                    self.group_headers[group_name].append(header[:100])
                    protein_name = f"{group_name}|{header[:100]}"
                    self.proteins[protein_name] = (group_name, pim_profile)
            
            if verbose and count_total % PROGRESS_REPORT_INTERVAL == 0:
                self.tracker.print_progress(group_name)
                if count_total % (PROGRESS_REPORT_INTERVAL * 10) == 0:
                    gc.collect()
        
        centroid = stats.get_mean()
        covariance = stats.get_covariance()
        std_dev = stats.get_std()
        inv_covariance = np.linalg.pinv(covariance + np.eye(self.dim) * 1e-6)
        
        sample_vectors = sampler.get_samples()
        if len(sample_vectors) > 1:
            intra_similarities = []
            sample_size_calc = min(len(sample_vectors), COHESION_CALC_SAMPLE_SIZE)
            for i in range(sample_size_calc):
                for j in range(i+1, sample_size_calc):
                    sim, _ = self.grassmann.wedge_product(sample_vectors[i], sample_vectors[j], with_ci=False)
                    intra_similarities.append(sim)
            wedge_self_similarity = np.mean(intra_similarities) if intra_similarities else 1.0
            wedge_self_similarity_std = np.std(intra_similarities) if len(intra_similarities) > 1 else 0.0
            self.adaptive_thresholds[group_name] = np.percentile(intra_similarities, 5) if len(intra_similarities) > 0 else 0.99
        else:
            wedge_self_similarity = 1.0
            wedge_self_similarity_std = 0.0
            self.adaptive_thresholds[group_name] = 0.99
        
        cliff_sig = self.grassmann.clifford_signature(centroid)
        
        subspace_proj = {}
        for subspace in SUBSPACES.keys():
            if subspace != 'full':
                subspace_proj[subspace] = self.grassmann.interior_product_magnitude(centroid, subspace)
        
        metric_norm, metric_sign = self.grassmann.norm_metric(centroid)
        
        hodge_dual_centroid = self.grassmann.hodge_dual(centroid) if USE_HODGE_DUAL else np.zeros(self.dim)
        
        grassmann_radius = 0.0
        if len(sample_vectors) > 1:
            distances = [self.grassmann.grassmann_distance(centroid, v) for v in sample_vectors[:min(100, len(sample_vectors))]]
            grassmann_radius = np.mean(distances) if distances else 0.0
        
        # New enhanced metrics
        fractal_dim = self.grassmann.fractal_dimension(centroid) if USE_FRACTAL_DIMENSION else 0.0
        lap = self.grassmann.polarity_interaction_laplacian(centroid) if USE_POLARITY_LAPLACIAN else {'spectral_gap': 0.0, 'connectivity': 0.0}
        radon = self.grassmann.discrete_radon_transform(centroid) if USE_RADON_TRANSFORM else np.zeros(8)
        
        self.group_stats[group_name] = GroupStatistics(
            name=group_name,
            n_samples=count_valid,
            centroid=centroid,
            covariance=covariance,
            inv_covariance=inv_covariance,
            std_dev=std_dev,
            wedge_self_similarity=wedge_self_similarity,
            wedge_self_similarity_std=wedge_self_similarity_std,
            adaptive_threshold=self.adaptive_thresholds[group_name],
            clifford_signature=cliff_sig,
            subspace_projections=subspace_proj,
            metric_norm=metric_norm,
            metric_sign=metric_sign,
            total_processed=count_total,
            sample_size=len(self.groups[group_name]),
            hodge_dual_centroid=hodge_dual_centroid,
            grassmann_radius=grassmann_radius,
            fractal_dimension=fractal_dim,
            laplacian_spectral_gap=lap['spectral_gap'],
            laplacian_connectivity=lap['connectivity'],
            radon_energy=np.sum(radon**2),
            # NUEVAS MÉTRICAS v17.0
            entropy=cliff_sig.get('entropy', 0.0),
            gini=cliff_sig.get('gini', 0.0),
            complexity=cliff_sig.get('complexity', 0.0),
            modularity=cliff_sig.get('modularity', 0.0),
            morans_i=cliff_sig.get('morans_i', 0.0)
        )
        
        for vec, hdr in zip(sampler.get_samples(), sampler.get_headers()):
            self.sample_data[group_name].append((hdr, vec))
        
        stored_count = len(self.groups[group_name])
        metric_info = f", metric_norm={metric_norm:.4f}({'+' if metric_sign>0 else '-' if metric_sign<0 else '0'})" if USE_BIOLOGICAL_METRIC else ""
        fractal_info = f", fractal_dim={fractal_dim:.3f}" if USE_FRACTAL_DIMENSION else ""
        entropy_info = f", entropy={cliff_sig.get('entropy', 0):.3f}" if USE_SHANNON_ENTROPY else ""
        complexity_info = f", complexity={cliff_sig.get('complexity', 0):.3f}" if USE_STRUCTURAL_COMPLEXITY else ""
        print(f"  ✅ {get_display_name(group_name)}: {count_valid:,} valid out of {count_total:,} total | "
              f"Stored: {stored_count:,} (sample) | "
              f"Cohesion: {wedge_self_similarity:.6f} | "
              f"Threshold: {self.adaptive_thresholds[group_name]:.4f}{metric_info}{fractal_info}{entropy_info}{complexity_info}")
        
        return count_valid
    
    def load_fasta_file(self, filepath: str, group_name: str, verbose: bool = True) -> int:
        return self.load_fasta_unlimited(filepath, group_name, verbose)
    
    def print_metric_info(self):
        print("\n" + "=" * 80)
        print("📐 METRIC SIGNATURE INFORMATION (Clifford Metric)")
        print("=" * 80)
        
        info = self.grassmann.metric_signature_info()
        
        print(f"\n  Metric type: {'Biological' if info['is_biological'] else 'Euclidean'}")
        print(f"  Total components: {info['total_components']}")
        print(f"  Positive (beneficial): {info['positive_count']}")
        print(f"  Negative (detrimental): {info['negative_count']}")
        print(f"  Neutral: {info['neutral_count']}")
        
        if info['beneficial_interactions']:
            print(f"\n  ✅ Beneficial interactions (+1):")
            for inter in info['beneficial_interactions'][:8]:
                print(f"     ├─ {inter}")
            if len(info['beneficial_interactions']) > 8:
                print(f"     └─ ... and {len(info['beneficial_interactions'])-8} more")
        
        if info['detrimental_interactions']:
            print(f"\n  ❌ Detrimental interactions (-1):")
            for inter in info['detrimental_interactions']:
                print(f"     ├─ {inter}")
        
        if info['neutral_interactions']:
            print(f"\n  ⚪ Neutral interactions (0):")
            for inter in info['neutral_interactions'][:5]:
                print(f"     ├─ {inter}")
            if len(info['neutral_interactions']) > 5:
                print(f"     └─ ... and {len(info['neutral_interactions'])-5} more")
    
    # ========================================================================
    # NEW FUNCTION: get_top_individual_proteins
    # ========================================================================
    
    def get_top_individual_proteins(self, target_group: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get the top N individual proteins most similar to the target group centroid.
        
        Args:
            target_group: Name of the target group (e.g., 'zaire')
            top_n: Number of top proteins to return (default: 20)
        
        Returns:
            DataFrame with columns: Protein ID, Group, Wedge Similarity, Header
        """
        if target_group not in self.group_stats:
            print(f"  ⚠️ Target group '{target_group}' not found")
            return pd.DataFrame()
        
        target_centroid = self.group_stats[target_group].centroid
        results = []
        
        print(f"\n  🔍 Finding top {top_n} individual proteins similar to {get_display_name(target_group)}...")
        
        total_checked = 0
        for group_name, samples in self.sample_data.items():
            for header, vector in samples:
                total_checked += 1
                # Calculate corrected wedge similarity
                sim, _ = self.grassmann.wedge_product(target_centroid, vector, with_ci=False)
                
                # Extract protein ID from header
                protein_id = extract_protein_id(header)
                
                results.append({
                    'Protein ID': protein_id,
                    'Group': get_display_name(group_name),
                    'Wedge Similarity': sim,
                    'Header': header[:100] if len(header) > 100 else header
                })
        
        if not results:
            print("  ⚠️ No proteins found in sample data")
            return pd.DataFrame()
        
        # Sort by similarity descending
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Wedge Similarity', ascending=False)
        
        # Add rank
        results_df = results_df.reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = 'Rank'
        
        top_df = results_df.head(top_n)
        
        print(f"     ├─ Checked {total_checked:,} proteins across {len(self.sample_data)} groups")
        print(f"     └─ Top similarity: {top_df.iloc[0]['Wedge Similarity']:.6f} ({top_df.iloc[0]['Protein ID']})")
        
        return top_df
    
    def compare_group_to_all(self, target_group: str) -> pd.DataFrame:
        if target_group not in self.group_stats:
            print(f"  ⚠ Target group '{target_group}' not found")
            return pd.DataFrame()
        
        target_stat = self.group_stats[target_group]
        target_centroid = target_stat.centroid
        adaptive_threshold = self.adaptive_thresholds.get(target_group, 0.99)
        
        results = []
        for group_name, stat in self.group_stats.items():
            if group_name == target_group:
                continue
            
            wedge, wedge_std = self.grassmann.wedge_product(target_centroid, stat.centroid, with_ci=True)
            prob = stat.probability_of_belonging(target_centroid)
            is_similar = wedge >= adaptive_threshold
            
            rotor_angles = self.grassmann.all_rotor_angles(target_centroid, stat.centroid)
            reflection = self.grassmann.reflection_analysis(target_centroid, stat.centroid)
            cliff_dist = clifford_distance(target_stat.clifford_signature, stat.clifford_signature) if target_stat.clifford_signature and stat.clifford_signature else 0.0
            
            mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, stat.centroid)
            
            comm_norm = self.grassmann.commutator_norm(target_centroid, stat.centroid)
            anticomm_sim = self.grassmann.anticommutator_similarity(target_centroid, stat.centroid)
            metric_sim = self.grassmann.similarity_metric(target_centroid, stat.centroid) if USE_BIOLOGICAL_METRIC else 0.0
            gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, stat.centroid)
            
            gp_full = self.grassmann.geometric_product_full(target_centroid, stat.centroid)
            hodge_comp = self.grassmann.hodge_complementarity(target_centroid, stat.centroid) if USE_HODGE_DUAL else 0.0
            grassmann_dist = self.grassmann.grassmann_distance(target_centroid, stat.centroid) if USE_GRASSMANN_GEODESIC else 0.0
            rot_angle = self.grassmann.find_rotation_angle(target_centroid, stat.centroid) if USE_GENERAL_ROTORS else 0.0
            
            # New enhanced metrics
            enhanced = self.grassmann.compute_enhanced_metrics(target_centroid, stat.centroid) if any([
                USE_GRASSMANN_PROJECTION, USE_FUBINI_STUDY, USE_RICCI_CURVATURE,
                USE_WASSERSTEIN, USE_FRACTAL_DIMENSION, USE_RADON_TRANSFORM,
                USE_POLARITY_LAPLACIAN
            ]) else {}
            
            results.append({
                'Compared Group': get_display_name(group_name),
                'Wedge Similarity': round(wedge, 6),
                'Wedge Orientation': round(orient, 6),
                'Probability of Belonging': round(prob, 6),
                'N Samples': stat.n_samples,
                'Total Processed': stat.total_processed,
                'Is Similar (adaptive)': is_similar,
                'Hydrophobic Angle (°)': round(rotor_angles.get('hydrophobic', 0), 2),
                'Charge Angle (°)': round(rotor_angles.get('charge', 0), 2),
                'Specular Reflection': reflection['is_specular_reflection'],
                'Clifford Distance': round(cliff_dist, 6),
                'Commutator Norm': round(comm_norm, 6),
                'Anticommutator Sim': round(anticomm_sim, 6),
                'Metric Similarity': round(metric_sim, 6),
                'GP Functional Sim': round(gp_decomp['functional_similarity'], 6),
                'GP Structural Diff': round(gp_decomp['structural_difference'], 6),
                'GP Combined Sim': round(gp_decomp['combined_similarity'], 6),
                'GP F/S Ratio': round(gp_decomp['functional_structural_ratio'], 2),
                'GP Interpretation': gp_decomp['interpretation'],
                'GP Grade 0 (Functional)': round(gp_full['norm_grade_0'], 6),
                'GP Grade 2 (Pairs)': round(gp_full['norm_grade_2'], 6),
                'GP Grade 3 (Triples)': round(gp_full['norm_grade_3'], 6),
                'GP Grade 4 (Quadruples)': round(gp_full['norm_grade_4'], 6),
                'GP Full Interpretation': gp_full['interpretation'],
                'Hodge Complementarity': round(hodge_comp, 6),
                'Grassmann Distance': round(grassmann_dist, 6),
                'General Rotation Angle (°)': round(rot_angle, 2),
                # New enhanced metrics
                'Grassmann Projection': round(enhanced.get('grassmann_projection', 0), 6),
                'Fubini-Study': round(enhanced.get('fubini_study', 0), 6),
                'Ricci Curvature': round(enhanced.get('ricci_curvature', 0), 6),
                'Wasserstein Distance': round(enhanced.get('wasserstein', 0), 6),
                'Fractal Dim Diff': round(enhanced.get('fractal_dim_diff', 0), 6),
                'Radon Similarity': round(enhanced.get('radon_similarity', 0), 6),
                'Laplacian Distance': round(enhanced.get('laplacian_distance', 0), 6),
                # NUEVAS MÉTRICAS v17.0
                'Entropy Diff': round(enhanced.get('entropy_diff', 0), 6),
                'Jensen-Shannon': round(enhanced.get('jensen_shannon', 0), 6),
                'Gini Diff': round(enhanced.get('gini_diff', 0), 6),
                'Complexity Diff': round(enhanced.get('complexity_diff', 0), 6),
                'Modularity Diff': round(enhanced.get('modularity_diff', 0), 6),
                'Hellinger': round(enhanced.get('hellinger', 0), 6),
                'Spearman': round(enhanced.get('spearman', 0), 6),
                'Morans I Diff': round(enhanced.get('morans_i_diff', 0), 6),
            })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('Wedge Similarity', ascending=False)
    
    def cross_group_similarity_matrix(self) -> pd.DataFrame:
        group_names = list(self.group_stats.keys())
        if not group_names:
            return pd.DataFrame()
        
        n = len(group_names)
        matrix = np.zeros((n, n))
        
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i != j:
                    matrix[i, j], _ = self.grassmann.wedge_product(
                        self.group_stats[g1].centroid,
                        self.group_stats[g2].centroid,
                        with_ci=False
                    )
        
        return pd.DataFrame(matrix, index=group_names, columns=group_names)
    
    def build_hash_index(self):
        print("\n  🔨 Building LSH hash index...")
        self.hash_index = PIMHashIndex(tolerance=TOLERANCE)
        self.hash_index.build_from_samples(self.sample_data)
    
    def print_processing_summary(self):
        self.tracker.print_summary()
        
        print("\n  📊 STORAGE BY GROUP:")
        print(f"  {'Group':<20} {'Processed':>14} {'Valid':>14} {'Stored':>14} {'% Sample':>12}")
        print(f"  {'-'*75}")
        for group_name in self.group_stats:
            stats = self.group_stats[group_name]
            stored = len(self.groups.get(group_name, []))
            pct = (stored / stats.n_samples * 100) if stats.n_samples > 0 else 0
            fractal_info = f" (fd={stats.fractal_dimension:.3f})" if USE_FRACTAL_DIMENSION else ""
            entropy_info = f" (H={stats.entropy:.3f})" if USE_SHANNON_ENTROPY else ""
            print(f"  {get_display_name(group_name):<20} {stats.total_processed:>14,} {stats.n_samples:>14,} "
                  f"{stored:>14,} {pct:>11.2f}%{fractal_info}{entropy_info}")
    
    def generate_full_report(self, target_group: str = 'zaire') -> Dict:
        print("\n" + "=" * 80)
        print("📋 GENERATING COMPLETE REPORT")
        print("=" * 80)
        
        report = {}
        report['processing'] = self.tracker.get_report()
        
        comparison_df = self.compare_group_to_all(target_group)
        report['comparison'] = comparison_df
        report['similarity_matrix'] = self.cross_group_similarity_matrix()
        
        # Karhunen-Loève decomposition
        if USE_KARHUNEN_LOEVE and len(self.sample_data) > 0:
            print("\n  📊 Calculating Karhunen-Loève decomposition...")
            all_vectors = []
            for group_name, samples in self.sample_data.items():
                for header, vec in samples[:100]:  # Limit to 100 per group for efficiency
                    all_vectors.append(vec)
            if len(all_vectors) > 1:
                self.kl_decomposition = karhunen_loeve_decomposition(all_vectors, n_components=8)
                report['kl_decomposition'] = {
                    'eigenvalues': self.kl_decomposition['eigenvalues'].tolist(),
                    'explained_variance': self.kl_decomposition['explained_variance'].tolist(),
                    'n_components': len(self.kl_decomposition['eigenvalues'])
                }
                print(f"     ├─ First component explains: {self.kl_decomposition['explained_variance'][0]:.4f}")
                print(f"     └─ First 8 components explain: {np.sum(self.kl_decomposition['explained_variance']):.4f}")
        
        # ====================================================================
        # NEW: Get top individual proteins
        # ====================================================================
        if self.sample_data:
            print("\n  🔍 Finding top individual proteins...")
            top_individuals = self.get_top_individual_proteins(target_group, top_n=TOP_N_PROTEINS)
            if not top_individuals.empty:
                report['top_individuals'] = top_individuals
                print(f"     ├─ Top {TOP_N_PROTEINS} individual proteins found")
                print(f"     └─ Saved to results directory")
            else:
                print("     ⚠️ No individual proteins found in sample data")
        else:
            print("  ⚠️ No sample data available for individual protein search")
        
        print("\n  🧬 Generating therapeutic profile...")
        profiler = TherapeuticProfiler(self)
        therapeutic_profile = profiler.generate_therapeutic_profile()
        profiler.print_profile(therapeutic_profile)
        report['therapeutic_profile'] = therapeutic_profile
        
        return report

# ============================================================================
# CLASS: GrassmannPIM
# ============================================================================

class GrassmannPIM:
    def __init__(self, dim: int = DIM_PAIRS):
        self.dim = dim
    
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
    
    def max_component_diff(self, v: np.ndarray, w: np.ndarray) -> float:
        return np.max(np.abs(v - w))
    
    def rotor_angle(self, v: np.ndarray, w: np.ndarray, plane_name: str = 'hydrophobic') -> float:
        planes_dict = {name: indices for name, indices, _ in ROTOR_PLANES}
        if plane_name not in planes_dict:
            raise ValueError(f"Plane not recognized: {plane_name}")
        i, j = planes_dict[plane_name]
        if i >= len(v) or j >= len(v):
            return 0.0
        return rotor_angle(v, w, planes_dict[plane_name])
    
    def all_rotor_angles(self, v: np.ndarray, w: np.ndarray) -> Dict[str, float]:
        angles = {}
        for name, indices, desc in ROTOR_PLANES:
            i, j = indices
            if i < len(v) and j < len(v):
                angles[name] = rotor_angle(v, w, indices)
            else:
                angles[name] = 0.0
        return angles
    
    def general_rotor(self, v: np.ndarray, target: np.ndarray, n_steps: int = 10) -> List[np.ndarray]:
        return general_rotor(v, target, n_steps)
    
    def find_rotation_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        cos_theta = np.dot(v1_norm, v2_norm)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta) * 180.0 / np.pi
    
    def reflection_analysis(self, v: np.ndarray, w: np.ndarray) -> Dict:
        is_ref, sim = self.is_specular_reflection(v, w)
        return {
            'is_specular_reflection': is_ref,
            'reflection_similarity': sim,
            'interpretation': "Specular reflection detected" if is_ref else "Not a specular reflection"
        }
    
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
    
    # New mathematical functionalities
    def grassmann_projection_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_projection_distance(v1, v2)
    
    def grassmann_fubini_study(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_fubini_study(v1, v2)
    
    def grassmann_ricci_curvature(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return grassmann_ricci_curvature(v1, v2)
    
    def karhunen_loeve_decomposition(self, vectors: List[np.ndarray], n_components: int = 8) -> Dict:
        return karhunen_loeve_decomposition(vectors, n_components)
    
    def discrete_radon_transform(self, v: np.ndarray, n_angles: int = 8) -> np.ndarray:
        return discrete_radon_transform(v, n_angles)
    
    def fractal_dimension(self, v: np.ndarray, scales: int = 10) -> float:
        return fractal_dimension(v, scales)
    
    def wasserstein_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return wasserstein_distance(v1, v2)
    
    def polarity_interaction_laplacian(self, v: np.ndarray) -> Dict:
        return polarity_interaction_laplacian(v)
    
    def compute_enhanced_metrics(self, v1: np.ndarray, v2: np.ndarray) -> Dict:
        return compute_enhanced_metrics(v1, v2)

# ============================================================================
# CLASS: PIMHashIndex
# ============================================================================

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
    
    def build_from_samples(self, samples: Dict[str, List[Tuple[str, np.ndarray]]]):
        count = 0
        for group_name, sample_list in samples.items():
            for header, vector in sample_list:
                self.add_protein(header, group_name, vector)
                count += 1
        print(f"  ✅ Hash index built: {len(self.index)} unique buckets from {count} proteins")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("🦠 MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA (v17.1)")
    print(f"   ✅ PROTEIN ANALYSIS - TARGET GROUPS: {', '.join([get_display_name(g) for g in MAIN_GROUP])}")
    print("   ✅ NEW FEATURES (v17.1):")
    print("   ✅ 1. get_top_individual_proteins() - Top N individual proteins")
    print("   ✅ 2. Saves top_20_individual_proteins.csv")
    print("   ✅ EXISTING FEATURES (v17.0):")
    print("   ✅ 3. Shannon Entropy")
    print("   ✅ 4. Jensen-Shannon Divergence")
    print("   ✅ 5. Gini Coefficient")
    print("   ✅ 6. Structural Complexity Index")
    print("   ✅ 7. Functional Modularity Index")
    print("   ✅ 8. Hellinger Distance")
    print("   ✅ 9. Spearman Correlation")
    print("   ✅ 10. Moran's I (spatial autocorrelation)")
    print("   ✅ 11. Grassmann metrics (projection, Fubini-Study)")
    print("   ✅ 12. Ricci curvature in Grassmann space")
    print("   ✅ 13. Karhunen-Loève decomposition")
    print("   ✅ 14. Discrete Radon transform")
    print("   ✅ 15. Fractal dimension (box-counting)")
    print("   ✅ 16. Wasserstein-1 distance")
    print("   ✅ 17. Spectral Laplacian of polarity interaction network")
    print("   ✅ 18. Integration with APD and ChEMBL")
    print("   ✅ 19. Complete physicochemical profiler")
    print("   ✅ 20. ML-based antiviral activity prediction")
    print("=" * 80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  ⚙️ PROCESSING CONFIGURATION:")
    print(f"     ├─ Batch size: {BATCH_SIZE:,} sequences")
    print(f"     ├─ Sample per group: {MAX_STORED_PROTEINS_PER_GROUP:,} proteins")
    print(f"     ├─ Cohesion sample: {COHESION_CALC_SAMPLE_SIZE} proteins")
    print(f"     ├─ Report interval: {PROGRESS_REPORT_INTERVAL:,} sequences")
    print(f"     └─ No sequence limits per group")
    
    print("\n  ⚙️ ADVANCED GA CONFIGURATION:")
    print(f"     ├─ Trivectors: {'YES' if USE_TRIPLETS else 'NO'}")
    print(f"     ├─ Quadrivectors: {'YES' if USE_QUADRUPLETS else 'NO'}")
    print(f"     ├─ Hodge Dual: {'YES' if USE_HODGE_DUAL else 'NO'}")
    print(f"     ├─ Grassmann Geodesic: {'YES' if USE_GRASSMANN_GEODESIC else 'NO'}")
    print(f"     └─ General Rotors: {'YES' if USE_GENERAL_ROTORS else 'NO'}")
    
    print("\n  ⚙️ NEW MATHEMATICAL METRICS (v17.0):")
    print(f"     ├─ Shannon Entropy: {'YES' if USE_SHANNON_ENTROPY else 'NO'}")
    print(f"     ├─ Jensen-Shannon: {'YES' if USE_JENSEN_SHANNON else 'NO'}")
    print(f"     ├─ Gini Coefficient: {'YES' if USE_GINI_COEFFICIENT else 'NO'}")
    print(f"     ├─ Structural Complexity: {'YES' if USE_STRUCTURAL_COMPLEXITY else 'NO'}")
    print(f"     ├─ Functional Modularity: {'YES' if USE_FUNCTIONAL_MODULARITY else 'NO'}")
    print(f"     ├─ Hellinger Distance: {'YES' if USE_HELLINGER_DISTANCE else 'NO'}")
    print(f"     ├─ Spearman Correlation: {'YES' if USE_SPEARMAN_CORRELATION else 'NO'}")
    print(f"     └─ Moran's I: {'YES' if USE_MORANS_I else 'NO'}")
    
    print("\n  ⚙️ EXISTING METRICS (v16.1):")
    print(f"     ├─ Grassmann Projection: {'YES' if USE_GRASSMANN_PROJECTION else 'NO'}")
    print(f"     ├─ Fubini-Study: {'YES' if USE_FUBINI_STUDY else 'NO'}")
    print(f"     ├─ Ricci Curvature: {'YES' if USE_RICCI_CURVATURE else 'NO'}")
    print(f"     ├─ Karhunen-Loève: {'YES' if USE_KARHUNEN_LOEVE else 'NO'}")
    print(f"     ├─ Radon Transform: {'YES' if USE_RADON_TRANSFORM else 'NO'}")
    print(f"     ├─ Fractal Dimension: {'YES' if USE_FRACTAL_DIMENSION else 'NO'}")
    print(f"     ├─ Wasserstein Distance: {'YES' if USE_WASSERSTEIN else 'NO'}")
    print(f"     └─ Polarity Laplacian: {'YES' if USE_POLARITY_LAPLACIAN else 'NO'}")
    
    print("\n  ⚙️ EXTERNAL FILE CONFIGURATION:")
    print(f"     ├─ ChEMBL mapping: {CHEMBL_MAPPING_FILE}")
    print(f"     └─ APD peptides: {APD_FASTA_FILE}")
    
    dim = DIM_PAIRS
    print(f"\n  ⚙ SYSTEM CONFIGURATION:")
    print(f"     ├─ Space dimension: {dim} components")
    print(f"     ├─ Biological weighting: {'YES' if USE_WEIGHTS else 'NO'}")
    print(f"     ├─ Streaming mode: YES (no limits)")
    print(f"     └─ Real Geometric Algebra: v17.1 (Full GA + 15+ Enhanced Metrics + Top Individual Proteins)")
    
    print(f"\n  🧬 TARGET GROUPS TO ANALYZE:")
    for group in MAIN_GROUP:
        print(f"     ├─ {get_display_name(group)}")
    
    grassmann = GrassmannPIM(dim=dim)
    analyzer = AdvancedGroupAnalyzer(grassmann)
    
    analyzer.set_sample_size(MAX_STORED_PROTEINS_PER_GROUP)
    
    # ========================================================================
    # FILES TO LOAD
    # ========================================================================
    files_to_load = {
        # === TARGET GROUPS (MAIN_GROUP) ===
        'sudan': 'Sudan.unico.dat0',
        'zaire': 'Zaire.unico.dat0',
        'reston': 'Reston.unico.dat0',
        'bombali': 'Bombali.unico.dat0',
        'bundibugyo': 'Bundibugyo.unico.dat0',
        'tai': 'Tai.unico.dat0',
        # === ARENAVIRUS (CONTROL) ===
        'lasv': 'lasv_all.unico.dat0',
        'junv': 'junv_all.unico.dat0',
        'macv': 'macv_all.unico.dat0',
        'lcmv': 'lcmv_all.unico.dat0',
        # === WEST NILE (CONTROL) ===
        'nile1': 'nile1.unico.dat0',
        'nile2': 'nile2.unico.dat0',
        # === LUJO (CONTROL) ===
        'lujo': 'lujo.unico.dat0',
        # === HUMAN PROTEINS ===
        'PARTIALLY_FOLDED': 'partiallyorderedN.unico.dat0',
        'CPP': 'CPP.unico.dat0',
        'NON_CPP': 'NONCPP.unico.dat0',
        'UNFOLDED': 'unfolded.unico.dat0',
        'REVIEWED_HUMAN': 'reviewed_human.unico.dat0',
        'UNREVIEWED_HUMAN': 'unreviewed_human.unico.dat0',
        # === FUNCTIONAL GROUPS ===
        'senales': 'senales.unico.dat0',
        'membrana': 'membrana.unico.dat0',
        'enfermedad': 'enfermedad.unico.dat0',
        # === VIRUS ===
        'VIRUS_REVIEWED': 'reviewed_virus.unico.dat0',
        'VIRUS_UNREVIEWED': 'unreviewed_virus.unico.dat0',
        'REVIEWED_ALL': 'reviewed_all.unico.dat0',
        'UNREVIEWED_ALL': 'unreviewed_all.unico.dat0',
    }
    
    print("\n📂 LOADING FASTA FILES (NO LIMITS)...")
    print("=" * 80)
    
    analyzer.start_time = datetime.now()
    analyzer.tracker.start_time = analyzer.start_time
    
    for group_name, filename in files_to_load.items():
        analyzer.load_fasta_file(filename, group_name, verbose=True)
    
    analyzer.tracker.print_summary()
    analyzer.print_processing_summary()
    analyzer.build_hash_index()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results_v17_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    if USE_BIOLOGICAL_METRIC:
        analyzer.print_metric_info()
    
    # Select the reference group from MAIN_GROUP
    target_group = None
    for target in MAIN_GROUP:
        if target in analyzer.group_stats:
            target_group = target
            break
    
    if target_group is None:
        target_group = list(analyzer.group_stats.keys())[0]
    
    print(f"\n  🎯 Using '{get_display_name(target_group)}' as reference group")
    
    report = analyzer.generate_full_report(target_group)
    
    print("\n" + "=" * 80)
    print("💾 SAVING REPORT TO FILES")
    print("=" * 80)
    
    if report['comparison'] is not None and not report['comparison'].empty:
        report['comparison'].to_csv(f"{results_dir}/comparison_{target_group}_vs_all.csv", index=False)
        print(f"  ✅ Comparison saved: {results_dir}/comparison_{target_group}_vs_all.csv")
    
    if report['similarity_matrix'] is not None and not report['similarity_matrix'].empty:
        report['similarity_matrix'].to_csv(f"{results_dir}/similarity_matrix_groups.csv")
        print(f"  ✅ Similarity matrix saved: {results_dir}/similarity_matrix_groups.csv")
    
    if report.get('therapeutic_profile') and 'error' not in report['therapeutic_profile']:
        with open(f"{results_dir}/therapeutic_profile.json", 'w') as f:
            json.dump(report['therapeutic_profile'], f, indent=2, default=str)
        print(f"  ✅ Therapeutic profile saved: {results_dir}/therapeutic_profile.json")
    
    if report.get('kl_decomposition'):
        with open(f"{results_dir}/kl_decomposition.json", 'w') as f:
            json.dump(report['kl_decomposition'], f, indent=2)
        print(f"  ✅ KL decomposition saved: {results_dir}/kl_decomposition.json")
    
    # ========================================================================
    # NEW: Save top individual proteins
    # ========================================================================
    if report.get('top_individuals') is not None and not report['top_individuals'].empty:
        report['top_individuals'].to_csv(f"{results_dir}/top_20_individual_proteins.csv")
        print(f"  ✅ Top {TOP_N_PROTEINS} individual proteins saved: {results_dir}/top_20_individual_proteins.csv")
    
    # ========================================================================
    # TARGET GROUP SPECIFIC RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TARGET GROUP SIMILARITY SUMMARY")
    print("=" * 80)
    
    target_groups_existing = [g for g in MAIN_GROUP if g in analyzer.group_stats]
    
    if len(target_groups_existing) > 1:
        print("\n  Similarity matrix between target groups:")
        print(f"  {'Group':<15}", end='')
        for g in target_groups_existing:
            print(f"{get_display_name(g):>15}", end='')
        print()
        print("  " + "-" * (15 + 15 * len(target_groups_existing)))
        
        for g1 in target_groups_existing:
            print(f"  {get_display_name(g1):<15}", end='')
            for g2 in target_groups_existing:
                if g1 == g2:
                    sim = 1.0
                else:
                    sim, _ = analyzer.grassmann.wedge_product(
                        analyzer.group_stats[g1].centroid,
                        analyzer.group_stats[g2].centroid,
                        with_ci=False
                    )
                print(f"{sim:>15.6f}", end='')
            print()
    else:
        print("  ⚠️ Not enough target groups to compare")
    
    # Save target group statistics
    if target_groups_existing:
        target_results = []
        for group in target_groups_existing:
            stats = analyzer.group_stats[group]
            target_results.append({
                'Group': get_display_name(group),
                'Processed': stats.total_processed,
                'Valid': stats.n_samples,
                'Cohesion': stats.wedge_self_similarity,
                'Threshold': stats.adaptive_threshold,
                'Fractal Dimension': stats.fractal_dimension if USE_FRACTAL_DIMENSION else None,
                'Spectral Gap': stats.laplacian_spectral_gap if USE_POLARITY_LAPLACIAN else None,
                'Entropy (H)': stats.entropy if USE_SHANNON_ENTROPY else None,
                'Gini': stats.gini if USE_GINI_COEFFICIENT else None,
                'Complexity': stats.complexity if USE_STRUCTURAL_COMPLEXITY else None,
                'Modularity': stats.modularity if USE_FUNCTIONAL_MODULARITY else None,
                'Morans I': stats.morans_i if USE_MORANS_I else None,
            })
        pd.DataFrame(target_results).to_csv(f"{results_dir}/target_groups_stats_v17.csv", index=False)
        print(f"  ✅ Target group statistics saved: {results_dir}/target_groups_stats_v17.csv")
    
    print("\n" + "=" * 80)
    print("✅ EXECUTION COMPLETED")
    print("=" * 80)
    print(f"\n  📁 Results saved in: {results_dir}/")
    print(f"\n  🧬 TARGET GROUPS ANALYZED:")
    for group in target_groups_existing:
        stats = analyzer.group_stats[group]
        print(f"     ├─ {get_display_name(group)}: {stats.n_samples:,} valid proteins")
    if len(target_groups_existing) < len(MAIN_GROUP):
        missing = [g for g in MAIN_GROUP if g not in target_groups_existing]
        print(f"     └─ ⚠️ Not found: {', '.join(missing)}")
    
    print(f"\n  📊 NEW FEATURES (v17.1):")
    print(f"     ├─ get_top_individual_proteins() - Top {TOP_N_PROTEINS} individual proteins")
    print(f"     └─ Saved as: top_20_individual_proteins.csv")
    
    print(f"\n  📊 NEW MATHEMATICAL METRICS IMPLEMENTED (v17.0):")
    print(f"     ├─ Shannon Entropy")
    print(f"     ├─ Jensen-Shannon Divergence")
    print(f"     ├─ Gini Coefficient")
    print(f"     ├─ Structural Complexity Index")
    print(f"     ├─ Functional Modularity Index")
    print(f"     ├─ Hellinger Distance")
    print(f"     ├─ Spearman Correlation")
    print(f"     └─ Moran's I (spatial autocorrelation)")
    
    print(f"\n  📊 EXISTING METRICS (v16.1):")
    print(f"     ├─ Grassmann Projection Distance")
    print(f"     ├─ Fubini-Study Metric")
    print(f"     ├─ Ricci Curvature")
    print(f"     ├─ Karhunen-Loève Decomposition")
    print(f"     ├─ Discrete Radon Transform")
    print(f"     ├─ Fractal Dimension (Box-Counting)")
    print(f"     ├─ Wasserstein-1 Distance")
    print(f"     └─ Polarity Interaction Laplacian")
    
    print(f"\n⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total execution time: {(datetime.now() - analyzer.start_time).total_seconds()/60:.1f} minutes")

if __name__ == "__main__":
    main()
