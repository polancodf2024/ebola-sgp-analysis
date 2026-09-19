#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN217.0v vs BioPython
VERSIÓN CORREGIDA - USA LA MISMA MÉTRICA QUE SGPMAIN217.0v.py

Rutas:
  - Programa SGPMAIN217.0v.py: /home/cpolanco/POLANCO/TAXONOMIAMAIN
  - Datos (config, FASTA, .dat0): /home/cpolanco/POLANCO/ARCHIVOMAESTRO
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import re
import importlib.util
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# 0. CONSTANTES DE RUTAS
# ============================================================================

ARCHIVOMAESTRO = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"
SGPMAIN_DIR = "/home/cpolanco/POLANCO/TAXONOMIAMAIN"
SGPMAIN_FILENAME = "SGPMAIN217.0v.py"

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN217.0v vs BioPython")
print("   📌 USANDO LA MISMA MÉTRICA QUE SGPMAIN217.0v.py (wedge_product)")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 ARCHIVOMAESTRO (datos): {ARCHIVOMAESTRO}")
print(f"📂 SGPMAIN_DIR (programa): {SGPMAIN_DIR}")

if not os.path.exists(ARCHIVOMAESTRO):
    print(f"❌ ARCHIVOMAESTRO no existe: {ARCHIVOMAESTRO}")
    sys.exit(1)

if not os.path.exists(SGPMAIN_DIR):
    print(f"❌ SGPMAIN_DIR no existe: {SGPMAIN_DIR}")
    sys.exit(1)

# ============================================================================
# 1. CARGAR CONFIG_EBOLA.json
# ============================================================================

CONFIG_PATH = os.path.join(ARCHIVOMAESTRO, "config_EBOLA.json")

if not os.path.exists(CONFIG_PATH):
    print(f"❌ Config no encontrado: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

print(f"✅ Configuración cargada: {CONFIG_PATH}")
print(f"   Versión: {CONFIG['metadata']['version']}")

USE_BOOTSTRAP = CONFIG['operation_control'].get('use_bootstrap', True)
BOOTSTRAP_CI = CONFIG['operation_control'].get('bootstrap_ci', 0.95)
APPLY_METRIC_WEIGHTS = CONFIG['operation_control'].get('apply_metric_weights', True)
APPLY_CLASSIFICATION = CONFIG['operation_control'].get('apply_classification', True)
VALIDATE_TARGET_RANGES = CONFIG['operation_control'].get('validate_target_ranges_in_characterization', True)

METRIC_WEIGHTS = CONFIG.get('metric_weights', {})
CLASSIFICATION_THRESHOLDS = CONFIG['classification_thresholds']['ebola']
TARGET_RANGES = CONFIG['target_ranges']['ebola']

PROCESSING_PARAMS = CONFIG.get('processing_params', {})
N_BOOTSTRAP = PROCESSING_PARAMS.get('n_bootstrap', 50)
MIN_SAMPLES_PER_GROUP = PROCESSING_PARAMS.get('min_samples_per_group', 2)
MAX_STORED_PROTEINS = PROCESSING_PARAMS.get('max_stored_proteins', 200)

DATA_PATH_CONFIG = CONFIG['data_paths']['base_dir']
match = re.match(r'\$\{(\w+):-([^}]+)\}', DATA_PATH_CONFIG)
if match:
    env_var, default = match.groups()
    DATA_PATH = os.environ.get(env_var, default)
else:
    DATA_PATH = DATA_PATH_CONFIG

if not os.path.exists(DATA_PATH):
    DATA_PATH = ARCHIVOMAESTRO

print(f"✅ DATA_PATH: {DATA_PATH}")
print(f"✅ Péptido base: {CONFIG['base_peptide_sequence']['ebola']['sequence']}")
print(f"✅ Péptido optimizado: {CONFIG['base_peptide_sequence']['ebola']['optimized_sequence']}")

# ============================================================================
# 2. IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no instalado. Ejecuta: pip install biopython")
    sys.exit(1)

# ============================================================================
# 3. IMPORTAR FUNCIONES DE SGPMAIN217.0v.py
# ============================================================================

SGPMAIN_PATH = os.path.join(SGPMAIN_DIR, SGPMAIN_FILENAME)

if not os.path.exists(SGPMAIN_PATH):
    candidates = [f for f in os.listdir(SGPMAIN_DIR)
                  if f.startswith('SGPMAIN') and f.endswith('.py')]
    if candidates:
        preferred = [c for c in candidates if '217' in c]
        SGPMAIN_PATH = os.path.join(SGPMAIN_DIR, preferred[0] if preferred else candidates[0])
    else:
        print(f"❌ No se encontró SGPMAIN*.py en {SGPMAIN_DIR}")
        sys.exit(1)

print(f"✅ Usando: {SGPMAIN_PATH}")

def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    sgp_module = import_from_file("sgpmain", SGPMAIN_PATH)
    print(f"✅ {os.path.basename(SGPMAIN_PATH)} importado correctamente")
    
    # ========================================================================
    # FUNCIONES REALES DE SGPMAIN217.0v.py (a nivel de módulo)
    # ========================================================================
    compute_pim_profile = sgp_module.compute_pim_profile
    shannon_entropy = sgp_module.shannon_entropy
    gini_coefficient = sgp_module.gini_coefficient
    jensen_shannon_divergence = sgp_module.jensen_shannon_divergence
    hellinger_distance = sgp_module.hellinger_distance
    spearman_correlation = sgp_module.spearman_correlation
    principal_angles = sgp_module.principal_angles
    geodesic_distance = sgp_module.geodesic_distance
    scalar_curvature = sgp_module.scalar_curvature
    pim_to_subspace = sgp_module.pim_to_subspace
    compare_pim_vectors = sgp_module.compare_pim_vectors  # ✅ FUNCIÓN CLAVE
    GrassmannPIM = sgp_module.GrassmannPIM                # ✅ CLASE CLAVE
    
    # ✅ v217.0: Instanciar GrassmannPIM para usar wedge_product
    grassmann_pim = GrassmannPIM(dim=16)
    
    print("✅ Funciones del pipeline 217.0 cargadas:")
    print("     ├─ compute_pim_profile")
    print("     ├─ shannon_entropy")
    print("     ├─ gini_coefficient")
    print("     ├─ jensen_shannon_divergence")
    print("     ├─ hellinger_distance")
    print("     ├─ spearman_correlation")
    print("     ├─ principal_angles")
    print("     ├─ geodesic_distance")
    print("     ├─ scalar_curvature")
    print("     ├─ pim_to_subspace")
    print("     ├─ compare_pim_vectors (CLAVE)")
    print("     └─ GrassmannPIM (CLASE) ✅ INSTANCIADA")
    
except Exception as e:
    print(f"❌ Error importando {SGPMAIN_PATH}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. CONFIGURACIÓN DE VIRUS ÉBOLA
# ============================================================================

EBOLA_VIRUSES = [
    {'name': 'sudan', 'display': 'EBOLA_SUDAN', 'pattern': r'Sudan|SUDAN|sudan'},
    {'name': 'zaire', 'display': 'EBOLA_ZAIRE', 'pattern': r'Zaire|ZAIRE|zaire'},
    {'name': 'reston', 'display': 'EBOLA_RESTON', 'pattern': r'Reston|RESTON|reston'},
    {'name': 'bombali', 'display': 'EBOLA_BOMBALI', 'pattern': r'[Bb]ombali'},
    {'name': 'bundibugyo', 'display': 'EBOLA_BUNDIBUGYO', 'pattern': r'Bundibugyo|BUNDIBUGYO|bundibugyo'},
    {'name': 'tai', 'display': 'EBOLA_TAI_FOREST', 'pattern': r'Tai|TAI|tai|Taï|TAÏ|taï|Tai Forest'},
]

# ============================================================================
# 5. CONFIGURACIÓN DE LOS 30 GRUPOS (NOMBRES REALES CORREGIDOS)
# ============================================================================

GROUP_FILES = {
    'EBOLA_SUDAN': os.path.join(DATA_PATH, 'Sudan.unico.dat0'),
    'EBOLA_ZAIRE': os.path.join(DATA_PATH, 'Zaire.unico.dat0'),
    'EBOLA_RESTON': os.path.join(DATA_PATH, 'Reston.unico.dat0'),
    'EBOLA_BOMBALI': os.path.join(DATA_PATH, 'Bombali.unico.dat0'),
    'EBOLA_BUNDIBUGYO': os.path.join(DATA_PATH, 'Bundibugyo.unico.dat0'),
    'EBOLA_TAI_FOREST': os.path.join(DATA_PATH, 'Tai.unico.dat0'),
    'LASV': os.path.join(DATA_PATH, 'lasv_all.unico.dat0'),
    'JUNV': os.path.join(DATA_PATH, 'junv_all.unico.dat0'),
    'MACV': os.path.join(DATA_PATH, 'macv_all.unico.dat0'),
    'LCMV': os.path.join(DATA_PATH, 'lcmv_all.unico.dat0'),
    'NILE1': os.path.join(DATA_PATH, 'nile1.unico.dat0'),
    'NILE2': os.path.join(DATA_PATH, 'nile2.unico.dat0'),
    'RVF1': os.path.join(DATA_PATH, 'RVF1.unico.dat0'),
    'RVF2': os.path.join(DATA_PATH, 'RVF2.unico.dat0'),
    'RVF3': os.path.join(DATA_PATH, 'RVF3.unico.dat0'),
    'RVF4': os.path.join(DATA_PATH, 'RVF4.unico.dat0'),
    'LUJO': os.path.join(DATA_PATH, 'lujo.unico.dat0'),
    'PARTIALLY_FOLDED': os.path.join(DATA_PATH, 'partiallyorderedN.unico.dat0'),
    'CPP': os.path.join(DATA_PATH, 'CPP.unico.dat0'),
    'NON_CPP': os.path.join(DATA_PATH, 'NONCPP.unico.dat0'),
    'UNFOLDED': os.path.join(DATA_PATH, 'unfolded.unico.dat0'),
    'REVIEWED_HUMAN': os.path.join(DATA_PATH, 'reviewed_human.unico.dat0'),
    'UNREVIEWED_HUMAN': os.path.join(DATA_PATH, 'unreviewed_human.unico.dat0'),
    'SIGNALS': os.path.join(DATA_PATH, 'senales.unico.dat0'),
    'MEMBRANE': os.path.join(DATA_PATH, 'membrana.unico.dat0'),
    'DISEASE': os.path.join(DATA_PATH, 'enfermedad.unico.dat0'),
    'VIRUS_REVIEWED': os.path.join(DATA_PATH, 'reviewed_virus.unico.dat0'),
    'VIRUS_UNREVIEWED': os.path.join(DATA_PATH, 'unreviewed_virus.unico.dat0'),
    'REVIEWED_ALL': os.path.join(DATA_PATH, 'reviewed_all.unico.dat0'),
    'UNREVIEWED_ALL': os.path.join(DATA_PATH, 'unreviewed_all.unico.dat0'),
}

GROUPS_TO_ANALYZE = list(GROUP_FILES.keys())

# ============================================================================
# 6. FUNCIONES DE LECTURA DE FASTA
# ============================================================================

def read_fasta_stream(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, ''.join(seq)

def extract_ebola_sequences(input_file, max_seq=1000):
    if not os.path.exists(input_file):
        print(f"❌ Archivo no encontrado: {input_file}")
        return {}
    
    print(f"\n📂 Leyendo archivo: {input_file}")
    
    sequences_by_virus = {virus['name']: [] for virus in EBOLA_VIRUSES}
    all_sequences = []
    
    for header, seq in read_fasta_stream(input_file):
        all_sequences.append((header, seq))
    
    print(f"  📊 Total de secuencias encontradas: {len(all_sequences)}")
    
    classified = 0
    unknown_count = 0
    
    for header, seq in all_sequences[:max_seq]:
        assigned = False
        for virus in EBOLA_VIRUSES:
            if re.search(virus['pattern'], header, re.IGNORECASE):
                sequences_by_virus[virus['name']].append((header, seq))
                assigned = True
                classified += 1
                break
        
        if not assigned:
            unknown_count += 1
            if 'unknown' not in sequences_by_virus:
                sequences_by_virus['unknown'] = []
            sequences_by_virus['unknown'].append((header, seq))
    
    print(f"  📊 Clasificados: {classified}, Desconocidos: {unknown_count}")
    
    for virus in EBOLA_VIRUSES:
        count = len(sequences_by_virus[virus['name']])
        if count > 0:
            print(f"     ├─ {virus['display']}: {count} secuencias")
    
    return sequences_by_virus

# ============================================================================
# 7. FUNCIONES DEL PIPELINE 217.0 (USANDO wedge_product)
# ============================================================================

def compute_sgp_pim(sequence, use_weights=True):
    try:
        return compute_pim_profile(sequence, use_weights=use_weights)
    except Exception as e:
        print(f"⚠️ Error en compute_sgp_pim: {e}")
        return np.zeros(16)

def compute_grassmann_similarity(v1, v2):
    """
    ✅ v217.0: Usa la MISMA métrica que SGPMAIN217.0v.py:
       wedge_product(v1, v2) = max(0, 1 - geodesic_distance / max_d)
       donde max_d = (π/2) * sqrt(k), k=2 → max_d ≈ 2.2214
    """
    try:
        sim, _ = grassmann_pim.wedge_product(v1, v2)
        return float(sim)
    except Exception as e:
        print(f"⚠️ Error en compute_grassmann_similarity: {e}")
        return 0.0

def get_virus_representative_pim(virus_sequences, max_seq=MAX_STORED_PROTEINS):
    if not virus_sequences:
        return None
    
    pims = []
    for header, seq in virus_sequences[:max_seq]:
        seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
        if len(seq_clean) < 10:
            continue
        pim = compute_sgp_pim(seq_clean, use_weights=True)
        if np.sum(pim) > 0.01:
            pims.append(pim)
    
    if not pims:
        return None
    
    avg_pim = np.mean(pims, axis=0)
    total = np.sum(avg_pim)
    if total > 0:
        avg_pim = avg_pim / total
    
    return avg_pim

def run_sgp_from_file(input_file, virus_name, groups_to_analyze, max_sequences=MAX_STORED_PROTEINS):
    print(f"\n  🧬 Ejecutando SGPMAIN 217.0 para {virus_name}...")
    
    sequences_by_virus = extract_ebola_sequences(input_file, max_seq=1000)
    
    if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
        print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
        return None
    
    virus_pim = get_virus_representative_pim(sequences_by_virus[virus_name], max_seq=max_sequences)
    
    if virus_pim is None or np.sum(virus_pim) < 0.01:
        print(f"  ❌ PIM inválido para {virus_name}")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
    print(f"     ├─ PIM del virus: dimensión {len(virus_pim)}, suma={np.sum(virus_pim):.4f}")
    
    entropy_val = shannon_entropy(virus_pim, normalize=True)
    print(f"     ├─ Entropía (normalizada): {entropy_val:.4f}")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = GROUP_FILES.get(group)
        if not group_file or not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {group}...")
        
        similarities = []
        count = 0
        for header, seq in read_fasta_stream(group_file):
            seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
            if len(seq_clean) < 10:
                continue
            
            pim_group = compute_sgp_pim(seq_clean, use_weights=True)
            if np.sum(pim_group) > 0.01:
                sim = compute_grassmann_similarity(virus_pim, pim_group)
                similarities.append(sim)
                count += 1
                if count >= max_sequences:
                    break
        
        if similarities:
            mean_sim = np.mean(similarities)
            results[group] = {
                'mean_similarity': mean_sim,
                'n': len(similarities),
                'insufficient_n': len(similarities) < MIN_SAMPLES_PER_GROUP
            }
            print(f"        └─ Similitud media: {mean_sim:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin PIMs válidos")
            results[group] = {
                'mean_similarity': 0.0,
                'n': 0,
                'insufficient_n': True
            }
    
    return results

# ============================================================================
# 8. FUNCIONES PARA BIOPYTHON
# ============================================================================

def extract_biopython_features(sequence):
    try:
        seq_str = ''.join([c for c in str(sequence).strip() if c.isalpha()])
        if len(seq_str) < 5:
            return None
        
        seq_obj = Seq(seq_str)
        analyzer = ProtParam.ProteinAnalysis(str(seq_obj))
        
        features = []
        
        try:
            aa_counts = analyzer.get_amino_acids_percent()
            for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                features.append(aa_counts.get(aa, 0.0))
        except:
            features.extend([0.0] * 20)
        
        try:
            mw = analyzer.molecular_weight()
            features.append(mw / len(seq_str))
        except:
            features.append(0.0)
        
        try:
            features.append(analyzer.isoelectric_point())
        except:
            features.append(7.0)
        
        try:
            features.append(analyzer.aromaticity())
        except:
            features.append(0.0)
        
        try:
            features.append(analyzer.instability_index())
        except:
            features.append(50.0)
        
        try:
            features.append(analyzer.gravy())
        except:
            features.append(0.0)
        
        try:
            sec_struct = analyzer.secondary_structure_fraction()
            features.extend(sec_struct)
        except:
            features.extend([0.0, 0.0, 0.0])
        
        try:
            charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
            net_charge = sum(charges.get(aa, 0) for aa in seq_str)
            features.append(net_charge / len(seq_str))
        except:
            features.append(0.0)
        
        features.append(np.log(len(seq_str) + 1))
        
        return np.array(features)
        
    except Exception as e:
        return None

def normalize_features(features_list):
    if not features_list:
        return features_list
    features_array = np.array(features_list)
    scaler = StandardScaler()
    return scaler.fit_transform(features_array)

def euclidean_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    dist = np.linalg.norm(v1_norm - v2_norm)
    max_dist = 2.0
    sim = 1 - (dist / max_dist)
    return max(0, min(1, sim))

def get_virus_representative_features(virus_sequences, max_seq=MAX_STORED_PROTEINS):
    if not virus_sequences:
        return None
    features_list = []
    for header, seq in virus_sequences[:max_seq]:
        features = extract_biopython_features(seq)
        if features is not None:
            features_list.append(features)
    if not features_list:
        return None
    return np.mean(features_list, axis=0)

def run_biopython_from_file(input_file, virus_name, groups_to_analyze, max_sequences=MAX_STORED_PROTEINS):
    print(f"\n  🔬 Ejecutando BioPython para {virus_name} (30 características)...")
    
    sequences_by_virus = extract_ebola_sequences(input_file, max_seq=1000)
    
    if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
        print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
        return None
    
    virus_features = get_virus_representative_features(sequences_by_virus[virus_name], max_seq=max_sequences)
    
    if virus_features is None:
        print(f"  ❌ Error extrayendo características de {virus_name}")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
    print(f"     ├─ Dimensiones: {len(virus_features)} características")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = GROUP_FILES.get(group)
        if not group_file or not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {group}...")
        
        vectors = []
        count = 0
        for header, seq in read_fasta_stream(group_file):
            features = extract_biopython_features(seq)
            if features is not None and len(features) > 0:
                vectors.append(features)
                count += 1
                if count >= max_sequences:
                    break
        
        if not vectors:
            results[group] = {'mean_similarity': 0.0, 'n': 0, 'insufficient_n': True}
            continue
        
        all_vectors = [virus_features] + vectors
        normalized_vectors = normalize_features(all_vectors)
        virus_norm = normalized_vectors[0]
        vectors_norm = normalized_vectors[1:]
        
        similarities = [euclidean_similarity(virus_norm, vec) for vec in vectors_norm]
        
        if similarities:
            mean_sim = np.mean(similarities)
            results[group] = {
                'mean_similarity': mean_sim,
                'n': len(similarities),
                'insufficient_n': len(similarities) < MIN_SAMPLES_PER_GROUP
            }
            print(f"        └─ Similitud media: {mean_sim:.6f} (n={len(similarities)})")
        else:
            results[group] = {'mean_similarity': 0.0, 'n': 0, 'insufficient_n': True}
    
    return results

# ============================================================================
# 9. FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================================================

def analyze_ebola_virus(input_file, virus_info, groups_to_analyze, max_sequences=MAX_STORED_PROTEINS):
    virus_name = virus_info['name']
    virus_display = virus_info['display']
    
    print("\n" + "=" * 80)
    print(f"🔬 ANALIZANDO: {virus_display}")
    print("=" * 80)
    
    if not os.path.exists(input_file):
        print(f"  ❌ Archivo no encontrado: {input_file}")
        return None
    
    print(f"  📌 Usando muestra de {max_sequences} secuencias por grupo")
    
    sgp_results = run_sgp_from_file(input_file, virus_name, groups_to_analyze, max_sequences)
    biopython_results = run_biopython_from_file(input_file, virus_name, groups_to_analyze, max_sequences)
    
    if biopython_results is None or len(biopython_results) == 0:
        print(f"\n❌ No se pudieron obtener resultados de BioPython para {virus_display}")
        return None
    
    comparacion = []
    groups_compared = set()
    
    if sgp_results:
        groups_compared = set(sgp_results.keys()) & set(biopython_results.keys())
    else:
        groups_compared = set(biopython_results.keys())
    
    print("\n" + "=" * 80)
    print(f"📋 TABLA COMPARATIVA: SGPMAIN 217.0 vs BioPython ({virus_display})")
    print("=" * 80)
    print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
    print("-" * 80)
    
    for grupo in sorted(groups_compared,
                        key=lambda x: sgp_results[x]['mean_similarity'] if sgp_results and x in sgp_results else 0,
                        reverse=True):
        sgp_data = sgp_results.get(grupo, {'mean_similarity': 0.0, 'n': 0, 'insufficient_n': True})
        bio_data = biopython_results.get(grupo, {'mean_similarity': 0.0, 'n': 0, 'insufficient_n': True})
        
        sgp_val = sgp_data['mean_similarity']
        biopy_val = bio_data['mean_similarity']
        diff = abs(sgp_val - biopy_val) if sgp_val > 0 else 1.0
        
        if sgp_data['insufficient_n']:
            interp = "⚠️ n insuficiente"
        elif sgp_val == 0:
            interp = "⚠️ Sin SGP"
        elif diff < 0.01:
            interp = "✅ Excelente"
        elif diff < 0.03:
            interp = "✔️ Buena"
        elif diff < 0.05:
            interp = "⚠️ Moderada"
        else:
            interp = "❌ Diferente"
        
        print(f"{grupo:<22} {sgp_val:>12.6f} {biopy_val:>12.6f} "
              f"{diff:>12.6f} {interp:>15}")
        
        comparacion.append({
            'Virus': virus_display,
            'Grupo': grupo,
            'SGPMAIN': sgp_val,
            'SGPMAIN_n': sgp_data['n'],
            'SGPMAIN_insufficient_n': sgp_data['insufficient_n'],
            'BioPython': biopy_val,
            'BioPython_n': bio_data['n'],
            'Diferencia': diff,
            'Interpretación': interp
        })
    
    return {
        'virus': virus_info,
        'sgp': sgp_results,
        'biopython': biopython_results,
        'comparacion': comparacion
    }

# ============================================================================
# 10. EJECUCIÓN PRINCIPAL
# ============================================================================

INPUT_FILE = os.path.join(ARCHIVOMAESTRO, "todo_ebola.fasta")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Archivo de entrada no encontrado: {INPUT_FILE}")
    sys.exit(1)

print(f"\n✅ Archivo de entrada: {INPUT_FILE}")

print("\n📂 Verificando archivos .dat0...")
found_files = 0
missing_files = []
for group, path in GROUP_FILES.items():
    if os.path.exists(path):
        found_files += 1
    else:
        missing_files.append((group, os.path.basename(path)))

print(f"  ✅ Archivos encontrados: {found_files} de {len(GROUP_FILES)}")
if missing_files:
    print(f"  ⚠️ Archivos faltantes ({len(missing_files)}):")
    for group, fname in missing_files[:10]:
        print(f"     - {group}: {fname}")

sequences_by_virus = extract_ebola_sequences(INPUT_FILE, max_seq=1000)
available_viruses = []

for virus in EBOLA_VIRUSES:
    if virus['name'] in sequences_by_virus and sequences_by_virus[virus['name']]:
        available_viruses.append(virus)

if not available_viruses:
    print("\n❌ No se encontraron secuencias de virus Ébola en el archivo")
    sys.exit(1)

print(f"\n  📊 Se analizarán {len(available_viruses)} virus Ébola")

# ============================================================================
# 11. ANALIZAR CADA VIRUS
# ============================================================================

all_results = {}
all_dataframes = []

for virus in available_viruses:
    result = analyze_ebola_virus(INPUT_FILE, virus, GROUPS_TO_ANALYZE, max_sequences=MAX_STORED_PROTEINS)
    if result and result['comparacion']:
        all_results[virus['name']] = result
        all_dataframes.append(pd.DataFrame(result['comparacion']))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"ebola_analysis_{virus['name']}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        df_comp = pd.DataFrame(result['comparacion'])
        df_comp.to_csv(f"{results_dir}/comparacion_sgp_biopython.csv", index=False)
        
        if result['sgp']:
            with open(f"{results_dir}/sgp_results.json", 'w') as f:
                json.dump(result['sgp'], f, indent=2)
        
        with open(f"{results_dir}/biopython_results.json", 'w') as f:
            json.dump(result['biopython'], f, indent=2)
        
        with open(f"{results_dir}/config_used.json", 'w') as f:
            json.dump(CONFIG, f, indent=2)
        
        print(f"\n  ✅ Resultados guardados en: {results_dir}/")

# ============================================================================
# 12. MATRIZ DE SIMILITUD
# ============================================================================

if len(all_results) >= 2:
    print("\n" + "=" * 80)
    print("🔍 MATRIZ DE SIMILITUD ENTRE VIRUS ÉBOLA")
    print("=" * 80)
    
    virus_names = list(all_results.keys())
    n = len(virus_names)
    
    sgp_matrix = np.zeros((n, n))
    for i, v1 in enumerate(virus_names):
        for j, v2 in enumerate(virus_names):
            if i != j:
                sgp1 = all_results[v1]['sgp']
                sgp2 = all_results[v2]['sgp']
                if sgp1 and sgp2:
                    common = set(sgp1.keys()) & set(sgp2.keys())
                    if common:
                        sims = [sgp1[g]['mean_similarity'] * sgp2[g]['mean_similarity'] for g in common]
                        sgp_matrix[i, j] = np.mean(sims)
            else:
                sgp_matrix[i, j] = 1.0
    
    bio_matrix = np.zeros((n, n))
    for i, v1 in enumerate(virus_names):
        for j, v2 in enumerate(virus_names):
            if i != j:
                bio1 = all_results[v1]['biopython']
                bio2 = all_results[v2]['biopython']
                if bio1 and bio2:
                    common = set(bio1.keys()) & set(bio2.keys())
                    if common:
                        sims = [bio1[g]['mean_similarity'] * bio2[g]['mean_similarity'] for g in common]
                        bio_matrix[i, j] = np.mean(sims)
            else:
                bio_matrix[i, j] = 1.0
    
    print("\n  📊 MATRIZ SGPMAIN 217.0 (usando wedge_product):")
    print(f"  {'':<16}", end='')
    for name in virus_names:
        print(f"{name:>16}", end='')
    print()
    print("  " + "-" * (16 + 16 * n))
    for i, v1 in enumerate(virus_names):
        print(f"  {v1:<16}", end='')
        for j in range(n):
            print(f"{sgp_matrix[i, j]:>16.4f}", end='')
        print()
    
    print("\n  📊 MATRIZ BIOPYTHON:")
    print(f"  {'':<16}", end='')
    for name in virus_names:
        print(f"{name:>16}", end='')
    print()
    print("  " + "-" * (16 + 16 * n))
    for i, v1 in enumerate(virus_names):
        print(f"  {v1:<16}", end='')
        for j in range(n):
            print(f"{bio_matrix[i, j]:>16.4f}", end='')
        print()

# ============================================================================
# 13. GUARDAR RESULTADOS COMBINADOS
# ============================================================================

if all_dataframes:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(f"ebola_comparison_complete_{timestamp}.csv", index=False)
    print(f"\n  ✅ Resultados combinados guardados: ebola_comparison_complete_{timestamp}.csv")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n  📊 Virus analizados: {len(all_results)} de {len(EBOLA_VIRUSES)}")
print(f"  📊 Grupos por virus: {len(GROUPS_TO_ANALYZE)}")
