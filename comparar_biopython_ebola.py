#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN212.0v vs BioPython
VERSIÓN CORREGIDA - IMPORTANDO FUNCIONES DE SGPMAIN212.0v.py
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN212.0v vs BioPython")
print("   Análisis de los 6 tipos de virus Ébola")
print("   📌 USANDO FUNCIONES INTERNAS DE SGPMAIN212.0v.py")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no está instalado. Ejecuta: pip install biopython")
    sys.exit(1)

# ============================================================================
# IMPORTAR FUNCIONES DE SGPMAIN212.0v.py (EN LUGAR DE EJECUTARLO EXTERNAMENTE)
# ============================================================================

# Ruta al archivo SGPMAIN212.0v.py
SGPMAIN_PATH = "SGPMAIN212.0v.py"

# Importar dinámicamente las funciones necesarias
import importlib.util

def import_from_file(module_name, file_path):
    """Importa un módulo desde un archivo .py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    sgp_module = import_from_file("sgpmain", SGPMAIN_PATH)
    print("✅ SGPMAIN212.0v importado correctamente")
    
    # Extraer las funciones necesarias
    compute_pim_profile = sgp_module.compute_pim_profile
    shannon_entropy = sgp_module.shannon_entropy
    gini_coefficient = sgp_module.gini_coefficient
    grassmann_distance = sgp_module.grassmann_distance
    hodge_complementarity = sgp_module.hodge_complementarity
    grassmann_ricci_curvature = sgp_module.grassmann_ricci_curvature
    wasserstein_distance = sgp_module.wasserstein_distance
    fractal_dimension = sgp_module.fractal_dimension
    renyi_entropy = sgp_module.renyi_entropy
    bhattacharyya_distance = sgp_module.bhattacharyya_distance
    jensen_shannon_divergence = sgp_module.jensen_shannon_divergence
    hellinger_distance = sgp_module.hellinger_distance
    POLARITY_MAP = sgp_module.POLARITY_MAP
    INTERACTIONS = sgp_module.INTERACTIONS
    INTERACTION_TO_IDX = sgp_module.INTERACTION_TO_IDX
    
    # Verificar que compute_pim_profile existe
    if hasattr(sgp_module, 'compute_pim_profile'):
        print("✅ compute_pim_profile disponible")
    else:
        print("❌ compute_pim_profile no encontrado")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error importando SGPMAIN212.0v.py: {e}")
    print("   Asegúrate de que SGPMAIN212.0v.py esté en el directorio actual")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN DE VIRUS ÉBOLA
# ============================================================================

EBOLA_VIRUSES = [
    {'name': 'sudan', 'display': 'EBOLA_SUDAN', 'pattern': r'Sudan|SUDAN|sudan'},
    {'name': 'zaire', 'display': 'EBOLA_ZAIRE', 'pattern': r'Zaire|ZAIRE|zaire'},
    {'name': 'reston', 'display': 'EBOLA_RESTON', 'pattern': r'Reston|RESTON|reston'},
    {'name': 'bombali', 'display': 'EBOLA_BOMBALI', 'pattern': r'Bombali|BOMBALI|bombali'},
    {'name': 'bundibugyo', 'display': 'EBOLA_BUNDIBUGYO', 'pattern': r'Bundibugyo|BUNDIBUGYO|bundibugyo'},
    {'name': 'tai', 'display': 'EBOLA_TAI_FOREST', 'pattern': r'Tai|TAI|tai|Taï|TAÏ|taï'},
]

# ============================================================================
# CONFIGURACIÓN DE GRUPOS
# ============================================================================

DATA_PATH = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"

GROUP_FILES = {
    'CPP': os.path.join(DATA_PATH, 'CPP.unico.dat0'),
    'NON_CPP': os.path.join(DATA_PATH, 'NONCPP.unico.dat0'),
    'UNFOLDED': os.path.join(DATA_PATH, 'unfolded.unico.dat0'),
    'PARTIALLY_FOLDED': os.path.join(DATA_PATH, 'partiallyorderedN.unico.dat0'),
    'REVIEWED_HUMAN': os.path.join(DATA_PATH, 'reviewed_human.unico.dat0'),
    'UNREVIEWED_HUMAN': os.path.join(DATA_PATH, 'unreviewed_human.unico.dat0'),
    'VIRUS_REVIEWED': os.path.join(DATA_PATH, 'reviewed_virus.unico.dat0'),
    'VIRUS_UNREVIEWED': os.path.join(DATA_PATH, 'unreviewed_virus.unico.dat0'),
    'REVIEWED_ALL': os.path.join(DATA_PATH, 'reviewed_all.unico.dat0'),
    'UNREVIEWED_ALL': os.path.join(DATA_PATH, 'unreviewed_all.unico.dat0'),
    'lujo': os.path.join(DATA_PATH, 'lujo.unico.dat0'),
    'lasv': os.path.join(DATA_PATH, 'lasv_all.unico.dat0'),
    'junv': os.path.join(DATA_PATH, 'junv_all.unico.dat0'),
    'macv': os.path.join(DATA_PATH, 'macv_all.unico.dat0'),
    'lcmv': os.path.join(DATA_PATH, 'lcmv_all.unico.dat0'),
    'nile1': os.path.join(DATA_PATH, 'nile1.unico.dat0'),
    'nile2': os.path.join(DATA_PATH, 'nile2.unico.dat0'),
    'enfermedad': os.path.join(DATA_PATH, 'enfermedad.unico.dat0'),
    'membrana': os.path.join(DATA_PATH, 'membrana.unico.dat0'),
    'senales': os.path.join(DATA_PATH, 'senales.unico.dat0'),
}

DISPLAY_NAMES = {
    'CPP': 'CPP',
    'NON_CPP': 'NON_CPP',
    'UNFOLDED': 'UNFOLDED',
    'PARTIALLY_FOLDED': 'PARTIALLY_FOLDED',
    'REVIEWED_HUMAN': 'REVIEWED_HUMAN',
    'UNREVIEWED_HUMAN': 'UNREVIEWED_HUMAN',
    'VIRUS_REVIEWED': 'VIRUS_REVIEWED',
    'VIRUS_UNREVIEWED': 'VIRUS_UNREVIEWED',
    'REVIEWED_ALL': 'REVIEWED_ALL',
    'UNREVIEWED_ALL': 'UNREVIEWED_ALL',
    'lujo': 'LUJO',
    'lasv': 'LASV',
    'junv': 'JUNV',
    'macv': 'MACV',
    'lcmv': 'LCMV',
    'nile1': 'NILE1',
    'nile2': 'NILE2',
    'enfermedad': 'DISEASE',
    'membrana': 'MEMBRANE',
    'senales': 'SIGNALS',
}

GROUPS_TO_ANALYZE = list(GROUP_FILES.keys())

# ============================================================================
# FUNCIONES DE LECTURA DE FASTA
# ============================================================================

def read_fasta_stream(filepath):
    """Lee archivo FASTA secuencialmente"""
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

def get_filename(group_name):
    return GROUP_FILES.get(group_name, f"{group_name}.unico.dat0")

def get_display_name(group_name):
    return DISPLAY_NAMES.get(group_name, group_name)

def extract_ebola_sequences(input_file, max_seq=30):
    """Extrae secuencias de Ébola del archivo (solo muestra)"""
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
    for header, seq in all_sequences[:max_seq]:
        assigned = False
        for virus in EBOLA_VIRUSES:
            if re.search(virus['pattern'], header, re.IGNORECASE):
                sequences_by_virus[virus['name']].append((header, seq))
                assigned = True
                classified += 1
                break
        
        if not assigned:
            if 'unknown' not in sequences_by_virus:
                sequences_by_virus['unknown'] = []
            sequences_by_virus['unknown'].append((header, seq))
    
    print("\n  📊 Distribución de secuencias por virus:")
    for virus in EBOLA_VIRUSES:
        count = len(sequences_by_virus[virus['name']])
        if count > 0:
            print(f"     ├─ {virus['display']}: {count} secuencias")
    
    return sequences_by_virus

# ============================================================================
# FUNCIONES PARA SGPMAIN (USANDO FUNCIONES IMPORTADAS)
# ============================================================================

def compute_sgp_pim(sequence, use_weights=True):
    """Calcula PIM usando la función importada de SGPMAIN212.0v.py"""
    try:
        pim = compute_pim_profile(sequence, use_weights=use_weights)
        return pim
    except Exception as e:
        print(f"⚠️ Error en compute_sgp_pim: {e}")
        return np.zeros(16)

def compute_grassmann_similarity(v1, v2):
    """Calcula similitud usando Grassmann distance"""
    try:
        dist = grassmann_distance(v1, v2)
        return max(0, 1 - dist)
    except:
        return 0.0

def get_virus_representative_pim(virus_sequences, max_seq=10):
    """Obtiene el PIM representativo de un virus (promedio)"""
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

def run_sgp_from_file(input_file, virus_name, groups_to_analyze, max_sequences=10):
    """Ejecuta SGPMAIN usando las funciones importadas"""
    print(f"\n  🧬 Ejecutando SGPMAIN212.0v para {virus_name}...")
    
    sequences_by_virus = extract_ebola_sequences(input_file, max_seq=max_sequences)
    
    if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
        print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
        return None
    
    virus_pim = get_virus_representative_pim(sequences_by_virus[virus_name], max_seq=max_sequences)
    
    if virus_pim is None or np.sum(virus_pim) < 0.01:
        print(f"  ❌ PIM inválido para {virus_name}")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
    print(f"     ├─ PIM del virus: dimensión {len(virus_pim)}, suma={np.sum(virus_pim):.4f}")
    print(f"     ├─ Entropía: {-np.sum(virus_pim * np.log2(virus_pim + 1e-10)):.4f}")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = get_filename(group)
        if not os.path.exists(group_file):
            continue
        
        display = get_display_name(group)
        print(f"     ├─ Procesando {display}...")
        
        # Leer secuencias del grupo
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
            results[group] = np.mean(similarities)
            print(f"        └─ Similitud media: {results[group]:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin PIMs válidos")
    
    return results

# ============================================================================
# FUNCIONES PARA BIOPYTHON
# ============================================================================

def extract_biopython_features_enhanced(sequence):
    """Extrae características fisicoquímicas usando BioPython"""
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
        
        try:
            hydrophobicity = {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            }
            avg_hydro = sum(hydrophobicity.get(aa, 0) for aa in seq_str) / len(seq_str)
            features.append(avg_hydro)
            
            flexibility = {
                'A': 0.0, 'R': 0.5, 'N': 0.3, 'D': 0.4, 'C': 0.1,
                'Q': 0.4, 'E': 0.5, 'G': 0.1, 'H': 0.3, 'I': 0.0,
                'L': 0.0, 'K': 0.5, 'M': 0.1, 'F': 0.0, 'P': 0.6,
                'S': 0.3, 'T': 0.2, 'W': 0.1, 'Y': 0.1, 'V': 0.0
            }
            avg_flex = sum(flexibility.get(aa, 0) for aa in seq_str) / len(seq_str)
            features.append(avg_flex)
            
            features.extend([0.0] * 18)
            
        except:
            features.extend([0.0] * 20)
        
        return np.array(features)
        
    except Exception as e:
        return None

def normalize_features(features_list):
    if not features_list:
        return features_list
    
    features_array = np.array(features_list)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features_array)
    return normalized

def euclidean_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    dist = np.linalg.norm(v1_norm - v2_norm)
    max_dist = np.sqrt(2)
    sim = 1 - (dist / max_dist)
    return max(0, min(1, sim))

def get_virus_representative_features(virus_sequences):
    if not virus_sequences:
        return None
    
    features_list = []
    for header, seq in virus_sequences:
        features = extract_biopython_features_enhanced(seq)
        if features is not None:
            features_list.append(features)
    
    if not features_list:
        return None
    
    avg_features = np.mean(features_list, axis=0)
    return avg_features

def run_biopython_from_file(input_file, virus_name, groups_to_analyze, max_sequences=10):
    """Ejecuta BioPython usando muestreo"""
    print(f"\n  🔬 Ejecutando BioPython para {virus_name} (50 características)...")
    
    sequences_by_virus = extract_ebola_sequences(input_file, max_seq=max_sequences)
    
    if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
        print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
        return None
    
    virus_features = get_virus_representative_features(sequences_by_virus[virus_name])
    
    if virus_features is None:
        print(f"  ❌ Error extrayendo características de {virus_name}")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
    print(f"     ├─ Dimensiones: {len(virus_features)} características")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = get_filename(group)
        if not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {get_display_name(group)}...")
        
        vectors = []
        count = 0
        for header, seq in read_fasta_stream(group_file):
            features = extract_biopython_features_enhanced(seq)
            if features is not None and len(features) > 0:
                vectors.append(features)
                count += 1
                if count >= max_sequences:
                    break
        
        if not vectors:
            print(f"        └─ ⚠️ Sin características válidas")
            continue
        
        all_vectors = [virus_features] + vectors
        normalized_vectors = normalize_features(all_vectors)
        virus_norm = normalized_vectors[0]
        vectors_norm = normalized_vectors[1:]
        
        similarities = []
        for vec in vectors_norm:
            sim = euclidean_similarity(virus_norm, vec)
            similarities.append(sim)
        
        if similarities:
            results[group] = np.mean(similarities)
            print(f"        └─ Similitud media: {results[group]:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin similitudes válidas")
    
    return results

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def analyze_ebola_virus(input_file, virus_info, groups_to_analyze, max_sequences=10):
    """Analiza un virus usando SGPMAIN (importado) y BioPython"""
    
    virus_name = virus_info['name']
    virus_display = virus_info['display']
    
    print("\n" + "=" * 80)
    print(f"🔬 ANALIZANDO: {virus_display}")
    print("=" * 80)
    
    if not os.path.exists(input_file):
        print(f"  ❌ Archivo no encontrado: {input_file}")
        return None
    
    print(f"  📌 Usando muestra de {max_sequences} secuencias por grupo")
    
    # SGPMAIN (usando funciones importadas)
    sgp_results = run_sgp_from_file(input_file, virus_name, groups_to_analyze, max_sequences)
    
    # BioPython
    biopython_results = run_biopython_from_file(input_file, virus_name, groups_to_analyze, max_sequences)
    
    if biopython_results is None or len(biopython_results) == 0:
        print(f"\n❌ No se pudieron obtener resultados de BioPython para {virus_display}")
        return None
    
    # Crear tabla comparativa
    comparacion = []
    groups_compared = set()
    
    if sgp_results:
        groups_compared = set(sgp_results.keys()) & set(biopython_results.keys())
    else:
        groups_compared = set(biopython_results.keys())
    
    print("\n" + "=" * 80)
    print(f"📋 TABLA COMPARATIVA: SGPMAIN212.0v vs BioPython ({virus_display})")
    print("=" * 80)
    print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
    print("-" * 80)
    
    for grupo in sorted(groups_compared, 
                        key=lambda x: sgp_results.get(x, 0) if sgp_results and x in sgp_results else 0, 
                        reverse=True):
        sgp_val = sgp_results.get(grupo, 0.0) if sgp_results else 0.0
        biopy_val = biopython_results.get(grupo, 0.0)
        diff = abs(sgp_val - biopy_val) if sgp_val > 0 else 1.0
        
        if sgp_val == 0:
            interp = "⚠️ Sin SGP"
        elif diff < 0.01:
            interp = "✅ Excelente"
        elif diff < 0.03:
            interp = "✔️ Buena"
        elif diff < 0.05:
            interp = "⚠️ Moderada"
        else:
            interp = "❌ Diferente"
        
        display_name = get_display_name(grupo)
        print(f"{display_name:<22} {sgp_val:>12.6f} {biopy_val:>12.6f} "
              f"{diff:>12.6f} {interp:>15}")
        
        comparacion.append({
            'Virus': virus_display,
            'Grupo': display_name,
            'Grupo_original': grupo,
            'SGPMAIN': sgp_val,
            'BioPython': biopy_val,
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
# EJECUCIÓN PRINCIPAL
# ============================================================================

INPUT_FILE = "todoebola.txt"

print("\n📂 Verificando archivo de entrada...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Archivo no encontrado: {INPUT_FILE}")
    sys.exit(1)

print(f"✅ Archivo encontrado: {INPUT_FILE}")

# Verificar archivos .dat0
print("\n📂 Verificando archivos .dat0...")
found_files = 0
for group in GROUP_FILES:
    if os.path.exists(GROUP_FILES[group]):
        found_files += 1
print(f"  ✅ Archivos encontrados: {found_files} de {len(GROUP_FILES)}")

# Verificar virus disponibles
sequences_by_virus = extract_ebola_sequences(INPUT_FILE, max_seq=30)
available_viruses = []

for virus in EBOLA_VIRUSES:
    if virus['name'] in sequences_by_virus and sequences_by_virus[virus['name']]:
        available_viruses.append(virus)

if not available_viruses:
    print("\n❌ No se encontraron secuencias de virus Ébola en el archivo")
    sys.exit(1)

print(f"\n  📊 Se analizarán {len(available_viruses)} virus Ébola")

# ============================================================================
# ANALIZAR CADA VIRUS
# ============================================================================

all_results = {}
all_dataframes = []

for virus in available_viruses:
    result = analyze_ebola_virus(INPUT_FILE, virus, GROUPS_TO_ANALYZE, max_sequences=10)
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
        
        print(f"\n  ✅ Resultados guardados en: {results_dir}/")

# ============================================================================
# MATRIZ DE SIMILITUD
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
                        sims = [sgp1[g] * sgp2[g] for g in common]
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
                        sims = [bio1[g] * bio2[g] for g in common]
                        bio_matrix[i, j] = np.mean(sims)
            else:
                bio_matrix[i, j] = 1.0
    
    print("\n  📊 MATRIZ SGPMAIN:")
    print(f"  {'':<16}", end='')
    for name in virus_names:
        print(f"{get_display_name(name):>16}", end='')
    print()
    print("  " + "-" * (16 + 16 * n))
    for i, v1 in enumerate(virus_names):
        print(f"  {get_display_name(v1):<16}", end='')
        for j in range(n):
            print(f"{sgp_matrix[i, j]:>16.4f}", end='')
        print()
    
    print("\n  📊 MATRIZ BIOPYTHON:")
    print(f"  {'':<16}", end='')
    for name in virus_names:
        print(f"{get_display_name(name):>16}", end='')
    print()
    print("  " + "-" * (16 + 16 * n))
    for i, v1 in enumerate(virus_names):
        print(f"  {get_display_name(v1):<16}", end='')
        for j in range(n):
            print(f"{bio_matrix[i, j]:>16.4f}", end='')
        print()

# ============================================================================
# GUARDAR RESULTADOS
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
