#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN212.0v vs BioPython (Propiedades Fisicoquímicas)
VERSIÓN OPTIMIZADA PARA ARCHIVOS MASIVOS
ARCHIVOS EN: /home/cpolanco/POLANCO/ARCHIVOMAESTRO
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import subprocess
import tempfile
import re
import random
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE RUTAS - ¡IMPORTANTE!
# ============================================================================

# Ruta donde están todos los archivos .dat0
DATA_PATH = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"

# Archivo de entrada con las secuencias de Ébola
INPUT_FILE = "todoebola.txt"  # O usa la ruta completa si está en otro lado

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN212.0v vs BioPython")
print(f"📂 Ruta de datos: {DATA_PATH}")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURACIÓN DE MUESTREO
# ============================================================================

MAX_SEQUENCES_PER_VIRUS = 10   # Número máximo de secuencias a procesar por virus
MAX_SEQUENCES_PER_GROUP = 30   # Número máximo de secuencias por grupo
SAMPLE_MODE = 'first'          # 'first': primeras N | 'random': N aleatorias
RANDOM_SEED = 42

print(f"\n📌 CONFIGURACIÓN DE MUESTREO:")
print(f"   ├─ Máx. secuencias por virus: {MAX_SEQUENCES_PER_VIRUS}")
print(f"   ├─ Máx. secuencias por grupo: {MAX_SEQUENCES_PER_GROUP}")
print(f"   └─ Modo de muestreo: {SAMPLE_MODE}")

# ============================================================================
# IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no está instalado. Ejecuta: pip install biopython")
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
# CONFIGURACIÓN DE SGPMAIN212.0v
# ============================================================================

SGPMAIN_SCRIPT = "SGPMAIN212.0v.py"

# 🔴 AHORA TODOS LOS ARCHIVOS APUNTAN A LA RUTA CORRECTA 🔴
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

# ============================================================================
# FUNCIONES OPTIMIZADAS
# ============================================================================

def check_sgp_main_available():
    """Verifica que SGPMAIN212.0v.py esté disponible"""
    global SGPMAIN_SCRIPT
    
    if os.path.exists(SGPMAIN_SCRIPT):
        return True
    
    try:
        result = subprocess.run(['which', 'SGPMAIN212.0v.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            SGPMAIN_SCRIPT = result.stdout.strip()
            return True
    except:
        pass
    
    return False

def get_filename(group_name):
    """Devuelve la ruta completa del archivo para un grupo"""
    return GROUP_FILES.get(group_name, os.path.join(DATA_PATH, f"{group_name}.unico.dat0"))

def get_display_name(group_name):
    """Devuelve el nombre legible de un grupo"""
    return DISPLAY_NAMES.get(group_name, group_name)

def count_sequences_fast(filepath):
    """Cuenta rápidamente el número de secuencias en un archivo FASTA"""
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def sample_sequences_fast(filepath, max_seq=10, mode='first', random_seed=42):
    """Toma una muestra de secuencias de un archivo FASTA masivo"""
    if not os.path.exists(filepath):
        return []
    
    print(f"     📂 Muestreando {os.path.basename(filepath)} (modo: {mode})...")
    
    total_seqs = None
    if mode == 'random':
        total_seqs = count_sequences_fast(filepath)
        print(f"     📊 Total de secuencias: {total_seqs:,}")
        if total_seqs <= max_seq:
            mode = 'first'
    
    sequences = []
    headers = []
    
    if mode == 'first':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = None
            seq = []
            count = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None and seq:
                        sequences.append((''.join(seq)))
                        headers.append(header)
                        count += 1
                        if count >= max_seq:
                            break
                    header = line[1:]
                    seq = []
                else:
                    seq.append(line)
            
            if header is not None and seq and count < max_seq:
                sequences.append((''.join(seq)))
                headers.append(header)
    
    else:  # mode == 'random'
        random.seed(random_seed)
        if total_seqs <= max_seq:
            indices_to_take = set(range(total_seqs))
        else:
            indices_to_take = set(random.sample(range(total_seqs), max_seq))
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = None
            seq = []
            current_idx = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None and seq and current_idx in indices_to_take:
                        sequences.append((''.join(seq)))
                        headers.append(header)
                    
                    header = line[1:]
                    seq = []
                    current_idx += 1
                else:
                    seq.append(line)
            
            if header is not None and seq and current_idx in indices_to_take:
                sequences.append((''.join(seq)))
                headers.append(header)
    
    print(f"     ✅ Muestra tomada: {len(sequences)} secuencias")
    return list(zip(headers, sequences))

def extract_ebola_sequences_optimized(input_file, max_seq=10, mode='first'):
    """Extrae secuencias de Ébola de forma optimizada"""
    if not os.path.exists(input_file):
        print(f"❌ Archivo no encontrado: {input_file}")
        return {}
    
    print(f"\n📂 Muestreando archivo: {input_file}")
    
    total = count_sequences_fast(input_file)
    print(f"  📊 Total de secuencias: {total:,}")
    
    if total > 10000:
        print(f"  ⚠️ Archivo muy grande. Usando muestreo ({mode} {min(max_seq, total)} secuencias)")
    else:
        max_seq = total
    
    sampled_sequences = sample_sequences_fast(input_file, max_seq, mode)
    print(f"  ✅ Muestra obtenida: {len(sampled_sequences)} secuencias")
    
    sequences_by_virus = {virus['name']: [] for virus in EBOLA_VIRUSES}
    
    for header, seq in sampled_sequences:
        assigned = False
        for virus in EBOLA_VIRUSES:
            if re.search(virus['pattern'], header, re.IGNORECASE):
                sequences_by_virus[virus['name']].append((header, seq))
                assigned = True
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

def run_sgp_main_for_sequence(sequence_file, output_file=None):
    """Ejecuta SGPMAIN212.0v.py para una secuencia"""
    if not check_sgp_main_available():
        return None
    
    if not os.path.exists(sequence_file):
        return None
    
    try:
        cmd = [sys.executable, SGPMAIN_SCRIPT, sequence_file]
        if output_file:
            cmd.extend(['-o', output_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return None
        
        base_name = os.path.splitext(os.path.basename(sequence_file))[0]
        possible_outputs = [
            f"{base_name}.dat0",
            f"{base_name}.unico.dat0",
            output_file
        ]
        
        for out in possible_outputs:
            if out and os.path.exists(out):
                return out
        
        dat_files = [f for f in os.listdir('.') if f.endswith('.dat0')]
        if dat_files:
            latest = max(dat_files, key=lambda f: os.path.getmtime(f))
            return latest
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

def compute_pim_with_sgp_main(sequence, temp_dir=None):
    """Calcula el vector PIM usando SGPMAIN212.0v.py"""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    fasta_file = os.path.join(temp_dir, "temp_seq.fasta")
    with open(fasta_file, 'w') as f:
        f.write(">temp\n")
        f.write(sequence + "\n")
    
    output_file = os.path.join(temp_dir, "temp_seq.unico.dat0")
    result_file = run_sgp_main_for_sequence(fasta_file, output_file)
    
    if not result_file or not os.path.exists(result_file):
        return None
    
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(parts) >= 16:
                pim_vector = [float(x) for x in parts[:16]]
                pim_array = np.array(pim_vector)
                total = np.sum(pim_array)
                if total > 0:
                    pim_array = pim_array / total
                return pim_array
        
        return None
        
    except Exception as e:
        return None

def compute_grassmann_similarity(v1, v2):
    """Calcula similitud usando producto wedge"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    
    n = len(v1_norm)
    wedge_sum = 0
    for i in range(n):
        for j in range(i+1, n):
            wedge_sum += (v1_norm[i] * v2_norm[j] - v1_norm[j] * v2_norm[i]) ** 2
    
    wedge_magnitude = np.sqrt(wedge_sum)
    max_wedge = np.sqrt(n * (n-1) / 2)
    similarity = 1 - min(1, wedge_magnitude / (max_wedge + 1e-10))
    
    return similarity

def get_virus_representative_pim(virus_sequences, max_seq=30):
    """Obtiene el PIM representativo de un virus"""
    if not virus_sequences:
        return None
    
    pims = []
    temp_dir = tempfile.mkdtemp()
    
    to_process = virus_sequences[:max_seq]
    print(f"     ├─ Procesando {len(to_process)} secuencias para PIM...")
    
    for i, (header, seq) in enumerate(to_process):
        seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
        if len(seq_clean) < 10:
            continue
        
        try:
            pim = compute_pim_with_sgp_main(seq_clean, temp_dir)
            if pim is not None and np.sum(pim) > 0.01:
                pims.append(pim)
        except:
            continue
        
        if (i + 1) % 5 == 0:
            print(f"     ├─ Procesadas {i+1}/{len(to_process)} secuencias...")
    
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    if not pims:
        return None
    
    print(f"     ├─ PIMs válidos: {len(pims)}")
    
    avg_pim = np.mean(pims, axis=0)
    total = np.sum(avg_pim)
    if total > 0:
        avg_pim = avg_pim / total
    
    return avg_pim

def get_group_pims_optimized(group_name, max_seq=50):
    """Obtiene PIMs de un grupo usando muestreo"""
    group_file = get_filename(group_name)
    if not os.path.exists(group_file):
        print(f"     ⚠️ Archivo no encontrado: {group_file}")
        return []
    
    print(f"     ├─ Muestreando {get_display_name(group_name)}...")
    
    sampled = sample_sequences_fast(group_file, max_seq, mode='first')
    
    if not sampled:
        return []
    
    pims = []
    temp_dir = tempfile.mkdtemp()
    
    for i, (header, seq) in enumerate(sampled):
        seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
        if len(seq_clean) < 10:
            continue
        
        try:
            pim = compute_pim_with_sgp_main(seq_clean, temp_dir)
            if pim is not None and np.sum(pim) > 0.01:
                pims.append(pim)
        except:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"     ├─ Procesadas {i+1}/{len(sampled)} secuencias...")
    
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return pims

# ============================================================================
# FUNCIONES BIOPYTHON
# ============================================================================

def extract_biopython_features_enhanced(sequence):
    """Extrae características fisicoquímicas con BioPython"""
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
        
    except:
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

def run_biopython_optimized(input_file, virus_name, groups_to_analyze, max_sequences_per_group=30):
    """Ejecuta BioPython usando muestreo optimizado"""
    print(f"\n  🔬 Ejecutando BioPython para {virus_name} (muestreo)...")
    
    sequences_by_virus = extract_ebola_sequences_optimized(
        input_file, 
        max_seq=MAX_SEQUENCES_PER_VIRUS, 
        mode=SAMPLE_MODE
    )
    
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
        
        sampled = sample_sequences_fast(group_file, max_seq=max_sequences_per_group, mode='first')
        
        if not sampled:
            continue
        
        vectors = []
        for header, seq in sampled:
            features = extract_biopython_features_enhanced(seq)
            if features is not None and len(features) > 0:
                vectors.append(features)
        
        if not vectors:
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
            print(f"     ├─ {get_display_name(group)}: {results[group]:.6f} (n={len(similarities)})")
    
    return results

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def run_sgp_optimized(input_file, virus_name, groups_to_analyze, max_sequences=30):
    """Ejecuta SGPMAIN212.0v optimizado con muestreo"""
    print(f"\n  🧬 Ejecutando SGPMAIN212.0v para {get_display_name(virus_name)} (muestreo)...")
    
    if not check_sgp_main_available():
        print(f"  ❌ No se encontró {SGPMAIN_SCRIPT}")
        return None
    
    sequences_by_virus = extract_ebola_sequences_optimized(
        input_file, 
        max_seq=MAX_SEQUENCES_PER_VIRUS, 
        mode=SAMPLE_MODE
    )
    
    if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
        print(f"  ⚠️ No se encontraron secuencias para {get_display_name(virus_name)}")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
    
    virus_pim = get_virus_representative_pim(sequences_by_virus[virus_name], max_seq=max_sequences)
    
    if virus_pim is None or np.sum(virus_pim) < 0.01:
        print(f"  ❌ PIM inválido para {get_display_name(virus_name)}")
        return None
    
    print(f"     ├─ PIM del virus: dimensión {len(virus_pim)}, suma={np.sum(virus_pim):.4f}")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = get_filename(group)
        if not os.path.exists(group_file):
            print(f"     ⚠️ Archivo no encontrado: {group_file}")
            continue
        
        display = get_display_name(group)
        print(f"     ├─ Procesando {display}...")
        
        group_pims = get_group_pims_optimized(group, max_seq=MAX_SEQUENCES_PER_GROUP)
        
        if not group_pims:
            print(f"        └─ ⚠️ Sin PIMs válidos")
            continue
        
        similarities = []
        for pim_group in group_pims:
            sim = compute_grassmann_similarity(virus_pim, pim_group)
            similarities.append(sim)
        
        if similarities:
            results[group] = np.mean(similarities)
            print(f"        └─ Similitud media: {results[group]:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin similitudes válidas")
    
    return results

def analyze_ebola_virus_optimized(input_file, virus_info, groups_to_analyze):
    """Analiza un virus usando muestreo optimizado"""
    
    virus_name = virus_info['name']
    virus_display = virus_info['display']
    
    print("\n" + "=" * 80)
    print(f"🔬 ANALIZANDO: {virus_display}")
    print("=" * 80)
    
    if not os.path.exists(input_file):
        print(f"  ❌ Archivo no encontrado: {input_file}")
        return None
    
    print(f"\n  📌 Usando muestra de {MAX_SEQUENCES_PER_VIRUS} secuencias por virus")
    print(f"  📌 Usando muestra de {MAX_SEQUENCES_PER_GROUP} secuencias por grupo")
    
    sgp_results = run_sgp_optimized(input_file, virus_name, groups_to_analyze, MAX_SEQUENCES_PER_VIRUS)
    
    biopython_results = run_biopython_optimized(input_file, virus_name, groups_to_analyze, MAX_SEQUENCES_PER_GROUP)
    
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
    print(f"📋 TABLA COMPARATIVA: SGPMAIN212.0v vs BioPython ({virus_display})")
    print("=" * 80)
    print(f"{'Grupo':<22} {'SGPMAIN212':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
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
            'SGPMAIN212': sgp_val,
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

print("\n📂 Verificando archivo de entrada...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Archivo no encontrado: {INPUT_FILE}")
    # Buscar en DATA_PATH
    INPUT_FILE = os.path.join(DATA_PATH, "todoebola.txt")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Archivo no encontrado en {DATA_PATH}")
        sys.exit(1)

print(f"✅ Archivo encontrado: {INPUT_FILE}")

# Verificar archivos .dat0
print("\n📂 Verificando archivos .dat0...")
found_files = 0
missing_files = 0

for group in GROUP_FILES:
    filepath = GROUP_FILES[group]
    if os.path.exists(filepath):
        found_files += 1
    else:
        missing_files += 1
        print(f"  ⚠️ No encontrado: {os.path.basename(filepath)}")

print(f"  ✅ Archivos encontrados: {found_files}")
if missing_files > 0:
    print(f"  ⚠️ Archivos faltantes: {missing_files}")

# Verificar SGPMAIN
print("\n📂 Verificando SGPMAIN212.0v...")
if not check_sgp_main_available():
    print(f"⚠️ No se encontró {SGPMAIN_SCRIPT}")
else:
    print(f"✅ SGPMAIN212.0v encontrado: {SGPMAIN_SCRIPT}")

# Obtener virus disponibles
sequences_by_virus = extract_ebola_sequences_optimized(
    INPUT_FILE, 
    max_seq=MAX_SEQUENCES_PER_VIRUS, 
    mode=SAMPLE_MODE
)

available_viruses = []
for virus in EBOLA_VIRUSES:
    if virus['name'] in sequences_by_virus and sequences_by_virus[virus['name']]:
        available_viruses.append(virus)

if not available_viruses:
    print("\n❌ No se encontraron secuencias de virus Ébola en el archivo")
    sys.exit(1)

print(f"\n  📊 Se analizarán {len(available_viruses)} virus Ébola")
print(f"  📊 Se compararán con {len(GROUP_FILES)} grupos")

# ============================================================================
# ANALIZAR CADA VIRUS
# ============================================================================

GROUPS_TO_ANALYZE = list(GROUP_FILES.keys())

all_results = {}
all_dataframes = []

for virus in available_viruses:
    result = analyze_ebola_virus_optimized(INPUT_FILE, virus, GROUPS_TO_ANALYZE)
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
    
    print("\n  📊 MATRIZ SGPMAIN212.0v:")
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
print(f"  📊 Archivo de entrada: {INPUT_FILE}")
print(f"  📌 Ruta de datos: {DATA_PATH}")
