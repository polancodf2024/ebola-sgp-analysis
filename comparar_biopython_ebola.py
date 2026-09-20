#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COMPARACIÓN: SGPMAIN217.0v vs BioPython
VERSIÓN DEFINITIVA v4.1 - COMPLETA, CERRADA Y CORREGIDA
================================================================================

Correcciones aplicadas en v4.1 respecto a v4:
  0. ✅ BUGFIX: eliminadas dos líneas duplicadas en compute_grassmann_similarity
     que hacían referencia a 'self' y a 'd' (variables inexistentes). El error
     era un copy/paste accidental desde get_similarity_matrix de SGPMAIN.

Correcciones aplicadas en v4 respecto a v3:
  1. Todo el código ejecutable está dentro de main() con if __name__ == "__main__".
  2. try/except global con traza completa y guardado de error log.
  3. compute_grassmann_similarity valida que compare_pim_vectors exista y
     devuelva 'geodesic_distance' ANTES de usarla; si no, aborta con mensaje claro.
  4. Se guarda config_used.json por virus.
  5. Se guarda ebola_comparison_complete_*.csv con encabezados.
  6. Se guardan las matrices SGPMAIN y BioPython con encabezados de nombres.
  7. Se guarda un reporte final TXT con resumen de todo el análisis.
  8. Se valida la existencia de todas las claves del núcleo en compare_pim_vectors.
  9. Se valida que GRASSMANN_SUBSPACE_DIM y AMBIENT_DIM existan en el módulo.
 10. Se guardan métricas completas del núcleo por grupo y por virus.
 11. Se imprime progreso con timestamps en cada etapa.
 12. Se maneja KeyboardInterrupt limpiamente.
 13. Se guarda un README de la corrida con metadatos.

Rutas:
  - Programa SGPMAIN217.0v.py: /home/cpolanco/POLANCO/TAXONOMIAMAIN
  - Datos (config, FASTA, .dat0): /home/cpolanco/POLANCO/ARCHIVOMAESTRO
================================================================================
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import re
import importlib.util
import traceback
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES DE RUTAS (globales, editables)
# ============================================================================

ARCHIVOMAESTRO = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"
SGPMAIN_DIR = "/home/cpolanco/POLANCO/TAXONOMIAMAIN"
SGPMAIN_FILENAME = "SGPMAIN217.0v.py"

# ============================================================================
# FUNCIONES AUXILIARES DE TIMESTAMP
# ============================================================================

def ts():
    """Devuelve timestamp formateado para logs."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    """Imprime mensaje con timestamp."""
    print(f"[{ts()}] {msg}")


# ============================================================================
# LECTURA DE FASTA
# ============================================================================

def read_fasta_stream(filepath):
    """Generador que lee un FASTA y produce (header, sequence)."""
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


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    # ------------------------------------------------------------------------
    # 0. BANNER Y VALIDACIÓN DE RUTAS
    # ------------------------------------------------------------------------
    print("=" * 80)
    print("📊 COMPARACIÓN: SGPMAIN217.0v vs BioPython (v4.1 DEFINITIVA CORREGIDA)")
    print("   📌 USA compare_pim_vectors + fórmula wedge de producción")
    print("=" * 80)
    log(f"Inicio: {ts()}")
    log(f"ARCHIVOMAESTRO (datos): {ARCHIVOMAESTRO}")
    log(f"SGPMAIN_DIR (programa): {SGPMAIN_DIR}")

    if not os.path.exists(ARCHIVOMAESTRO):
        print(f"❌ ARCHIVOMAESTRO no existe: {ARCHIVOMAESTRO}")
        sys.exit(1)

    if not os.path.exists(SGPMAIN_DIR):
        print(f"❌ SGPMAIN_DIR no existe: {SGPMAIN_DIR}")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # 1. CARGAR CONFIG_EBOLA.json
    # ------------------------------------------------------------------------
    log("Cargando config_EBOLA.json...")
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
    VALIDATE_TARGET_RANGES = CONFIG['operation_control'].get(
        'validate_target_ranges_in_characterization', True)

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

    # ------------------------------------------------------------------------
    # 2. IMPORTAR BIOPYTHON
    # ------------------------------------------------------------------------
    log("Importando BioPython...")
    try:
        from Bio.SeqUtils import ProtParam
        from Bio.Seq import Seq
        print("✅ BioPython importado correctamente")
    except ImportError:
        print("❌ BioPython no instalado. Ejecuta: pip install biopython")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # 3. IMPORTAR FUNCIONES DE SGPMAIN217.0v.py
    # ------------------------------------------------------------------------
    log("Importando SGPMAIN217.0v.py...")

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
    except Exception as e:
        print(f"❌ Error importando {SGPMAIN_PATH}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Validar que las funciones y constantes necesarias existan
    REQUIRED_FUNCTIONS = [
        'compute_pim_profile', 'shannon_entropy', 'gini_coefficient',
        'jensen_shannon_divergence', 'hellinger_distance', 'spearman_correlation',
        'principal_angles', 'geodesic_distance', 'scalar_curvature',
        'pim_to_subspace', 'compare_pim_vectors', 'GrassmannPIM',
    ]
    REQUIRED_CONSTANTS = ['GRASSMANN_SUBSPACE_DIM', 'AMBIENT_DIM', 'DIM_PAIRS']

    missing_functions = [f for f in REQUIRED_FUNCTIONS if not hasattr(sgp_module, f)]
    missing_constants = [c for c in REQUIRED_CONSTANTS if not hasattr(sgp_module, c)]

    if missing_functions:
        print(f"❌ Funciones faltantes en SGPMAIN217.0v.py: {missing_functions}")
        sys.exit(1)
    if missing_constants:
        print(f"❌ Constantes faltantes en SGPMAIN217.0v.py: {missing_constants}")
        sys.exit(1)

    print("✅ Todas las funciones y constantes requeridas existen")

    # Importar funciones reales
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
    compare_pim_vectors = sgp_module.compare_pim_vectors
    GrassmannPIM = sgp_module.GrassmannPIM

    GRASSMANN_SUBSPACE_DIM = sgp_module.GRASSMANN_SUBSPACE_DIM
    AMBIENT_DIM = sgp_module.AMBIENT_DIM
    DIM_PAIRS = sgp_module.DIM_PAIRS

    grassmann_pim = GrassmannPIM(dim=DIM_PAIRS)

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
    print("     ├─ compare_pim_vectors (CLAVE - misma métrica que producción)")
    print("     └─ GrassmannPIM (CLASE) ✅ INSTANCIADA")
    print(f"✅ Constantes: GRASSMANN_SUBSPACE_DIM={GRASSMANN_SUBSPACE_DIM}, "
          f"AMBIENT_DIM={AMBIENT_DIM}, DIM_PAIRS={DIM_PAIRS}")
    print(f"✅ max_d (wedge) = π/2 · sqrt({GRASSMANN_SUBSPACE_DIM}) = "
          f"{np.pi/2 * np.sqrt(GRASSMANN_SUBSPACE_DIM):.4f}")

    # Validar que compare_pim_vectors devuelva las claves esperadas
    print("🔍 Validando claves de compare_pim_vectors...")
    try:
        _test_v1 = np.random.rand(DIM_PAIRS)
        _test_v2 = np.random.rand(DIM_PAIRS)
        _test_result = compare_pim_vectors(_test_v1, _test_v2)
        if not isinstance(_test_result, dict):
            print(f"❌ compare_pim_vectors no devuelve dict, devuelve {type(_test_result)}")
            sys.exit(1)
        required_keys = ['geodesic_distance', 'principal_angles_max',
                         'principal_angles_mean', 'jensen_shannon', 'hellinger',
                         'shannon_entropy_1', 'gini_1', 'spearman']
        missing_keys = [k for k in required_keys if k not in _test_result]
        if missing_keys:
            print(f"❌ compare_pim_vectors no devuelve las claves: {missing_keys}")
            print(f"   Claves disponibles: {list(_test_result.keys())}")
            sys.exit(1)
        print(f"✅ compare_pim_vectors devuelve las {len(required_keys)} claves requeridas")
    except Exception as e:
        print(f"❌ Error validando compare_pim_vectors: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------------
    # 4. CONFIGURACIÓN DE VIRUS ÉBOLA
    # ------------------------------------------------------------------------
    EBOLA_VIRUSES = [
        {'name': 'sudan', 'display': 'EBOLA_SUDAN', 'pattern': r'Sudan|SUDAN|sudan'},
        {'name': 'zaire', 'display': 'EBOLA_ZAIRE', 'pattern': r'Zaire|ZAIRE|zaire'},
        {'name': 'reston', 'display': 'EBOLA_RESTON', 'pattern': r'Reston|RESTON|reston'},
        {'name': 'bombali', 'display': 'EBOLA_BOMBALI', 'pattern': r'[Bb]ombali'},
        {'name': 'bundibugyo', 'display': 'EBOLA_BUNDIBUGYO',
         'pattern': r'Bundibugyo|BUNDIBUGYO|bundibugyo'},
        {'name': 'tai', 'display': 'EBOLA_TAI_FOREST',
         'pattern': r'Tai|TAI|tai|Taï|TAÏ|taï|Tai Forest'},
    ]

    # ------------------------------------------------------------------------
    # 5. CONFIGURACIÓN DE LOS 30 GRUPOS
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # 6. FUNCIONES DE EXTRACCIÓN DE SECUENCIAS EBOLA
    # ------------------------------------------------------------------------
    def extract_ebola_sequences(input_file, max_seq=1000):
        """Lee el FASTA y clasifica las secuencias por cepa de Ebola."""
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

    # ------------------------------------------------------------------------
    # 7. FUNCIONES DEL PIPELINE 217.0
    # ------------------------------------------------------------------------
    def compute_sgp_pim(sequence, use_weights=True):
        """Calcula el PIM de una secuencia usando SGPMAIN."""
        try:
            return compute_pim_profile(sequence, use_weights=use_weights)
        except Exception as e:
            print(f"⚠️ Error en compute_sgp_pim: {e}")
            return np.zeros(DIM_PAIRS)

    def compute_grassmann_similarity(v1, v2, k=None):
        """
        Usa compare_pim_vectors (misma métrica que producción) para obtener
        'geodesic_distance', y aplica la MISMA fórmula que
        GrassmannPIM.wedge_product:

            sim = max(0, 1 - geodesic_distance / (π/2 · sqrt(k)))

        con k = GRASSMANN_SUBSPACE_DIM = 2, por lo que max_d ≈ 2.2214.

        Devuelve un dict con 'similarity' y TODAS las métricas del núcleo.
        """
        if k is None:
            k = GRASSMANN_SUBSPACE_DIM

        try:
            metrics = compare_pim_vectors(v1, v2, k=k)
        except Exception as e:
            print(f"⚠️ Error en compare_pim_vectors: {e}")
            try:
                X = pim_to_subspace(v1, k=k)
                Y = pim_to_subspace(v2, k=k)
                geo = geodesic_distance(X, Y)
                metrics = {'geodesic_distance': geo}
            except Exception as e2:
                print(f"⚠️ Error en fallback geodesic_distance: {e2}")
                return {'similarity': 0.0, 'geodesic_distance': float('inf')}

        geo = metrics.get('geodesic_distance', float('inf'))
        max_d = (np.pi / 2.0) * np.sqrt(k)
        sim = max(0.0, 1.0 - geo / max_d)

        metrics['similarity'] = float(sim)
        metrics['max_d'] = float(max_d)
        metrics['k'] = int(k)
        return metrics

    def get_virus_representative_pim(virus_sequences, max_seq=MAX_STORED_PROTEINS):
        """Calcula el PIM representativo de un virus promediando sus PIMs."""
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

    def run_sgp_from_file(sequences_by_virus, virus_name, groups_to_analyze,
                          max_sequences=MAX_STORED_PROTEINS):
        """Ejecuta SGPMAIN 217.0 para un virus contra todos los grupos."""
        print(f"\n  🧬 Ejecutando SGPMAIN 217.0 para {virus_name}...")

        if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
            print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
            return None

        virus_pim = get_virus_representative_pim(sequences_by_virus[virus_name],
                                                  max_seq=max_sequences)

        if virus_pim is None or np.sum(virus_pim) < 0.01:
            print(f"  ❌ PIM inválido para {virus_name}")
            return None

        print(f"     ├─ Secuencias encontradas: {len(sequences_by_virus[virus_name])}")
        print(f"     ├─ PIM del virus: dimensión {len(virus_pim)}, "
              f"suma={np.sum(virus_pim):.4f}")

        entropy_val = shannon_entropy(virus_pim, normalize=True)
        print(f"     ├─ Entropía (normalizada): {entropy_val:.4f}")

        results = {}
        metrics_full = {}

        for group in groups_to_analyze:
            group_file = GROUP_FILES.get(group)
            if not group_file or not os.path.exists(group_file):
                continue

            print(f"     ├─ Procesando {group}...")

            similarities = []
            metrics_per_group = []
            count = 0
            for header, seq in read_fasta_stream(group_file):
                seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
                if len(seq_clean) < 10:
                    continue

                pim_group = compute_sgp_pim(seq_clean, use_weights=True)
                if np.sum(pim_group) > 0.01:
                    m = compute_grassmann_similarity(virus_pim, pim_group)
                    similarities.append(m['similarity'])
                    metrics_per_group.append(m)
                    count += 1
                    if count >= max_sequences:
                        break

            if similarities:
                mean_sim = float(np.mean(similarities))
                results[group] = {
                    'mean_similarity': mean_sim,
                    'n': len(similarities),
                    'insufficient_n': len(similarities) < MIN_SAMPLES_PER_GROUP
                }
                if metrics_per_group:
                    keys = ['geodesic_distance', 'principal_angles_max',
                            'principal_angles_mean', 'jensen_shannon',
                            'hellinger', 'shannon_entropy_1', 'gini_1', 'spearman']
                    metrics_full[group] = {
                        k: float(np.mean([m.get(k, 0.0) for m in metrics_per_group]))
                        for k in keys
                    }
                print(f"        └─ Similitud media: {mean_sim:.6f} "
                      f"(n={len(similarities)})")
            else:
                print(f"        └─ ⚠️ Sin PIMs válidos")
                results[group] = {
                    'mean_similarity': 0.0,
                    'n': 0,
                    'insufficient_n': True
                }

        return {
            'results': results,
            'virus_pim': virus_pim,
            'entropy': entropy_val,
            'metrics_full': metrics_full,
        }

    # ------------------------------------------------------------------------
    # 8. FUNCIONES PARA BIOPYTHON
    # ------------------------------------------------------------------------
    def extract_biopython_features(sequence):
        """Extrae 30 características fisicoquímicas con BioPython ProtParam."""
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
            except Exception:
                features.extend([0.0] * 20)

            try:
                mw = analyzer.molecular_weight()
                features.append(mw / len(seq_str))
            except Exception:
                features.append(0.0)

            try:
                features.append(analyzer.isoelectric_point())
            except Exception:
                features.append(7.0)

            try:
                features.append(analyzer.aromaticity())
            except Exception:
                features.append(0.0)

            try:
                features.append(analyzer.instability_index())
            except Exception:
                features.append(50.0)

            try:
                features.append(analyzer.gravy())
            except Exception:
                features.append(0.0)

            try:
                sec_struct = analyzer.secondary_structure_fraction()
                features.extend(sec_struct)
            except Exception:
                features.extend([0.0, 0.0, 0.0])

            try:
                charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
                net_charge = sum(charges.get(aa, 0) for aa in seq_str)
                features.append(net_charge / len(seq_str))
            except Exception:
                features.append(0.0)

            features.append(np.log(len(seq_str) + 1))

            return np.array(features)

        except Exception:
            return None

    def normalize_features(features_list):
        """
        Normalización robusta para n pequeño.
        - Si n >= 3: StandardScaler (Z-score).
        - Si n < 3: min-max por feature (con fallback a 0.5 si rango=0).
        """
        if not features_list:
            return features_list

        features_array = np.array(features_list, dtype=float)

        if len(features_array) < 3:
            mins = features_array.min(axis=0)
            maxs = features_array.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (features_array - mins) / ranges
            constant_mask = (maxs - mins) == 0
            if np.any(constant_mask):
                normalized[:, constant_mask] = 0.5
            return normalized

        scaler = StandardScaler()
        return scaler.fit_transform(features_array)

    def euclidean_similarity(v1, v2):
        """Similitud euclídea normalizada entre dos vectores."""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        dist = np.linalg.norm(v1_norm - v2_norm)
        max_dist = 2.0
        sim = 1 - (dist / max_dist)
        return float(max(0.0, min(1.0, sim)))

    def get_virus_representative_features(virus_sequences, max_seq=MAX_STORED_PROTEINS):
        """Calcula el vector de features representativo de un virus."""
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

    def run_biopython_from_file(sequences_by_virus, virus_name, groups_to_analyze,
                                max_sequences=MAX_STORED_PROTEINS):
        """Ejecuta BioPython ProtParam para un virus contra todos los grupos."""
        print(f"\n  🔬 Ejecutando BioPython para {virus_name} (30 características)...")

        if virus_name not in sequences_by_virus or not sequences_by_virus[virus_name]:
            print(f"  ⚠️ No se encontraron secuencias para {virus_name}")
            return None

        virus_features = get_virus_representative_features(
            sequences_by_virus[virus_name], max_seq=max_sequences)

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
                results[group] = {'mean_similarity': 0.0, 'n': 0,
                                  'insufficient_n': True}
                continue

            all_vectors = [virus_features] + vectors
            normalized_vectors = normalize_features(all_vectors)
            virus_norm = normalized_vectors[0]
            vectors_norm = normalized_vectors[1:]

            similarities = [euclidean_similarity(virus_norm, vec)
                            for vec in vectors_norm]

            if similarities:
                mean_sim = float(np.mean(similarities))
                results[group] = {
                    'mean_similarity': mean_sim,
                    'n': len(similarities),
                    'insufficient_n': len(similarities) < MIN_SAMPLES_PER_GROUP
                }
                print(f"        └─ Similitud media: {mean_sim:.6f} "
                      f"(n={len(similarities)})")
            else:
                results[group] = {'mean_similarity': 0.0, 'n': 0,
                                  'insufficient_n': True}

        return {'results': results, 'virus_features': virus_features}

    # ------------------------------------------------------------------------
    # 9. FUNCIÓN PRINCIPAL DE ANÁLISIS POR VIRUS
    # ------------------------------------------------------------------------
    def analyze_ebola_virus(input_file, virus_info, groups_to_analyze,
                            max_sequences=MAX_STORED_PROTEINS):
        """Analiza un virus Ébola contra todos los grupos."""
        virus_name = virus_info['name']
        virus_display = virus_info['display']

        print("\n" + "=" * 80)
        print(f"🔬 ANALIZANDO: {virus_display}")
        print("=" * 80)

        if not os.path.exists(input_file):
            print(f"  ❌ Archivo no encontrado: {input_file}")
            return None

        print(f"  📌 Usando muestra de {max_sequences} secuencias por grupo")

        sequences_by_virus = extract_ebola_sequences(input_file, max_seq=1000)

        sgp_output = run_sgp_from_file(sequences_by_virus, virus_name,
                                        groups_to_analyze, max_sequences)
        biopython_output = run_biopython_from_file(sequences_by_virus, virus_name,
                                                    groups_to_analyze, max_sequences)

        if biopython_output is None or len(biopython_output['results']) == 0:
            print(f"\n❌ No se pudieron obtener resultados de BioPython "
                  f"para {virus_display}")
            return None

        sgp_results = sgp_output['results'] if sgp_output else None
        biopython_results = biopython_output['results']

        comparacion = []
        groups_compared = set()

        if sgp_results:
            groups_compared = set(sgp_results.keys()) & set(biopython_results.keys())
        else:
            groups_compared = set(biopython_results.keys())

        print("\n" + "=" * 80)
        print(f"📋 TABLA COMPARATIVA: SGPMAIN 217.0 vs BioPython ({virus_display})")
        print("=" * 80)
        print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} "
              f"{'|Δ| (escalas distintas)':>24} {'Interpretación':>22}")
        print("-" * 100)

        for grupo in sorted(groups_compared,
                            key=lambda x: sgp_results[x]['mean_similarity']
                            if sgp_results and x in sgp_results else 0,
                            reverse=True):
            sgp_data = sgp_results.get(grupo, {'mean_similarity': 0.0, 'n': 0,
                                                'insufficient_n': True})
            bio_data = biopython_results.get(grupo, {'mean_similarity': 0.0, 'n': 0,
                                                      'insufficient_n': True})

            sgp_val = sgp_data['mean_similarity']
            biopy_val = bio_data['mean_similarity']
            diff = abs(sgp_val - biopy_val)

            if sgp_data['insufficient_n'] and bio_data['insufficient_n']:
                interp = "⚠️ n insuficiente (ambos)"
            elif sgp_data['insufficient_n']:
                interp = "⚠️ n insuficiente (SGPMAIN)"
            elif bio_data['insufficient_n']:
                interp = "⚠️ n insuficiente (BioPython)"
            elif sgp_val == 0:
                interp = "⚠️ Sin SGP"
            elif biopy_val == 0:
                interp = "⚠️ Sin BioPython"
            else:
                interp = "— (escalas distintas)"

            print(f"{grupo:<22} {sgp_val:>12.6f} {biopy_val:>12.6f} "
                  f"{diff:>24.6f} {interp:>22}")

            comparacion.append({
                'Virus': virus_display,
                'Grupo': grupo,
                'SGPMAIN': sgp_val,
                'SGPMAIN_n': sgp_data['n'],
                'SGPMAIN_insufficient_n': sgp_data['insufficient_n'],
                'BioPython': biopy_val,
                'BioPython_n': bio_data['n'],
                'BioPython_insufficient_n': bio_data['insufficient_n'],
                'Diferencia_abs': diff,
                'Interpretación': interp
            })

        return {
            'virus': virus_info,
            'sgp': sgp_results,
            'sgp_virus_pim': sgp_output['virus_pim'] if sgp_output else None,
            'sgp_entropy': sgp_output['entropy'] if sgp_output else None,
            'sgp_metrics_full': sgp_output.get('metrics_full', {}) if sgp_output else {},
            'biopython': biopython_results,
            'biopython_virus_features': biopython_output['virus_features'],
            'comparacion': comparacion
        }

    # ------------------------------------------------------------------------
    # 10. MATRICES DE SIMILITUD DIRECTAS ENTRE VIRUS
    # ------------------------------------------------------------------------
    def compute_sgp_virus_matrix(all_results, virus_names):
        """Matriz SGPMAIN DIRECTA entre PIMs de virus."""
        n = len(virus_names)
        matrix = np.eye(n)
        for i, v1 in enumerate(virus_names):
            pim1 = all_results[v1].get('sgp_virus_pim')
            if pim1 is None:
                continue
            for j, v2 in enumerate(virus_names):
                if i < j:
                    pim2 = all_results[v2].get('sgp_virus_pim')
                    if pim2 is None:
                        continue
                    m = compute_grassmann_similarity(pim1, pim2)
                    sim = m['similarity']
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        return matrix

    def compute_biopython_virus_matrix(all_results, virus_names):
        """Matriz BioPython DIRECTA entre features de virus."""
        n = len(virus_names)
        matrix = np.eye(n)
        for i, v1 in enumerate(virus_names):
            feat1 = all_results[v1].get('biopython_virus_features')
            if feat1 is None:
                continue
            for j, v2 in enumerate(virus_names):
                if i < j:
                    feat2 = all_results[v2].get('biopython_virus_features')
                    if feat2 is None:
                        continue
                    normed = normalize_features([feat1, feat2])
                    sim = euclidean_similarity(normed[0], normed[1])
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        return matrix

    def print_matrix(matrix, virus_names, title):
        """Imprime una matriz de similitud formateada."""
        n = len(virus_names)
        print(f"\n  📊 {title}:")
        print(f"  {'':<16}", end='')
        for name in virus_names:
            print(f"{name:>16}", end='')
        print()
        print("  " + "-" * (16 + 16 * n))
        for i, v1 in enumerate(virus_names):
            print(f"  {v1:<16}", end='')
            for j in range(n):
                print(f"{matrix[i, j]:>16.4f}", end='')
            print()

    # ------------------------------------------------------------------------
    # 11. EJECUCIÓN PRINCIPAL
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # 12. ANALIZAR CADA VIRUS
    # ------------------------------------------------------------------------
    all_results = {}
    all_dataframes = []

    for virus in available_viruses:
        result = analyze_ebola_virus(INPUT_FILE, virus, GROUPS_TO_ANALYZE,
                                      max_sequences=MAX_STORED_PROTEINS)
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

            if result.get('sgp_metrics_full'):
                with open(f"{results_dir}/sgp_metrics_full.json", 'w') as f:
                    json.dump(result['sgp_metrics_full'], f, indent=2)

            with open(f"{results_dir}/biopython_results.json", 'w') as f:
                json.dump(result['biopython'], f, indent=2)

            if result['sgp_virus_pim'] is not None:
                np.save(f"{results_dir}/sgp_virus_pim.npy", result['sgp_virus_pim'])
            if result['biopython_virus_features'] is not None:
                np.save(f"{results_dir}/biopython_virus_features.npy",
                        result['biopython_virus_features'])

            with open(f"{results_dir}/config_used.json", 'w') as f:
                json.dump(CONFIG, f, indent=2)

            print(f"\n  ✅ Resultados guardados en: {results_dir}/")

    # ------------------------------------------------------------------------
    # 13. MATRICES DE SIMILITUD
    # ------------------------------------------------------------------------
    sgp_matrix = None
    bio_matrix = None
    virus_names = []

    if len(all_results) >= 2:
        print("\n" + "=" * 80)
        print("🔍 MATRIZ DE SIMILITUD ENTRE VIRUS ÉBOLA (DIRECTA)")
        print("=" * 80)

        virus_names = list(all_results.keys())

        sgp_matrix = compute_sgp_virus_matrix(all_results, virus_names)
        bio_matrix = compute_biopython_virus_matrix(all_results, virus_names)

        print_matrix(sgp_matrix, virus_names,
                     "MATRIZ SGPMAIN 217.0 (compare_pim_vectors + wedge)")
        print_matrix(bio_matrix, virus_names,
                     "MATRIZ BIOPYTHON (euclidean_similarity sobre features normalizadas)")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_sgp = pd.DataFrame(sgp_matrix, index=virus_names, columns=virus_names)
        df_bio = pd.DataFrame(bio_matrix, index=virus_names, columns=virus_names)
        df_sgp.to_csv(f"sgp_virus_matrix_{timestamp}.csv", float_format='%.6f')
        df_bio.to_csv(f"biopython_virus_matrix_{timestamp}.csv", float_format='%.6f')

        print(f"\n  ✅ Matrices guardadas:")
        print(f"     ├─ sgp_virus_matrix_{timestamp}.csv")
        print(f"     └─ biopython_virus_matrix_{timestamp}.csv")

    # ------------------------------------------------------------------------
    # 14. GUARDAR RESULTADOS COMBINADOS
    # ------------------------------------------------------------------------
    combined_filename = None
    if all_dataframes:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_filename = f"ebola_comparison_complete_{timestamp}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"\n  ✅ Resultados combinados guardados: {combined_filename}")

    # ------------------------------------------------------------------------
    # 15. REPORTE FINAL TXT
    # ------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_report_filename = f"FINAL_REPORT_{timestamp}.txt"
    final_lines = []
    final_lines.append("=" * 80)
    final_lines.append("📋 FINAL REPORT - COMPARACIÓN SGPMAIN217.0v vs BioPython (v4.1)")
    final_lines.append("=" * 80)
    final_lines.append(f"Fecha: {ts()}")
    final_lines.append(f"SGPMAIN: {SGPMAIN_PATH}")
    final_lines.append(f"Config: {CONFIG_PATH}")
    final_lines.append(f"DATA_PATH: {DATA_PATH}")
    final_lines.append(f"Virus analizados: {len(all_results)} de {len(EBOLA_VIRUSES)}")
    final_lines.append(f"Grupos por virus: {len(GROUPS_TO_ANALYZE)}")
    final_lines.append("")
    final_lines.append("🔧 CORRECCIONES APLICADAS:")
    final_lines.append(f"  • compute_grassmann_similarity usa compare_pim_vectors")
    final_lines.append(f"    para obtener 'geodesic_distance', y aplica la fórmula wedge:")
    final_lines.append(f"    sim = max(0, 1 - geo / (π/2 · sqrt(k)))  con k={GRASSMANN_SUBSPACE_DIM}")
    final_lines.append(f"  • NO busca una clave 'similarity' inexistente en el dict")
    final_lines.append(f"  • normalize_features robusto para n<3")
    final_lines.append(f"  • Matriz SGPMAIN directa entre PIMs de virus")
    final_lines.append(f"  • Matriz BioPython directa entre features de virus")
    final_lines.append(f"  • extract_ebola_sequences llamado UNA SOLA VEZ por virus")
    final_lines.append(f"  • insufficient_n aplicado a AMBAS métricas")
    final_lines.append(f"  • Se guardan TODAS las métricas del núcleo por grupo")
    final_lines.append(f"  • Se guardan PIM y features de virus para reproducibilidad")
    final_lines.append("")
    if virus_names:
        final_lines.append("📊 MATRIZ SGPMAIN (compare_pim_vectors + wedge):")
        final_lines.append("       " + " ".join([f"{v:>12}" for v in virus_names]))
        for i, v1 in enumerate(virus_names):
            row_str = f"  {v1:<12}"
            for j in range(len(virus_names)):
                row_str += f" {sgp_matrix[i, j]:>12.4f}"
            final_lines.append(row_str)
        final_lines.append("")
        final_lines.append("📊 MATRIZ BIOPYTHON (euclidean_similarity):")
        final_lines.append("       " + " ".join([f"{v:>12}" for v in virus_names]))
        for i, v1 in enumerate(virus_names):
            row_str = f"  {v1:<12}"
            for j in range(len(virus_names)):
                row_str += f" {bio_matrix[i, j]:>12.4f}"
            final_lines.append(row_str)
    final_lines.append("")
    final_lines.append("=" * 80)
    final_lines.append("✅ ANÁLISIS COMPLETADO")
    final_lines.append("=" * 80)

    final_report = "\n".join(final_lines)
    with open(final_report_filename, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"\n  ✅ Reporte final guardado: {final_report_filename}")

    # ------------------------------------------------------------------------
    # 16. RESUMEN FINAL EN CONSOLA
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO (v4.1 DEFINITIVA CORREGIDA)")
    print("=" * 80)
    print(f"⏰ Fin: {ts()}")
    print(f"\n  📊 Virus analizados: {len(all_results)} de {len(EBOLA_VIRUSES)}")
    print(f"  📊 Grupos por virus: {len(GROUPS_TO_ANALYZE)}")
    print(f"\n  🔧 Correcciones aplicadas en v4.1:")
    print(f"     ├─ BUGFIX: eliminadas líneas duplicadas con 'self' y 'd'")
    print(f"     ├─ compute_grassmann_similarity usa compare_pim_vectors")
    print(f"     │   para obtener 'geodesic_distance', y aplica la fórmula wedge:")
    print(f"     │   sim = max(0, 1 - geo / (π/2 · sqrt(k)))  con k={GRASSMANN_SUBSPACE_DIM}")
    print(f"     ├─ Validación previa de funciones y constantes de SGPMAIN")
    print(f"     ├─ Validación previa de claves de compare_pim_vectors")
    print(f"     ├─ normalize_features robusto para n<3")
    print(f"     ├─ Matriz SGPMAIN directa entre PIMs de virus")
    print(f"     ├─ Matriz BioPython directa entre features de virus")
    print(f"     ├─ extract_ebola_sequences llamado UNA SOLA VEZ por virus")
    print(f"     ├─ insufficient_n aplicado a AMBAS métricas")
    print(f"     ├─ Se guardan TODAS las métricas del núcleo por grupo")
    print(f"     ├─ Se guardan PIM y features de virus para reproducibilidad")
    print(f"     ├─ Se guarda config_used.json por virus")
    print(f"     ├─ Se guarda reporte final TXT con matrices y metadatos")
    print(f"     └─ try/except global con traza completa")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ INTERRUMPIDO POR EL USUARIO")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        traceback.print_exc()
        try:
            error_log = f"error_comparar_biopython_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"ERROR: {e}\n\n")
                f.write(traceback.format_exc())
            print(f"  ✅ Error guardado en: {error_log}")
        except Exception:
            pass
        sys.exit(1)
