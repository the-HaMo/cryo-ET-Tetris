"""
Benchmark de comparación entre SAWLC y Tetris 3D
================================================

Compara el rendimiento de ambos algoritmos de inserción de proteínas
midiendo tiempo de ejecución y número de proteínas insertadas para
diferentes niveles de ocupancia objetivo.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Rutas a módulos hermanos (src/tetris_3d y src/sawlc)
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "tetris_3d"))
sys.path.insert(0, str(_SRC / "sawlc"))

# Parámetros comunes para ambos algoritmos
COMMON_PARAMS = {
    'VOI_SHAPE': (500, 500, 250),
    'VOXEL_SIZE': 10.0,  # A/vx
    'PROTEINS_LIST': [
        "in_10A/4v4r_10A.pns",
        "in_10A/3j9i_10A.pns",
        "in_10A/5mrc_10A.pns",
    ],
    'SEED': 42
}

# Niveles de ocupancia a evaluar (%)
OCCUPANCY_LEVELS = [10, 15]


class BenchmarkTetris3D:
    """Wrapper para ejecutar Tetris 3D con ocupancia objetivo"""

    def __init__(self, params: dict):
        self.params = params
        np.random.seed(params['SEED'])

    def run(self, target_occupancy: float, save_output: bool = False) -> Dict:
        from tetris import Tetris3D, xp
        from image_processing_3d import ImageProcessing3D
        from parser_3d import Parser3D
        from insert_proteins_tetris import pick_seed, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING

        start_time = time.time()
        data_dir = Path(__file__).resolve().parents[2] / "data"

        # Cargar y recortar proteínas (ordenadas por ocupación interna descendente)
        from insert_proteins_tetris import sorted_proteinSizes
        molecules = []
        for p_path in sorted_proteinSizes(self.params['PROTEINS_LIST']):
            vol, _ = Parser3D.load_protein(str(data_dir / p_path), str(data_dir))
            threshold = vol.max() * PROTEIN_ISO_THRESHOLD_RATIO
            coords = xp.argwhere(vol > threshold)
            if coords.size == 0:
                continue
            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0) + 1
            molecules.append((os.path.basename(p_path), vol[int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)]))

        if not molecules:
            raise ValueError("No se encontraron proteínas")

        global_threshold = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
        tetris = Tetris3D(dimensions=self.params['VOI_SHAPE'], threshold=global_threshold)
        allowed_mask = xp.ones(self.params['VOI_SHAPE'], dtype=bool)

        inserted = 0
        saturated = False
        print(f"\n[TETRIS] Objetivo: {target_occupancy*100:.1f}%")

        for name, volume in molecules:
            if float(tetris.get_occupancy()) >= target_occupancy:
                break
            box_size = max(volume.shape)
            target = pick_seed(allowed_mask, tetris.output_volume, global_threshold, box_size)
            failures = 0
            while failures < TRIES_CLUSTERING:
                if target is None or float(tetris.get_occupancy()) >= target_occupancy:
                    break
                rotated, _ = ImageProcessing3D.randomly_rotate(volume)
                rotated_bin = ImageProcessing3D.smooth_and_binarize(rotated, 1.5, global_threshold)
                template, _, _ = ImageProcessing3D.create_in_shell(rotated_bin, (0, 2), penalty=100)
                res = tetris.insert_molecule_3d(template, rotated, name, allowed_mask, target, box_size)
                if res == 'inserted':
                    inserted += 1
                    failures = 0
                    target = tetris.all_coordinates[-1]
                else:
                    failures += 1
                    target = pick_seed(allowed_mask, tetris.output_volume, global_threshold, box_size)

        current_occupancy = float(tetris.get_occupancy())
        if current_occupancy < target_occupancy:
            saturated = True
            print(f"[TETRIS] SATURACIÓN: {current_occupancy*100:.1f}% ({inserted} proteínas)")
        else:
            print(f"[TETRIS] Ocupancia alcanzada: {current_occupancy*100:.1f}% ({inserted} proteínas)")

        return {
            'time': time.time() - start_time,
            'proteins_inserted': inserted,
            'final_occupancy': current_occupancy,
            'saturated': saturated,
        }


class BenchmarkSAWLC:
    """Wrapper para ejecutar SAWLC hasta saturación y reportar ocupancia alcanzada"""

    def __init__(self, params: dict):
        self.params = params
        np.random.seed(params['SEED'])

    def run(self, target_occupancy: float, save_output: bool = False) -> Dict:
        import sys as _sys
        _sys.path.insert(0, str(_SRC / "sawlc"))
        from insert_proteins_in_membranes import insert_proteins_in_membrane, sorted_proteinSizes

        start_time = time.time()
        output_dir = Path(__file__).resolve().parents[2] / "data" / "data_generated" / "output" / "benchmark_sawlc_tmp"

        print(f"\n[SAWLC] Objetivo referencia: {target_occupancy*100:.1f}% (SAWLC corre hasta saturación)")

        try:
            proteins = sorted_proteinSizes(self.params['PROTEINS_LIST'])
            result = insert_proteins_in_membrane(None, proteins, str(output_dir), membrane_id=0)
            saturated = result is None
            # Leer ocupancia desde el volumen generado si está disponible
            final_occupancy = 0.0
            proteins_inserted = 0
            if result:
                import mrcfile, numpy as _np
                den_files = list(Path(result['output_dir']).glob("*.mrc"))
                if den_files:
                    with mrcfile.open(str(den_files[0]), mode='r') as mrc:
                        vol = mrc.data.astype(_np.float32)
                    final_occupancy = float(_np.count_nonzero(vol > 0) / vol.size)
            print(f"[SAWLC] Completado: {final_occupancy*100:.1f}%")
        except Exception as e:
            print(f"[SAWLC] Error: {e}")
            final_occupancy = 0.0
            proteins_inserted = 0
            saturated = True

        return {
            'time': time.time() - start_time,
            'proteins_inserted': proteins_inserted,
            'final_occupancy': final_occupancy,
            'saturated': saturated,
        }


def run_benchmark() -> Tuple[Dict, Dict]:
    """
    Ejecuta el benchmark completo para ambos algoritmos
    
    Returns:
        Tupla (resultados_tetris, resultados_sawlc)
    """
    results_tetris = {}
    results_sawlc = {}
    
    print("="*80)
    print("BENCHMARK: TETRIS 3D vs SAWLC")
    print("="*80)
    print(f"VOI Shape: {COMMON_PARAMS['VOI_SHAPE']}")
    print(f"Voxel Size: {COMMON_PARAMS['VOXEL_SIZE']} Å")
    print(f"Proteínas: {COMMON_PARAMS['PROTEINS_LIST']}")
    print(f"Niveles de ocupancia: {OCCUPANCY_LEVELS}%")
    print("="*80)
    
    # Crear benchmarks
    tetris_benchmark = BenchmarkTetris3D(COMMON_PARAMS)
    sawlc_benchmark = BenchmarkSAWLC(COMMON_PARAMS)
    
    both_saturated = False
    
    for occupancy_pct in OCCUPANCY_LEVELS:
        if both_saturated:
            print(f"\n[SKIP] Ambos algoritmos saturados. Deteniendo benchmark.")
            break
        
        occupancy = occupancy_pct / 100.0
        
        print(f"\n{'='*80}")
        print(f"Evaluando ocupancia objetivo: {occupancy_pct}%")
        print(f"{'='*80}")
        
        # Ejecutar Tetris 3D
        try:
            result_tetris = tetris_benchmark.run(occupancy)
            results_tetris[occupancy_pct] = result_tetris
        except Exception as e:
            print(f"[TETRIS] ERROR: {e}")
            results_tetris[occupancy_pct] = {
                'time': 0,
                'proteins_inserted': 0,
                'final_occupancy': 0,
                'saturated': True
            }
        
        # Ejecutar SAWLC
        try:
            result_sawlc = sawlc_benchmark.run(occupancy)
            results_sawlc[occupancy_pct] = result_sawlc
        except Exception as e:
            print(f"[SAWLC] ERROR: {e}")
            results_sawlc[occupancy_pct] = {
                'time': 0,
                'proteins_inserted': 0,
                'final_occupancy': 0,
                'saturated': True
            }
        
        # Verificar si ambos saturaron
        if results_tetris[occupancy_pct]['saturated'] and results_sawlc[occupancy_pct]['saturated']:
            both_saturated = True
    
    return results_tetris, results_sawlc


def plot_results(results_tetris: Dict, results_sawlc: Dict, output_dir: str):
    """Genera gráficas comparativas"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraer datos
    occupancies = sorted(results_tetris.keys())
    
    tetris_times = [results_tetris[occ]['time'] for occ in occupancies]
    tetris_proteins = [results_tetris[occ]['proteins_inserted'] for occ in occupancies]
    tetris_final_occ = [results_tetris[occ]['final_occupancy'] * 100 for occ in occupancies]
    
    sawlc_times = [results_sawlc[occ]['time'] for occ in occupancies]
    sawlc_proteins = [results_sawlc[occ]['proteins_inserted'] for occ in occupancies]
    sawlc_final_occ = [results_sawlc[occ]['final_occupancy'] * 100 for occ in occupancies]
    
    # Configurar estilo
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación: Tetris 3D vs SAWLC', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Tiempo vs Ocupancia
    ax1 = axes[0, 0]
    ax1.plot(occupancies, tetris_times, 'o-', label='Tetris 3D', linewidth=2, markersize=8)
    ax1.plot(occupancies, sawlc_times, 's-', label='SAWLC', linewidth=2, markersize=8)
    ax1.set_xlabel('Ocupancia objetivo (%)', fontsize=12)
    ax1.set_ylabel('Tiempo (s)', fontsize=12)
    ax1.set_title('Tiempo de ejecución vs Ocupancia', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Proteínas insertadas vs Ocupancia
    ax2 = axes[0, 1]
    ax2.plot(occupancies, tetris_proteins, 'o-', label='Tetris 3D', linewidth=2, markersize=8)
    ax2.plot(occupancies, sawlc_proteins, 's-', label='SAWLC', linewidth=2, markersize=8)
    ax2.set_xlabel('Ocupancia objetivo (%)', fontsize=12)
    ax2.set_ylabel('Número de proteínas', fontsize=12)
    ax2.set_title('Proteínas insertadas vs Ocupancia', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Eficiencia (proteínas/segundo)
    ax3 = axes[1, 0]
    tetris_efficiency = [p/t if t > 0 else 0 for p, t in zip(tetris_proteins, tetris_times)]
    sawlc_efficiency = [p/t if t > 0 else 0 for p, t in zip(sawlc_proteins, sawlc_times)]
    ax3.plot(occupancies, tetris_efficiency, 'o-', label='Tetris 3D', linewidth=2, markersize=8)
    ax3.plot(occupancies, sawlc_efficiency, 's-', label='SAWLC', linewidth=2, markersize=8)
    ax3.set_xlabel('Ocupancia objetivo (%)', fontsize=12)
    ax3.set_ylabel('Proteínas / segundo', fontsize=12)
    ax3.set_title('Eficiencia de inserción', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Ocupancia final alcanzada
    ax4 = axes[1, 1]
    ax4.plot(occupancies, tetris_final_occ, 'o-', label='Tetris 3D', linewidth=2, markersize=8)
    ax4.plot(occupancies, sawlc_final_occ, 's-', label='SAWLC', linewidth=2, markersize=8)
    ax4.plot(occupancies, occupancies, 'k--', label='Objetivo', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Ocupancia objetivo (%)', fontsize=12)
    ax4.set_ylabel('Ocupancia final (%)', fontsize=12)
    ax4.set_title('Precisión de ocupancia', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Gráfica guardada en: {output_dir}/benchmark_comparison.png")
    plt.close()


def save_results(results_tetris: Dict, results_sawlc: Dict, output_dir: str):
    """Guarda resultados en JSON"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'parameters': COMMON_PARAMS,
        'tetris_3d': results_tetris,
        'sawlc': results_sawlc
    }
    
    output_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[SAVE] Resultados guardados en: {output_file}")


def print_summary(results_tetris: Dict, results_sawlc: Dict):
    """Imprime resumen de resultados"""
    
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    
    print(f"\n{'Ocupancia':<12} {'Tetris Time':<15} {'Tetris Proteins':<18} {'SAWLC Time':<15} {'SAWLC Proteins'}")
    print("-"*80)
    
    for occ in sorted(results_tetris.keys()):
        t_time = results_tetris[occ]['time']
        t_prot = results_tetris[occ]['proteins_inserted']
        s_time = results_sawlc[occ]['time']
        s_prot = results_sawlc[occ]['proteins_inserted']
        
        print(f"{occ:>3}%         {t_time:>8.2f}s       {t_prot:>8}          {s_time:>8.2f}s       {s_prot:>8}")
    
    print("="*80)


if __name__ == '__main__':
    # Directorio de salida
    output_dir = Path(__file__).resolve().parents[2] / "data" / "data_generated" / "output" / "benchmark_results"
    
    # Ejecutar benchmark
    results_tetris, results_sawlc = run_benchmark()
    
    # Guardar resultados
    save_results(results_tetris, results_sawlc, output_dir)
    
    # Generar gráficas
    plot_results(results_tetris, results_sawlc, output_dir)
    
    # Imprimir resumen
    print_summary(results_tetris, results_sawlc)
    
    print("\n[DONE] Benchmark completado exitosamente!")
