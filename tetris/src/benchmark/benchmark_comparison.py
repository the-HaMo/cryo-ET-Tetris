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

# Añadir paths para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'tetris_3d'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sawlc'))

# Parámetros comunes para ambos algoritmos
COMMON_PARAMS = {
    'VOI_SHAPE': (300, 300, 250),
    'VOXEL_SIZE': 10.0,  # A/vx
    'PROTEINS_LIST': [
        "in_10A/4v4r_10A.pns",
        "in_10A/3j9i_10A.pns",
        "in_10A/5mrc_10A.pns",
    ],
    'SEED': 42
}

# Niveles de ocupancia a evaluar (%)
OCCUPANCY_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50]


class BenchmarkTetris3D:
    """Wrapper para ejecutar Tetris 3D con ocupancia objetivo"""
    
    def __init__(self, params: dict):
        from image_processing_3d import ImageProcessing3D
        from parser_3d import Parser3D
        
        self.params = params
        self.ImageProcessing3D = ImageProcessing3D
        self.Parser3D = Parser3D
        np.random.seed(params['SEED'])
    
    def run(self, target_occupancy: float, save_output: bool = False) -> Dict:
        """
        Ejecuta Tetris 3D hasta alcanzar ocupancia objetivo
        
        Returns:
            Dict con: time, proteins_inserted, final_occupancy, saturated
        """
        from tetris import Tetris3D
        
        start_time = time.time()
        
        # Rutas
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        
        # Cargar proteínas
        molecules = {}
        for protein_path in self.params['PROTEINS_LIST']:
            full_path = os.path.join(data_dir, protein_path)
            if os.path.exists(full_path):
                name = os.path.basename(protein_path)
                try:
                    volume, _ = self.Parser3D.load_protein(full_path, data_dir)
                    molecules[name] = volume
                except Exception as e:
                    print(f"Error cargando {name}: {e}")
        
        if len(molecules) == 0:
            raise ValueError("No se encontraron proteínas")
        
        # Crear Tetris 3D
        tetris = Tetris3D(
            dimensions=self.params['VOI_SHAPE'],
            sigma=1.5,
            threshold=1,
            insertion_distances=(-2, 0)
        )
        
        # Insertar moléculas
        mol_list = list(molecules.values())
        mol_names = list(molecules.keys())
        inserted = 0
        saturated = False
        
        print(f"\n[TETRIS] Objetivo: {target_occupancy*100:.1f}%")
        
        while True:
            mol_idx = inserted % len(mol_list)
            mol_name = mol_names[mol_idx]
            
            success = tetris.insert_molecule_3d(mol_list[mol_idx], mol_name=mol_name)
            
            if success:
                inserted += 1
                current_occupancy = tetris.get_occupancy()
                
                # Verificar si alcanzamos objetivo
                if current_occupancy >= target_occupancy:
                    print(f"[TETRIS] Ocupancia alcanzada: {current_occupancy*100:.1f}% ({inserted} proteínas)")
                    break
            else:
                # Saturación
                current_occupancy = tetris.get_occupancy()
                saturated = True
                print(f"[TETRIS] SATURACIÓN: {current_occupancy*100:.1f}% ({inserted} proteínas)")
                break
        
        elapsed_time = time.time() - start_time
        
        return {
            'time': elapsed_time,
            'proteins_inserted': inserted,
            'final_occupancy': current_occupancy,
            'saturated': saturated
        }


class BenchmarkSAWLC:
    """Wrapper para ejecutar SAWLC con ocupancia objetivo"""
    
    def __init__(self, params: dict):
        self.params = params
        np.random.seed(params['SEED'])
    
    def run(self, target_occupancy: float, save_output: bool = False) -> Dict:
        """
        Ejecuta SAWLC hasta alcanzar ocupancia objetivo
        
        Returns:
            Dict con: time, proteins_inserted, final_occupancy, saturated
        """
        from polnet.tomogram import SynthTomo
        
        start_time = time.time()
        
        # Rutas
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = Path(base_dir) / 'data'
        
        print(f"\n[SAWLC] Objetivo: {target_occupancy*100:.1f}%")
        
        # Crear tomograma sintético
        tomo = SynthTomo(
            id=0,
            mbs_file_list=[],
            hns_file_list=[],
            pns_file_list=self.params['PROTEINS_LIST'],
            pms_file_list=[],
        )
        
        # Generar muestra con ocupancia objetivo
        try:
            tomo.gen_sample(
                data_path=data_dir,
                shape=self.params['VOI_SHAPE'],
                v_size=self.params['VOXEL_SIZE'],
                offset=(4, 4, 4),
                target_occupancy=target_occupancy,  # Pasar ocupancia objetivo
                verbosity=False
            )
            
            # Obtener estadísticas
            final_occupancy = tomo.get_occupancy() if hasattr(tomo, 'get_occupancy') else target_occupancy
            proteins_inserted = tomo.get_num_proteins() if hasattr(tomo, 'get_num_proteins') else 0
            saturated = False
            
            print(f"[SAWLC] Completado: {final_occupancy*100:.1f}% ({proteins_inserted} proteínas)")
            
        except Exception as e:
            print(f"[SAWLC] Error o saturación: {e}")
            final_occupancy = 0
            proteins_inserted = 0
            saturated = True
        
        elapsed_time = time.time() - start_time
        
        return {
            'time': elapsed_time,
            'proteins_inserted': proteins_inserted,
            'final_occupancy': final_occupancy,
            'saturated': saturated
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
    plt.style.use('seaborn-v0_8-darkgrid')
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'data', 'data_generated', 'output', 'benchmark_results')
    
    # Ejecutar benchmark
    results_tetris, results_sawlc = run_benchmark()
    
    # Guardar resultados
    save_results(results_tetris, results_sawlc, output_dir)
    
    # Generar gráficas
    plot_results(results_tetris, results_sawlc, output_dir)
    
    # Imprimir resumen
    print_summary(results_tetris, results_sawlc)
    
    print("\n[DONE] Benchmark completado exitosamente!")
