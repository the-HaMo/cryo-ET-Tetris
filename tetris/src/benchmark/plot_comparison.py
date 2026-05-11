import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
from pathlib import Path

# Directorio donde están los logs generados por los benchmarks
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "data_generated" / "output"
LOGS_TETRIS = _OUTPUT_DIR / "logs_tetris"
LOGS_SAWLC  = _OUTPUT_DIR / "logs_sawlc"

# Base de la membrana detectada en tus logs para el Tomograma 3
MEMBRANE_LEVEL = 11.5450

def parse_detailed_log(file_path, algo_type):
    """Extrae métricas finales y detalle de monómeros por tipo."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except: return None

    data = {'num_types': 0, 'total_occ': 0, 'time_min': 0, 'total_monomers': 0, 'monomers_detail': {}}

    # 1. Tiempos y patrones según algoritmo
    if algo_type == 'Tetris':
        time_m = re.search(r"DONE: .*? in (?:(\d+)m )?([\d.]+)s", content)
        pattern = r"After type \d+ \((.*?)\):.*?num_monomers=(\d+)"
    else:
        time_m = re.search(r"TOTAL PROCESSING TIME: (?:(\d+)min )?([\d.]+)s", content)
        pattern = r"After type \d+ \((.*?)\):.*?monomers=(\d+)"

    if time_m:
        data['time_min'] = (float(time_m.group(1)) if time_m.group(1) else 0) + float(time_m.group(2))/60

    # 2. Ocupancia FINAL (Último valor real del log)
    occ_values = re.findall(r"total_occ\s*[:=]\s*([\d.]+)%", content)
    if occ_values:
        data['total_occ'] = float(occ_values[-1])

    # 3. Detalle de Monómeros por tipo
    lines = re.findall(pattern, content)
    for name, count in lines:
        short_name = name.split('_')[0]
        c = int(count)
        data['monomers_detail'][short_name] = c
        data['total_monomers'] += c

    data['num_types'] = len(lines)
    return data

def load_data(directory, label):
    if not os.path.exists(directory): return []
    files = sorted([f for f in os.listdir(directory) if 'tomo3_den' in f and f.endswith('.txt')],
                   key=lambda x: int(re.search(r'den(\d+)', x).group(1)))
    return [parse_detailed_log(os.path.join(directory, f), label) for f in files]

# --- PROCESAMIENTO ---
tetris_results = load_data(LOGS_TETRIS, 'Tetris')
sawlc_results  = load_data(LOGS_SAWLC,  'SAWLC')

df_t = pd.DataFrame(tetris_results)
df_s = pd.DataFrame(sawlc_results)

# Mapeo de colores único por nombre de proteína
all_prots = sorted(list(set([p for s in tetris_results for p in s['monomers_detail'].keys()] +
                            [p for s in sawlc_results  for p in s['monomers_detail'].keys()])))
color_map = dict(zip(all_prots, plt.cm.tab20(np.linspace(0, 1, len(all_prots)))))

# --- GENERACIÓN DE GRÁFICAS ---
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

fig.suptitle('TETRIS vs SAWLC\nTomograma 3', fontsize=22, fontweight='bold')
plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.90)

for ax in axs.flat:
    ax.set_xlabel('Simulaciones', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 12))
    ax.grid(True, alpha=0.3)

# 1. SATURACIÓN TOTAL
ax0 = axs[0,0]
ax0.axhline(y=MEMBRANE_LEVEL, color='black', linestyle=':', alpha=0.5, label='Membrana (11.5%)')
if not df_t.empty:
    ax0.plot(df_t['num_types'].values, df_t['total_occ'].values, 'o-', color='blue', lw=3, label='Tetris (Total)')
if not df_s.empty:
    ax0.plot(df_s['num_types'].values, df_s['total_occ'].values, 's--', color='orange', lw=3, label='SAWLC (Total)')
ax0.set_title('Saturación Alcanzada', fontsize=14, fontweight='bold')
ax0.set_ylabel('Ocupancia Total (%)')
ax0.set_ylim(0, 60); ax0.legend()

# 2. TIEMPO DE EJECUCIÓN
axs[0,1].plot(df_t['num_types'].values, df_t['time_min'].values, 'o-', color='red', label='Tetris')
axs[0,1].plot(df_s['num_types'].values, df_s['time_min'].values, 's--', color='orange', label='SAWLC')
axs[0,1].set_title('Tiempo de Ejecución', fontsize=14, fontweight='bold')
axs[0,1].set_ylabel('Minutos'); axs[0,1].legend()

# 3. POBLACIÓN DE PROTEÍNAS
ax_bar = axs[1,0]
width = 0.35
for sim_list, offset, edge_c in [(tetris_results, -width/2, 'white'), (sawlc_results, width/2, 'black')]:
    for sim in sim_list:
        bottom = 0
        for p in all_prots:
            val = sim['monomers_detail'].get(p, 0)
            if val > 0:
                ax_bar.bar(sim['num_types'] + offset, val, width, bottom=bottom,
                           color=color_map[p], edgecolor=edge_c, lw=0.5)
                bottom += val

ax_bar.set_title('Población de Proteínas', fontsize=14, fontweight='bold')
ax_bar.set_ylabel('Cantidad de Monómeros')
from matplotlib.lines import Line2D
ax_bar.legend(handles=[Line2D([0], [0], color=color_map[p], lw=6, label=p) for p in all_prots],
              title="Proteínas", loc='upper left', fontsize='xx-small', ncol=2)

# 4. RENDIMIENTO
vel_t = df_t['total_monomers'].values / df_t['time_min'].values
vel_s = df_s['total_monomers'].values / df_s['time_min'].values
axs[1,1].plot(df_t['num_types'].values, vel_t, 'o-', color='green', label='Tetris')
axs[1,1].plot(df_s['num_types'].values, vel_s, 's--', color='gray', label='SAWLC')
axs[1,1].set_title('Rendimiento (Proteínas/min)', fontsize=14, fontweight='bold')
axs[1,1].legend()

output_path = _OUTPUT_DIR / "comparativa_tetris_sawlc.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[PLOT] Guardado en: {output_path}")
plt.show()
