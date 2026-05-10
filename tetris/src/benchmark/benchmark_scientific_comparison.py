"""
Benchmark científico SAWLC vs Tetris 3D

Objetivo:
- Comparar densidad de empaquetamiento (packing fraction)
- Comparar eficiencia computacional (tiempo e inserciones)

Metodología:
1) Mismas condiciones para ambos algoritmos:
   - Mismo VOI (300x300x250)
   - Misma proteína base (.pns)
   - Mismos targets de ocupancia
2) Métricas principales:
   - ocupancia final binaria (%)
   - proteínas insertadas
   - tiempo total (s)
   - tiempo por inserción exitosa (s/proteína)
   - proteínas por segundo
   - pmer_fails (solo SAWLC)
3) Escenarios propuestos:
   - target 15%
   - target 25%

Notas:
- Este script no reemplaza benchmark_comparison.py ni benchmark_algorithm_selection.py.
- Restaura automáticamente scripts y parámetros .pns al finalizar.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
OUT_BASE = DATA / "data_generated" / "output"
OUT_REPORT = OUT_BASE / "benchmark_scientific"

SAWLC_SCRIPT = SRC / "sawlc" / "all_features.py"
TETRIS_SCRIPT = SRC / "tetris_3d" / "tetris.py"

SAWLC_CWD = SRC / "sawlc"
TETRIS_CWD = SRC / "tetris_3d"

SAWLC_CMD = ["python", "all_features.py"]
TETRIS_CMD = ["python", "tetris.py"]

SAWLC_LABEL = OUT_BASE / "output_sawlc" / "tomos" / "tomo_000_lbl.mrc"
TETRIS_LABEL = OUT_BASE / "output_tetris" / "tetris_3d_output_labels.mrc"

# Configuración del benchmark
PROTEIN_FILE = "in_10A/4v4r_10A.pns"
TARGETS_PERCENT = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0]  # [15.0, 25.0] --- IGNORE ---
REPEATS_PER_TARGET = 2

# SAWLC en modo casi monómero: PMER_L_MAX muy bajo
FORCE_SAWLC_SHORT_CHAIN = True
SHORT_CHAIN_PMER_L_MAX = 1.0


@dataclass
class RunMetrics:
    algorithm: str
    target_percent: float
    repeat_id: int
    runtime_seconds: float
    occupancy_percent_binary: Optional[float]
    proteins_inserted: Optional[int]
    time_per_insertion_seconds: Optional[float]
    proteins_per_second: Optional[float]
    pmer_fails: Optional[int]
    stop_reason: str
    saturated: bool
    exit_code: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_pns_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_pns_lines(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_pns_value(path: Path, key: str, value: str) -> None:
    lines = read_pns_lines(path)
    out_lines: List[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith(key):
            out_lines.append(f"{key} = {value}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append(f"{key} = {value}")
    write_pns_lines(path, out_lines)


def get_pns_value(path: Path, key: str) -> Optional[str]:
    for line in read_pns_lines(path):
        line = line.strip()
        if line.startswith(key):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def set_proteins_list_in_script(script_path: Path, proteins: List[str]) -> None:
    text = read_text(script_path)
    block = "PROTEINS_LIST = [\n" + "\n".join([f'    "{p}",' for p in proteins]) + "\n]\n"
    new_text, count = re.subn(
        r"PROTEINS_LIST\s*=\s*\[(?:.|\n)*?\]\n",
        block,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"No se pudo actualizar PROTEINS_LIST en {script_path}")
    write_text(script_path, new_text)


def compute_binary_occupancy_percent(label_path: Path) -> Optional[float]:
    if not label_path.exists():
        return None
    with mrcfile.open(label_path, permissive=True) as mrc:
        vol = np.asarray(mrc.data)
    if vol.size == 0:
        return None
    return 100.0 * np.count_nonzero(vol > 0) / vol.size


def parse_inserted(stdout: str, algo: str) -> Optional[int]:
    if algo == "sawlc":
        m = re.search(r"Total proteinas insertadas:\s*(\d+)", stdout)
    else:
        m = re.search(r"Proteínas insertadas:\s*(\d+)", stdout)
        if m is None:
            m = re.search(r"Proteinas insertadas:\s*(\d+)", stdout)
    return int(m.group(1)) if m else None


def parse_pmer_fails(stdout: str) -> Optional[int]:
    m = re.search(r"Pmer fails:\s*(\d+)", stdout)
    return int(m.group(1)) if m else None


def parse_stop_reason(stdout: str, algo: str, occ: Optional[float], target_percent: float) -> str:
    s = stdout.lower()
    if algo == "tetris":
        if occ is not None and occ >= target_percent:
            return "target-reached"
        if "ocupancia objetivo" in s:
            return "target-reached-internal"
        if "saturación" in s or "saturacion" in s:
            return "saturation"
        return "unknown"

    # SAWLC: inferencia por resultado final
    if occ is not None and occ >= target_percent:
        return "target-reached"
    if "pmer fails" in s:
        return "attempt-limit"
    return "unknown"


def run_algo(algo: str, target_percent: float, repeat_id: int) -> RunMetrics:
    if algo == "sawlc":
        cmd, cwd, label = SAWLC_CMD, SAWLC_CWD, SAWLC_LABEL
    else:
        cmd, cwd, label = TETRIS_CMD, TETRIS_CWD, TETRIS_LABEL

    start = time.time()
    done = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    runtime = time.time() - start

    stdout = done.stdout or ""
    stderr = done.stderr or ""

    occ = compute_binary_occupancy_percent(label)
    inserted = parse_inserted(stdout, algo)
    pmer_fails = parse_pmer_fails(stdout) if algo == "sawlc" else None

    stop_reason = parse_stop_reason(stdout, algo, occ, target_percent)
    saturated = stop_reason in {"saturation", "attempt-limit"}

    tpi = None if (inserted is None or inserted <= 0 or runtime <= 0) else runtime / inserted
    pps = None if (inserted is None or runtime <= 0) else inserted / runtime

    stamp = f"{algo}_t{int(target_percent)}_r{repeat_id}"
    (OUT_REPORT / f"{stamp}.stdout.log").write_text(stdout, encoding="utf-8")
    (OUT_REPORT / f"{stamp}.stderr.log").write_text(stderr, encoding="utf-8")

    return RunMetrics(
        algorithm=algo,
        target_percent=target_percent,
        repeat_id=repeat_id,
        runtime_seconds=runtime,
        occupancy_percent_binary=occ,
        proteins_inserted=inserted,
        time_per_insertion_seconds=tpi,
        proteins_per_second=pps,
        pmer_fails=pmer_fails,
        stop_reason=stop_reason,
        saturated=saturated,
        exit_code=done.returncode,
    )


def mean_valid(values: List[Optional[float]]) -> float:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else float("nan")


def std_valid(values: List[Optional[float]]) -> float:
    clean = [v for v in values if v is not None]
    return float(np.std(clean)) if clean else float("nan")


def print_table(rows: List[RunMetrics]) -> None:
    print("\n" + "=" * 144)
    print("BENCHMARK CIENTÍFICO: SAWLC vs TETRIS")
    print("=" * 144)
    print(
        f"{'Algo':<8} {'Target%':>8} {'Rep':>5} {'Occ%':>8} {'Prot':>8} {'Time(s)':>10} "
        f"{'s/Prot':>10} {'Prot/s':>10} {'Fails':>8} {'Stop':>14} {'Sat?':>6} {'Exit':>6}"
    )
    print("-" * 144)

    for r in rows:
        occ = "NA" if r.occupancy_percent_binary is None else f"{r.occupancy_percent_binary:.2f}"
        prot = "NA" if r.proteins_inserted is None else str(r.proteins_inserted)
        tpi = "NA" if r.time_per_insertion_seconds is None else f"{r.time_per_insertion_seconds:.4f}"
        pps = "NA" if r.proteins_per_second is None else f"{r.proteins_per_second:.3f}"
        fails = "NA" if r.pmer_fails is None else str(r.pmer_fails)
        sat = "yes" if r.saturated else "no"

        print(
            f"{r.algorithm:<8} {r.target_percent:>8.1f} {r.repeat_id:>5d} {occ:>8} {prot:>8} {r.runtime_seconds:>10.2f} "
            f"{tpi:>10} {pps:>10} {fails:>8} {r.stop_reason:>14} {sat:>6} {r.exit_code:>6}"
        )

    print("=" * 144)


def build_summary(rows: List[RunMetrics]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {"sawlc": {}, "tetris": {}}
    for algo in ("sawlc", "tetris"):
        vals_algo = [r for r in rows if r.algorithm == algo]
        for target in sorted({r.target_percent for r in vals_algo}):
            vals = [r for r in vals_algo if r.target_percent == target]
            out[algo][str(target)] = {
                "mean_occupancy_percent": mean_valid([r.occupancy_percent_binary for r in vals]),
                "std_occupancy_percent": std_valid([r.occupancy_percent_binary for r in vals]),
                "mean_runtime_seconds": mean_valid([r.runtime_seconds for r in vals]),
                "std_runtime_seconds": std_valid([r.runtime_seconds for r in vals]),
                "mean_time_per_insertion_seconds": mean_valid([r.time_per_insertion_seconds for r in vals]),
                "mean_proteins_per_second": mean_valid([r.proteins_per_second for r in vals]),
                "mean_proteins_inserted": mean_valid([None if r.proteins_inserted is None else float(r.proteins_inserted) for r in vals]),
                "mean_pmer_fails": mean_valid([None if r.pmer_fails is None else float(r.pmer_fails) for r in vals]),
            }
    return out


def generate_plots(rows: List[RunMetrics]) -> None:
    def rows_by(algo: str, target: float) -> List[RunMetrics]:
        return [r for r in rows if r.algorithm == algo and r.target_percent == target]

    targets = sorted({r.target_percent for r in rows})

    # Curvas promedio por target
    saw_occ = [mean_valid([r.occupancy_percent_binary for r in rows_by("sawlc", t)]) for t in targets]
    tet_occ = [mean_valid([r.occupancy_percent_binary for r in rows_by("tetris", t)]) for t in targets]

    saw_time = [mean_valid([r.runtime_seconds for r in rows_by("sawlc", t)]) for t in targets]
    tet_time = [mean_valid([r.runtime_seconds for r in rows_by("tetris", t)]) for t in targets]

    saw_tpi = [mean_valid([r.time_per_insertion_seconds for r in rows_by("sawlc", t)]) for t in targets]
    tet_tpi = [mean_valid([r.time_per_insertion_seconds for r in rows_by("tetris", t)]) for t in targets]

    saw_fail = [mean_valid([None if r.pmer_fails is None else float(r.pmer_fails) for r in rows_by("sawlc", t)]) for t in targets]

    # Estilo robusto: usar seaborn si está disponible, si no fallback seguro
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        available = set(plt.style.available)
        if "seaborn-darkgrid" in available:
            plt.style.use("seaborn-darkgrid")
        elif "ggplot" in available:
            plt.style.use("ggplot")
        else:
            plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark científico SAWLC vs Tetris", fontsize=15, fontweight="bold")

    # 1) Packing fraction
    ax = axes[0, 0]
    ax.plot(targets, saw_occ, "o-", linewidth=2, label="SAWLC")
    ax.plot(targets, tet_occ, "s-", linewidth=2, label="Tetris")
    ax.plot(targets, targets, "k--", alpha=0.5, label="Objetivo")
    ax.set_title("Densidad alcanzada (packing fraction)")
    ax.set_xlabel("Ocupancia objetivo (%)")
    ax.set_ylabel("Ocupancia final binaria (%)")
    ax.legend()

    # 2) Tiempo total
    ax = axes[0, 1]
    ax.plot(targets, saw_time, "o-", linewidth=2, label="SAWLC")
    ax.plot(targets, tet_time, "s-", linewidth=2, label="Tetris")
    ax.set_title("Tiempo total por target")
    ax.set_xlabel("Ocupancia objetivo (%)")
    ax.set_ylabel("Tiempo (s)")
    ax.legend()

    # 3) Tiempo por inserción exitosa
    ax = axes[1, 0]
    ax.plot(targets, saw_tpi, "o-", linewidth=2, label="SAWLC")
    ax.plot(targets, tet_tpi, "s-", linewidth=2, label="Tetris")
    ax.set_title("Tiempo por inserción exitosa")
    ax.set_xlabel("Ocupancia objetivo (%)")
    ax.set_ylabel("s / proteína")
    ax.legend()

    # 4) Fallos
    ax = axes[1, 1]
    w = 0.6
    x = np.arange(len(targets), dtype=float)
    ax.bar(x, saw_fail, width=w, label="SAWLC pmer_fails")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in targets])
    ax.set_title("Fallos de SAWLC (pmer_fails)")
    ax.set_xlabel("Ocupancia objetivo (%)")
    ax.set_ylabel("valor promedio")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "benchmark_scientific_plots.png", dpi=180)
    plt.close()


def main() -> None:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    sawlc_original = read_text(SAWLC_SCRIPT)
    tetris_original = read_text(TETRIS_SCRIPT)

    pns_path = DATA / PROTEIN_FILE
    if not pns_path.exists():
        raise FileNotFoundError(f"No existe proteína de benchmark: {pns_path}")

    original_occ = get_pns_value(pns_path, "PMER_OCC")
    original_lmax = get_pns_value(pns_path, "PMER_L_MAX")

    rows: List[RunMetrics] = []

    try:
        # Igualar proteína en ambos algoritmos
        set_proteins_list_in_script(SAWLC_SCRIPT, [PROTEIN_FILE])
        set_proteins_list_in_script(TETRIS_SCRIPT, [PROTEIN_FILE])

        # SAWLC casi monómero
        if FORCE_SAWLC_SHORT_CHAIN:
            set_pns_value(pns_path, "PMER_L_MAX", str(SHORT_CHAIN_PMER_L_MAX))

        for target in TARGETS_PERCENT:
            set_pns_value(pns_path, "PMER_OCC", str(target / 100.0))

            for rep in range(1, REPEATS_PER_TARGET + 1):
                print(f"\n[RUN] target={target:.1f}% | rep={rep}")
                rows.append(run_algo("sawlc", target, rep))
                rows.append(run_algo("tetris", target, rep))

        print_table(rows)
        summary = build_summary(rows)
        generate_plots(rows)

        report = {
            "config": {
                "protein_file": PROTEIN_FILE,
                "targets_percent": TARGETS_PERCENT,
                "repeats_per_target": REPEATS_PER_TARGET,
                "force_sawlc_short_chain": FORCE_SAWLC_SHORT_CHAIN,
                "short_chain_pmer_l_max": SHORT_CHAIN_PMER_L_MAX,
                "voi_shape": [300, 300, 250],
            },
            "rows": [asdict(r) for r in rows],
            "summary": summary,
            "plots": [str(OUT_REPORT / "benchmark_scientific_plots.png")],
        }

        out_json = OUT_REPORT / "benchmark_scientific_report.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n[OK] Reporte JSON: {out_json}")
        print(f"[OK] Gráficas: {OUT_REPORT / 'benchmark_scientific_plots.png'}")

    finally:
        # Restaurar scripts
        write_text(SAWLC_SCRIPT, sawlc_original)
        write_text(TETRIS_SCRIPT, tetris_original)

        # Restaurar .pns
        if original_occ is not None:
            set_pns_value(pns_path, "PMER_OCC", original_occ)
        if original_lmax is not None:
            set_pns_value(pns_path, "PMER_L_MAX", original_lmax)

        print("[RESTORE] Scripts y .pns restaurados")


if __name__ == "__main__":
    main()
