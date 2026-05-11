"""
Curva de ocupancia acumulada: Tetris CPU vs Tetris GPU vs SAWLC.

Genera una gráfica con las tres curvas de ocupancia vs tiempo sobre el mismo eje.
En una máquina sin GPU sólo se produce la curva Tetris CPU.
"""

import os
import sys
import shutil
import subprocess
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

_BENCHMARK = Path(__file__).resolve().parent
_SRC       = _BENCHMARK.parent
_DATA      = _SRC.parent / "data"
_OUT       = _DATA / "data_generated" / "output"

sys.path.insert(0, str(_SRC / "tetris_3d"))
sys.path.insert(0, str(_SRC / "sawlc"))

_PROFILE_SCRIPT  = _BENCHMARK / "profile_test.py"
_PROFILE_TXT_SRC = _BENCHMARK / "profile_occupancy_breakdown.txt"

Timeline = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_profile_txt(path: Path) -> Tuple[Timeline, float]:
    """Lee el .txt generado por profile_test.py → (timeline, total_time)."""
    timeline: Timeline = []
    total_time = 0.0
    in_data = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("total_time_seconds="):
                total_time = float(line.split("=")[1])
            elif line == "timeline_seconds,occupancy_percent":
                in_data = True
            elif in_data and "," in line:
                t_s, occ_s = line.split(",", 1)
                try:
                    timeline.append((float(t_s), float(occ_s)))
                except ValueError:
                    pass
    return timeline, total_time


# ---------------------------------------------------------------------------
# Tetris (CPU / GPU via subprocess)
# ---------------------------------------------------------------------------

def _run_tetris(label: str, force_cpu: bool) -> Tuple[Timeline, float]:
    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"\n[PROFILE] Ejecutando Tetris {label}…")
    subprocess.run(
        [sys.executable, str(_PROFILE_SCRIPT)],
        cwd=str(_BENCHMARK),
        env=env,
        check=False,
    )

    if not _PROFILE_TXT_SRC.exists():
        print(f"[PROFILE] No se encontró {_PROFILE_TXT_SRC}")
        return [], 0.0

    dest = _BENCHMARK / f"profile_{label.lower()}_timeline.txt"
    shutil.copy(_PROFILE_TXT_SRC, dest)
    return _parse_profile_txt(dest)


# ---------------------------------------------------------------------------
# SAWLC (inline con hilo de sondeo de ocupancia)
# ---------------------------------------------------------------------------

def _poll_voi(sample_ref: list, timeline: Timeline, stop: threading.Event,
              t0: float, interval: float = 2.0) -> None:
    """Hilo: sondea sample._SyntheticSample__voi cada `interval` segundos."""
    while not stop.is_set():
        try:
            sample = sample_ref[0]
            if sample is not None:
                voi = sample._SyntheticSample__voi  # True = libre
                occ = 100.0 * float(np.count_nonzero(~voi)) / voi.size
                timeline.append((time.perf_counter() - t0, occ))
        except Exception:
            pass
        stop.wait(interval)


def _run_sawlc() -> Tuple[Timeline, float]:
    from insert_proteins_tetris import (
        MEMBRANES_PATH, MEMBRANE_FILES, PROTEINS_LIST,
        VOI_SHAPE, ROOT_PATH, sorted_proteinSizes,
    )
    from polnet.sample import SyntheticSample, PnFile
    import lio

    print("\n[PROFILE] Ejecutando SAWLC…")

    if MEMBRANE_FILES:
        membrane_volume = lio.load_mrc(
            str(MEMBRANES_PATH / MEMBRANE_FILES[0])
        ).astype("float32")
    else:
        membrane_volume = np.zeros(VOI_SHAPE, dtype="float32")

    membrane_mask = membrane_volume > 0
    shape = membrane_volume.shape

    sample = SyntheticSample(shape=shape, v_size=10, offset=(0, 0, 0))
    voi = sample._SyntheticSample__voi
    voi[membrane_mask] = False
    sample._SyntheticSample__bg_voi = voi.copy()

    timeline: Timeline = []
    stop = threading.Event()
    sample_ref = [sample]
    t0 = time.perf_counter()

    thread = threading.Thread(
        target=_poll_voi, args=(sample_ref, timeline, stop, t0), daemon=True
    )
    thread.start()

    try:
        for p_path in sorted_proteinSizes(PROTEINS_LIST):
            pn_params = PnFile().load(ROOT_PATH / p_path)
            sample.add_set_cproteins(
                params=pn_params,
                data_path=ROOT_PATH,
                surf_dec=0.9,
                mmer_tries=20,
                pmer_tries=100,
                verbosity=False,
            )
    finally:
        stop.set()
        thread.join()

    total_time = time.perf_counter() - t0
    # Punto final garantizado
    final_voi = sample._SyntheticSample__voi
    final_occ = 100.0 * float(np.count_nonzero(~final_voi)) / final_voi.size
    timeline.append((total_time, final_occ))
    return timeline, total_time


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(timelines: dict, output_path: Path) -> None:
    colors = {"Tetris GPU": "#9b30f0", "Tetris CPU": "#1f77b4", "SAWLC": "#ff7f0e"}
    styles = {"Tetris GPU": "-",       "Tetris CPU": "--",        "SAWLC": "-."}

    fig, ax = plt.subplots(figsize=(12, 5))

    for label, (tl, _) in timelines.items():
        if not tl:
            continue
        times = [t for t, _ in tl]
        occs  = [o for _, o in tl]
        ax.plot(times, occs,
                linestyle=styles.get(label, "-"),
                color=colors.get(label, "gray"),
                linewidth=2.5, label=label)

    ax.set_xlabel("Tiempo (segundos)", fontsize=12)
    ax.set_ylabel("Ocupancia (%)", fontsize=12)
    ax.set_title("Curva de ocupancia acumulada: Tetris CPU vs GPU vs SAWLC", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n[PROFILE] Gráfica guardada: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from tetris import GPU_AVAILABLE

    _OUT.mkdir(parents=True, exist_ok=True)
    timelines: dict = {}

    if GPU_AVAILABLE:
        tl, tt = _run_tetris("GPU", force_cpu=False)
        if tl:
            timelines["Tetris GPU"] = (tl, tt)

        tl, tt = _run_tetris("CPU", force_cpu=True)
        if tl:
            timelines["Tetris CPU"] = (tl, tt)
    else:
        tl, tt = _run_tetris("CPU", force_cpu=False)
        if tl:
            timelines["Tetris CPU"] = (tl, tt)

    tl, tt = _run_sawlc()
    if tl:
        timelines["SAWLC"] = (tl, tt)

    if timelines:
        _plot(timelines, _OUT / "profile_comparison.png")
    else:
        print("[PROFILE] No hay datos para graficar.")


if __name__ == "__main__":
    main()
