"""
Curva de ocupancia acumulada: Tetris CPU vs Tetris GPU vs SAWLC.

Genera una gráfica con las tres curvas de ocupancia vs tiempo.
En máquinas sin GPU sólo se produce la curva Tetris CPU.
No modifica ningún script fuente.
"""
from __future__ import annotations

import multiprocessing as mp, os, re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

_BENCHMARK = Path(__file__).resolve().parent
_SRC       = _BENCHMARK.parent
_DATA      = _SRC.parent / "data"
_OUT       = _DATA / "data_generated" / "output"

Timeline = List[Tuple[float, float]]


# ─── Worker Tetris (nivel de módulo — requerido por multiprocessing.spawn) ────

def _tetris_profile_worker(force_cpu: bool, q) -> None:
    """Corre Tetris con monkey-patch en _correlate y devuelve (timeline, total_time)."""
    import sys as _sys, time as _t
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        _sys.modules["cupy"]  = None  # type: ignore[assignment]
        _sys.modules["cupyx"] = None  # type: ignore[assignment]
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    import numpy as _np
    from image_processing_3d import ImageProcessing3D
    from parser_3d import Parser3D
    from tetris import Tetris3D, xp
    from insert_proteins_tetris import (
        MEMBRANES_PATH, MEMBRANE_FILES, PROTEINS_LIST,
        VOI_SHAPE, ROOT_PATH, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING,
        sorted_proteinSizes,
    )
    import lio

    def _pick_seed(allowed_mask, output_volume, threshold, box_size):
        half = box_size // 2
        z, y, x = output_volume.shape
        empty  = allowed_mask & (output_volume <= threshold)
        viable = xp.zeros_like(empty, dtype=bool)
        viable[half:z-half, half:y-half, half:x-half] = \
            empty[half:z-half, half:y-half, half:x-half]
        cands = xp.argwhere(viable)
        if len(cands) == 0:
            return None
        return tuple(int(v) for v in cands[_np.random.randint(0, len(cands))])

    # Cargar volumen
    if MEMBRANE_FILES:
        mem_vol = lio.load_mrc(str(MEMBRANES_PATH / MEMBRANE_FILES[0])).astype("float32")
    else:
        mem_vol = _np.zeros(VOI_SHAPE, dtype="float32")
    allowed = xp.asarray(~(mem_vol > 0))

    # Cargar proteínas
    molecules = []
    for p_path in sorted_proteinSizes(PROTEINS_LIST):
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p_path), str(ROOT_PATH))
        vol = xp.asarray(vol)
        coords = xp.argwhere(vol > vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        if coords.size == 0:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        molecules.append((p_path, vol[int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)]))

    if not molecules:
        q.put(([], 0.0))
        return

    g_thresh = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris   = Tetris3D(dimensions=mem_vol.shape, threshold=g_thresh)
    tetris.output_volume[~allowed] = 500.0

    timeline: Timeline = []
    t0_global = _t.perf_counter()

    for _, (_, vol) in enumerate(molecules, 1):
        bsize = max(vol.shape)
        seed  = _pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        fails = 0
        while fails < TRIES_CLUSTERING:
            if seed is None:
                break
            rot, _  = ImageProcessing3D.randomly_rotate(vol)
            rbin    = ImageProcessing3D.smooth_and_binarize(rot, 1.5, g_thresh)
            tmpl, _, _ = ImageProcessing3D.create_in_shell(rbin, (0, 2), penalty=100)
            res = tetris.insert_molecule_3d(tmpl, rot, "", allowed, seed, bsize)
            if res == "inserted":
                occ = float(tetris.get_occupancy() * 100.0)
                timeline.append((_t.perf_counter() - t0_global, occ))
                fails = 0
                seed  = tetris.all_coordinates[-1]
            else:
                fails += 1
                seed  = _pick_seed(allowed, tetris.output_volume, g_thresh, bsize)

    total_time = _t.perf_counter() - t0_global
    q.put((timeline, total_time))


# ─── SAWLC worker (nivel de módulo — requerido por multiprocessing.spawn) ─────

def _sawlc_profile_worker(q) -> None:
    """Corre SAWLC con verbosity=True, envía líneas de progreso por la cola."""
    import sys as _sys, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "sawlc"))

    import numpy as _np
    from insert_proteins_tetris import (
        MEMBRANES_PATH, MEMBRANE_FILES, PROTEINS_LIST,
        VOI_SHAPE, ROOT_PATH, sorted_proteinSizes,
    )
    from polnet.sample import SyntheticSample, PnFile
    import lio

    if MEMBRANE_FILES:
        mem_vol = lio.load_mrc(str(MEMBRANES_PATH / MEMBRANE_FILES[0])).astype("float32")
    else:
        mem_vol = _np.zeros(VOI_SHAPE, dtype="float32")

    mem_mask = mem_vol > 0
    sample   = SyntheticSample(shape=mem_vol.shape, v_size=10, offset=(0, 0, 0))
    voi      = sample._SyntheticSample__voi
    voi[mem_mask] = False
    sample._SyntheticSample__bg_voi = voi.copy()

    t0 = _t.perf_counter()

    class _LineQueue:
        def __init__(self): self._buf = ""
        def write(self, text):
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    q.put(("line", line, _t.perf_counter() - t0))
        def flush(self): pass

    _sys.stdout = _LineQueue()
    try:
        for p_path in sorted_proteinSizes(PROTEINS_LIST):
            pn = PnFile().load(ROOT_PATH / p_path)
            sample.add_set_cproteins(params=pn, data_path=ROOT_PATH,
                                     surf_dec=0.9, mmer_tries=20,
                                     pmer_tries=100, verbosity=True)
    finally:
        _sys.stdout = _sys.__stdout__

    total_time = _t.perf_counter() - t0
    final_voi  = sample._SyntheticSample__voi
    final_occ  = 100.0 * float(_np.count_nonzero(~final_voi)) / final_voi.size
    q.put(("done", total_time, final_occ))


def _run_sawlc() -> Tuple[Timeline, float]:
    print("\n[PROFILE] Ejecutando SAWLC…")
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=_sawlc_profile_worker, args=(q,))
    p.start()

    timeline: Timeline = []
    total_time = 0.0
    while True:
        msg = q.get()
        if msg[0] == "done":
            _, total_time, final_occ = msg
            if not timeline or timeline[-1][1] != final_occ:
                timeline.append((total_time, final_occ))
            break
        if msg[0] == "line":
            _, line, t = msg
            m = re.search(r"Ocupancia\s+([\d.]+)%", line)
            if m:
                timeline.append((t, float(m.group(1))))
                print(f"  [SAWLC] t={t:.1f}s occ={m.group(1)}%")

    p.join()
    return timeline, total_time


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot(timelines: dict, output_path: Path) -> None:
    colors = {"Tetris GPU": "#9b30f0", "Tetris CPU": "#1f77b4", "SAWLC": "#ff7f0e"}
    styles = {"Tetris GPU": "-",       "Tetris CPU": "--",       "SAWLC": "-."}

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, (tl, _) in timelines.items():
        if not tl:
            continue
        ax.plot([t for t, _ in tl], [o for _, o in tl],
                linestyle=styles.get(label, "-"),
                color=colors.get(label, "gray"),
                linewidth=2.5, label=label)

    ax.set_xlabel("Tiempo (segundos)", fontsize=12)
    ax.set_ylabel("Ocupancia (%)",      fontsize=12)
    ax.set_title("Curva de ocupancia acumulada: Tetris CPU vs GPU vs SAWLC", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n[PROFILE] Gráfica guardada: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys as _sys
    _sys.path.insert(0, str(_SRC / "tetris_3d"))
    from tetris import GPU_AVAILABLE

    _OUT.mkdir(parents=True, exist_ok=True)
    timelines: dict = {}

    ctx = mp.get_context("spawn")

    def _run_tetris(label: str, force_cpu: bool) -> Optional[Tuple[Timeline, float]]:
        print(f"\n[PROFILE] Ejecutando Tetris {label}…")
        q = ctx.Queue()
        p = ctx.Process(target=_tetris_profile_worker, args=(force_cpu, q))
        p.start(); p.join()
        tl, tt = q.get()
        return (tl, tt) if tl else None

    if GPU_AVAILABLE:
        result = _run_tetris("GPU", force_cpu=False)
        if result:
            timelines["Tetris GPU"] = result
        result = _run_tetris("CPU", force_cpu=True)
        if result:
            timelines["Tetris CPU"] = result
    else:
        result = _run_tetris("CPU", force_cpu=False)
        if result:
            timelines["Tetris CPU"] = result

    tl, tt = _run_sawlc()
    if tl:
        timelines["SAWLC"] = (tl, tt)

    if timelines:
        _plot(timelines, _OUT / "profile_comparison.png")
    else:
        print("[PROFILE] No hay datos para graficar.")


if __name__ == "__main__":
    main()
