"""
COMPARATIVA FINAL: TETRIS (CPU/GPU) vs SAWLC
4 gráficas variando el número de tipos de proteína (N = 1..len(PROTEINS_ALL)).

Ejecutar dos veces:
  Con membrana:  MEMBRANE_FILE = "tomo_mem_lbls_3.mrc"
  Sin membrana:  MEMBRANE_FILE = None

Resultados cacheados en cache_tomo{TOMO_ID}_{escenario}.json.
Borrar el JSON para forzar re-ejecución.
No modifica ningún script fuente.
"""
from __future__ import annotations

import io, json, multiprocessing as mp, os, re, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_BENCHMARK = Path(__file__).resolve().parent
_SRC       = _BENCHMARK.parent
_OUTPUT    = _BENCHMARK.parents[1] / "data" / "data_generated" / "output"

# ─── Configuración ────────────────────────────────────────────────────────────
TOMO_ID        = 3
MEMBRANE_FILE: Optional[str] = "tomo_mem_lbls_3.mrc"   # None → sin membrana
MEMBRANE_LEVEL = 11.5450   # % base membrana para línea punteada

PROTEIN_ALL = [
    "in_10A/5mrc_10A.pns",
    "in_10A/4v94_10A.pns",
    "in_10A/4v4r_10A.pns",
    "in_10A/4cr2_10A.pns",
    "in_10A/3d2f_10A.pns",
    "in_10A/3cf3_10A.pns",
    "in_10A/2uv8_10A.pns",
    "in_10A/2cg9_10A.pns",
    "in_10A/1u6g_10A.pns",
    "in_10A/1s3x_10A.pns",
    "in_10A/1qvr_10A.pns"
]
# ──────────────────────────────────────────────────────────────────────────────

_escenario = Path(MEMBRANE_FILE).stem if MEMBRANE_FILE else "empty"
_CACHE     = _BENCHMARK / f"cache_tomo{TOMO_ID}_{_escenario}.json"


# ─── Workers (nivel de módulo — requerido por multiprocessing.spawn) ──────────

def _tetris_worker(proteins: list, membrane_file, force_cpu: bool, q) -> None:
    """Corre Tetris hasta saturación y devuelve métricas por la cola."""
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import sys as _sys, os as _os, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    import numpy as _np
    from tetris import Tetris3D, xp
    from image_processing_3d import ImageProcessing3D
    from parser_3d import Parser3D
    from insert_proteins_tetris import (
        pick_seed, sorted_proteinSizes, crop_volume,
        ROOT_PATH, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING,
    )
    import lio

    start   = _t.time()
    _mem_dir = ROOT_PATH / "templates" / "membranes"

    if membrane_file:
        mem_path = _mem_dir / membrane_file
        mem = lio.load_mrc(str(mem_path)).astype("float32") if mem_path.exists() \
              else _np.zeros((500, 500, 250), dtype="float32")
    else:
        mem = _np.zeros((500, 500, 250), dtype="float32")

    allowed   = xp.asarray(~(xp.asarray(mem) > 0))
    tot_vox   = allowed.size
    mem_occ   = float(xp.count_nonzero(~allowed) / tot_vox)

    mols = []
    for p in sorted_proteinSizes(proteins):
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p), str(ROOT_PATH))
        vol_c  = crop_volume(vol, vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        mols.append((_os.path.basename(p), vol_c))

    if not mols:
        q.put({"occ": 0.0, "time_min": 0.0, "monomers": {}, "total_monomers": 0})
        return

    g_thresh = mols[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris   = Tetris3D(dimensions=tuple(mem.shape), threshold=g_thresh)
    tetris.output_volume[~allowed] = 500.0

    monomers: dict = {}
    total = 0

    for _, (name, vol) in enumerate(mols, 1):
        before = total
        bsize  = max(vol.shape)
        seed   = pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        fails  = 0
        while fails < TRIES_CLUSTERING:
            if seed is None:
                break
            rot, _  = ImageProcessing3D.randomly_rotate(vol)
            rbin    = ImageProcessing3D.smooth_and_binarize(rot, 1.5, g_thresh)
            tmpl, _, _ = ImageProcessing3D.create_in_shell(rbin, (0, 2), penalty=100)
            res = tetris.insert_molecule_3d(tmpl, rot, name, allowed, seed, bsize)
            if res == "inserted":
                total += 1; fails = 0
                seed = tetris.all_coordinates[-1]
            else:
                fails += 1
                seed = pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        key = name.split("_")[0]
        monomers[key] = monomers.get(key, 0) + (total - before)

    q.put({"occ":           float(tetris.get_occupancy()) * 100.0,
           "time_min":      (_t.time() - start) / 60.0,
           "monomers":      monomers,
           "total_monomers": sum(monomers.values())})


def _sawlc_worker(proteins: list, membrane_file, q) -> None:
    """Corre SAWLC hasta saturación y devuelve métricas por la cola."""
    import sys as _sys, io as _io, re as _re, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "sawlc"))

    from insert_proteins_in_membranes import (
        insert_proteins_in_membrane, sorted_proteinSizes, MEMBRANES_PATH, OUT_DIR,
    )

    sorted_p = sorted_proteinSizes(proteins)
    t0  = _t.time()
    buf = _io.StringIO()
    old = _sys.stdout; _sys.stdout = buf
    try:
        if membrane_file:
            mem_path = MEMBRANES_PATH / membrane_file
            insert_proteins_in_membrane(mem_path, sorted_p, str(OUT_DIR), 0)
        else:
            insert_proteins_in_membrane(None, sorted_p, str(OUT_DIR), 0)
    finally:
        _sys.stdout = old

    elapsed = _t.time() - t0
    txt     = buf.getvalue()

    monomers: dict = {}
    for m in _re.finditer(r"After type \d+ \((\S+?)\):.*?monomers=(\d+)", txt):
        k = m.group(1).split("_")[0]
        monomers[k] = monomers.get(k, 0) + int(m.group(2))
    occ_v = _re.findall(r"total_occ\s*[:=]\s*([\d.]+)%", txt)

    q.put({"occ":           float(occ_v[-1]) if occ_v else 0.0,
           "time_min":      elapsed / 60.0,
           "monomers":      monomers,
           "total_monomers": sum(monomers.values())})


# ─── Ejecución de simulaciones ────────────────────────────────────────────────

def _spawn(target, args) -> dict:
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=target, args=(*args, q))
    p.start(); p.join()
    return q.get()


def _run_all_sims() -> dict:
    sims = []
    for n in range(1, len(PROTEINS_ALL) + 1):
        proteins = PROTEINS_ALL[:n]
        short    = [p.split("/")[-1] for p in proteins]
        print(f"\n{'─'*60}\n[SIM] N={n}: {short}\n{'─'*60}")

        print("[SIM] SAWLC…")
        sawlc = _spawn(_sawlc_worker, (proteins, MEMBRANE_FILE))
        print(f"      occ={sawlc['occ']:.2f}%  t={sawlc['time_min']:.2f}min")

        print("[SIM] Tetris GPU…")
        t_gpu = _spawn(_tetris_worker, (proteins, MEMBRANE_FILE, False))
        print(f"      occ={t_gpu['occ']:.2f}%  t={t_gpu['time_min']:.2f}min")

        print("[SIM] Tetris CPU…")
        t_cpu = _spawn(_tetris_worker, (proteins, MEMBRANE_FILE, True))
        print(f"      occ={t_cpu['occ']:.2f}%  t={t_cpu['time_min']:.2f}min")

        sims.append({"n_types": n, "proteins": proteins,
                     "sawlc": sawlc, "tetris_gpu": t_gpu, "tetris_cpu": t_cpu})

    return {"config": {"tomo_id": TOMO_ID, "membrane_file": MEMBRANE_FILE,
                       "proteins": PROTEINS_ALL},
            "sims": sims}


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot(cache: dict) -> None:
    sims = cache["sims"]
    xs   = [s["n_types"] for s in sims]

    def col(algo, key):
        return [s[algo][key] for s in sims]

    t_cpu_ppm = [s["tetris_cpu"]["total_monomers"] / t if (t := s["tetris_cpu"]["time_min"]) > 0 else 0 for s in sims]
    t_gpu_ppm = [s["tetris_gpu"]["total_monomers"] / t if (t := s["tetris_gpu"]["time_min"]) > 0 else 0 for s in sims]
    s_ppm     = [s["sawlc"]["total_monomers"]      / t if (t := s["sawlc"]["time_min"])      > 0 else 0 for s in sims]

    all_prots = sorted({p for s in sims
                        for algo in ("tetris_cpu", "tetris_gpu", "sawlc")
                        for p in s[algo]["monomers"]})
    cmap = dict(zip(all_prots, plt.cm.tab20(np.linspace(0, 1, max(len(all_prots), 1)))))

    titulo  = f"COMPARATIVA FINAL: TETRIS (CPU/GPU) vs SAWLC\n"
    titulo += f"Tomograma {TOMO_ID} — " + ("con membrana" if MEMBRANE_FILE else "sin membrana")

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(titulo, fontsize=18, fontweight="bold")
    plt.subplots_adjust(hspace=0.40, wspace=0.28, top=0.88)

    for ax in axs.flat:
        ax.set_xlabel("Simulaciones (Nº Tipos de Proteína)", fontsize=11, fontweight="bold")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

    # 1. Saturación
    ax = axs[0, 0]
    if MEMBRANE_FILE:
        ax.axhline(MEMBRANE_LEVEL, color="black", ls=":", alpha=0.5,
                   label=f"Membrana ({MEMBRANE_LEVEL:.1f}%)")
    ax.plot(xs, col("tetris_cpu", "occ"), "o-",  color="#1f77b4", lw=2.5, label="Tetris CPU")
    ax.plot(xs, col("tetris_gpu", "occ"), "D-",  color="#9b30f0", lw=2.5, label="Tetris GPU")
    ax.plot(xs, col("sawlc",      "occ"), "s--", color="#ff7f0e", lw=2.5, label="SAWLC")
    ax.set_title("Saturación Alcanzada",  fontsize=13, fontweight="bold")
    ax.set_ylabel("Ocupancia Total (%)")
    ax.legend(fontsize=9)

    # 2. Tiempo
    ax = axs[0, 1]
    ax.plot(xs, col("tetris_cpu", "time_min"), "o-",  color="#1f77b4", lw=2.5, label="Tetris CPU")
    ax.plot(xs, col("tetris_gpu", "time_min"), "D-",  color="#9b30f0", lw=2.5, label="Tetris GPU")
    ax.plot(xs, col("sawlc",      "time_min"), "s--", color="#ff7f0e", lw=2.5, label="SAWLC")
    ax.set_title("Tiempo de Ejecución",   fontsize=13, fontweight="bold")
    ax.set_ylabel("Minutos")
    ax.legend(fontsize=9)

    # 3. Monómeros (barras apiladas)
    ax = axs[1, 0]
    w  = 0.25
    offsets = {"tetris_cpu": -w, "tetris_gpu": 0.0, "sawlc": w}
    edges   = {"tetris_cpu": "white", "tetris_gpu": "white", "sawlc": "black"}
    for algo, off in offsets.items():
        bottom = np.zeros(len(sims))
        for prot in all_prots:
            vals = np.array([s[algo]["monomers"].get(prot, 0) for s in sims], dtype=float)
            if vals.any():
                ax.bar(np.array(xs, dtype=float) + off, vals, w, bottom=bottom,
                       color=cmap[prot], edgecolor=edges[algo], lw=0.4)
                bottom += vals
    ax.set_title("Población de Monómeros", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cantidad de Monómeros")
    ax.legend(handles=[Line2D([0],[0], color=cmap[p], lw=6, label=p) for p in all_prots],
              title="Proteínas", loc="upper left", fontsize="xx-small", ncol=2)

    # 4. Rendimiento
    ax = axs[1, 1]
    ax.plot(xs, t_cpu_ppm, "o-",  color="#1f77b4", lw=2.5, label="Tetris CPU")
    ax.plot(xs, t_gpu_ppm, "D-",  color="#9b30f0", lw=2.5, label="Tetris GPU")
    ax.plot(xs, s_ppm,     "s--", color="#ff7f0e", lw=2.5, label="SAWLC")
    ax.set_title("Rendimiento (Proteínas/min)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Proteínas / min")
    ax.legend(fontsize=9)

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT / f"comparativa_tomo{TOMO_ID}_{_escenario}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] Guardado: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if _CACHE.exists():
        print(f"[CACHE] Cargando {_CACHE}")
        cache = json.loads(_CACHE.read_text(encoding="utf-8"))
        cfg   = cache.get("config", {})
        if cfg.get("proteins") != PROTEINS_ALL or cfg.get("membrane_file") != MEMBRANE_FILE:
            print("[CACHE] Config cambió — re-ejecutando…")
            cache = _run_all_sims()
            _CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    else:
        print("[RUN] Sin cache — ejecutando simulaciones…")
        cache = _run_all_sims()
        _CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"[CACHE] Guardado: {_CACHE}")

    _plot(cache)


if __name__ == "__main__":
    main()
