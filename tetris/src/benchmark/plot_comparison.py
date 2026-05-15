"""
COMPARATIVA FINAL: TETRIS (CPU/GPU) vs SAWLC
Una sola simulación con todas las proteínas. Métricas acumuladas por tipo.

Resultados cacheados en cache_tomo{TOMO_ID}_{escenario}.json.
Borrar el JSON para forzar re-ejecución.
No modifica ningún script fuente.
"""
from __future__ import annotations

import json, multiprocessing as mp, re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

_BENCHMARK = Path(__file__).resolve().parent
_SRC       = _BENCHMARK.parent
_OUTPUT    = _BENCHMARK.parents[1] / "data" / "data_generated" / "output"

# ─── Configuración ────────────────────────────────────────────────────────────
TOMO_ID        = 3
MEMBRANE_FILE: Optional[str] = None   # None → sin membrana
MEMBRANE_LEVEL = 11.5450

PROTEINS_ALL: List[str] = [
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
    "in_10A/1qvr_10A.pns",
]
# ──────────────────────────────────────────────────────────────────────────────

_escenario = Path(MEMBRANE_FILE).stem if MEMBRANE_FILE else "empty"
_CACHE     = _BENCHMARK / f"cache_tomo{TOMO_ID}_{_escenario}.json"


# ─── Workers (nivel de módulo — requerido por multiprocessing.spawn) ──────────

def _tetris_worker(proteins: list, membrane_file, force_cpu: bool, q) -> None:
    """Corre Tetris con todas las proteínas y devuelve métricas por tipo."""
    import sys as _sys, os as _os, io as _io, time as _t
    from pathlib import Path as _P

    # GPU control must happen before any GPU-dependent import
    if force_cpu:
        _os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        _sys.modules["cupy"]  = None  # type: ignore[assignment]
        _sys.modules["cupyx"] = None  # type: ignore[assignment]

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    # import insert_proteins_tetris FIRST — runs its USE_GPU control code before cupy is imported
    from insert_proteins_tetris import (
        pick_seed, sorted_proteinSizes, crop_volume,
        ROOT_PATH, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING,
    )

    import numpy as _np
    from tetris import Tetris3D, xp
    from image_processing_3d import ImageProcessing3D
    from parser_3d import Parser3D
    import lio

    t_global = _t.time()
    _mem_dir = ROOT_PATH / "templates" / "membranes"

    if membrane_file:
        mem_path = _mem_dir / membrane_file
        mem = lio.load_mrc(str(mem_path)).astype("float32") if mem_path.exists() \
              else _np.zeros((500, 500, 250), dtype="float32")
    else:
        mem = _np.zeros((500, 500, 250), dtype="float32")

    allowed = xp.asarray(~(mem > 0))

    # Cargar y recortar proteínas en CPU, luego subir a GPU si procede
    mols = []
    for p in sorted_proteinSizes(proteins):
        vol_np, _ = Parser3D.load_protein(str(ROOT_PATH / p), str(ROOT_PATH))
        vol_c     = crop_volume(vol_np, vol_np.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        mols.append((_os.path.basename(p), xp.asarray(vol_c)))

    if not mols:
        q.put({"occ": 0.0, "time_min": 0.0, "monomers": {}, "total_monomers": 0, "per_type": []})
        return

    g_thresh = max(float(v.max()) for _, v in mols) * PROTEIN_ISO_THRESHOLD_RATIO
    tetris   = Tetris3D(dimensions=tuple(mem.shape), threshold=g_thresh)
    tetris.output_volume[~allowed] = 500.0

    monomers: dict = {}
    total   = 0
    per_type: list = []

    for _, (name, vol) in enumerate(mols, 1):
        t_type = _t.time()
        before = total
        bsize  = max(vol.shape)
        seed   = pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        fails  = 0

        buf = _io.StringIO()
        old = _sys.stdout; _sys.stdout = buf
        try:
            while fails < TRIES_CLUSTERING:
                if seed is None:
                    break
                rot, _     = ImageProcessing3D.randomly_rotate(vol)
                rbin       = ImageProcessing3D.smooth_and_binarize(rot, 1.5, g_thresh)
                tmpl, _, _ = ImageProcessing3D.create_in_shell(rbin, (0, 2), penalty=100)
                res = tetris.insert_molecule_3d(tmpl, rot, name, allowed, seed, bsize)
                if res == "inserted":
                    total += 1; fails = 0
                    seed = tetris.all_coordinates[-1]
                else:
                    fails += 1
                    seed = pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        finally:
            _sys.stdout = old

        key     = name.split("_")[0]
        n       = total - before
        occ_now = float(tetris.get_occupancy()) * 100.0
        monomers[key] = monomers.get(key, 0) + n
        per_type.append({
            "name":      key,
            "monomers":  n,
            "time_s":    _t.time() - t_type,
            "occ_after": occ_now,
        })
        tag = "CPU" if force_cpu else "GPU"
        _sys.__stdout__.write(f"  [Tetris {tag}] {key}  ins={n}  occ={occ_now:.2f}%\n")
        _sys.__stdout__.flush()

    q.put({
        "occ":            float(tetris.get_occupancy()) * 100.0,
        "time_min":       (_t.time() - t_global) / 60.0,
        "monomers":       monomers,
        "total_monomers": sum(monomers.values()),
        "per_type":       per_type,
    })


def _sawlc_worker(proteins: list, membrane_file, q) -> None:
    """Corre SAWLC con polnet, proteína a proteína, y devuelve métricas por tipo."""
    import sys as _sys, io as _io, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "sawlc"))
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    import numpy as _np
    from polnet.sample import SyntheticSample, PnFile
    from insert_proteins_tetris import ROOT_PATH, MEMBRANES_PATH, sorted_proteinSizes
    import lio

    t_global = _t.time()

    if membrane_file:
        mem_path = MEMBRANES_PATH / membrane_file
        mem_vol  = lio.load_mrc(str(mem_path)).astype("float32") if mem_path.exists() \
                   else _np.zeros((500, 500, 250), dtype="float32")
    else:
        mem_vol = _np.zeros((500, 500, 250), dtype="float32")

    mem_mask = mem_vol > 0
    sample   = SyntheticSample(shape=mem_vol.shape, v_size=10, offset=(0, 0, 0))
    voi      = sample._SyntheticSample__voi
    voi[mem_mask] = False
    sample._SyntheticSample__bg_voi = voi.copy()

    monomers: dict = {}
    per_type: list = []

    for p_path in sorted_proteinSizes(proteins):
        t_type = _t.time()

        buf = _io.StringIO()
        old = _sys.stdout; _sys.stdout = buf
        try:
            pn = PnFile().load(ROOT_PATH / p_path)
            sample.add_set_cproteins(params=pn, data_path=ROOT_PATH,
                                     surf_dec=0.9, mmer_tries=20,
                                     pmer_tries=100, verbosity=True)
        finally:
            _sys.stdout = old

        txt = buf.getvalue()
        m_total = re.search(r"Total proteinas insertadas:\s*(\d+)", txt)
        n_inserted = int(m_total.group(1)) if m_total else sum(
            int(m) for m in re.findall(r"Paso\s+\d+:\s+(\d+)\s+proteinas", txt)
        )

        voi_now  = sample._SyntheticSample__voi
        occ_now  = 100.0 * float(_np.count_nonzero(~voi_now)) / voi_now.size

        key = p_path.split("/")[-1].split("_")[0]
        monomers[key] = monomers.get(key, 0) + n_inserted
        per_type.append({
            "name":      key,
            "monomers":  n_inserted,
            "time_s":    _t.time() - t_type,
            "occ_after": occ_now,
        })
        print(f"  [SAWLC] {key}  ins={n_inserted}  occ={occ_now:.2f}%")

    final_voi = sample._SyntheticSample__voi
    occ       = 100.0 * float(_np.count_nonzero(~final_voi)) / final_voi.size

    q.put({
        "occ":            occ,
        "time_min":       (_t.time() - t_global) / 60.0,
        "monomers":       monomers,
        "total_monomers": sum(monomers.values()),
        "per_type":       per_type,
    })


# ─── Ejecución ────────────────────────────────────────────────────────────────

def _spawn(target, args) -> dict:
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=target, args=(*args, q))
    p.start(); p.join()
    if q.empty():
        raise RuntimeError(f"Worker terminó sin resultados (exit code {p.exitcode})")
    result = q.get()
    if "error" in result:
        raise RuntimeError(f"Worker falló:\n{result['error']}")
    return result


def _run_single_sim() -> dict:
    import sys as _sys
    _sys.path.insert(0, str(_SRC / "tetris_3d"))
    from tetris import GPU_AVAILABLE

    print(f"\n{'─'*60}\n[SIM] {len(PROTEINS_ALL)} proteínas\n{'─'*60}")

    if GPU_AVAILABLE:
        print("\n[SIM] Tetris GPU…")
        t_gpu = _spawn(_tetris_worker, (PROTEINS_ALL, MEMBRANE_FILE, False))
        print(f"      occ={t_gpu['occ']:.2f}%  t={t_gpu['time_min']:.2f}min  "
              f"total={t_gpu['total_monomers']}")
    else:
        print("\n[SIM] GPU no disponible — omitiendo Tetris GPU")
        t_gpu = None

    print("\n[SIM] Tetris CPU…")
    t_cpu = _spawn(_tetris_worker, (PROTEINS_ALL, MEMBRANE_FILE, True))
    print(f"      occ={t_cpu['occ']:.2f}%  t={t_cpu['time_min']:.2f}min  "
          f"total={t_cpu['total_monomers']}")

    print("\n[SIM] SAWLC…")
    sawlc = _spawn(_sawlc_worker, (PROTEINS_ALL, MEMBRANE_FILE))
    print(f"      occ={sawlc['occ']:.2f}%  t={sawlc['time_min']:.2f}min  "
          f"total={sawlc['total_monomers']}")

    return {
        "config": {"tomo_id": TOMO_ID, "membrane_file": MEMBRANE_FILE,
                   "proteins": PROTEINS_ALL},
        "sawlc":      sawlc,
        "tetris_gpu": t_gpu,
        "tetris_cpu": t_cpu,
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot(cache: dict) -> None:
    algos = {"Tetris CPU": cache["tetris_cpu"], "SAWLC": cache["sawlc"]}
    if cache.get("tetris_gpu"):
        algos = {"Tetris CPU": cache["tetris_cpu"],
                 "Tetris GPU": cache["tetris_gpu"],
                 "SAWLC":      cache["sawlc"]}
    colors = {"Tetris CPU": "#1f77b4", "Tetris GPU": "#9b30f0", "SAWLC": "#ff7f0e"}
    marks  = {"Tetris CPU": "o",       "Tetris GPU": "D",       "SAWLC": "s"}
    lstyle = {"Tetris CPU": "-",       "Tetris GPU": "-",       "SAWLC": "--"}

    all_names = [e["name"] for e in list(algos.values())[0]["per_type"]]
    xs  = list(range(1, len(all_names) + 1))
    xsa = np.array(xs, dtype=float)

    titulo = f"COMPARATIVA FINAL: TETRIS (CPU/GPU) vs SAWLC\nTomograma {TOMO_ID}"

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.subplots_adjust(hspace=0.42, wspace=0.28, top=0.88)

    for ax in axs.flat:
        ax.set_xticks(xs)
        ax.set_xticklabels(all_names, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Tipos de Proteína", fontsize=10)
        ax.grid(True, alpha=0.3)

    # 1. Saturación Alcanzada — ocupancia acumulada por tipo (líneas)
    ax = axs[0, 0]
    if MEMBRANE_FILE:
        ax.axhline(MEMBRANE_LEVEL, color="black", ls=":", alpha=0.5,
                   label=f"Membrana ({MEMBRANE_LEVEL:.1f}%)")
    for label, data in algos.items():
        occs = [e["occ_after"] for e in data["per_type"]]
        ax.plot(xs, occs, marker=marks[label], ls=lstyle[label],
                color=colors[label], lw=2.5, label=label)
    ax.set_title("Saturación Alcanzada", fontsize=13, fontweight="bold")
    ax.set_ylabel("Ocupancia Total (%)")
    ax.legend(fontsize=9)

    # 2. Tiempo de Ejecución — tiempo acumulado por tipo (líneas)
    ax = axs[0, 1]
    for label, data in algos.items():
        cumtime = np.cumsum([e["time_s"] / 60.0 for e in data["per_type"]]).tolist()
        ax.plot(xs, cumtime, marker=marks[label], ls=lstyle[label],
                color=colors[label], lw=2.5, label=label)
    ax.set_title("Tiempo de Ejecución", fontsize=13, fontweight="bold")
    ax.set_ylabel("Minutos")
    ax.legend(fontsize=9)

    # 3. Población de Monómeros — barras apiladas acumuladas por tipo de proteína
    ax = axs[1, 0]
    w   = 0.25
    off = {"Tetris CPU": -w, "Tetris GPU": 0.0, "SAWLC": w}
    prot_colors = plt.cm.tab20(np.linspace(0, 1, len(all_names)))
    first_algo  = list(algos.keys())[0]
    for label, data in algos.items():
        bottom = np.zeros(len(xs))
        xpos   = xsa + off[label]
        for i, entry in enumerate(data["per_type"]):
            heights        = np.zeros(len(xs))
            heights[i:]    = entry["monomers"] or 0
            ax.bar(xpos, heights, w, bottom=bottom,
                   color=prot_colors[i], alpha=0.85,
                   label=all_names[i] if label == first_algo else "")
            bottom += heights
    ax.set_title("Población de Monómeros", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cantidad")
    handles = [plt.Rectangle((0, 0), 1, 1, color=prot_colors[i])
               for i in range(len(all_names))]
    ax.legend(handles, all_names, title="Tipos de Proteína",
              fontsize=7, ncol=2, loc="upper left")

    # 4. Rendimiento — throughput por tipo (líneas)
    ax = axs[1, 1]
    for label, data in algos.items():
        tput = [(e["monomers"] or 0) / (e["time_s"] / 60.0)
                if e["time_s"] > 0 else 0
                for e in data["per_type"]]
        ax.plot(xs, tput, marker=marks[label], ls=lstyle[label],
                color=colors[label], lw=2.5, label=label)
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
    current_config = {"tomo_id": TOMO_ID, "membrane_file": MEMBRANE_FILE,
                      "proteins": PROTEINS_ALL}

    if _CACHE.exists():
        cache = json.loads(_CACHE.read_text(encoding="utf-8"))
        if cache.get("config") == current_config:
            print(f"[CACHE] Config coincide — cargando {_CACHE}")
            _plot(cache)
            return
        print("[CACHE] Config cambió — re-ejecutando…")

    cache = _run_single_sim()
    _CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"[CACHE] Guardado: {_CACHE}")
    _plot(cache)


if __name__ == "__main__":
    main()
