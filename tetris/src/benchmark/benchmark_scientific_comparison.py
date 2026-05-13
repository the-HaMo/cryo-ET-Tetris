"""
Benchmark científico SAWLC vs Tetris 3D
========================================
Compara densidad de empaquetamiento y eficiencia computacional para distintos
targets de ocupancia. Ambos algoritmos usan el mismo VOI y la misma proteína.

No modifica ningún script fuente. Solo parchea el fichero .pns (dato).
"""
from __future__ import annotations

import json, multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
SRC        = ROOT / "src"
DATA       = ROOT / "data"
OUT_BASE   = DATA / "data_generated" / "output"
OUT_REPORT = OUT_BASE / "benchmark_scientific"

# Configuración
PROTEIN_FILE = [
    "in_10A/2uv8_10A.pns",
    "in_10A/5mrc_10A.pns",
    "in_10A/4v4r_10A.pns",
    "in_10A/4v94_10A.pns",
    "in_10A/4cr2_10A.pns",
    "in_10A/1qvr_10A.pns",
    "in_10A/3cf3_10A.pns",
    "in_10A/2cg9_10A.pns",
    "in_10A/1u6g_10A.pns",
    "in_10A/3d2f_10A.pns",
    "in_10A/1s3x_10A.pns"
]
TARGETS_PERCENT   = [2.5, 5.0] #, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0]
REPEATS_PER_TARGET = 1
BENCHMARK_VOI_SHAPE = (500, 500, 250)   # reducir si hay OOM; usar (300,300,250) en GPU

FORCE_SAWLC_SHORT_CHAIN = False
SHORT_CHAIN_PMER_L_MAX  = 1
#

@dataclass
class RunMetrics:
    algorithm:                    str
    target_percent:               float
    repeat_id:                    int
    runtime_seconds:              float
    occupancy_percent:            Optional[float]
    proteins_inserted:            Optional[int]
    time_per_insertion_seconds:   Optional[float]
    proteins_per_second:          Optional[float]
    pmer_fails:                   Optional[int]
    stop_reason:                  str
    saturated:                    bool


# ─── .pns helpers (dato, no código fuente) ───────────────────────────────────

def _set_pns(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, replaced = [], False
    for ln in lines:
        if ln.strip().startswith(key):
            out.append(f"{key} = {value}"); replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(f"{key} = {value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

def _get_pns(path: Path, key: str) -> Optional[str]:
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln.startswith(key):
            parts = ln.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


# ─── Workers (nivel de módulo — requerido por multiprocessing.spawn) ──────────

def _tetris_sci_worker(proteins: list, voi_shape: tuple, target_occ: float, q) -> None:
    """Corre Tetris hasta target_occ y devuelve métricas."""
    import sys as _sys, os as _os, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    from tetris import Tetris3D, xp
    from image_processing_3d import ImageProcessing3D
    from parser_3d import Parser3D
    from insert_proteins_tetris import (
        pick_seed, sorted_proteinSizes, crop_volume,
        ROOT_PATH, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING,
    )

    start   = _t.time()
    allowed = xp.ones(voi_shape, dtype=bool)

    mols = []
    for p in sorted_proteinSizes(proteins):
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p), str(ROOT_PATH))
        vol_c  = crop_volume(vol, vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        mols.append((_os.path.basename(p), vol_c))

    if not mols:
        q.put({"occ": 0.0, "inserted": 0, "runtime": _t.time()-start,
               "stop": "no-molecules", "pmer_fails": None})
        return

    g_thresh    = mols[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris      = Tetris3D(dimensions=voi_shape, threshold=g_thresh)
    total       = 0
    target_hit  = False
    seed_fails  = 0

    for _, (name, vol) in enumerate(mols, 1):
        if float(tetris.get_occupancy()) * 100.0 >= target_occ:
            target_hit = True; break
        bsize  = max(vol.shape)
        seed   = pick_seed(allowed, tetris.output_volume, g_thresh, bsize)
        if seed is None:
            seed_fails += 1
            continue
        fails  = 0
        while fails < TRIES_CLUSTERING:
            if float(tetris.get_occupancy()) * 100.0 >= target_occ:
                target_hit = True; break
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
                if seed is None:
                    seed_fails += 1
                    break
        else:
            seed_fails += 1  # agotó TRIES_CLUSTERING sin insertar
        if target_hit: break

    q.put({"occ":      float(tetris.get_occupancy()) * 100.0,
           "inserted": total,
           "runtime":  _t.time() - start,
           "stop":     "target-reached" if target_hit else "saturation",
           "pmer_fails": seed_fails})


def _sawlc_sci_worker(proteins: list, voi_shape: tuple, target_occ: float, q) -> None:
    """Corre SAWLC con polnet y se detiene al alcanzar target_occ."""
    import sys as _sys, io as _io, re as _re, time as _t
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "sawlc"))
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tetris_3d"))

    import numpy as _np
    from polnet.sample import SyntheticSample, PnFile
    from insert_proteins_tetris import ROOT_PATH, sorted_proteinSizes

    start  = _t.time()
    sample = SyntheticSample(shape=voi_shape, v_size=10, offset=(0, 0, 0))

    buf = _io.StringIO()
    old = _sys.stdout; _sys.stdout = buf
    stop_reason = "saturation"
    try:
        for p_path in sorted_proteinSizes(proteins):
            voi_check = sample._SyntheticSample__voi
            if 100.0 * float(_np.count_nonzero(~voi_check)) / voi_check.size >= target_occ:
                stop_reason = "target-reached"
                break
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
        _sys.stdout = old

    elapsed = _t.time() - start
    txt     = buf.getvalue()

    voi  = sample._SyntheticSample__voi
    occ  = 100.0 * float(_np.count_nonzero(~voi)) / voi.size

    m_ins   = _re.search(r"Total proteinas insertadas:\s*(\d+)", txt)
    m_fails = _re.search(r"Pmer fails:\s*(\d+)", txt)
    inserted   = int(m_ins.group(1))   if m_ins   else None
    pmer_fails = int(m_fails.group(1)) if m_fails else None

    if stop_reason == "saturation" and pmer_fails:
        stop_reason = "attempt-limit"
    q.put({"occ": occ, "inserted": inserted, "runtime": elapsed,
           "stop": stop_reason, "pmer_fails": pmer_fails})


# ─── Orquestación ─────────────────────────────────────────────────────────────

def _spawn(target, args) -> dict:
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=target, args=(*args, q))
    p.start(); p.join()
    return q.get()


def _run_algo(algo: str, proteins: list, target_pct: float, rep: int) -> RunMetrics:
    print(f"\n  [{algo.upper()}] target={target_pct:.1f}% rep={rep}…")

    if algo == "tetris":
        d = _spawn(_tetris_sci_worker, (proteins, BENCHMARK_VOI_SHAPE, target_pct))
    else:
        d = _spawn(_sawlc_sci_worker,  (proteins, BENCHMARK_VOI_SHAPE, target_pct))

    rt  = d["runtime"]
    ins = d["inserted"]
    occ = d["occ"]
    tpi = (rt / ins)     if ins and ins > 0 else None
    pps = (ins / rt)     if ins and rt > 0  else None
    sat = d["stop"] not in {"target-reached"}

    print(f"  [{algo.upper()}] occ={occ:.2f}%  ins={ins}  t={rt:.1f}s  stop={d['stop']}")
    return RunMetrics(
        algorithm=algo, target_percent=target_pct, repeat_id=rep,
        runtime_seconds=rt, occupancy_percent=occ,
        proteins_inserted=ins, time_per_insertion_seconds=tpi,
        proteins_per_second=pps, pmer_fails=d["pmer_fails"],
        stop_reason=d["stop"], saturated=sat,
    )


# ─── Tabla / resumen ──────────────────────────────────────────────────────────

def _mean(vals):
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else float("nan")

def _std(vals):
    clean = [v for v in vals if v is not None]
    return float(np.std(clean)) if clean else float("nan")


def _print_table(rows: list) -> None:
    print("\n" + "="*130)
    print("BENCHMARK CIENTÍFICO: SAWLC vs TETRIS")
    print("="*130)
    hdr = f"{'Algo':<8} {'Target%':>8} {'Rep':>4} {'Occ%':>8} {'Prot':>7} {'Time(s)':>9} {'s/Prot':>9} {'Prot/s':>9} {'Fails':>7} {'Stop':>14} {'Sat':>5}"
    print(hdr); print("-"*130)
    for r in rows:
        print(
            f"{r.algorithm:<8} {r.target_percent:>8.1f} {r.repeat_id:>4d}"
            f" {'NA' if r.occupancy_percent is None else f'{r.occupancy_percent:.2f}':>8}"
            f" {'NA' if r.proteins_inserted is None else str(r.proteins_inserted):>7}"
            f" {r.runtime_seconds:>9.2f}"
            f" {'NA' if r.time_per_insertion_seconds is None else f'{r.time_per_insertion_seconds:.4f}':>9}"
            f" {'NA' if r.proteins_per_second is None else f'{r.proteins_per_second:.3f}':>9}"
            f" {'NA' if r.pmer_fails is None else str(r.pmer_fails):>7}"
            f" {r.stop_reason:>14} {'yes' if r.saturated else 'no':>5}"
        )
    print("="*130)


# ─── Gráficas ─────────────────────────────────────────────────────────────────

def _generate_plots(rows: list) -> None:
    targets = sorted({r.target_percent for r in rows})

    def by(algo, t):
        return [r for r in rows if r.algorithm == algo and r.target_percent == t]

    saw_occ      = [_mean([r.occupancy_percent  for r in by("sawlc",  t)]) for t in targets]
    tet_occ      = [_mean([r.occupancy_percent  for r in by("tetris", t)]) for t in targets]
    saw_occ_std  = [_std( [r.occupancy_percent  for r in by("sawlc",  t)]) for t in targets]
    tet_occ_std  = [_std( [r.occupancy_percent  for r in by("tetris", t)]) for t in targets]

    saw_time     = [_mean([r.runtime_seconds    for r in by("sawlc",  t)]) for t in targets]
    tet_time     = [_mean([r.runtime_seconds    for r in by("tetris", t)]) for t in targets]
    saw_time_std = [_std( [r.runtime_seconds    for r in by("sawlc",  t)]) for t in targets]
    tet_time_std = [_std( [r.runtime_seconds    for r in by("tetris", t)]) for t in targets]

    saw_tpi      = [_mean([r.time_per_insertion_seconds for r in by("sawlc",  t)]) for t in targets]
    tet_tpi      = [_mean([r.time_per_insertion_seconds for r in by("tetris", t)]) for t in targets]
    saw_tpi_std  = [_std( [r.time_per_insertion_seconds for r in by("sawlc",  t)]) for t in targets]
    tet_tpi_std  = [_std( [r.time_per_insertion_seconds for r in by("tetris", t)]) for t in targets]

    saw_fail     = [_mean([float(r.pmer_fails) if r.pmer_fails is not None else float("nan")
                           for r in by("sawlc", t)]) for t in targets]
    saw_fail_std = [_std( [float(r.pmer_fails) if r.pmer_fails is not None else float("nan")
                           for r in by("sawlc", t)]) for t in targets]

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SAWLC vs Tetris", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    t_ext     = [0] + list(targets)
    saw_occ_x = [0.0] + saw_occ;  saw_std_x = [0.0] + saw_occ_std
    tet_occ_x = [0.0] + tet_occ;  tet_std_x = [0.0] + tet_occ_std
    ax.errorbar(t_ext, saw_occ_x, yerr=saw_std_x, fmt="o-", lw=2, capsize=4, label="SAWLC")
    ax.errorbar(t_ext, tet_occ_x, yerr=tet_std_x, fmt="s-", lw=2, capsize=4, label="Tetris")
    ax.axline((0, 0), slope=1, color="k", ls="--", lw=1.5, alpha=0.8, zorder=10, label="Objetivo (y=x)")
    ax.set_title("Densidad alcanzada (packing fraction)")
    ax.set_xlabel("Ocupancia objetivo (%)"); ax.set_ylabel("Ocupancia final binaria (%)")
    ax.legend()

    ax = axes[0, 1]
    ax.errorbar(targets, saw_time, yerr=saw_time_std, fmt="o-", lw=2, capsize=4, label="SAWLC")
    ax.errorbar(targets, tet_time, yerr=tet_time_std, fmt="s-", lw=2, capsize=4, label="Tetris")
    ax.set_title("Tiempo total por target")
    ax.set_xlabel("Ocupancia objetivo (%)"); ax.set_ylabel("Tiempo (s)")
    ax.legend()

    ax = axes[1, 0]
    ax.errorbar(targets, saw_tpi, yerr=saw_tpi_std, fmt="o-", lw=2, capsize=4, label="SAWLC")
    ax.errorbar(targets, tet_tpi, yerr=tet_tpi_std, fmt="s-", lw=2, capsize=4, label="Tetris")
    ax.set_title("Tiempo por inserción exitosa")
    ax.set_xlabel("Ocupancia objetivo (%)"); ax.set_ylabel("s / proteína")
    ax.legend()

    ax = axes[1, 1]
    tet_fail     = [_mean([float(r.pmer_fails) if r.pmer_fails is not None else float("nan")
                           for r in by("tetris", t)]) for t in targets]
    tet_fail_std = [_std( [float(r.pmer_fails) if r.pmer_fails is not None else float("nan")
                           for r in by("tetris", t)]) for t in targets]
    x = np.arange(len(targets), dtype=float)
    w = 0.3
    ax.bar(x - w/2, saw_fail, w, yerr=saw_fail_std, capsize=4, label="SAWLC pmer_fails")
    ax.bar(x + w/2, tet_fail, w, yerr=tet_fail_std, capsize=4, label="Tetris seed_fails")
    ax.set_xticks(x); ax.set_xticklabels([str(int(t)) for t in targets])
    ax.set_title("Fallos de inserción acumulados")
    ax.set_xlabel("Ocupancia objetivo (%)"); ax.set_ylabel("Fallos (promedio)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUT_REPORT / "benchmark_scientific_plots.png"
    plt.savefig(out, dpi=180); plt.close()
    print(f"\n[OK] Gráfica: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    proteins = PROTEIN_FILE if isinstance(PROTEIN_FILE, list) else [PROTEIN_FILE]

    current_config = {
        "protein_file":    PROTEIN_FILE,
        "targets_percent": TARGETS_PERCENT,
        "repeats":         REPEATS_PER_TARGET,
        "voi_shape":       list(BENCHMARK_VOI_SHAPE),
        "sawlc_short_chain": FORCE_SAWLC_SHORT_CHAIN,
    }
    out_json = OUT_REPORT / "benchmark_scientific_report.json"

    # ── Intentar cargar caché ─────────────────────────────────────────────────
    if out_json.exists():
        cached = json.loads(out_json.read_text(encoding="utf-8"))
        if cached.get("config") == current_config:
            print(f"[CACHE] Config coincide — cargando resultados de {out_json}")
            rows = [RunMetrics(**r) for r in cached["rows"]]
            _print_table(rows)
            _generate_plots(rows)
            return
        print("[CACHE] Config cambió — re-ejecutando simulaciones…")

    # ── Ejecutar simulaciones ─────────────────────────────────────────────────
    pns_paths = [DATA / p for p in proteins]
    missing = [str(p) for p in pns_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No existen: {missing}")

    orig_occs  = {p: _get_pns(DATA / p, "PMER_OCC")   for p in proteins}
    orig_lmaxs = {p: _get_pns(DATA / p, "PMER_L_MAX") for p in proteins}

    rows: list = []
    try:
        if FORCE_SAWLC_SHORT_CHAIN:
            for p in proteins:
                _set_pns(DATA / p, "PMER_L_MAX", str(SHORT_CHAIN_PMER_L_MAX))

        for target in TARGETS_PERCENT:
            for p in proteins:
                _set_pns(DATA / p, "PMER_OCC", str(target / 100.0))
            for rep in range(1, REPEATS_PER_TARGET + 1):
                print(f"\n[RUN] target={target:.1f}% | rep={rep}")
                rows.append(_run_algo("sawlc",  proteins, target, rep))
                rows.append(_run_algo("tetris", proteins, target, rep))

        _print_table(rows)
        _generate_plots(rows)

        summary: Dict = {"sawlc": {}, "tetris": {}}
        for algo in ("sawlc", "tetris"):
            for t in TARGETS_PERCENT:
                vals = [r for r in rows if r.algorithm == algo and r.target_percent == t]
                summary[algo][str(t)] = {
                    "mean_occ":        _mean([r.occupancy_percent for r in vals]),
                    "std_occ":         _std( [r.occupancy_percent for r in vals]),
                    "mean_runtime":    _mean([r.runtime_seconds   for r in vals]),
                    "mean_tpi":        _mean([r.time_per_insertion_seconds for r in vals]),
                    "mean_pps":        _mean([r.proteins_per_second        for r in vals]),
                    "mean_inserted":   _mean([float(r.proteins_inserted) if r.proteins_inserted else None for r in vals]),
                    "mean_pmer_fails": _mean([float(r.pmer_fails) if r.pmer_fails else None for r in vals]),
                }

        report = {"config": current_config, "rows": [asdict(r) for r in rows], "summary": summary}
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Reporte: {out_json}")

    finally:
        for p in proteins:
            if orig_occs.get(p):  _set_pns(DATA / p, "PMER_OCC",   orig_occs[p])
            if orig_lmaxs.get(p): _set_pns(DATA / p, "PMER_L_MAX", orig_lmaxs[p])
        print("[RESTORE] .pns restaurados")


if __name__ == "__main__":
    main()
