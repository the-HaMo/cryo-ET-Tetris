"""Profile Tetris 3D and render occupancy vs time with a time breakdown bar."""

import sys, time
from pathlib import Path

# Add tetris_3d directory to path so we can import its modules
tetris_3d_dir = Path(__file__).parent.parent / "tetris_3d"
sys.path.insert(0, str(tetris_3d_dir))

# insert_proteins_tetris controla USE_GPU y bloquea CuPy si es necesario
# antes de que image_processing_3d y tetris se importen
from insert_proteins_tetris import (
    USE_GPU,
    MEMBRANES_PATH,
    MEMBRANE_FILES,
    OUT_DIR,
    PROTEINS_LIST,
    PROTEIN_ISO_THRESHOLD_RATIO,
    MEMBRANE_INTENSITY_SCALE,
    VOI_VSIZE,
    TRIES_CLUSTERING,
    VOI_SHAPE,
    ROOT_PATH,
    sorted_proteinSizes,
    pick_seed,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from image_processing_3d import ImageProcessing3D
from parser_3d import Parser3D
import tetris as tetris_mod
from tetris import Tetris3D, xp, GPU_AVAILABLE
import lio


def _sync_if_gpu():
    if GPU_AVAILABLE:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def _plot_occupancy_timeline(timeline, totals, total_time, output_path):
    labels = [
        "seed_pick",
        "rotation_affine",
        "template",
        "fft_correlation",
        "insert_validation",
        "other",
    ]
    if totals.get("other", 0.0) <= 0:
        labels = [label for label in labels if label != "other"]
    colors = {
        "seed_pick": "#1b9e77",
        "rotation_affine": "#66a61e",
        "template": "#e6ab02",
        "fft_correlation": "#171adf",
        "insert_validation": "#ff0000",
        "other": "#aaaaaa",
    }

    fig, ax = plt.subplots(figsize=(10, 4.0))

    if not timeline:
        ax.text(0.5, 0.5, "No occupancy samples", transform=ax.transAxes, ha="center")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    times = [t for t, _occ in timeline]
    occs = [occ for _t, occ in timeline]
    if times[-1] < total_time:
        times = times + [total_time]
        occs = occs + [occs[-1]]
    t_grid = np.linspace(0.0, total_time, 1000)
    occ_grid = np.interp(t_grid, times, occs)
    ax.plot(times, occs, color="#1f77b4", linewidth=2)

    # Color under the curve based on total time per task.
    t_start = 0.0
    legend_patches = []
    for label in labels:
        seg = totals.get(label, 0.0)
        if seg <= 0:
            continue
        t_end = t_start + seg
        mask = (t_grid >= t_start) & (t_grid <= t_end)
        ax.fill_between(
            t_grid,
            occ_grid,
            where=mask,
            color=colors[label],
            alpha= 0.7,
            interpolate=True,
        )
        pct = (seg / total_time) * 100.0 if total_time > 0 else 0.0
        legend_patches.append(Patch(color=colors[label], label=f"{label} ({pct:.1f}%)"))
        t_start = t_end

    ax.set_ylabel("Occupancy (%)")
    ax.set_xlabel("Seconds")
    ax.set_title(f"Occupancy vs time (total: {total_time:.2f}s)")
    ax.set_xlim(0, total_time)
    ax.legend(handles=legend_patches, ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    output_dir = OUT_DIR

    start_time = time.perf_counter()
    if MEMBRANE_FILES:
        membrane_path = MEMBRANES_PATH / MEMBRANE_FILES[0]
        membrane_volume = lio.load_mrc(str(membrane_path)).astype("float32")
        print(f"[PROFILE] Using membrane: {membrane_path.name}")
    else:
        membrane_volume = np.zeros(VOI_SHAPE, dtype="float32")
        print(f"[PROFILE] No membrane — using empty volume {VOI_SHAPE}")
    allowed_mask = xp.asarray(~(membrane_volume > 0))

    molecules = []
    for p_path in sorted_proteinSizes(PROTEINS_LIST):
        protein_path = ROOT_PATH / p_path
        vol, _ = Parser3D.load_protein(str(protein_path), str(ROOT_PATH))
        vol = xp.asarray(vol)
        coords = xp.argwhere(vol > vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        if coords.size == 0:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        vol_cropped = vol[int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)]
        molecules.append((p_path, vol_cropped))

    if not molecules:
        print("[PROFILE] No molecules found to insert.")
        return

    global_threshold = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris_obj = Tetris3D(dimensions=membrane_volume.shape, threshold=global_threshold)
    tetris_obj.output_volume[~allowed_mask] = 500.0

    totals = {
        "seed_pick": 0.0,
        "rotation_affine": 0.0,
        "template": 0.0,
        "fft_correlation": 0.0,
        "insert_molecule_total": 0.0,
        "occupancy": 0.0,
    }
    timeline = []

    original_correlate = tetris_mod._correlate

    def _timed_correlate(local_bin, template):
        t0 = time.perf_counter()
        out = original_correlate(local_bin, template)
        _sync_if_gpu()
        totals["fft_correlation"] += time.perf_counter() - t0
        return out

    tetris_mod._correlate = _timed_correlate

    class _Null:
        def write(self, _): pass
        def flush(self): pass

    _real_out = sys.stdout

    try:
        for _type_idx, (_name, volume) in enumerate(molecules, start=1):
            box_size   = max(volume.shape)
            n_before   = len(tetris_obj.all_coordinates)

            t0 = time.perf_counter()
            current_target = pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)
            totals["seed_pick"] += time.perf_counter() - t0

            consecutive_failures = 0
            while consecutive_failures < TRIES_CLUSTERING:
                if current_target is None:
                    break

                t0 = time.perf_counter()
                rotated, _ = ImageProcessing3D.randomly_rotate(volume)
                _sync_if_gpu()
                totals["rotation_affine"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                rotated_bin = ImageProcessing3D.smooth_and_binarize(rotated, 1.5, global_threshold)
                template, _, _ = ImageProcessing3D.create_in_shell(rotated_bin, (0, 2), penalty=100)
                _sync_if_gpu()
                totals["template"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                sys.stdout = _Null()
                res = tetris_obj.insert_molecule_3d(template, rotated, _name, allowed_mask, current_target, box_size)
                sys.stdout = _real_out
                _sync_if_gpu()
                totals["insert_molecule_total"] += time.perf_counter() - t0

                if res == "inserted":
                    t0 = time.perf_counter()
                    occ = float(tetris_obj.get_occupancy() * 100.0)
                    _sync_if_gpu()
                    totals["occupancy"] += time.perf_counter() - t0
                    timeline.append((time.perf_counter() - start_time, occ))
                    consecutive_failures = 0
                    current_target = tetris_obj.all_coordinates[-1]
                else:
                    consecutive_failures += 1
                    t0 = time.perf_counter()
                    current_target = pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)
                    totals["seed_pick"] += time.perf_counter() - t0

            n_ins = len(tetris_obj.all_coordinates) - n_before
            occ   = float(tetris_obj.get_occupancy() * 100.0)
            key   = str(_name).split("/")[-1].split("_")[0]
            label = "GPU" if GPU_AVAILABLE else "CPU"
            print(f"  [Tetris {label}] {key}  ins={n_ins}  occ={occ:.2f}%")
    finally:
        sys.stdout = _real_out
        tetris_mod._correlate = original_correlate

    total_time = time.perf_counter() - start_time
    totals["insert_validation"] = max(totals["insert_molecule_total"] - totals["fft_correlation"], 0.0)
    accounted = (
        totals["seed_pick"]
        + totals["rotation_affine"]
        + totals["template"]
        + totals["fft_correlation"]
        + totals["insert_validation"]
    )
    totals["other"] = max(total_time - accounted, 0.0)
    totals["insert_molecule_total"] = 0.0

    output_path = Path(__file__).with_name("profile_occupancy_breakdown.png")
    log_path = Path(__file__).with_name("profile_occupancy_breakdown.txt")
    _plot_occupancy_timeline(timeline, totals, total_time, output_path)
    print(f"[PROFILE] Saved chart: {output_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"total_time_seconds={total_time:.6f}\n")
        for key in ["seed_pick", "rotation_affine", "template", "fft_correlation", "insert_validation", "other"]:
            handle.write(f"{key}_seconds={totals[key]:.6f}\n")
        handle.write("timeline_seconds,occupancy_percent\n")
        for t, occ in timeline:
            handle.write(f"{t:.6f},{occ:.6f}\n")
    print(f"[PROFILE] Saved log: {log_path}")


if __name__ == "__main__":
    main()
