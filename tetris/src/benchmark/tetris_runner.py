"""
Wrapper de Tetris para el benchmark científico.
Lee TARGET_OCC y VOI_SHAPE como variables parchables por benchmark_scientific_comparison.py.
No modifica insert_proteins_tetris.py.
Formato de salida compatible con los parsers del benchmark.
"""
import sys, os, time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "tetris_3d"))

from tetris import Tetris3D, xp, GPU_AVAILABLE as HAS_GPU
from image_processing_3d import ImageProcessing3D
from parser_3d import Parser3D
from insert_proteins_tetris import (
    pick_seed, sorted_proteinSizes, crop_volume,
    ROOT_PATH, PROTEIN_ISO_THRESHOLD_RATIO, TRIES_CLUSTERING,
)
import lio

# --- Parámetros parchables por el benchmark ---
PROTEINS_LIST = [
    "in_10A/2uv8_10A.pns",
]
TARGET_OCC = 0.075
VOI_SHAPE = (500, 500, 250)
# ----------------------------------------------

def main() -> None:
    start = time.time()

    membrane_volume = xp.zeros(VOI_SHAPE, dtype="float32")
    allowed_mask = xp.ones(VOI_SHAPE, dtype=bool)

    molecules = []
    for p_path in sorted_proteinSizes(PROTEINS_LIST):
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p_path), str(ROOT_PATH))
        vol_c = crop_volume(vol, vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        molecules.append((os.path.basename(p_path), vol_c))

    if not molecules:
        print("DONE: 0 inserted in 0m 0.00s")
        return

    global_threshold = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris = Tetris3D(dimensions=VOI_SHAPE, threshold=global_threshold)

    total_inserted = 0
    target_reached = False

    for name, volume in molecules:
        if TARGET_OCC > 0 and float(tetris.get_occupancy()) >= TARGET_OCC:
            target_reached = True
            break

        box_size = max(volume.shape)
        seed = pick_seed(allowed_mask, tetris.output_volume, global_threshold, box_size)
        consecutive_failures = 0

        while consecutive_failures < TRIES_CLUSTERING:
            if seed is None:
                break
            if TARGET_OCC > 0 and float(tetris.get_occupancy()) >= TARGET_OCC:
                target_reached = True
                break

            rotated, _ = ImageProcessing3D.randomly_rotate(volume)
            rotated_bin = ImageProcessing3D.smooth_and_binarize(rotated, 1.5, global_threshold)
            template, _, _ = ImageProcessing3D.create_in_shell(rotated_bin, (0, 2), penalty=100)

            res = tetris.insert_molecule_3d(template, rotated, name, allowed_mask, seed, box_size)
            if res == "inserted":
                total_inserted += 1
                consecutive_failures = 0
                seed = tetris.all_coordinates[-1]
            else:
                consecutive_failures += 1
                seed = pick_seed(allowed_mask, tetris.output_volume, global_threshold, box_size)

        if target_reached:
            break

    final_occ = float(tetris.get_occupancy())
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = elapsed % 60

    stop = "target-reached" if target_reached else "saturation"
    print(f"total_occ = {final_occ * 100:.4f}%")
    print(f"stop_reason = {stop}")
    print(f"DONE: {total_inserted} inserted in {mins}m {secs:.2f}s")


if __name__ == "__main__":
    main()
