import os, sys, time, numpy as np
from pathlib import Path
from tetris import Tetris3D, xp, GPU_AVAILABLE as HAS_GPU
from image_processing_3d import ImageProcessing3D
from parser_3d import Parser3D
import lio

# CONFIGURACIÓN
ROOT_PATH = Path(__file__).resolve().parents[2] / "data"
MEMBRANES_PATH = ROOT_PATH / "templates" / "membranes"

MEMBRANE_FILES = [
    # "tomo_mem_lbls_0.mrc",
    # "tomo_mem_lbls_1.mrc",
    # "tomo_mem_lbls_2.mrc",
    "tomo_mem_lbls_3.mrc",
]

PROTEINS_LIST = [
    # "in_10A/4v4r_10A.pns",
    # "in_10A/3j9i_10A.pns",
    "in_10A/5mrc_10A.pns",
    # "in_10A/4v7r_10A.pns",
    "in_10A/2uv8_10A.pns",
    # "in_10A/4v94_10A.pns",
    # "in_10A/4cr2_10A.pns",
    # "in_10A/3qm1_10A.pns",
    # "in_10A/3h84_10A.pns",
    # "in_10A/3gl1_10A.pns",
    # "in_10A/3d2f_10A.pns",
    # "in_10A/3cf3_10A.pns",
    # "in_10A/2cg9_10A.pns",
    # "in_10A/1u6g_10A.pns",
    # "in_10A/1s3x_10A.pns",
    # "in_10A/1qvr_10A.pns",
]


OUT_DIR = ROOT_PATH / "data_generated" / "output" / "output_proteins_tetris"
PROTEIN_ISO_THRESHOLD_RATIO = 0.08
MEMBRANE_INTENSITY_SCALE = 0.35
VOI_VSIZE = 10
TRIES_CLUSTERING = 10
VOI_SHAPE = (500, 500, 250)  

def sorted_proteinSizes(proteins_list):
    def internal_occupancy(p_path):
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p_path), str(ROOT_PATH))
        threshold = vol.max() * PROTEIN_ISO_THRESHOLD_RATIO
        return np.count_nonzero(vol > threshold) / vol.size
    return sorted(proteins_list, key=internal_occupancy, reverse=True)

def crop_volume(vol, threshold):
    """densidad real de la proteína"""
    coords = np.argwhere(vol > threshold)
    if coords.size == 0: return vol
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return vol[z0:z1, y0:y1, x0:x1]

def pick_seed(allowed_mask, output_volume, threshold, box_size):
    """Muestreo uniforme para llenar bordes igual que el centro."""
    half = box_size // 2
    z_dim, y_dim, x_dim = output_volume.shape
    empty = allowed_mask & (output_volume <= threshold)
    viable = xp.zeros_like(empty, dtype=bool)
    viable[half:z_dim-half, half:y_dim-half, half:x_dim-half] = empty[half:z_dim-half, half:y_dim-half, half:x_dim-half]
    candidates = xp.argwhere(viable)
    if len(candidates) == 0: return None
    return tuple(int(x) for x in candidates[np.random.randint(0, len(candidates))])

def insert_proteins_in_membrane(membrane_mrc_path, proteins_list, output_dir, membrane_id):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    label = os.path.basename(str(membrane_mrc_path)) if membrane_mrc_path else "empty_volume"
    print(f"\nProcessing Tomogram: {label}")
    start_time = time.time()
    if membrane_mrc_path is None:
        membrane_volume = np.zeros(VOI_SHAPE, dtype=np.float32)
    else:
        membrane_volume = lio.load_mrc(str(membrane_mrc_path)).astype(np.float32)
    allowed_mask = xp.asarray(~(membrane_volume > 0))
    total_voxels = allowed_mask.size
    membrane_occ = float(xp.count_nonzero(~allowed_mask) / total_voxels)
    print(f"\nMembrane occupancy before proteins: {membrane_occ*100:.4f}%")
    
    molecules = []
    for p_path in proteins_list:
        vol, _ = Parser3D.load_protein(str(ROOT_PATH / p_path), str(ROOT_PATH))
        vol_cropped = crop_volume(vol, vol.max() * PROTEIN_ISO_THRESHOLD_RATIO)
        molecules.append((os.path.basename(p_path), vol_cropped))

    global_threshold = molecules[0][1].max() * PROTEIN_ISO_THRESHOLD_RATIO
    tetris_obj = Tetris3D(dimensions=membrane_volume.shape, threshold=global_threshold)
    tetris_obj.output_volume[~allowed_mask] = 500.0 

    total_inserted = 0
    per_type_summary = []
    for type_idx, (name, volume) in enumerate(molecules, start=1):
        box_size = max(volume.shape)
        inserted_before = total_inserted
        current_target = pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)
        if current_target is not None:
            print(f"[!] new seed {current_target}")
        consecutive_failures = 0
        
        while consecutive_failures < TRIES_CLUSTERING:
            if current_target is None: break
            rotated, _ = ImageProcessing3D.randomly_rotate(volume)
            rotated_bin = ImageProcessing3D.smooth_and_binarize(rotated, 1.5, global_threshold)
            template, _, _ = ImageProcessing3D.create_in_shell(rotated_bin, (0, 2), penalty=100)
            
            res = tetris_obj.insert_molecule_3d(template, rotated, name, allowed_mask, current_target, box_size)
            if res == 'inserted':
                total_inserted += 1
                consecutive_failures = 0
                current_target = tetris_obj.all_coordinates[-1]
            else:
                consecutive_failures += 1
                current_target = pick_seed(allowed_mask, tetris_obj.output_volume, global_threshold, box_size)
                if current_target is not None:
                    print(f"[!] No space. New seed {current_target}")

        num_monomers = total_inserted - inserted_before
        total_occ = float(tetris_obj.get_occupancy())
        proteins_occ = max(float(total_occ) - membrane_occ, 0.0)
        inserted_vox = int(round(proteins_occ * total_voxels))
        per_type_summary.append((type_idx, name, inserted_vox, num_monomers, proteins_occ, total_occ))

    final = tetris_obj.output_volume.copy()
    final[~allowed_mask] = 0.0
    combined = final + ((~allowed_mask).astype(np.float32) * (MEMBRANE_INTENSITY_SCALE * final.max()))
    combined_cpu = combined.get() if HAS_GPU else combined
    tomo_idx = Path(membrane_mrc_path).stem.split("_")[-1] if membrane_mrc_path else "empty"
    num_types = len(proteins_list)
    output_mrc = Path(output_dir) / f"tomo{tomo_idx}_den{num_types}.mrc"
    lio.write_mrc(combined_cpu, str(output_mrc), v_size=VOI_VSIZE)

    print(f"\nMembrane occupancy before proteins: {membrane_occ*100:.4f}%")
    for type_idx, name, inserted_vox, num_monomers, proteins_occ, total_occ in per_type_summary:
        print(
            f"After type {type_idx:02d} ({name}): "
            f"inserted_vox={inserted_vox}, "
            f"num_monomers={num_monomers}, "
            f"proteins_occ={proteins_occ*100:.4f}%, "
            f"total_occ={total_occ*100:.4f}%"
        )

    print(f"total_occ = {tetris_obj.get_occupancy()*100:.4f}%")

    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    print(f"DONE: {total_inserted} inserted in {total_minutes}m {total_seconds:.2f}s")

if __name__ == "__main__":
    if MEMBRANE_FILES:
        insert_proteins_in_membrane(MEMBRANES_PATH / MEMBRANE_FILES[0], sorted_proteinSizes(PROTEINS_LIST), OUT_DIR, 0)
    else:
        insert_proteins_in_membrane(None, sorted_proteinSizes(PROTEINS_LIST), OUT_DIR, 0)