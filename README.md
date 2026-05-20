# Tetris — Protein Insertion for Synthetic Cryo-ET Tomograms

This repository contains the implementation, benchmarking suite, and evaluation scripts for **Tetris**, a protein insertion algorithm developed as part of a Bachelor's Thesis at the Universidad de Murcia (Facultad de Informática).

Tetris is designed as a high-throughput alternative to the SAWLC (Self-Avoiding Worm-Like Chain) insertion module used in [PolNet](https://github.com/anmartinezs/polnet). It uses 3D FFT cross-correlation to find, at each step, the globally optimal placement position for a new macromolecule within a simulation volume — achieving higher volumetric occupancy and, with GPU acceleration, shorter runtimes than SAWLC.

The algorithm is adapted from [TemplateLearning](https://github.com/MohamadHarastani/TemplateLearning), which introduced the original FFT-based packing strategy. This repository extends it to operate inside volumes that already contain biological structures (lipid membranes), integrates it with the PolNet simulation pipeline, and evaluates it against SAWLC.

---

## Repository structure

```
tetris/
├── setup_env.sh                # Environment setup script (Linux/Mac)
├── setup_env.bat               # Environment setup script (Windows)
├── requirements.txt            # CPU-only dependencies
├── requirements-gpu.txt        # GPU dependencies (CUDA 12)
│
├── data/
│   ├── templates/
│   │   └── membranes/          # (create folder before) Membrane MRC label files (tomo_mem_lbls_*.mrc)
│   ├── in_10A/                 # Protein descriptor files at 10 Å resolution (.pns)
│   └── data_generated/         # Output directory (density + label MRCs)
│
└── src/
    ├── tetris_3d/              # Core Tetris algorithm
    │   ├── tetris.py           # Tetris3D class (FFT correlation, placement logic)
    │   ├── insert_proteins_tetris.py   # Main entry point — configuration & run
    │   ├── image_processing_3d.py      # 3D morphological utilities
    │   ├── parser_3d.py        # Protein .pns file parser
    │   └── lio.py              # MRC I/O utilities
    │
    ├── sawlc/
    │   └── insert_proteins_in_membranes.py   # SAWLC insertion wrapper (PolNet)
    │
    └── benchmark/
        ├── profile_test.py                   # Per-phase timing profiler
        ├── profile_comparison.py             # Occupancy-vs-time: Tetris CPU/GPU vs SAWLC
        ├── benchmark_scientific_comparison.py # Target-sweep comparison (2.5%–57.5%)
        └── sawlc-cpu-gpu/
            ├── plot_comparison.py            # 4-panel comparison from pre-recorded logs
            ├── logs_tetris/                  # Tetris CPU execution logs (1–11 protein types)
            ├── logs_tetris_gpu/              # Tetris GPU execution logs
            └── logs_sawlc/                   # SAWLC execution logs
```

---

## Installation

The `setup_env.sh` script (Linux/Mac) handles everything: it creates a virtual environment, asks whether you have an NVIDIA GPU, and installs the appropriate dependencies automatically.

```bash
cd tetris
bash setup_env.sh
```

On Windows, use `setup_env.bat` instead.

Once the environment is created, activate it before running any script:

```bash
source .venv/bin/activate
```

---

## Quick start

### 1. Run Tetris

Edit `src/tetris_3d/insert_proteins_tetris.py` to configure your run:

```python
USE_GPU = True                  # set to False to use CPU only

MEMBRANE_FILES = [
    "tomo_mem_lbls_3.mrc",      # membrane MRC files inside data/templates/membranes/
]

PROTEINS_LIST = [
    "in_10A/2uv8_10A.pns",     # protein descriptor files inside data/
    "in_10A/1s3x_10A.pns",
    # ...
]

TRIES_CLUSTERING = 10           # consecutive failures before moving to next protein type
VOI_SHAPE = (500, 500, 250)     # output tomogram shape in voxels
```

Then run from the `src/tetris_3d/` directory:

```bash
cd tetris/src/tetris_3d
python insert_proteins_tetris.py
```

Output files are written to `data/data_generated/output/output_proteins_tetris/`:
- `tomo<i>_den<n>.mrc` — density volume
- `tomo<i>_lbl<n>.mrc` — label volume (0 = empty, 1 = membrane, 2 = protein)

### 2. Run SAWLC

```bash
cd tetris/src/sawlc
python insert_proteins_in_membranes.py
```

---

## Benchmarks

All benchmark scripts are run from `src/benchmark/` and share the same experimental setup: 11 protein types at 10 Å resolution.

### Occupancy-vs-time curve

Runs Tetris GPU, Tetris CPU, and SAWLC sequentially on the same volume and produces a cumulative occupancy vs time line chart.

```bash
cd tetris/src/benchmark
python profile_comparison.py
```

GPU/CPU mode for Tetris is controlled via the `USE_GPU` flag in `insert_proteins_tetris.py`, or without editing via the environment variable:

```bash
TETRIS_USE_GPU=0 python profile_comparison.py   # force CPU
```

### Per-phase profiler

Instruments Tetris with fine-grained timers and produces an occupancy-vs-time chart coloured by phase (seed selection, rotation, template construction, FFT correlation, overlap validation).

```bash
python profile_test.py
```

### Target-sweep comparison

Sweeps target occupancies from 2.5% to 57.5% in 2.5% steps, running both Tetris and SAWLC 4 times at each target. Results are cached in a JSON file to avoid redundant computation.

```bash
python benchmark_scientific_comparison.py
```

### Multi-protein-type comparison (from logs)

Reads pre-recorded execution logs (one per number of active protein types, 1–11) and produces a 4-panel figure: total occupancy, execution time, monomer population, and throughput.

```bash
cd tetris/src/benchmark/sawlc-cpu-gpu
python plot_comparison.py
```

---

## Algorithm overview

Tetris reframes protein placement as an optimisation problem. At each insertion step:

1. A seed coordinate is sampled uniformly from the free volume (`pick_seed`).
2. A local bounding box is extracted around the seed.
3. A binary occupancy map is constructed by thresholding the existing density at a global isosurface value (`min peak intensity × 0.08`).
4. An **in-shell template** is built by morphologically dilating the new protein mask and subtracting the protein itself — encoding the ideal contact zone.
5. The **3D FFT cross-correlation** between the shell template and the binary occupancy map is computed in O(N log N), producing a score map where high values indicate tight-packing positions.
6. The position with the maximum valid score is selected and the protein is placed.

Membrane pre-loading marks membrane voxels with a high density and penalises any candidate position that overlaps them, restricting insertion to biologically permitted regions.

GPU acceleration (CuPy) targets the two dominant bottlenecks: the global `argwhere` inside `pick_seed` and the FFT cross-correlation, delivering a ~×3 speedup over the CPU implementation.

---

## Protein set (10 Å resolution)

| Protein | Dimensions | Occupied voxels | Internal occupancy |
|---|:---:|:---:|:---:|
| 4v4r_10A | 66×66×66 | 5,440 | 1.89% |
| 5mrc_10A | 72×72×72 | 7,599 | 2.04% |
| 2uv8_10A | 48×48×48 | 7,458 | 6.74% |
| 4v94_10A | 72×72×72 | 4,985 | 1.34% |
| 4cr2_10A | 66×66×66 | 3,499 | 1.22% |
| 3d2f_10A | 62×62×62 |   739 | 0.31% |
| 3cf3_10A | 54×54×54 |   814 | 0.52% |
| 2cg9_10A | 52×52×52 |   576 | 0.41% |
| 1u6g_10A | 56×56×56 |   741 | 0.42% |
| 1s3x_10A | 46×46×46 |   156 | 0.16% |
| 1qvr_10A | 56×56×56 |   957 | 0.54% |

---

## Related work

- **TemplateLearning** (original algorithm this work is based on): [github.com/MohamadHarastani/TemplateLearning](https://github.com/MohamadHarastani/TemplateLearning)
- **PolNet**: [github.com/anmartinezs/polnet](https://github.com/anmartinezs/polnet)
- **SAWLC**: A. Martínez-Sánchez et al., *Simulating the cellular context of structural studies*, 2024.
