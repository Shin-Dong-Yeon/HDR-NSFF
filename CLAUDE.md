# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HDR-NSFF** is a research project for 4D dynamic scene reconstruction with HDR video rendering. It extends Neural Radiance Fields (NeRF) with Neural Space-Time Flow (NSFF) for dynamic scenes, multi-exposure HDR fusion, and camera response function (CRF) learning.

## Key Commands

### Build CUDA Extension (required for optical flow)
```bash
cd nsff_scripts/alt_cuda_corr/
python setup.py build_ext --inplace
```

### Download Pre-trained Models
```bash
bash nsff_scripts/download_models.sh
```

### Preprocessing Pipeline
```bash
# 1. Extract camera poses
python nsff_scripts/save_poses_nerf.py --data_path <dataset_path>

# 2. Organize frames
python nsff_scripts/preprocess.py --src_dir <video_data>

# 3. Depth estimation (choose one)
python nsff_scripts/run_midas.py --data_path <data> --resize_height <H>
python nsff_scripts/run_midas_hdr.py --data_path <data> --resize_height <H>

# 4. Optical flow
python nsff_scripts/run_flows_video.py --model models/raft-things.pth --data_path <data> --step 1 --skip_moseg
python nsff_scripts/run_flows_video_hdr.py --model <raft_model> --data_path <data> --step 1

# 5. Gamma correction (for HDR multi-exposure data)
python nsff_scripts/run_gamma_correction.py --data_path <data>
python nsff_scripts/run_gamma_correction_3exp.py --data_path <data>  # 3-exposure variant
```

### Training
```bash
cd nsff_exp/
python run_nerf.py --config configs/<config_file.txt>
```

### Evaluation / Rendering
```bash
cd nsff_exp/
python evaluation_gopro.py --config configs/<config_file.txt>
python evaluation_dslr.py --config configs/<config_file.txt>
python evaluation_gopro_time.py --config configs/<config_file.txt>

# Render specific modes
python run_nerf.py --config <config> --render_lockcam --render_tm wb --target_idx 2
python run_nerf.py --config <config> --render_bt --target_idx 20   # Space-time interpolation
python run_nerf.py --config <config> --render_train                # Render training views
```

### HDR Metrics
```bash
cd nsff_exp/
python hdr_evaluation/hdr_video_mertens.py --src_dir <ldr_image_dir>
python hdr_evaluation/eval.py  # Compute PSNR/SSIM/LPIPS
```

### Fine-tuning
```bash
cd nsff_exp/
python finetune.py --config <config_file> [additional args]
```

No test suite or linter is configured — this is a research codebase.

## Architecture

### Data Flow

```
Video Frames (GoPro/DSLR/Blender)
    ↓
nsff_scripts/ [PREPROCESSING]
  save_poses_nerf.py → Camera poses (COLMAP/Blender metadata)
  preprocess.py      → Frame extraction, resizing, organization
  run_midas*.py      → Monocular depth maps
  run_flows_video*.py→ Optical flow (RAFT, accelerated via alt_cuda_corr)
  run_gamma_*.py     → Exposure compensation
    ↓
nsff_exp/ [TRAINING & RENDERING]
  run_nerf.py        → Main training loop
  render_utils.py    → Volumetric rendering, tone mapping
  cam_param.py       → HDR camera parameter learning
    ↓
nsff_exp/hdr_evaluation/ [METRICS]
  eval.py            → PSNR/SSIM/LPIPS
```

### Core Modules (all in `nsff_exp/`)

- **`run_nerf.py`** (1981 lines) — Entry point: loads data, trains model, checkpoints, loss computation. Integrates all components.
- **`render_utils.py`** (2316 lines) — Ray casting, volumetric rendering, scene flow warping, depth-guided sampling, tone mapping (Reinhard, Filmic, WB).
- **`run_nerf_helpers.py`** — NeRF MLP architecture (8-layer, width 256, skip connections), positional encoding (frequency-based).
- **`cam_param.py`** (37KB) — Camera Response Function (CRF) parameterization, white balance estimation, leaky thresholding, inverse tone mapping.
- **`load_llff.py` / `load_blender.py`** — Dataset loaders for real (LLFF format) and synthetic (Blender) data.
- **`config.py`** — 156+ config arguments via `configargparse`.
- **`difix_updater.py`** (52KB) — Iterative 3D geometry refinement (enabled via `use_difix = True`).
- **`models/`** — LPIPS perceptual loss (VGG/AlexNet features).

### Loss Functions

| Loss | Config Key | Purpose |
|------|-----------|---------|
| Photometric | (base) | MSE/MAE on rendered vs. target RGB |
| Depth | `w_depth` | Align with MiDas/Depth-Anything estimates |
| Optical Flow | `w_optical_flow` | Match predicted flow with RAFT estimates |
| Perceptual | `w_percept` | LPIPS feature-level similarity |
| CRF | `w_crf` | Camera response function monotonicity |
| White Balance | `w_unit` | White balance unit constraints |
| Scene Flow | implicit | Smoothness and probability regularization |

### Config System

Configs live in `nsff_exp/configs/` as `.txt` files parsed by `configargparse`. Key parameters:

```
dataset_type = llff           # or 'blender'
disp_model = midas            # midas / aligned / depthcrafter / depth-anything
render_tm = tm                # tm (tone mapping) / wb (white balance) / no
share_crf = False             # Shared CRF across frames
use_difix = True              # Enable iterative 3D refinement
use_percept = True            # Enable perceptual loss
chain_sf = True               # Chain scene flow across frames
start_frame / end_frame / step_frame  # Frame range selection
```

Sub-directories organize configs by dataset: `configs/gopro/`, `configs/lego/`, `configs/tank/`, `configs/raft-mask/`.

### Symbolic Links

The repo uses symlinks to large data stored on a GPU server:
- `logs/` → `/node_data/shindy/projects/hdr-4d/logs/hdr-nsff/`
- `dataset_gpu/` → `/node_data/shindy/projects/hdr-4d/dataset_gpu`
- `dataset_home/` → `/node_data/shindy/projects/hdr-4d/dataset_home`

## Tech Stack

- **PyTorch** — Neural rendering, optimization (Adam with LR scheduling)
- **RAFT** (pre-trained `raft-things.pth`) — Optical flow, with custom CUDA correlation kernel in `nsff_scripts/alt_cuda_corr/`
- **MiDas / Depth-Anything-V2** — Monocular depth estimation
- **OpenCV** — Image I/O, Mertens HDR fusion, tone mapping
- **configargparse** — Hierarchical config management
- **LPIPS** (`lpips` package) — Perceptual loss computation
- **Kornia** — Camera geometry (`create_meshgrid`)
- **Wandb / TensorBoard** — Experiment tracking
