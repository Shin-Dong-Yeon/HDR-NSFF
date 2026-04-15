# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HDR-NSFF** is a research project for 4D dynamic scene reconstruction with HDR video rendering. Extends NeRF + NSFF for dynamic scenes, multi-exposure HDR fusion, and camera response function (CRF) learning.

## Key Commands

### Build CUDA Extension (required for optical flow)
```bash
cd nsff_scripts/alt_cuda_corr/
python setup.py build_ext --inplace
```

### Preprocessing Pipeline
```bash
# 1. Extract camera poses (COLMAP)
python nsff_scripts/save_poses_nerf.py --data_path <dataset_path>

# 2. Resize images → outputs images_{W}x{H}/
python nsff_scripts/resize_images.py --data_path <data> --resize_height <H>

# 3. Metric depth estimation → outputs depth-anything/ (npy) and depth-anything-viz/ (png)
python nsff_exp/Depth-Anything-V2/metric_depth/run.py --encoder vitl \
    --load-from nsff_exp/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --max-depth 20 --data-dir <dataset_path> --save-numpy

# 4. Gamma correction (for HDR multi-exposure data)
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
python evaluation_gopro_time_36.py --config configs/<config_file.txt>

# Render modes
python run_nerf.py --config <config> --render_lockcam --target_idx 2
python run_nerf.py --config <config> --render_bt --target_idx 20
python run_nerf.py --config <config> --render_train
```

No test suite or linter is configured — this is a research codebase.

## Architecture

### Data Flow

```
Video Frames (GoPro/DSLR)
    ↓
nsff_scripts/ [PREPROCESSING]
  save_poses_nerf.py   → Camera poses via COLMAP
  resize_images.py     → Resized frames (images_{W}x{H}/)
  Depth-Anything-V2    → Metric depth maps (depth-anything/)
  run_gamma_*.py       → Exposure compensation for HDR
    ↓
nsff_exp/ [TRAINING & RENDERING]
  run_nerf.py          → Main training loop, loss computation
  render_utils.py      → Volumetric rendering, scene flow warping, tone mapping
  cam_param.py         → CRF + white balance learning
    ↓
evaluation_*.py [METRICS]
  evaluation_gopro.py / evaluation_dslr.py → PSNR/SSIM/LPIPS
  evaluation_gopro_time*.py                → Temporal interpolation evaluation
```

### Core Modules (`nsff_exp/`)

- **`run_nerf.py`** — Entry point: data loading, training loop, checkpointing, loss computation.
- **`render_utils.py`** — Ray casting, volumetric rendering, scene flow warping, tone mapping (Reinhard/Filmic/WB).
- **`run_nerf_helpers.py`** — NeRF MLP (8-layer, width 256, skip connections), positional encoding.
- **`cam_param.py`** — CRF parameterization, white balance estimation, leaky thresholding, inverse tone mapping.
- **`config.py`** — All config arguments via `configargparse`. All evaluation scripts share this via `from config import config_parser`.
- **`difix_updater.py`** — Iterative 3D geometry refinement (enabled via `use_difix = True`).
- **`load_llff.py` / `load_blender.py`** — Dataset loaders.

### Loss Functions

| Loss | Config Key | Purpose |
|------|-----------|---------|
| Photometric | (base) | MSE/MAE on rendered vs. target RGB |
| Depth | `w_depth` | Align with Depth-Anything estimates |
| Optical Flow | `w_optical_flow` | Match predicted flow |
| Perceptual | `w_percept` | LPIPS feature-level similarity |
| Scene Flow Cycle | `w_cycle` | Bidirectional scene flow consistency |
| Scene Flow Reg | `w_sf_reg` | Encourage small flow magnitude |
| Scene Flow Smooth | `w_sm` | Spatial smoothness + least kinetic energy |
| Prob Reg | `w_prob_reg` | Disocclusion weight regularization |
| CRF Smooth | `w_crf_smooth` | CRF smoothness |

### Config System

Configs in `nsff_exp/configs/` as `.txt` files. Key parameters:

```
dataset_type = llff           # or 'blender'
disp_model = depth-anything   # midas / aligned / depth-anything
render_tm = tm                # tm / wb / no
share_crf = False
use_difix = True              # iterative 3D refinement
use_percept = True
chain_sf = True               # 5-frame scene flow chain
start_frame / end_frame / step_frame
```

Sub-directories: `configs/gopro/`, `configs/lego/`, `configs/tank/`.
Public configs: `config_pub/`.

### Symbolic Links

- `logs/` → `/node_data/shindy/projects/hdr-4d/logs/hdr-nsff/`
- `dataset_gpu/` → `/node_data/shindy/projects/hdr-4d/dataset_gpu`
- `dataset_home/` → `/node_data/shindy/projects/hdr-4d/dataset_home`

## Tech Stack

- **PyTorch** — Neural rendering, Adam optimizer with LR decay
- **Depth-Anything-V2** — Metric monocular depth (`nsff_exp/Depth-Anything-V2/`; checkpoint `.pth` files excluded by `.gitignore`)
- **RAFT** — Optical flow (`raft-things.pth`), with custom CUDA kernel in `nsff_scripts/alt_cuda_corr/`
- **configargparse** — Hierarchical config (file + CLI args)
- **Kornia** — Camera geometry (`create_meshgrid`)
- **Wandb / TensorBoard** — Experiment tracking
