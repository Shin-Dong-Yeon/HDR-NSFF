# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HDR-NSFF** is a research project for 4D dynamic scene reconstruction with HDR video rendering. Extends NeRF + NSFF for dynamic scenes, multi-exposure HDR fusion, and camera response function (CRF) learning.

## Conda Environments

Two separate environments are required (torch version incompatibility):

| Env | Torch | Purpose |
|-----|-------|---------|
| `hdr-nsff-sam2` | 2.5.1+cu121 | NeRF training, SAM2 mask generation, JupyterLab |
| `dino-tracker` | 2.1.0+cu121 | DINO-tracker semantic flow only |

Setup scripts: `nsff_scripts/setup_hdr_nsff_sam2.sh`, `nsff_scripts/setup_dino_tracker.sh`

## Full Preprocessing Pipeline

```
Raw Video
  ↓ (1) SAM2 notebook       → masks/, motion_masks/
  ↓ (2) COLMAP              → sparse/0/, dense/images/, dense/motion_masks/
  ↓ (3) save_poses_nerf.py  → dense/poses_bounds.npy
  ↓ (4) resize_images.py    → dense/images_{W}x{H}/
  ↓ (5) Depth-Anything-V2   → dense/depth-anything/
  ↓ (6) run_gamma_*.py      → (HDR only) exposure-corrected frames
  ↓ (7) DINO-tracker        → dense/dino-tracker/semantic_flow/semantic_flows_i1_{fwd,bwd}.npy
  ↓ (8) run_flows_video.py  → dense/semantic_flow_i1/*.npz
```

### Step-by-step commands

```bash
# (1) SAM2 motion masks — run interactively in JupyterLab (hdr-nsff-sam2 env)
#     nsff_scripts/sam2_motion_mask.ipynb
#     Output: <data_path>/masks/  and  <data_path>/motion_masks/

# (2) COLMAP (install via: sudo apt-get install colmap)
bash nsff_scripts/run_colmap_0.sh          # edit data_path inside first

# (3) Camera poses
conda activate hdr-nsff-sam2
python nsff_scripts/save_poses_nerf.py --data_path <dense_path>

# (4) Resize images → images_{W}x{H}/
python nsff_scripts/resize_images.py --data_path <dense_path> --resize_height <H>

# (5) Metric depth → depth-anything/
python nsff_exp/Depth-Anything-V2/metric_depth/run.py --encoder vitl \
    --load-from nsff_exp/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --max-depth 20 --data-dir <dense_path> --save-numpy

# (6) Gamma correction (HDR multi-exposure data)
python nsff_scripts/run_gamma_correction.py --data_path <dense_path>
python nsff_scripts/run_gamma_correction_3exp.py --data_path <dense_path>  # 3-exposure variant

# (7)+(8) DINO-tracker semantic flow (dino-tracker env, edit data_path inside first)
bash nsff_scripts/run_dino-tracker_0.sh
```

### Build CUDA Extension (required for optical flow)
```bash
cd nsff_scripts/alt_cuda_corr/
python setup.py build_ext --inplace
```

## Training & Evaluation

```bash
cd nsff_exp/
python run_nerf.py --config configs/<config_file.txt>

# Evaluation
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

### Core Modules (`nsff_exp/`)

- **`run_nerf.py`** — Entry point: data loading, training loop, checkpointing, loss computation.
- **`render_utils.py`** — Ray casting, volumetric rendering, scene flow warping, tone mapping (Reinhard/Filmic/WB).
- **`run_nerf_helpers.py`** — NeRF MLP (8-layer, width 256, skip connections), positional encoding.
- **`cam_param.py`** — CRF parameterization, white balance estimation, leaky thresholding, inverse tone mapping.
- **`config.py`** — All config arguments via `configargparse`. All evaluation scripts share this via `from config import config_parser`.
- **`difix_updater.py`** — Iterative 3D geometry refinement (enabled via `use_difix = True`).
- **`load_llff.py` / `load_blender.py`** — Dataset loaders.

### DINO-tracker (`nsff_scripts/dino-tracker/`)

Semantic optical flow via DINO features. Included in repo (not a submodule).

Key custom files (modified from upstream):
- **`models/extractor.py`** — Fixed DINOv2 `attn_drop` hook registration (new DINOv2 stores it as float, not `nn.Module`; fallback to `proj_drop`).
- **`models/model_inference.py`** — Extended with vectorized flow methods: `compute_framewise_flow_tensor()` / `_bwd()` (foreground-masked) and `compute_framewise_flow_tensor_all_pixels()` / `_bwd()`.
- **`inference_semantic_flow.py`** — Runs fwd/bwd semantic flow inference; saves to `<data-path>/dino-tracker/semantic_flow/semantic_flows_i{interval}_{fwd,bwd}.npy`.

Pipeline (called by `run_dino-tracker_0.sh`):
1. `preprocessing/main_preprocessing.py` → `dino_embeddings/`, `dino_best_buddies/`
2. `train.py` → `models/dino_tracker/` checkpoints
3. `inference_semantic_flow.py` → `semantic_flow/semantic_flows_i1_fwd.npy`, `*_bwd.npy`
4. `nsff_scripts/run_flows_video.py --semantic_flow` → `semantic_flow_i1/*.npz` (per-frame pair)

### SAM2 (`nsff_scripts/sam2_repo/`)

Interactive notebook `sam2_motion_mask.ipynb` for foreground mask generation.
- Images must be JPEG for SAM2's `load_video_frames`; the notebook auto-converts PNG → temp JPEG dir.
- Uses `matplotlib.use('module://ipympl.backend_nbagg')` for remote server compatibility (not `%matplotlib widget`).
- Output: `motion_masks/` (fg=255) and `masks/` (fg=0, for COLMAP `--mask_path`).

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
- **SAM2** — Interactive video segmentation for motion masks (`nsff_scripts/sam2_repo/`)
- **DINO-tracker** — Semantic optical flow via DINOv2 features (`nsff_scripts/dino-tracker/`)
- **Depth-Anything-V2** — Metric monocular depth (`nsff_exp/Depth-Anything-V2/`; checkpoint `.pth` files excluded by `.gitignore`)
- **RAFT** — Optical flow (`raft-things.pth`), with custom CUDA kernel in `nsff_scripts/alt_cuda_corr/`
- **COLMAP** — Camera pose estimation and image undistortion (install via `sudo apt-get install colmap`)
- **configargparse** — Hierarchical config (file + CLI args)
- **Kornia** — Camera geometry (`create_meshgrid`)
- **Wandb / TensorBoard** — Experiment tracking
