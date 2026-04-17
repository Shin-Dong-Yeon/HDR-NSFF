#!/usr/bin/env bash
# =============================================================================
# run_colmap_0.sh
#
# Full COLMAP pipeline for hdr-nsff:
#   1. Automatic reconstruction  (feature extraction + matching + mapping)
#   2. Image undistortion        (→ dense/images/)
#   3. Motion mask undistortion  (→ dense/motion_masks/)
#
# Prerequisites:
#   - SAM2 notebook already run → <data_path>/masks/ and <data_path>/motion_masks/
#   - colmap installed (conda install -c conda-forge colmap)
#   - hdr-nsff-sam2 conda env activated
#
# Usage:
#   bash nsff_scripts/run_colmap_0.sh
#
# Directory layout expected:
#   <data_path>/
#     images/         ← raw frames (.jpg or .png)
#     masks/          ← COLMAP mask format (fg=0, bg=255) from SAM2 notebook
#     motion_masks/   ← motion masks (fg=255) from SAM2 notebook
#
# Output:
#   <data_path>/
#     sparse/0/       ← COLMAP sparse model
#     dense/
#       images/       ← undistorted frames
#       motion_masks/ ← undistorted motion masks (fg=255, grayscale PNG)
#       sparse/
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# CONFIGURATION — edit data_path before running
# ---------------------------------------------------------------------------
data_path="/home/shindy/projects/hdr-4d/hdr-nsff/data/hdr-gopro/bear_thread_test"

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
echo "================================================================"
echo " COLMAP pipeline"
echo " data_path : $data_path"
echo "================================================================"

# Sanity checks
[ -d "$data_path/images" ]       || { echo "ERROR: $data_path/images not found"; exit 1; }
[ -d "$data_path/masks" ]        || { echo "ERROR: $data_path/masks not found (run SAM2 notebook first)"; exit 1; }
[ -d "$data_path/motion_masks" ] || { echo "ERROR: $data_path/motion_masks not found (run SAM2 notebook first)"; exit 1; }

# ---------------------------------------------------------------------------
# 1. COLMAP automatic reconstruction
#    --single_camera 1 : all frames share one camera model (GoPro / fixed lens)
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] COLMAP automatic reconstruction ..."
if [ -d "$data_path/sparse/0" ]; then
    echo "  sparse/0 already exists — skipping reconstruction."
else
    colmap automatic_reconstructor \
        --workspace_path "$data_path" \
        --image_path     "$data_path/images" \
        --mask_path      "$data_path/masks" \
        --single_camera  1
fi

# ---------------------------------------------------------------------------
# 2. Undistort images → dense/images/
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] COLMAP image undistortion (images → dense/images/) ..."
if [ -d "$data_path/dense/images" ]; then
    echo "  dense/images already exists — skipping."
else
    colmap image_undistorter \
        --image_path  "$data_path/images" \
        --input_path  "$data_path/sparse/0" \
        --output_path "$data_path/dense" \
        --output_type COLMAP
fi

# ---------------------------------------------------------------------------
# 3. Undistort motion masks → dense/motion_masks/
#    Uses preprocess.py --gopro_undistortion which:
#      - Reads masks/ (fg=0), inverts → motion_masks/ (fg=255)
#      - Runs colmap image_undistorter on motion_masks/
#      - Moves result → dense/motion_masks/
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Motion mask undistortion (→ dense/motion_masks/) ..."
if [ -d "$data_path/dense/motion_masks" ]; then
    echo "  dense/motion_masks already exists — skipping."
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hdr-nsff-sam2

    python "${SCRIPT_DIR}/preprocess.py" \
        --gopro_undistortion \
        --src_dir "$data_path"
fi

echo ""
echo "================================================================"
echo " Done!"
echo ""
echo " Outputs:"
echo "   $data_path/sparse/0/        ← COLMAP sparse model"
echo "   $data_path/dense/images/    ← undistorted frames"
echo "   $data_path/dense/motion_masks/ ← undistorted motion masks"
echo ""
echo " Next: run save_poses_nerf.py then run_nerf.py"
echo "================================================================"
