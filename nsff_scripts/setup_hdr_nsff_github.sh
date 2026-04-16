#!/usr/bin/env bash
# =============================================================================
# setup_hdr_nsff_github.sh
#
# Creates the `hdr-nsff-github` conda environment that supports:
#   - hdr-nsff  (NeRF training)
#   - SAM2      (interactive motion mask generation)
#   - dino-tracker (semantic optical flow)
#
# Also clones SAM2 and dino-tracker into nsff_scripts/ if not already present.
#
# Usage:
#   cd nsff_scripts/
#   bash setup_hdr_nsff_github.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="hdr-nsff-github"
PYTHON_VERSION="3.10"

# CUDA 12.1 wheels (compatible with driver CUDA 13.x via backward compat)
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
TORCH_VER="2.5.1+cu121"
TORCHVISION_VER="0.20.1+cu121"
TORCHAUDIO_VER="2.5.1+cu121"
XFORMERS_VER="0.0.29.post1"

echo "================================================================"
echo " HDR-NSFF-GitHub environment setup"
echo " ENV : $ENV_NAME"
echo " DIR : $SCRIPT_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/7] Conda env '${ENV_NAME}' already exists — skipping creation."
else
    echo "[1/7] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate inside the script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 12.1)
# ---------------------------------------------------------------------------
echo "[2/7] Installing PyTorch ${TORCH_VER} ..."
pip install --quiet \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    "torchaudio==${TORCHAUDIO_VER}" \
    --index-url "${TORCH_INDEX}"

# xformers — used by dino-tracker for efficient attention
pip install --quiet "xformers==${XFORMERS_VER}" --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# 3. Install hdr-nsff dependencies
# ---------------------------------------------------------------------------
echo "[3/7] Installing hdr-nsff dependencies ..."
pip install --quiet \
    configargparse \
    kornia \
    lpips \
    einops \
    imageio \
    imageio-ffmpeg \
    "numpy>=1.24,<2.0" \
    opencv-python \
    pillow \
    tqdm \
    wandb \
    tensorboard \
    matplotlib \
    scipy \
    scikit-image \
    pandas \
    pyyaml \
    diffusers \
    transformers \
    accelerate \
    ffmpeg-python \
    mediapy

# ---------------------------------------------------------------------------
# 4. Install SAM2
# ---------------------------------------------------------------------------
echo "[4/7] Setting up SAM2 ..."
# NOTE: repo is cloned as sam2_repo (not sam2) to avoid Python package name conflict.
# The sam2 Python package is inside sam2_repo/sam2/, which conflicts if the parent
# directory is also named sam2.
SAM2_DIR="${SCRIPT_DIR}/sam2_repo"

if [ ! -d "${SAM2_DIR}" ]; then
    echo "  Cloning SAM2 into ${SAM2_DIR} ..."
    git clone https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
else
    echo "  SAM2 already cloned at ${SAM2_DIR} — pulling latest ..."
    git -C "${SAM2_DIR}" pull --ff-only || true
fi

pip install --quiet -e "${SAM2_DIR}"

# Download SAM2 checkpoint (sam2.1 hiera large, ~900 MB)
SAM2_CKPT_DIR="${SAM2_DIR}/checkpoints"
SAM2_CKPT="${SAM2_CKPT_DIR}/sam2.1_hiera_large.pt"
# Reinstall torch AFTER dino-tracker requirements (which downgrade it)
echo "  Re-pinning torch to ${TORCH_VER} after dino-tracker install ..."
pip install --quiet --force-reinstall \
    "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
    "torchaudio==${TORCHAUDIO_VER}" "xformers==${XFORMERS_VER}" \
    --index-url "${TORCH_INDEX}"
pip install --quiet "numpy==1.26.4" "packaging<26"
# Fix potential numpy dist-info duplicates
find "${CONDA_PREFIX}/lib/python${PYTHON_VERSION}/site-packages" \
    -maxdepth 1 -name "numpy-*.dist-info" | sort -rV | tail -n +2 | xargs rm -rf || true
if [ ! -f "${SAM2_CKPT}" ]; then
    echo "  Downloading SAM2 checkpoint (~900 MB) ..."
    mkdir -p "${SAM2_CKPT_DIR}"
    # Official checkpoint URL from SAM2 README
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "${SAM2_CKPT}"
    echo "  Saved: ${SAM2_CKPT}"
else
    echo "  SAM2 checkpoint already present."
fi

# SAM2 also needs ipykernel for notebook use
pip install --quiet ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 5. Install dino-tracker
# ---------------------------------------------------------------------------
echo "[5/7] Setting up dino-tracker ..."
DINO_DIR="${SCRIPT_DIR}/dino-tracker"

if [ ! -d "${DINO_DIR}" ]; then
    echo "  Cloning dino-tracker into ${DINO_DIR} ..."
    git clone https://github.com/AssafSinger94/dino-tracker.git "${DINO_DIR}"
else
    echo "  dino-tracker already cloned at ${DINO_DIR} — pulling latest ..."
    git -C "${DINO_DIR}" pull --ff-only || true
fi

# Install dino-tracker deps — skip strict torch/xformers version pins
pip install --quiet \
    antialiased_cnns \
    einops \
    mediapy

# Install dino-tracker itself (editable)
pip install --quiet -e "${DINO_DIR}"

# Copy our custom scripts into dino-tracker
echo "  Copying custom inference scripts into dino-tracker ..."
cp "${SCRIPT_DIR}/inference_flow_01.py"   "${DINO_DIR}/inference_flow_01.py"
cp "${SCRIPT_DIR}/model_inference_ext.py" "${DINO_DIR}/models/model_inference.py"

# Download dino-tracker pretrained features (DinoV2 loads automatically via torch.hub)
# No manual checkpoint needed.

# ---------------------------------------------------------------------------
# 6. Build RAFT CUDA extension (for standard optical flow in hdr-nsff)
# ---------------------------------------------------------------------------
echo "[6/7] Building RAFT CUDA extension ..."
RAFT_EXT_DIR="${SCRIPT_DIR}/alt_cuda_corr"
if [ -d "${RAFT_EXT_DIR}" ]; then
    pushd "${RAFT_EXT_DIR}" > /dev/null
    # CUDA_HOME: torch 2.5.1 built against cu121; use system cuda-12.4 headers
    # TORCH_CUDA_ARCH_LIST: RTX A6000 = sm_86
    CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST="8.6" \
        python setup.py build_ext --inplace
    popd > /dev/null
else
    echo "  WARNING: ${RAFT_EXT_DIR} not found — skipping RAFT CUDA build."
fi

# ---------------------------------------------------------------------------
# 7. Update notebook and script paths to new repo locations
# ---------------------------------------------------------------------------
echo "[7/7] Done."

echo ""
echo "================================================================"
echo " Setup complete!"
echo ""
echo " conda activate ${ENV_NAME}"
echo ""
echo " Repo paths:"
echo "   SAM2         : ${SAM2_DIR}"
echo "   dino-tracker : ${DINO_DIR}"
echo ""
echo " Update these variables in your scripts/notebooks:"
echo "   SAM2_REPO_PATH  = '${SAM2_DIR}'"
echo "   SAM2_CHECKPOINT = '${SAM2_CKPT}'"
echo ""
echo " Notebook: jupyter lab nsff_scripts/sam2_motion_mask.ipynb"
echo "           (select kernel: ${ENV_NAME})"
echo "================================================================"
