#!/usr/bin/env bash
# =============================================================================
# setup_hdr_nsff_sam2.sh
#
# Creates the `hdr-nsff-sam2` conda environment:
#   - hdr-nsff  (NeRF training with HDR)
#   - SAM2      (interactive motion mask generation)
#
# Usage:
#   cd nsff_scripts/
#   bash setup_hdr_nsff_sam2.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="hdr-nsff-sam2"
PYTHON_VERSION="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "================================================================"
echo " hdr-nsff-sam2 environment setup"
echo " ENV : $ENV_NAME  |  DIR : $SCRIPT_DIR"
echo "================================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Env '${ENV_NAME}' already exists — skipping."
else
    echo "[1/6] Creating conda env '${ENV_NAME}' ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2. PyTorch (CUDA 12.1) + xformers
# ---------------------------------------------------------------------------
echo "[2/6] Installing PyTorch 2.5.1+cu121 ..."
pip install --quiet \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    xformers==0.0.29.post1 \
    --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# 3. hdr-nsff dependencies
#    Pin diffusers==0.25.1 + huggingface_hub==0.25.1 (Difix3D requires PositionNet
#    which was removed in diffusers>=0.26; huggingface_hub>=0.26 removes cached_download)
# ---------------------------------------------------------------------------
echo "[3/6] Installing hdr-nsff dependencies ..."
pip install --quiet \
    configargparse \
    kornia \
    lpips \
    einops \
    imageio \
    imageio-ffmpeg \
    "numpy==1.26.4" \
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
    ffmpeg-python \
    "diffusers==0.25.1" \
    "transformers==4.38.0" \
    "huggingface_hub==0.25.1" \
    "accelerate==1.12.0" \
    "peft==0.9.0" \
    "packaging<26"

# ---------------------------------------------------------------------------
# 4. SAM2
#    Cloned as sam2_repo/ (NOT sam2/) to avoid Python package name shadow.
# ---------------------------------------------------------------------------
echo "[4/6] Setting up SAM2 ..."
SAM2_DIR="${SCRIPT_DIR}/sam2_repo"

if [ ! -d "${SAM2_DIR}" ]; then
    echo "  Cloning SAM2 ..."
    git clone https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
else
    echo "  SAM2 already cloned — pulling ..."
    git -C "${SAM2_DIR}" pull --ff-only || true
fi

pip install --quiet \
    hydra-core \
    iopath

pip install --quiet -e "${SAM2_DIR}"

# Download SAM2 checkpoint (~900 MB)
SAM2_CKPT="${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "${SAM2_CKPT}" ]; then
    echo "  Downloading SAM2 checkpoint ..."
    mkdir -p "${SAM2_DIR}/checkpoints"
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "${SAM2_CKPT}"
else
    echo "  SAM2 checkpoint already present."
fi

# ---------------------------------------------------------------------------
# 5. Build RAFT CUDA extension (for optical flow in hdr-nsff)
#    RTX A6000 = sm_86 / torch built against CUDA 12.1 → use cuda-12.4 headers
# ---------------------------------------------------------------------------
echo "[5/6] Building RAFT CUDA extension ..."
RAFT_DIR="${SCRIPT_DIR}/alt_cuda_corr"
if [ -d "${RAFT_DIR}" ]; then
    pushd "${RAFT_DIR}" > /dev/null
    CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST="8.6" \
        python setup.py build_ext --inplace
    popd > /dev/null
else
    echo "  WARNING: ${RAFT_DIR} not found — skipping."
fi

# ---------------------------------------------------------------------------
# 6. Register Jupyter kernel
# ---------------------------------------------------------------------------
echo "[6/6] Registering Jupyter kernel ..."
pip install --quiet ipykernel
python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "${ENV_NAME}"

echo ""
echo "================================================================"
echo " Done!  conda activate ${ENV_NAME}"
echo " SAM2 repo       : ${SAM2_DIR}"
echo " SAM2 checkpoint : ${SAM2_CKPT}"
echo " Notebook kernel : ${ENV_NAME}"
echo "================================================================"
