#!/usr/bin/zsh -i
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=0

# ── Data path ──────────────────────────────────────────────────────────────
export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/bear_thread/dense"

# ── Set up dino-tracker input directories ──────────────────────────────────
mkdir -p "$data_path/dino-tracker"
# Remove stale symlinks if they exist
rm -f "$data_path/dino-tracker/video"
rm -f "$data_path/dino-tracker/masks"
ln -s "$data_path/images"        "$data_path/dino-tracker/video"
ln -s "$data_path/motion_masks"  "$data_path/dino-tracker/masks"

# ── Switch to dino-tracker repo ────────────────────────────────────────────
NSFF_SCRIPTS_DIR="$(pwd)"
cd ../../dino-tracker-temp/dino-tracker
conda activate dino-tracker

OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export dino_path="$data_path/dino-tracker"

# ── Copy our custom scripts into dino-tracker ─────────────────────────────
cp "$NSFF_SCRIPTS_DIR/inference_flow_01.py"  ./inference_flow_01.py
cp "$NSFF_SCRIPTS_DIR/model_inference_ext.py" ./models/model_inference.py

# ── Run DINO-tracker pipeline ──────────────────────────────────────────────
python ./preprocessing/main_preprocessing.py \
    --config ./config/preprocessing.yaml \
    --data-path "$dino_path"

python ./train.py \
    --config ./config/train.yaml \
    --data-path "$dino_path"

python ./inference_flow_01.py \
    --config ./config/train.yaml \
    --data-path "$dino_path" \
    --interval 1

# ── Convert trajectory flow → semantic_flow_i{step} npz ───────────────────
cd "$NSFF_SCRIPTS_DIR"
export PYTHONPATH="$OLD_PYTHONPATH"
conda activate nsff

python run_flows_video.py \
    --semantic_flow \
    --data_path "$data_path" \
    --skip_of \
    --skip_moseg
