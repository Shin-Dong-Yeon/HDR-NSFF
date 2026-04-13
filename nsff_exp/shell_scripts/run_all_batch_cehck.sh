#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export DATA_PATH="/node_data/shindy/projects/hdr-4d/NSFF/dataset/blender-lego/v3/v3_mix3"
export CUDA_VISIBLE_DEVICES=1
export final_height=400

# Activate the 4dgs environment and process data with ns-process-data
conda activate 4dgs
ns-process-data images --data $DATA_PATH/images --output-dir $DATA_PATH/output --no-gpu

# Create necessary directories and copy files
mkdir -p $DATA_PATH/dense/sparse
cp $DATA_PATH/output/colmap/sparse/0/*.bin $DATA_PATH/dense/sparse
cp -r $DATA_PATH/images $DATA_PATH/dense

# Activate the nsff environment and run the necessary scripts
conda activate nsff
cd ../nsff_scripts
python save_poses_nerf.py --data_path $DATA_PATH/dense
python run_midas.py --data_path $DATA_PATH/dense --resize_height $final_height
python run_flows_video.py --model models/raft-things.pth --data_path $DATA_PATH/dense

# Navigate to nsff_exp directory and start training and rendering
cd ../nsff_exp

export config_file="/home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/lego_v2/config_lego_v3_mix3_hdr_batch_check.txt"

python run_nerf.py --config $config_file --final_height $final_height
# python run_nerf.py --config configs/own_config_file.txt --render_bt --target_idx 10 --final_height 400
# python run_nerf.py --config configs/own_config_file.txt --render_lockcam_slowmo --target_idx 8 --final_height 400
# python run_nerf.py --config configs/own_config_file.txt --render_slowmo_bt --target_idx 10 --final_height 400

echo "Start slowmo_bt full"
python run_nerf.py --config $config_file --final_height $final_height --render_slowmo_bt --target_idx 10 --render_mode full

echo "Start slowmo_bt rigid"
python run_nerf.py --config $config_file --final_height $final_height --render_slowmo_bt --target_idx 10 --render_mode rigid

echo "Start slowmo_bt dynamic"
python run_nerf.py --config $config_file --final_height $final_height --render_slowmo_bt --target_idx 10 --render_mode dynamic

echo "All commands done"

