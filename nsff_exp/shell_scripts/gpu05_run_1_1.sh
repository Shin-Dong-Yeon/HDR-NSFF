#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
# export DATA_PATH="/node_data/shindy/projects/hdr-4d/NSFF/dataset/real/jun_running_multi_f"
export CUDA_VISIBLE_DEVICES=1
export final_height=480

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/config_wave_hand_1_start_1_250102.txt"
python run_nerf.py --config $config_file --final_height $final_height
echo "Start slowmo_bt full"
python run_nerf.py --config $config_file --final_height $final_height --render_slowmo_bt --render_mode cb
echo "All commands done"

# echo "All commands done"

