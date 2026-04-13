#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
# export DATA_PATH="/node_data/shindy/projects/hdr-4d/NSFF/dataset/real/jun_running_multi_f"
export CUDA_VISIBLE_DEVICES=5
export final_height=480

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/real/dslr/config_rist_2_hdr_skip_loss_1_241022.txt"
python run_nerf.py --config $config_file --final_height $final_height
echo "Start slowmo_bt full"
python run_nerf.py --config $config_file --final_height $final_height --render_slowmo_bt --render_mode full

echo "All commands done"

# echo "All commands done"

