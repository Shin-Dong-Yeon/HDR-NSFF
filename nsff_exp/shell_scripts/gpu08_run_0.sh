#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=0
# export final_height=400

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/raft-exp/side_walk_1_exp_hdr_250306.txt"
python run_nerf.py --config $config_file #--final_height $final_height
python evaluation_gopro.py --config $config_file
# echo "Start slowmo_bt rendering"
python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 

echo "All commands done"
# echo "All commands done"
