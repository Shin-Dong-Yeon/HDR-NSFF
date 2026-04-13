#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=1
# export final_height=400

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_tf_hdr_difix_251123.txt"
python run_nerf.py --config $config_file
python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30

# # HDR-NSFF
# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/difix3d/difix3d_i2/config_tube_toss_1_i2_l1_tf_hdr_full_difix_200k_bal0.9_mv_mask_percept_250908.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30







