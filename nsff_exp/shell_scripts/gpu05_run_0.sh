#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=0
# export final_height=400

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250914/config_bear_thread_i4_tf_hdr_difix_251121_0.5.txt"
python run_nerf.py --config $config_file
python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30

# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/difix3d/difix3d_inter_pose/config_jumping_jack_1_tf_hdr_w.1_difix_gt_full_200k_bal0.9_weights_map_th.9_3dil_percept_inter_250906.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30


echo "All commands done"
# echo "All commands done"
