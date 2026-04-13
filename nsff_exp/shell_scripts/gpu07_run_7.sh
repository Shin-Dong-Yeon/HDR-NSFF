#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=7

echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251129/config_bag_4_1_i4_tf_hdr_rw.1_difix_pw.001_251203.txt"
python run_nerf.py --config $config_file #--i_img 1000
python evaluation_gopro.py --config $config_file
python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
python run_nerf.py --config $config_file --render_bt --target_idx 20

# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251129/config_ball_touch_3_2_i4_tf_hdr_rw.1_difix_251202.txt"
# python run_nerf.py --config $config_file #--i_img 1000
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 20















# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250914/config_bear_thread_i2_tf_hdr_difix_wopercept_250915.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# # # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 11 --render_tm wb

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250914/config_bear_thread_i2_tf_hdr_difix_250915.txt"

# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# python run_nerf.py --config $config_file --render_bt --target_idx 20
# python run_nerf.py --config $config_file --render_bt --target_idx 30
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30
# python run_nerf.py --config $config_file --render_interpolate

