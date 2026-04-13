#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=5

# HDR-NSFF
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251128/config_fire1_i8_tf_hdr_rw.1_difix_pw.001_251203.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 2 --render_tm wb 

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/l1_loss/config_pointing_walk_tf_hdr_base_l1_L4_250830.txt"
# python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 20 --render_tm wb
# python run_nerf.py --config $config_file --render_bt --target_idx 20
# python run_nerf.py --config $config_file --render_bt --target_idx 20
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file

# HDR-NSFF
# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/blender/2509/config_lego_step_10_of_hdr_250923.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python evaluation_gopro_time_36.py --config $config_file



# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/256/config_side_walk_1_tf_256_hdr_250526.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_bt --target_idx 30

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/256/config_sit_down_up_shin_2_tf_256_hdr_250526.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_bt --target_idx 30

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/256/config_tube_toss_1_tf_256_hdr_250526.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_bt --target_idx 30

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/256/config_two_arm_swing_shin_1_tf_256_hdr_250526.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_bt --target_idx 30



# python run_nerf.py --config $config_file #--final_height $final_height
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 50
# python run_nerf.py --config $config_file --render_train
# python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 10

# echo "Start slowmo_bt rendering"
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 

# python evaluation_gopro.py --config $config_file #--final_height $final_height
# python evaluation_gopro_time.py --config $config_file
# echo "All commands done"
# echo "All commands done"

