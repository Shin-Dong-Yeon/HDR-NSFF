#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=1

export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251129/config_ball_touch_3_2_i4_tf_hdr_rw.1_difix_pw.001_251203.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 2 --render_tm wb 
# python evaluation_gopro_time_36.py --config $config_file




# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/tf_rev/config_jumping_jack_1_tf_rev_hdr_250428.txt"
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2



# # HDR-NSFF
# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/blender/standup/config_standup_i2_tf_hdr_250305.txt"
# python run_nerf.py --config $config_file
# python evaluation.py --config $config_file
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# # python run_nerf.py --config $config_file #--final_height $final_height
# echo "Start slowmo_bt rendering"
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 
# python evaluation_gopro_time.py --config $config_file #--final_height $final_height
# python evaluation_gopro.py --config $config_file
echo "All commands done"
# echo "All commands done"

