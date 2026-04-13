#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=2

export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251129/config_bag_4_1_i4_tf_hdr_rw.1_difix_pw.001_251203.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
python run_nerf.py --config $config_file --render_lockcam --target_idx 2
python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
# python run_nerf.py --config $config_file --rrender_lockcam_slowmo --target_idx 2 --render_tm wb 


# HDR-NSFF
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_nsff_tf_250919.txt"
# python evaluation_gopro_time_36.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_of_hdr_250918.txt"
# python evaluation_gopro_time_36.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_tf_hdr_250919.txt"
# python evaluation_gopro_time_36.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_tf_hdr_difix_250921.txt"
# python evaluation_gopro_time_36.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_pointing_walk_1_i2_l1_nsff_tf_250908.txt"
# python evaluation_gopro_time_36.py --config $config_file



# # HDR-NSFF
# echo "run HDR-NSFF!"
# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_pointing_walk_1_i2_l1_of_hdr_250908.txt"
# python run_nerf.py --config $config_file
# python evaluation_gopro.py --config $config_file
# # python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30



# python run_nerf.py --config $config_file #--final_height $final_height
# python run_nerf.py --config $config_file --render_train
# python evaluation_gopro.py --config $config_file
# # echo "Start slowmo_bt rendering"
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 

# python evaluation_gopro.py --config $config_file #--final_height $final_height
# python evaluation_gopro_time.py --config $config_file
echo "All commands done"
# echo "All commands done"



# python run_nerf.py --config $config_file #--final_height $final_height

# # echo "Start slowmo_bt rendering"
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# # python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# # python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# # python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 
# # python evaluation_gopro_time.py --config $config_file
# echo "All commands done"
# # echo "All commands done"


# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/blender/lego/config_lego_step_10_tf_hdr_share_wb_250303.txt"
# python evaluation_gopro.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250213/config_jumping_jack_1_nsff_tf_250219.txt"
# python evaluation_gopro.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250213/config_one_arm_swing_shin_1_nsff_tf_250219.txt"
# python evaluation_gopro.py --config $config_file

# export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250213/config_pointing_walk_nsff_tf_250219.txt"
# python evaluation_gopro.py --config $config_file