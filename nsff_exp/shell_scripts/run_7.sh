#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=7

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250914/config_bear_thread_i4_tf_hdr_250916.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_dog_shake_2_i4_tf_hdr_250919.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/250915/config_tumbler_shake_i4_tf_hdr_250916.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_big_jump_1_i2_l1_tf_hdr_250907.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_jumping_jack_1_i2_l1_tf_hdr_250910.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_pointing_walk_1_i2_l1_tf_hdr_250907.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_side_walk_1_i2_l1_tf_hdr_250907.txt"
python run_nerf.py --config $config_file --render_train
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/i2/l1/config_tube_toss_1_i2_l1_tf_hdr_250907.txt"
python run_nerf.py --config $config_file --render_train





# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 20 --render_tm wb 
# python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 2 --render_tm wb 


# python evaluation_gopro_time_36.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 10
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# python run_nerf.py --config $config_file --render_interpolate




# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30


# python run_nerf.py --config $config_file #--final_height $final_height
# python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 10
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 30
# # python run_nerf.py --config $config_file --render_slowmo_bt --target_idx 50
# python run_nerf.py --config $config_file --render_train
# python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 10

# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 10 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 30 #--final_height $final_height 
# python run_nerf.py --config $config_file --render_slowmo_bt --render_mode cb --target_idx 50 #--final_height $final_height 
# python evaluation_gopro_time.py --config $config_file
# python evaluation_gopro.py --config $config_file
echo "All commands done"
# echo "All commands done"