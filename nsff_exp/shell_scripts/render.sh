#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

export config_file="/home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_car_black_v1_hdr_exp.txt"

python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 10 --final_height 360
python run_nerf.py --config $config_file --render_bt --target_idx 10 --final_height 360

export config_file = "/home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_car_black_v1_hdr.txt"
python run_nerf.py --config $config_file --render_lockcam_slowmo --target_idx 10 --final_height 360
python run_nerf.py --config $config_file --render_bt --target_idx 10 --final_height 360


# python run_nerf.py --config /home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_street_v1_hdr.txt --render_lockcam_slowmo  --target_idx 10 --final_height 360
# python run_nerf.py --config /home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_terminal_v1_hdr.txt --render_lockcam_slowmo  --target_idx 10 --final_height 360
# python run_nerf.py --config /home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_trees_v1_hdr.txt --render_lockcam_slowmo  --target_idx 10 --final_height 360
# python run_nerf.py --config /home/shindy/projects/hdr-4d/hdr_NSFF-master/nsff_exp/configs/DeepHDRVideo/config_walking_v1_hdr.txt --render_lockcam_slowmo  --target_idx 10 --final_height 360

# CUDA_VISIBLE_DEVICES=2,3 python run_nerf.py --config configs/config_kid-running_vanilla_mix.txt --render_slowmo_bt  --target_idx 10 --render_mode 0
# CUDA_VISIBLE_DEVICES=2,3 python run_nerf.py --config configs/config_kid-running_vanilla_mix.txt --render_slowmo_bt  --target_idx 10 --render_mode 1
# CUDA_VISIBLE_DEVICES=2,3 python run_nerf.py --config configs/config_kid-running_vanilla_mix.txt --render_slowmo_bt  --target_idx 10 --render_mode 2
