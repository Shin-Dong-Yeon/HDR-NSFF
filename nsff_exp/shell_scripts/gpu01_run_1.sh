#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# Set the data path environment variable
export CUDA_VISIBLE_DEVICES=1
# export final_height=400

# HDR-NSFF
echo "run HDR-NSFF!"
export config_file="/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/configs/gopro/251128/config_laptop_i8_of_hdr_rw.1_251201.txt"
# python run_nerf.py --config $config_file #--i_img 1000
# python evaluation_gopro.py --config $config_file
# python run_nerf.py --config $config_file --render_lockcam --render_tm wb --target_idx 2
# python run_nerf.py --config $config_file --render_bt --target_idx 20
python evaluation_gopro_time_36.py --config $config_file
