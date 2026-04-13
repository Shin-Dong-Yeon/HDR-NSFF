#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=1

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251128/open_laptop/i8/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy