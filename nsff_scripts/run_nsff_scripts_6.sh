#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=6

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250911/ball_leg_4_2/i2/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 480
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 40 --data-dir $data_path --save-numpy

                                                
# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 7 --end_frame 31
# python ../nsff_exp/images2video.py --src_dir $data_path
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter