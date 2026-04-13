#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=2


export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/ball_touch_3_2/i4/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/ball_touch_5_1/i3/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy                                                   

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/jumping_jack_1"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/pointing_walk"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/sit_down_shin_1"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/sit_down_up_shin_2"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/tube_toss_1"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 400
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 2 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 3 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 4 --skip_moseg
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg #--model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# python mask_inverse.py --data_path $data_path

# python ../nsff_exp/images2video.py --src_dir $data_path
# python ../nsff_exp/images2video.py --src_dir $data_path
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/two_arm_swing_shin/dense"
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg