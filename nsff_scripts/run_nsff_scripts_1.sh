#!/usr/bin/env zsh
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=4

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250911/basket_ball_1/i2/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251128/fire_1/i8/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/sliding_4_1/i4/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/sliding_4_2/i4/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/ball_touch_5_1/i4/dense"
conda activate nsff
python save_poses_nerf.py --data_path $data_path
python run_midas.py --data_path $data_path --resize_height 360
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
conda activate dino-tracker
python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
                                                   --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
                                                   --max-depth 20 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251128/uslung/i8/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250915/dog_shake_2/i4/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/dslr/250909/thnx_bear_6/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/dslr/250909/thnx_bear_7/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy                                                   

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/480/jumping_jack_1/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl\
#                                                    --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/crop/jump_down_1/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/crop/tube_shake_1/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 400
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy                                                   

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/dslr/250512/robin_1/dense"
# conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/dslr/250512/robin_3/dense"
# conda activate nsff
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy                                                                                                      
                                                   
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/dslr/250512/robin_3/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/i8/jumping_jack_1/dense"
# # python save_poses_nerf.py --data_path $data_path
# # python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy     

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/i8/two_arm_swing_shin_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy                                                   
# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/crop/one_arm_swing_shin_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/crop/side_walk_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/crop/two_arm_swing_shin_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy



# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/HDR-Hexplane-dataset/colmap/tank/cam_1/ver2/dense"
# # python save_poses_nerf.py --data_path $data_path
# # python run_midas.py --data_path $data_path --resize_height 360
# # python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/time_eval/side_walk_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/time_eval/two_arm_swing_shin_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 40 --data-dir $data_path --save-numpy



# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/time_eval/pointing_walk/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/time_eval/sit_down_up_shin_2/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/time_eval/tube_toss_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\
#                                                    --max-depth 80 --data-dir $data_path --save-numpy                                                   
# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/side_walk_1/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy                                                                                                     
# # conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/one_arm_swing_shin/dense"
# # python save_poses_nerf.py --data_path $data_path
# # python run_midas.py --data_path $data_path --resize_height 360
# # python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/side_walk_mok_2/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy
# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/side_walk_mok_3/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# conda activate nsff
# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/two_arm_swing_shin/dense"
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# conda activate dino-tracker
# python ../../Depth-Anything-V2/metric_depth/run.py --encoder vitl --load-from ./depth-anything-v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
#                                                    --max-depth 20 --data-dir $data_path --save-numpy

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/test/two_arm_swing/i1"
# # python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 360
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 1 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 2 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 3 --skip_moseg
# python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 4 --skip_moseg
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg 
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg --step 2

# python mask_inverse.py --data_path $data_path

# python ../nsff_exp/images2video.py --src_dir $data_path
# python ../nsff_exp/images2video.py --src_dir $data_path
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/HDR-Hexplane-dataset/test/standup_128"
# python save_poses_nerf.py --data_path $data_path
# # python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/HDR-Hexplane-dataset/test/punch_128"
# python save_poses_nerf.py --data_path $data_path
# # python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/HDR-Hexplane-dataset/test/jump_jack_v5_128"
# python save_poses_nerf.py --data_path $data_path
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg --step 2
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg --step 3

# export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/HDR-Hexplane-dataset/test/punch_128"
# python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg