#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0

export data_path="../dataset/real/dslr/umbrella_raise/dense"

conda activate nsff
# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 480
# python run_flows_video.py --model models/raft-exposure.pth --data_path $data_path --epi_threshold 2.0

python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 99 --end_frame 123
conda activate depthcrafter
python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="../dataset/real/dslr/umbrella_raise_exp/dense"
# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 99 --end_frame 123
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

export data_path="../dataset/real/dslr/changeup_arm_swing_3/dense"
python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 0 --end_frame 24
conda activate depthcrafter
python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="../dataset/real/dslr/changeup_arm_swing_3_exp/dense"
# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 0 --end_frame 24
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

export data_path="../dataset/real/dslr/changeup_walk_1/dense"
python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 7 --end_frame 31
conda activate depthcrafter
python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

export data_path="../dataset/real/dslr/grass_walk_3/dense"
python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 3 --end_frame 27
conda activate depthcrafter
python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="../dataset/real/dslr/grass_walk_3_exp/dense"
# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 3 --end_frame 27
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

export data_path="../dataset/real/dslr/rist_2/dense"
python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 73 --end_frame 97
conda activate depthcrafter
python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

# export data_path="../dataset/real/dslr/rist_2_exp/dense"
# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 73 --end_frame 97
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter

