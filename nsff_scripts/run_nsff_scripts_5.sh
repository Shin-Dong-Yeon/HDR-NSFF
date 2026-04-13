#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=5

export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset/HDR-Hexplane-dataset/colmap_2/mutant/cam_3/full/dense"

# python save_poses_nerf.py --data_path $data_path
# python run_midas.py --data_path $data_path --resize_height 400
python run_flows_video.py --model models/raft-things.pth --data_path $data_path --step 2 --skip_moseg
# python mask_inverse.py --data_path $data_path

# python ../nsff_exp/images2video.py --src_dir $data_path --start_frame 7 --end_frame 31
# python ../nsff_exp/images2video.py --src_dir $data_path
# conda activate depthcrafter
# python ../../DepthCrafter/run.py --video-path $data_path/video/*.mp4 --save-folder $data_path/disp_depthcrafter