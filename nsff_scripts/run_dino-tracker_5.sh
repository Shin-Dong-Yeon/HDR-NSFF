#!/usr/bin/zsh -i
source ~/anaconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=5

# Set 1 ------------------------------------------------------------------------------------------------
export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/ball_touch_3_2/i4/dense"
#-------------------------------------------------------------------------------------------------------
mkdir -p "$data_path/dino-tracker/video"
mkdir -p "$data_path/dino-tracker/masks"
cp -r "$data_path/images/"* "$data_path/dino-tracker/video/"
cp -r "$data_path/motion_masks/"* "$data_path/dino-tracker/masks/"

cd ../../dino-tracker-temp/dino-tracker
conda activate dino-tracker

OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH=`pwd`:$PYTHONPATH
export dino_path="$data_path/dino-tracker"

python ./preprocessing/main_preprocessing.py --config ./config/preprocessing.yaml --data-path $dino_path
python ./train.py --config ./config/train.yaml --data-path $dino_path
python ./inference_flow_01.py --config ./config/train.yaml --data-path $dino_path

cd ../../hdr-nsff/nsff_scripts/
export PYTHONPATH=$OLD_PYTHONPATH
conda activate nsff
python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg
#-------------------------------------------------------------------------------------------------------

# Set 1 ------------------------------------------------------------------------------------------------
export data_path="/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/251129/ball_touch_5_1/i3/dense"
#-------------------------------------------------------------------------------------------------------
mkdir -p "$data_path/dino-tracker/video"
mkdir -p "$data_path/dino-tracker/masks"
cp -r "$data_path/images/"* "$data_path/dino-tracker/video/"
cp -r "$data_path/motion_masks/"* "$data_path/dino-tracker/masks/"

cd ../../dino-tracker-temp/dino-tracker
conda activate dino-tracker

OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH=`pwd`:$PYTHONPATH
export dino_path="$data_path/dino-tracker"

python ./preprocessing/main_preprocessing.py --config ./config/preprocessing.yaml --data-path $dino_path
python ./train.py --config ./config/train.yaml --data-path $dino_path
python ./inference_flow_01.py --config ./config/train.yaml --data-path $dino_path

cd ../../hdr-nsff/nsff_scripts/
export PYTHONPATH=$OLD_PYTHONPATH
conda activate nsff
python run_flows_video.py --traj_flow --data_path $data_path --skip_of --skip_moseg
#-------------------------------------------------------------------------------------------------------