import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import cv2

import math
from render_utils import *
from run_nerf_helpers import *

from load_llff import load_nvidia_data
from load_llff import load_hdr_blender_data
import cam_param
import skimage.measure
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')

    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    
    # HDR-NSFF add-ons ##
    parser.add_argument("--use_posenc_mask", action='store_true',
                        help='use positional encoding mask')
    parser.add_argument("--reg_end_ratio", type=float, default=0.9,
                        help='warmup end iteration')
    parser.add_argument("--sf_multires", type=int, default=0,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--debug", action='store_true', help='debugging mode')
    parser.add_argument("--aligned_depth", action='store_true', help='use aligned_depth instead midas')
    parser.add_argument("--disp_model", type=str, default='midas', help='options: midas / aligned / depthcrafter / depth-anything')
    parser.add_argument("--sparse", action='store_true', help='sparse input or else')
    parser.add_argument("--unit_sf_bidirection", action='store_true', help='change bidirection loss as unit bidirection loss')
    
    parser.add_argument("--half_res", action='store_true', default=True, help='option for blender dataset')
    parser.add_argument("--testkip", type=int, default=1, help='option for blender dataset')
    
    parser.add_argument("--render_tm", type=str, default='tm', help='options: tm, wb, no')
    parser.add_argument("--skip_sf_reg_loss", action='store_true', default=False, help='option for not using sf_reg_loss')
    parser.add_argument("--skip_sf_sm_loss", action='store_true', default=False, help='option for not using sf_sm_loss')
    parser.add_argument("--skip_prob_reg_loss", action='store_true', default=False, help='option for not using prob_reg_loss')
    parser.add_argument("--traj_flow", action='store_true', default=False, help='option for using traj_flow')
    parser.add_argument("--w_traj_flow", type=float, default=0.02, help='weights of traj_flow loss')
    parser.add_argument("--decay_traj_flow_w", action='store_true', default=False, help='decay traj_flow weights')
    parser.add_argument("--step_frame", type=int, default=1, help='step frame for loading data')
    parser.add_argument("--render_train", action='store_true', default=False, help='render train data')
    parser.add_argument("--share_crf", action='store_true', default=False, help='share crf')
    parser.add_argument("--share_wb", action='store_true', default=False, help='share just 3 wb')
    parser.add_argument("--log_scale", action='store_true', default=False, help='use log scale crf')
    parser.add_argument("--unit_loss", action='store_true', default=False, help='use unit loss')
    parser.add_argument("--w_crf", type=float, default=0.001, help='weights of crf loss')
    parser.add_argument("--w_unit", type=float, default=0.001, help='weights of unit loss')
    parser.add_argument("--ref_idx", type=int, default=0, help='reference index for crf')
    parser.add_argument("--disp_loss", type=str, default=None, help='options: depth-anything-o1 / depth-anything-o2')
    parser.add_argument("--no_chain_5frames", action='store_true', default=False, help='options: True / False')
    parser.add_argument("--chain_5frames_flow", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--custom_times", type=int, default=None, help='render at just custom times')
    parser.add_argument("--leaky_th_h", type=float, default=0.9, help='leaky threshold high')
    parser.add_argument("--leaky_th_l", type=float, default=0.15, help='leaky threshold high')
    parser.add_argument("--sat_mask_0", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--sat_mask_3", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--viz_crf", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--fine_tune", action='store_true', default=False)
    parser.add_argument("--render_dy", action='store_true', default=False)

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='how many iteration do you want to run')
    parser.add_argument("--multi_camera", action='store_true', default=False, help='when use multi-camera setting')


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--use_tone_mapping", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument("--use_tone_mapping_camparam", action='store_true', default=False,
                        help='for test in rendering, should be merge use_tone_mapping')
    parser.add_argument("--use_initialize", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument('--use_sat_mask', action='store_true', default=False)
    parser.add_argument('--tone_mapping', type=str, choices=["piece_wise", "nn", "hdr_hexplane"], default = "piece_wise")
    # parser.add_argument('--render_mode', type=int, default=0,
    #                     help='0:combined, 1:rigit, 2: dynamic')
    parser.add_argument('--render_mode', type=str, default='full',
                        help='full / cb / dynamic /rigid')
    return parser


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)
    mask = mask[..., np.newaxis] 

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, data_range=1.0, channel_axis=2, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid


def compute_mse_np(pred: np.ndarray,
                   gt:   np.ndarray,
                   mask: np.ndarray) -> float:
    """
    pred, gt: np.ndarray of shape (..., C)
    mask:     np.ndarray of shape same as pred without the last axis
    """
    # 1) mask 에 채널 축 추가
    mask_rep = mask[..., np.newaxis]    # (H, W) → (H, W, 1); 
                                        # 일반화하려면 pred.ndim-mask.ndim 만큼 반복해도 됩니다.

    # 2) 제곱 오차에 마스크 적용
    diff2 = (pred - gt)**2 * mask_rep

    # 3) 유효 픽셀 개수
    num = mask_rep.sum() + 1e-8

    # 4) MSE 리턴
    return float(diff2.sum() / num)


def evaluation():
    
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_nvidia_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)


        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # if not isinstance(i_test, list):
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'hdr-blender':
        
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_hdr_blender_data(args.datadir,
                                                                 args.start_frame, args.end_frame, args.step_frame,
                                                                 args.factor,
                                                                 target_idx=target_idx,
                                                                 recenter=True, bd_factor=.9,
                                                                 spherify=args.spherify,
                                                                 final_height=args.final_height)
          
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded hdr-blender', images.shape, render_poses.shape, hwf, args.datadir)
        
        i_test = []
        i_val = []
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()

    
    
        

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, 
                                                 args.end_frame)
    expname = args.expname
    
    def imread(f):
        # if f.endswith('png'):
        #     return imageio.imread(f, apply_gamma=False)
        # else:
        return imageio.imread(f)
    
    eval_files = sorted(glob(os.path.join(basedir, expname, 'render-train_dy_*', 'images', '*.png')))
    eval_imgs = [imread(f)[...,:3]/255. for f in eval_files]
    eval_imgs = np.stack(eval_imgs, -1).astype(np.float32)
    
    mask_dir = os.path.join(args.datadir, 'motion_masks')
    mask_files = [os.path.join(mask_dir, f) 
                for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]
    masks = [cv2.resize(imread(f)/255., (images.shape[2], images.shape[1]),
                        interpolation=cv2.INTER_NEAREST) for f in mask_files]
    masks = np.stack(masks, -1)
    masks = np.float32(masks > 1e-3)
    
    mse = 0.
    psnr = 0.
    
    save_error_dir = os.path.join(glob(os.path.join(basedir, expname, 'render-train_dy_*'))[0], 'error_map')
    os.makedirs(save_error_dir, exist_ok=True)
    
    for img_i in i_train:
        
        post_i = img_i * 3
        prev_i = post_i + 1
        
        post_dy = eval_imgs[..., post_i]
        prev_dy = eval_imgs[..., prev_i]
        
        gt_img = images[img_i]
        mask = masks[..., img_i]
        mse += compute_mse_np(post_dy, gt_img, mask) if img_i != (len(i_train) - 1) else 0.
        mse += compute_mse_np(prev_dy, gt_img, mask) if img_i != 0 else 0.
        
        psnr += calculate_psnr(post_dy, gt_img, mask) if img_i != (len(i_train) - 1) else 0.
        psnr += calculate_psnr(prev_dy, gt_img, mask) if img_i != 0 else 0.
        
        error_post = np.abs(post_dy - gt_img) * mask[..., None]
        error_prev = np.abs(prev_dy - gt_img) * mask[..., None]

        # 평균 오차 맵 (RGB -> grayscale)
        error_post_map = np.mean(error_post, axis=-1)
        error_prev_map = np.mean(error_prev, axis=-1)

        # 시각화 및 저장
        plt.imsave(os.path.join(save_error_dir, f"{img_i:03d}_post_error.png"), error_post_map, cmap='hot', vmin=0, vmax=0.2)
        plt.imsave(os.path.join(save_error_dir, f"{img_i:03d}_prev_error.png"), error_prev_map, cmap='hot', vmin=0, vmax=0.2)
        
        
    print(f'MSE: {(mse / ((len(i_train) - 1)*2)):.4f}')
    print(f'PSNR: {(psnr / ((len(i_train) - 1)*2)):.4f}')
    
    return
   
if __name__=='__main__':
    evaluation()
