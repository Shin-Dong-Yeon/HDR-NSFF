import os, sys
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
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

import torchvision.models as models


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
    parser.add_argument("--disp_model", type=str, default='midas', help='options: midas / aligned / depthcrafter')
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
    parser.add_argument("--share_wb", action='store_true', default=False, help='share wb')
    parser.add_argument("--log_scale", action='store_true', default=False, help='use log scale crf')
    parser.add_argument("--unit_loss", action='store_true', default=False, help='use unit loss')
    parser.add_argument("--w_crf", type=float, default=0.001, help='weights of crf loss')
    parser.add_argument("--ref_idx", type=int, default=0, help='reference index for crf')
    
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
    parser.add_argument("--lrate_decay", type=int, default=250, 
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
                        help='do not use normalized device coordinates (set for non-forward facing scenes), \
                              Currently only support NDC reconstruction for forward facing scenes')
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
    parser.add_argument("--w_sf_reg", type=float, default=0.01, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')
    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 10000')
    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=24)
    parser.add_argument("--disp_loss", type=str, default=None, help='options: depth-anything-o1 / depth-anything-o2')

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
    parser.add_argument('--tone_mapping', type=str, choices=["piece_wise", "nn"], default = "piece_wise")
    
    parser.add_argument('--render_mode', type=str, default='full',
                        help='full / cb / dynamic /rigid')

    return parser

class VGGFeatures(nn.Module):
    def __init__(self, selected_layers):
        super(VGGFeatures, self).__init__()
        self.selected_layers = selected_layers
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        for param in self.vgg.parameters():
            param.rqeuires_grad = False
        
    def forward(self, x):
        features = {}
        
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[name] = x
        return features

def compute_content_loss(features_generated, features_target):
    loss = 0.0
    for key in features_target.keys():
        loss += torch.nn.functional.mse_loss(features_generated[key], features_target[key])
    return loss

def finetune_hdr_nsff(render_kwargs_train, Cam_param, pose, img_idx_embed, ldr_target_img, device, num_steps=300, lr=5e-5):
    selected_layers = ['1', '3', '6', '8']
    feature_extractor = VGGFeatures(selected_layers).to(device)
    
    optimizer = render_kwargs_train['optimizer']
    grad_vars = render_kwargs_train['grad_vars']
    
    render_kwargs_train['network_fn'].train()
    if render_kwargs_train['network_fine'] is not None:
        render_kwargs_train['network_fine'].train()
        
    for step in range(num_steps):
        optimizer.zero_grad()
        
        ret = render(img_idx_embed, 0, False, )
    

def main():
    
    import pdb; pdb.set_trace()
    
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

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, \
        start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    if args.use_tone_mapping == True:
        # #FIXME: 이 부분 고쳐야함!
        N, H, W, C = images.shape
        Cam_param = cam_param.CamParam(N, H, W, device=device, gts=images, initialize=args.use_initialize, tone_mapping=args.tone_mapping, share_crf=args.share_crf, share_wb=args.share_wb, ref_idx=args.ref_idx, log_scale=args.log_scale, unit_loss=args.unit_loss)
        # optim_cam = Cam_param.optimizer(l_rate = args.lrate)
        optim_cam_wb, optim_cam_crf = Cam_param.optimizer(l_rate = args.lrate)

        basedir = args.basedir
        expname = args.expname
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            
            print('Reloading CamParam from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            Cam_param.load_ckpt(ckpt)
    else:
        Cam_param = None
    
    # if args.use_tone_mapping_camparam == True:
    #     Cam_param, optim_cam = create_CamParam(args, device, images, test=True)
    # else:
    #     Cam_param = None
    #     optim_cam = None

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    num_img = float(images.shape[0])
    poses = torch.Tensor(poses).to(device)

    evaluation_render_dir = os.path.join(basedir, expname, 'evaluation')
    # evaluation_gt_dir = os.path.join(evaluation_render_dir, 'gt')
    os.makedirs(evaluation_render_dir, exist_ok=True)
    # os.makedirs(evaluation_gt_dir, exist_ok=True)
    os.makedirs(os.path.join(evaluation_render_dir, 'hdr'), exist_ok=True)
    os.makedirs(os.path.join(evaluation_render_dir, 'ldr'), exist_ok=True)
    
    # selected_layers = ['1','3','6','8']
    selected_layers = ['1','3']
    feature_extractor = VGGFeatures(selected_layers).to(device)
    
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # optimizer = render_kwargs_train['optimizer']
    # grad_vars = render_kwargs_train['grad_vars']
    
    import pdb; pdb.set_trace()
    
    render_kwargs_train['network_fn'].train()
    if render_kwargs_train.get('network_fine', None):
        render_kwargs_train['network_fine'].train()
        
    # render_kwargs_train['network_fn'] = nn.DataParallel(render_kwargs_train['network_fn'])
    # if render_kwargs_train['network_fine'] is not None:
    #     render_kwargs_train['network_fine'] = nn.DataParallel(render_kwargs_train['network_fine'])  
        
    num_steps = 300
    for step in range(num_steps):
        optimizer.zero_grad()
        
        img_i = 8

        img_idx_embed = img_i/num_img * 2. -1.0
        # pred_ldr_dir = os.path.join(evaluation_render_dir, 'ldr', f'{img_i:05d}')
        # pred_hdr_exr_dir = os.path.join(evaluation_render_dir, 'hdr_exr', f'{img_i:05d}')
        # pred_hdr_tm_dir = os.path.join(evaluation_render_dir, 'hdr_tm', f'{img_i:05d}')
        # os.makedirs(pred_hdr_exr_dir, exist_ok=True)
        # os.makedirs(pred_hdr_tm_dir, exist_ok=True)
        # os.makedirs(pred_ldr_dir, exist_ok=True)
        
        gt_ldr = images[img_i]
        
        c2w = poses[img_i]
        ret = render(img_idx_embed, 
                     chain_bwd, 
                     chain_5frames,
                     num_img, H, W, focal,
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)
        # ret = render(img_idx_embed, 0, False,
        #             num_img,
        #             H, W, focal,
        #             chunk=1024, c2w=c2w[:3,:4],
        #             **render_kwargs_train)
        
        if Cam_param is not None:
            rgb_hdr_ = ret['rgb_map_ref']
            rgb_ldr = Cam_param.RAD2LDR_img(rgb_hdr_, img_i)
            
        features_generated = feature_extractor(rgb_ldr)
        features_target = feature_extractor(gt_ldr)
        
        loss = compute_content_loss(features_generated, features_target)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"[step {step}] Loss: {loss.item():.4f}")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
