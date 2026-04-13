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

from config import config_parser

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

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
        Cam_param = cam_param.CamParam(N, H, W, device=device, gts=images, initialize=args.use_initialize, \
                                        tone_mapping=args.tone_mapping, share_crf=args.share_crf, \
                                        share_wb=args.share_wb, ref_idx=args.ref_idx, log_scale=args.log_scale, unit_loss=args.unit_loss)
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

    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        count = 0.
        total_psnr_dy = 0.
        total_ssim_dy = 0.
        total_lpips_dy = 0.
        t = time.time()
        
        mid = 0
        high = 2
        low = 1

        if args.ft_path is not None and args.ft_path!='None':
            eval_dir = f'evaluation_' + args.ft_path.split('/')[-2] + '_' + os.path.basename(args.ft_path)
        else:
            eval_dir = 'evaluation'
        evaluation_render_dir = os.path.join(basedir, expname, eval_dir)
            
        # evaluation_gt_dir = os.path.join(evaluation_render_dir, 'gt')
        os.makedirs(evaluation_render_dir, exist_ok=True)
        # os.makedirs(evaluation_gt_dir, exist_ok=True)
        os.makedirs(os.path.join(evaluation_render_dir, 'hdr'), exist_ok=True)
        os.makedirs(os.path.join(evaluation_render_dir, 'ldr'), exist_ok=True)


        # for img_i in i_train[10:37]:
        if args.eval_start_time != 0:
            i_train = i_train[args.eval_start_time:]

        for img_i in i_train:

            img_idx_embed = img_i/num_img * 2. -1.0
            pred_ldr_dir = os.path.join(evaluation_render_dir, 'ldr', f'{img_i:05d}')
            pred_hdr_exr_dir = os.path.join(evaluation_render_dir, 'hdr_exr', f'{img_i:05d}')
            pred_hdr_tm_dir = os.path.join(evaluation_render_dir, 'hdr_tm', f'{img_i:05d}')
            os.makedirs(pred_hdr_exr_dir, exist_ok=True)
            os.makedirs(pred_hdr_tm_dir, exist_ok=True)
            os.makedirs(pred_ldr_dir, exist_ok=True)

            for camera_i in range(0,9):
                print(time.time() - t)
                t = time.time()

                print(img_i, camera_i)
                # if img_i % 9 == camera_i:
                #     continue

                c2w = poses[camera_i]
                ret = render(img_idx_embed, 0, False,
                             num_img, 
                             H, W, focal, 
                             chunk=1024*16, c2w=c2w[:3,:4], 
                             **render_kwargs_test)
                                
                if Cam_param is not None:
                    rgb_hdr_ = ret['rgb_map_ref']
                    # if args.log_scale:
                    #     print("exponential!!")
                    #     rgb_hdr_=torch.exp(rgb_hdr_)
                    # save_tonemap(rgb_hdr_, 'reinhard', os.path.join(pred_hdr_tm_dir, f'{camera_i+1:05d}.png'))
                    # imageio.imwrite(os.path.join(pred_hdr_exr_dir, f'{camera_i+1:05d}.exr'), rgb_hdr_.cpu().numpy(), format='EXR')
                    cv2.imwrite(os.path.join(pred_hdr_exr_dir, f'{camera_i+1:05d}.exr'), rgb_hdr_.cpu().numpy().astype(np.float32)[...,::-1])
                    mu = torch.tensor(50, dtype=rgb_hdr_.dtype, device=rgb_hdr_.device)
                    rgb_hdr = torch.log(1 + mu * rgb_hdr_) / torch.log(1 + mu)
                    rgb8_hdr = to8b(rgb_hdr.cpu().numpy())
                    imageio.imwrite(os.path.join(pred_hdr_tm_dir, f'{camera_i+1:05d}.png'), rgb8_hdr)   
                    rgb_ldr = Cam_param.RAD2LDR_img(ret['rgb_map_ref'], camera_i).cpu().numpy()
                else:
                    rgb_ldr = ret['rgb_map_ref'].cpu().numpy()

                rgb8_ldr = to8b(rgb_ldr)
                imageio.imwrite(os.path.join(pred_ldr_dir, f'{camera_i+1:05d}.png'), rgb8_ldr)

        #         gt_img_path = os.path.join(args.datadir, 
        #                                 'mv_images', 
        #                                 f'{img_i:05d}', 
        #                                 f'{camera_i+1:05d}.png')
        #         gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
        #         gt_img = cv2.resize(gt_img, 
        #                             dsize=(rgb_ldr.shape[1], rgb_ldr.shape[0]), 
        #                             interpolation=cv2.INTER_AREA)
        #         gt_img = np.float32(gt_img) / 255
        #         rgb_ldr = np.float32(rgb_ldr)# / 255


        #         psnr = peak_signal_noise_ratio(gt_img, rgb_ldr)
        #         ssim = structural_similarity(gt_img, rgb_ldr, data_range=1.0, channel_axis=2)

        #         gt_img_0 = im2tensor(gt_img).cuda()
        #         rgb_0 = im2tensor(rgb_ldr).cuda()
                
        #         lpips = model.forward(gt_img_0, rgb_0)
        #         lpips = lpips.item()

        #         print('psnr:', psnr, 'ssim:', ssim, 'lpips:',lpips)

        #         total_psnr += psnr
        #         total_ssim += ssim
        #         total_lpips += lpips
        #         count += 1

        #         dynamic_mask_path = os.path.join(args.datadir, 
        #                                 'mv_masks', 
        #                                 f'{img_i:05d}', 
        #                                 f'{camera_i+1:05d}.png')
        #         print(dynamic_mask_path)
        #         dynamic_mask = np.float32(cv2.imread(dynamic_mask_path) > 1e-3)#/255.
        #         dynamic_mask = cv2.resize(dynamic_mask, 
        #                                 dsize=(rgb_ldr.shape[1], rgb_ldr.shape[0]), 
        #                                 interpolation=cv2.INTER_NEAREST)
                
        #         dynamic_mask_0 = torch.Tensor(dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

        #         dynamic_ssim = calculate_ssim(gt_img, 
        #                                       rgb_ldr, 
        #                                       dynamic_mask)
        #         dynamic_psnr = calculate_psnr(gt_img, 
        #                                       rgb_ldr, 
        #                                       dynamic_mask)

        #         dynamic_lpips = model.forward(gt_img_0, 
        #                                       rgb_0, 
        #                                       dynamic_mask_0).item()
        #         print('psnr_dy:', dynamic_psnr, 'ssim_dy:', dynamic_ssim, 'lpips_dy:',dynamic_lpips)
                
        #         total_psnr_dy += dynamic_psnr
        #         total_ssim_dy += dynamic_ssim
        #         total_lpips_dy += dynamic_lpips

        # mean_psnr = total_psnr / count
        # mean_ssim = total_ssim / count
        # mean_lpips = total_lpips / count 
        
        # print('mean_psnr', mean_psnr)
        # print('mean_ssim', mean_ssim)
        # print('mean_lpips', mean_lpips)

        # mean_psnr_dy = total_psnr_dy / count
        # mean_ssim_dy = total_ssim_dy / count
        # mean_lpips_dy= total_lpips_dy / count

        # print('mean_psnr dy', mean_psnr_dy)
        # print('mean_ssim dy', mean_ssim_dy)
        # print('mean_lpips dy', mean_lpips_dy)

   
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()
