import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid
from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from load_blender import *
import cam_param
import wandb
import torchvision.utils as vutils
import torchvision
from torchvision import models, transforms
from tqdm import tqdm
import imageio.v3 as iio
from percept import *
from difix_updater import * 

from config import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mae = lambda x, y : torch.mean(torch.abs(x - y))

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords = load_llff_data(args.datadir, 
                                                            args.start_frame, args.end_frame, args.step_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height,
                                                            disp_model=args.disp_model)
        
        print('loaded disp model: ', args.disp_model)

        HWF = poses[:,:3,-1]
        mean_focal = np.mean(poses[:,2,-1])
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'blender':
        target_idx = args.target_idx
        images, poses, render_poses, hwf, i_split, \
        depths, masks, ref_c2w, motion_coords, near, far = load_blender_data(args.datadir,
                                                                  args.start_frame, args.end_frame,
                                                                  args.half_res, args.testskip,
                                                                  factor=args.factor,
                                                                  target_idx=target_idx,
                                                                  final_height=args.final_height,
                                                                  disp_model=args.disp_model
                                                                  )
        
        i_val = []
        i_test = []
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
               
        images = images[..., :3]
            
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()

    if args.multi_camera == True:
        # Instrinsics for render
        H, W, _ = hwf
        H, W = int(H), int(W)
        hwf = [H, W, mean_focal]
        
    else:
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    # args.expname = args.expname + '_sigma_rgb-%.2f'%(args.sigma_rgb) \
                # + '_use-rgb-w_' + str(args.use_rgb_w) + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict) # add bds_dict to kwargs dictionaries
    render_kwargs_test.update(bds_dict)
    
    if args.use_tone_mapping == True:
        N, H, W, C = images.shape
        Cam_param = cam_param.CamParam(N, H, W, device=device, gts=images, initialize=args.use_initialize, 
                                       tone_mapping=args.tone_mapping, share_crf=args.share_crf, share_wb=args.share_wb, 
                                       ref_idx=args.ref_idx, log_scale=args.log_scale, unit_loss=args.unit_loss)
        optim_cam_wb, optim_cam_crf = Cam_param.optimizer(l_rate = args.lrate)
        camparam_frozen = False
        crf_stop_step = args.crf_stop_step
        
        basedir = args.basedir
        expname = args.expname
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
            FT = True
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            
            print('Reloading CamParam from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            Cam_param.load_ckpt(ckpt)
            optim_cam_wb.load_state_dict(ckpt['optim_cam_wb_state_dict'])
            if ckpt['optim_cam_crf_state_dict'] is not None:
                if optim_cam_crf is None:
                    optim_cam_crf = torch.optim.Adam(Cam_param.crf_params, lr=args.lrate, betas=(0.9, 0.999))
                optim_cam_crf.load_state_dict(ckpt['optim_cam_crf_state_dict'])
        
        if args.viz_crf:
            viz_crf_dir =os.path.join(basedir, expname, 'crf_viz')
            os.makedirs(viz_crf_dir, exist_ok=True)

            for i in range(int(images.shape[0])):
                path_i = os.path.join(viz_crf_dir, f'crf_{i:05d}.png')
                Cam_param.visualize_crf(i, path_i)
                print(path_i)
            wb_path = os.path.join(viz_crf_dir, 'wb.png')
            Cam_param.save_wb_plot(save_path=wb_path)
                           
            sys.exit(0)

    else:
        Cam_param = None
        optim_cam_crf = None
        optim_cam_wb = None
        
    if args.fine_tune:
        N_iters = global_step + 100001 # Continue
        #-----------------------------------------------
        # global_step = 0
        # start = 0
        # N_iters=100001  # Start with zero
        #-----------------------------------------------
        print('Fine-tuning!!!')
        expname_FT= '/fine-tuned_350k'
        expname = expname + expname_FT
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        
        if Cam_param is not None:
            for param in Cam_param.wb.parameters():
                param.requires_grad = False
            for param in Cam_param.CRF.parameters():
                param.requires_grad = False
        FT = True
        
        hdr_enh_paths = sorted(glob(os.path.join(basedir, args.expname, f'render-train_{args.N_iters}', 'enhanced', 'hdr_exr', '*_6000.exr')))
        hdr_list = []
        for path in hdr_enh_paths:
            hdr_np = iio.imread(path)
            if hdr_np is None:
                raise FileNotFoundError(f"Failed to laod {path}")
            hdr_tensor = torch.from_numpy(hdr_np)
            hdr_list.append(hdr_tensor)
            
        hdr_enh = torch.stack(hdr_list, dim=0) # [36, 360, 641, 3]
    else:
        N_iters = args.N_iters
        hdr_enh=None
        FT = False
    
    if args.render_bt:
        print('RENDER VIEW INTERPOLATION')
        
        render_poses = torch.Tensor(render_poses).to(device)
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        # os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, args.render_mode, args.render_tm,
                               gt_imgs=images, savedir=testsavedir, camparam=Cam_param,
                               render_factor=args.render_factor)

        return

    if args.render_lockcam_slowmo:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        # os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo(ref_c2w, num_img, hwf, 
                            args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor,
                            target_idx=target_idx,
                            camparam=Cam_param, render_tm=args.render_tm)

            return 
        
    if args.render_lockcam_slowmo_full:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, f'render-lockcam-slowmo-full-{target_idx:02d}')
        # os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo_full(ref_c2w, num_img, hwf, 
                            args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor,
                            target_idx=target_idx,
                            camparam=Cam_param, render_tm=args.render_tm)

            return

    if args.render_slowmo_bt:
        print('RENDER SLOW MOTION')

        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        if args.custom_times:
            custom_times = args.custom_times
        else:
            custom_times=None

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}_{}'.format('test' if args.render_test else 'path', start, str(args.render_mode)))
            # os.makedirs(testsavedir, exist_ok=True)
            if FT:
                testsavedir += '_FT_'
            
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            if args.use_tone_mapping_camparam == True and args.render_tm == 'wb':
                render_slowmo_bt(depths, render_poses, bt_poses, 
                                hwf, args.chunk, render_kwargs_test, args.render_mode, args.render_tm,
                                gt_imgs=images, savedir=testsavedir, camparam=Cam_param,
                                render_factor=args.render_factor, 
                                target_idx=args.target_idx, custom_times=custom_times, log_scale=args.log_scale,
                                start=start)
            if args.use_tone_mapping_camparam == True and args.render_tm == 'manual':
                render_slowmo_bt(depths, render_poses, bt_poses, 
                                hwf, args.chunk, render_kwargs_test, args.render_mode, args.render_tm,
                                gt_imgs=images, savedir=testsavedir, camparam=Cam_param,
                                render_factor=args.render_factor, 
                                target_idx=args.target_idx, custom_times=custom_times, log_scale=args.log_scale,
                                start=start)
            else: 
                render_slowmo_bt(depths, render_poses, bt_poses, 
                                hwf, args.chunk, render_kwargs_test, args.render_mode, args.render_tm,
                                gt_imgs=images, savedir=testsavedir, 
                                render_factor=args.render_factor, 
                                target_idx=args.target_idx,
                                start=start)
                # print('Done rendering', i,testsavedir)

        return
    
    if args.render_train:
        print("RENDER TRAINING DATA")
        RENDER_DEPTH=False
        RENDER_IMG=False
        RENDER_IMG_3EXP=True
        RENDER_WEIGHTS_MAP=False

        def _to_uint8_01(arr01):
            """[0,1] -> uint8"""
            return (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)

        def depth_to_norm01(depth):
            """depth: (H,W) float32. 2-98 percentile 정규화 + invalid=0"""
            d = depth.copy()
            valid = np.isfinite(d) & (d > 0)
            if not np.any(valid):
                return np.zeros_like(d, dtype=np.float32)
            p2, p98 = np.percentile(d[valid], [2, 98])
            # p2==p98 방어
            if p98 <= p2:
                norm = np.zeros_like(d, dtype=np.float32)
                norm[valid] = 1.0
                norm[~valid] = 0.0
                return norm
            norm = (d - p2) / (p98 - p2)
            norm = np.clip(norm, 0.0, 1.0)
            norm[~valid] = 0.0
            return norm.astype(np.float32)

        def apply_colormap01(img01, cmap=cv2.COLORMAP_JET):
            """[0,1] float -> RGB uint8 with colormap"""
            u8 = _to_uint8_01(img01)
            # OpenCV는 BGR 반환 → RGB로 변환
            bgr = cv2.applyColorMap(u8, cmap)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        
        with torch.no_grad():
                        
            num_img = poses.shape[0]
            
            testsavedir = os.path.join(basedir, expname, 'render-train_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            
            save_hdr_tm_dir = os.path.join(testsavedir, 'hdr_tm')
            save_hdr_exr_dir = os.path.join(testsavedir, 'hdr_exr')
            save_ldr_dir = os.path.join(testsavedir, 'ldr')
            os.makedirs(save_hdr_tm_dir, exist_ok=True)
            os.makedirs(save_hdr_exr_dir, exist_ok=True)
            os.makedirs(save_ldr_dir, exist_ok=True)
            if RENDER_DEPTH:
                save_depth_dir = os.path.join(testsavedir, 'depth')
                save_depth_viz_dir = os.path.join(testsavedir, 'depth_viz')
                os.makedirs(save_depth_dir, exist_ok=True)
                os.makedirs(save_depth_viz_dir, exist_ok=True)
            if RENDER_WEIGHTS_MAP:
                save_weights_viz_dir = os.path.join(testsavedir, 'weights_viz')
                os.makedirs(save_weights_viz_dir, exist_ok=True)
            
            for img_i in range(num_img):
                print(f'Rendering img index: {img_i}')
                img_idx_embed = img_i/num_img * 2. - 1.0                
                pose = torch.Tensor(poses[img_i, :3,:4]).to(device)
                ret = render(img_idx_embed, 
                            0, False,
                            num_img, H, W, focal, 
                            chunk=1024*16, 
                            c2w=pose,
                            **render_kwargs_test)
                
                # rgb_map_ref = (torch.clamp(ret['rgb_map_ref'], 0., 1.) * 255).cpu().numpy().astype(np.uint8)
                if RENDER_DEPTH:
                    depth_map_ref = ret['depth_map_ref'].cpu().numpy()
                    np.save(os.path.join(save_depth_dir, f'{img_i:05d}_depth.npy'), depth_map_ref)
                    # 2) viz (gray & colormap)
                    depth01 = depth_to_norm01(depth_map_ref)
                    # 그레이스케일 (0~1)
                    imageio.imwrite(os.path.join(save_depth_viz_dir, f'{img_i:05d}_depth_gray.png'),
                                    _to_uint8_01(depth01))
                    # 컬러맵
                    depth_color = apply_colormap01(depth01, cmap=cv2.COLORMAP_JET)
                    imageio.imwrite(os.path.join(save_depth_viz_dir, f'{img_i:05d}_depth_jet.png'),
                                    depth_color)
                    
                if RENDER_IMG:
                    print(save_ldr_dir, '{:05d}.png'.format(img_i))

                    rgb_hdr_ = ret['rgb_map_ref']
                    rgb8_ldr = to8b(Cam_param.RAD2LDR_img(rgb_hdr_, img_i).cpu().numpy())
                    
                    mu = torch.tensor(50).to(device)
                    rgb_map_ref = torch.log(1 + mu * rgb_hdr_) / torch.log(1 + mu)
                    rgb = (torch.clamp(rgb_map_ref, 0., 1.) * 255).cpu().numpy().astype(np.uint8)
                    imageio.imwrite(os.path.join(save_hdr_tm_dir, f'{img_i:05d}.png'), rgb)
                    imageio.imwrite(os.path.join(save_ldr_dir, '{:05d}.png'.format(img_i)), rgb8_ldr)

                if RENDER_IMG_3EXP:
                    print(save_ldr_dir, '{:05d}.png'.format(img_i))

                    rgb_hdr_ = ret['rgb_map_ref']
                                        
                    mu = torch.tensor(50).to(device)
                    rgb_map_ref = torch.log(1 + mu * rgb_hdr_) / torch.log(1 + mu)
                    rgb = (torch.clamp(rgb_map_ref, 0., 1.) * 255).cpu().numpy().astype(np.uint8)

                    rgb8_ldr_0 = to8b(Cam_param.RAD2LDR_img(rgb_hdr_, 0).cpu().numpy())
                    rgb8_ldr_1 = to8b(Cam_param.RAD2LDR_img(rgb_hdr_, 1).cpu().numpy())
                    rgb8_ldr_2 = to8b(Cam_param.RAD2LDR_img(rgb_hdr_, 2).cpu().numpy())

                    imageio.imwrite(os.path.join(save_hdr_tm_dir, f'{img_i:05d}.png'), rgb)
                    imageio.imwrite(os.path.join(save_ldr_dir, '{:05d}_0.png'.format(img_i)), rgb8_ldr_0)
                    imageio.imwrite(os.path.join(save_ldr_dir, '{:05d}_1.png'.format(img_i)), rgb8_ldr_1)
                    imageio.imwrite(os.path.join(save_ldr_dir, '{:05d}_2.png'.format(img_i)), rgb8_ldr_2)

                
                if RENDER_WEIGHTS_MAP:
                    import pdb; pdb.set_trace()
                    # weights_map_ref = ret['weights_map_dd']

                    w = ret['weights_map_dd']
                    if isinstance(w, torch.Tensor):
                        w = w.detach().cpu().numpy()
                    # (H,W)로 가정. 채널이 있으면 평균
                    if w.ndim == 3:
                        w = np.mean(w, axis=-1)
                    w = w.astype(np.float32)

                    # min-max 정규화
                    w_min, w_max = np.min(w), np.max(w)
                    w01 = (w - w_min) / (w_max - w_min + 1e-8)

                    # ----- Threshold & Mask -----
                    thr = 0.5   # 원하는 threshold 값
                    mask = (w01 >= thr).astype(np.uint8)  # 0/1 mask
                    mask_u8 = mask * 255  # PNG 저장용 (0 or 255)
                    imageio.imwrite(os.path.join(save_weights_viz_dir, f'{img_i:05d}_weights_mask.png'), mask_u8)

                    # ----- Gray / Color 저장 -----
                    imageio.imwrite(os.path.join(save_weights_viz_dir, f'{img_i:05d}_weights_gray.png'),
                                    _to_uint8_01(w01))
                    w_color = apply_colormap01(w01, cmap=cv2.COLORMAP_JET)
                    imageio.imwrite(os.path.join(save_weights_viz_dir, f'{img_i:05d}_weights_jet.png'), w_color)
                    
                
        return
    
    if args.render_dy:
        print("RENDER DYNAMIC")
        
        with torch.no_grad():
            num_img = poses.shape[0]
            
            testsavedir = os.path.join(basedir, expname, 'render-train_dy_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            
            save_img_dir = os.path.join(testsavedir, 'images')
            os.makedirs(save_img_dir, exist_ok=True)
            
            pbar = tqdm(range(num_img), desc=f"Rendering dynamic img")
            render_kwargs_test['inference'] = False
            for img_i in pbar:
                pbar.set_description(f"Rendering img index: {img_i}")
                # print(f'Rendering img index: {img_i}')
                img_idx_embed = img_i/num_img * 2. -1.0
                pose = torch.from_numpy(poses[img_i, :3,:4]).to(device)
                
                ret = render(img_idx_embed,
                             0, False,
                             num_img, H, W, focal,
                             chunk=1024*16,
                             c2w=pose,
                             **render_kwargs_test)
                
                ref_dy = (torch.clamp(Cam_param.RAD2LDR_img(ret['rgb_map_ref_dy'], img_i), 0., 1.) * 255).cpu().numpy().astype(np.uint8)
                post_dy = (torch.clamp(Cam_param.RAD2LDR_img(ret['rgb_map_post_dy'], img_i), 0., 1.) * 255).cpu().numpy().astype(np.uint8)
                prev_dy = (torch.clamp(Cam_param.RAD2LDR_img(ret['rgb_map_prev_dy'], img_i), 0., 1.) * 255).cpu().numpy().astype(np.uint8)
                
                imageio.imwrite(os.path.join(save_img_dir, f'{img_i:05d}_ref.png'), ref_dy)
                imageio.imwrite(os.path.join(save_img_dir, f'{img_i:05d}_post.png'), post_dy)
                imageio.imwrite(os.path.join(save_img_dir, f'{img_i:05d}_prev.png'), prev_dy)
                
        return
    
    if args.render_interpolate:
        
        with torch.no_grad():
        
            poses = torch.Tensor(poses).to(device)
            num_img = poses.shape[0]
            render_mid_exp=True
            if args.ft_path is not None:
                ft_basename = args.ft_path.split('/')[-1].split('.')[0]
                expname = expname + '_' + ft_basename
            if render_mid_exp:
                render_dir = os.path.join(basedir, expname, 'render_inter_mid_exp')
            else:
                render_dir = os.path.join(basedir, expname, 'render_interpolate')
            os.makedirs(os.path.join(render_dir, 'hdr_tm'), exist_ok=True)
            os.makedirs(os.path.join(render_dir, 'ldr'), exist_ok=True)
            os.makedirs(os.path.join(render_dir, 'hdr_exr'), exist_ok=True)
            os.makedirs(render_dir, exist_ok=True)
            for img_i in i_train[:-1]:
                
                ratio=0.5
                
                img_idx_embed_1 = img_i/num_img * 2. -1.0
                img_idx_embed_2 = (img_i + 1) / num_img * 2.  -1.0
                print(img_i)
                
                pose_1 = poses[img_i]
                pose_2 = poses[img_i + 1]
                render_pose = (pose_1 + pose_2)/2
                R_w2t = render_pose[:3, :3].transpose(0,1)
                t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])
                
                
                ret1 = render_sm(img_idx_embed_1, 0, False,
                                num_img,
                                H, W, focal,
                                chunk=1024*16,
                                c2w=render_pose,
                                **render_kwargs_test)
                
                ret2 = render_sm(img_idx_embed_2, 0, False,
                                    num_img,
                                    H, W, focal,
                                    chunk=1024*16,
                                    c2w=render_pose,
                                    **render_kwargs_test)
                
                T_i = torch.ones((1, H, W))
                final_rgb = torch.zeros((3, H, W))
                num_sample = ret1['raw_rgb'].shape[2]
                
                for j in range(0, num_sample):
                    splat_alpha_dy_1, splat_rgb_dy_1, \
                    splat_alpha_rig_1, splat_rgb_rig_1 = splat_rgb_img(ret1, ratio, R_w2t,
                                                                        t_w2t, j, H, W,
                                                                        focal, True)
                    splat_alpha_dy_2, splat_rgb_dy_2, \
                    splat_alpha_rig_2, splat_rgb_rig_2 = splat_rgb_img(ret2, 1. - ratio, R_w2t,
                                                                        t_w2t, j, H, W,
                                                                        focal, False)
                    
                    final_rgb += T_i * (splat_alpha_dy_1 * splat_rgb_dy_1 + \
                                        splat_alpha_rig_1 * splat_rgb_rig_1) * (1.0 - ratio)
                    final_rgb += T_i * (splat_alpha_dy_2 * splat_rgb_dy_2 + \
                                        splat_alpha_rig_2 * splat_rgb_rig_2) * ratio
                    
                    alpha_1_final = (1.0 - (1. - splat_alpha_dy_1) * (1. - splat_alpha_rig_1)) * (1. - ratio)
                    alpha_2_final = (1.0 - (1. - splat_alpha_dy_2) * (1. - splat_alpha_rig_2)) * ratio
                    alpha_final = alpha_1_final + alpha_2_final
                    
                    T_i = T_i * (1.0 - alpha_final + 1e-10)
                    
                if Cam_param is not None:
                    # tone_map = img_i // args.step_frame + 2
                    if render_mid_exp:
                        tone_map = img_i - (img_i % 3)
                    else:  
                        tone_map = img_i
                    rgb8_ldr = to8b(Cam_param.RAD2LDR_img(final_rgb.permute(1,2,0), tone_map).cpu().numpy())
                    
                    mu = torch.tensor(50, dtype=final_rgb.dtype, device=final_rgb.device)
                    final_rgb = torch.log(1 + mu * final_rgb) / torch.log(1 + mu)
                    rgb8_hdr = to8b(final_rgb.permute(1,2,0).cpu().numpy())
                    
                    imageio.imwrite(os.path.join(render_dir, 'hdr_tm', '{:05d}.png'.format(img_i)), rgb8_hdr)
                    imageio.imwrite(os.path.join(render_dir, 'ldr', '{:05d}.png'.format(img_i)), rgb8_ldr)
                    cv2.imwrite(os.path.join(render_dir, '{:05d}.exr'.format(img_i)), final_rgb.permute(1,2,0).cpu().numpy().astype(np.float32)[...,::-1])
                    # imageio.imwrite(os.path.join(render_dir, 'hdr_exr', '{:05d}.exr'.format(img_i)),
                    #                 final_rgb.permute(1,2,0).cpu().numpy().astype(np.float32))
                    
                else:
                    rgb8_ldr = to8b(final_rgb.permute(1,2,0).cpu().numpy())
                    imageio.imwrite(os.path.join(render_dir, 'ldr', '{:05d}.png'.format(img_i)), rgb8_ldr)
                    
        return
        
    if args.render_lockcam:
        print('RENDER JUST LOCKCAM')
        
        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx', target_idx)
        
        testsavedir = os.path.join(basedir, expname, f'render-lockcam-{target_idx:03d}-{start:06d}')
        with torch.no_grad():
            H, W, focal =hwf
            
            t = time.time()
            
            # if Cam_param is not None:
            #     savedir += f'_{args.render_tm}'
            save_hdr_tm_dir = os.path.join(testsavedir, 'hdr_tm')
            save_depth_dir = os.path.join(testsavedir, 'depths')
            save_hdr_exr_dir = os.path.join(testsavedir, 'hdr_exr')
            if Cam_param is not None and args.render_tm == 'wb':
                # save_ldr_dir = os.path.join(testsavedir, 'ldr')
                dir_ldr_0 = os.path.join(testsavedir, 'ldr', '0')
                dir_ldr_1 = os.path.join(testsavedir, 'ldr', '1')
                dir_ldr_2 = os.path.join(testsavedir, 'ldr', '2')
                os.makedirs(dir_ldr_0, exist_ok=True)
                os.makedirs(dir_ldr_1, exist_ok=True)
                os.makedirs(dir_ldr_2, exist_ok=True)
                
            os.makedirs(save_hdr_tm_dir, exist_ok=True)
            os.makedirs(save_depth_dir, exist_ok=True)        
            os.makedirs(save_hdr_exr_dir, exist_ok=True)
            
            pose = torch.Tensor(poses[target_idx, :3,:4]).to(device)
            for img_i in range(int(num_img)):
                print('render lock camera time ', img_i, time.time() - t)
                t = time.time()
                img_idx_embed = img_i/num_img * 2. -1.0
                ret = render(img_idx_embed,
                                0, False,
                                num_img, H, W, focal,
                                chunk=1024*16,
                                c2w=pose,
                                **render_kwargs_test)
                
                depth_map_ref = ret['depth_map_ref'].cpu().numpy()
                np.save(os.path.join(save_depth_dir, f'{img_i:05d}_depth.npy'), depth_map_ref)
                
                rgb_hdr_ = ret['rgb_map_ref']
                mu = torch.tensor(50).to(device)
                hdr_tm = torch.log(1 + mu * rgb_hdr_) / torch.log(1 + mu)
                # imageio.imwrite(os.path.join(save_hdr_exr_dir, f'{img_i:05d}.exr'), rgb_hdr_.cpu().numpy())
                cv2.imwrite(os.path.join(save_hdr_exr_dir, f'{img_i:05d}.exr'), rgb_hdr_.cpu().numpy().astype(np.float32)[...,::-1])
                imageio.imwrite(os.path.join(save_hdr_tm_dir, f'{img_i:05d}.jpg'), to8b(hdr_tm.cpu().numpy()))
                
                if Cam_param is not None and args.render_tm == 'wb':
                    
                    ldr_0 = Cam_param.RAD2LDR_img(rgb_hdr_, 0)
                    ldr_1 = Cam_param.RAD2LDR_img(rgb_hdr_, 1)
                    ldr_2 = Cam_param.RAD2LDR_img(rgb_hdr_, 2)
                    
                    rgb8_0 = to8b(ldr_0.cpu().numpy())
                    rgb8_1 = to8b(ldr_1.cpu().numpy())
                    rgb8_2 = to8b(ldr_2.cpu().numpy())
                    
                    filename_0 = os.path.join(dir_ldr_0, '{:03d}_0.jpg'.format(img_i))
                    filename_1 = os.path.join(dir_ldr_1, '{:03d}_1.jpg'.format(img_i))
                    filename_2 = os.path.join(dir_ldr_2, '{:03d}_2.jpg'.format(img_i))
                    
                    imageio.imwrite(filename_0, rgb8_0)
                    imageio.imwrite(filename_1, rgb8_1)
                    imageio.imwrite(filename_2, rgb8_2)
        return
 
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)
    masks = torch.Tensor(masks).to(device)

    poses = torch.Tensor(poses).to(device)
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    chain_bwd = 0
    l1_flag = args.use_render_l1

    # Distinguish debugging mode
    debug_mode = 1 if args.expname.split('_')[-2] == 'debug' else 0
    debug_mode = 1 if args.debug else 0

    if not debug_mode:
        wandb.init(
            project = "HDR-NSFF",
            name = args.expname if not FT else expname_FT,
            dir = '/node_data/shindy/projects/hdr-4d/wandb',
            resume="allow"
        )
    
    if args.use_percept:
        perceptual_net = VGGPerceptual().to(device)
        perceptual_net.eval()
        patch_sz = args.patch_size
        N_rand = patch_sz * patch_sz
        
        # if Cam_param is not None:
        #     for param in Cam_param.wb.parameters():
        #         param.requires_grad = False
        #     for param in Cam_param.CRF.parameters():
        #         param.requires_grad = False

    if args.stage == 2:
        args.w_sf_reg = 0
        args.w_cycle = 0
        args.w_optical_flow = 0
        args.w_sm = 0
        args.w_prob_reg = 0
        args.w_depth = 0
        args.w_entropy = 0
    
    use_difix_flag=False
    if args.use_difix:
        sys.path.append(os.path.join(os.path.dirname(__file__), "Difix3D"))
        from src.pipeline_difix import DifixPipeline

        # difix_pipe= DifixPipeline.from_pretrained(
        #     "stabilityai/sd-turbo"
        # )
        # difix_pipe.to(device).eval()

        difix_pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        difix_pipe.to(device)

        difix_save_root = os.path.join(basedir, expname, 'difix_views')

        difix_updater = DifixUpdater(H, W, focal,
                                        render, render_kwargs_test,
                                        images, poses,
                                        args.difix_kview,
                                        args.difix_kview_step,
                                        difix_save_root,
                                        args.datadir,
                                        args.difix_use_depth,
                                        args.mask_warp,
                                        motion_masks=masks,
                                        weights_map_ray=args.difix_weights_map_ray,
                                        use_mv_mask=args.difix_use_mv_mask,
                                        inter_pose=args.difix_inter_pose,
                                        )
        
        if args.difix_start_step is not None:
            difix_start_step = args.difix_start_step
            next_refresh = difix_start_step
            difix_log_step = (global_step // args.difix_interval) * args.difix_interval
            if difix_log_step != 0 and difix_log_step <= difix_start_step:
                difix_log_step = difix_start_step
                next_refresh = difix_log_step + args.difix_interval
        
        else:
            difix_log_step = (global_step // args.difix_interval) * args.difix_interval
            next_refresh = difix_log_step + args.difix_interval
        
            difix_start_step = args.difix_interval

        if args.difix_ver2 or args.difix_ver3:
            images_orig = images
            poses_orig = poses

            if args.difix_use_depth:
                depths_orig = depths

            ### For Balanced Sampling
            # initial
            is_syn = 1
            use_difix_flag = False
            prev_img_i = 0 # initial
            #########################
            
        if os.path.exists(difix_save_root) or difix_log_step != 0:
            if args.difix_ver2:
                import pdb; pdb.set_trace()
                difix_updater.update(difix_log_step, difix_pipe, images_orig, Cam_param, args.difix_gt_ref, args.difix_gt_ref_ratio)
                images_syn = difix_updater.synthetic_imgs.to(images.device)
                poses_syn = difix_updater.synthetic_poses.to(poses.device)
                images = torch.cat([images_orig, images_syn], 0)
                poses = torch.cat([poses_orig, poses_syn], 0)

                if args.difix_use_depth:
                    print("update difix depth!")
                    depths_syn = difix_updater.synthetic_depths.to(depths.device)
                    depths = torch.cat([depths_orig, depths_syn], 0)
            elif args.difix_ver3:
                import pdb; pdb.set_trace()
                difix_updater.update_3(difix_log_step, difix_pipe, images_orig, Cam_param, args.difix_gt_ref)

            else:
                difix_updater.refresh(difix_log_step, difix_pipe, images, Cam_param)

    for i in range(start, N_iters):
        freeze_cam_this_iter = False
        chain_bwd = 1 - chain_bwd
        time0 = time.time()
        print('expname ', expname, ' chain_bwd ', chain_bwd, 
             ' lindisp ', args.lindisp, ' decay_iteration ', decay_iteration)
        # print('Random FROM SINGLE IMAGE')
        
        if args.use_freq_reg:
            render_kwargs_train['current_iter'] = i
            render_kwargs_test['current_iter'] = i
        else:
            # set to None to disable trick
            render_kwargs_train['current_iter'] = None
            render_kwargs_test['current_iter'] = None

        # Random from one image
        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        if args.use_difix and args.difix_ver2 and i > difix_start_step:
            use_difix_flag=True

            if args.balanced_sampling:
                is_syn = 0 if (np.random.rand() < args.difix_gt_ratio) else 1
            else:
                is_syn = np.random.choice([0,len(i_train)])
            
            if args.use_tone_mapping and args.difix_freeze_cam and (Cam_param is not None):
                if (args.use_difix and args.difix_ver2 and i > difix_start_step) and (is_syn != 0):
                    Cam_param.freeze_for_iter()
                    freeze_cam_this_iter = True
                else:
                    Cam_param.unfreeze_for_iter()

            if is_syn != 0:
                which_syn = np.random.choice(args.difix_kview)
                img_i_difix = len(i_train) + img_i * args.difix_kview + which_syn

                # Freeze Static, Only train dynamic network
                if args.difix_freeze_rigid:
                    for p in render_kwargs_train['network_rigid'].parameters():
                        p.requires_grad=False
                    render_kwargs_train['network_rigid'].eval()

                # if args.difix_freeze_cam:
                #     if Cam_param is not None:
                #         for param in Cam_param.wb.parameters():
                #             param.requires_grad = False
                #         for param in Cam_param.CRF.parameters():
                #             param.requires_grad = False
                #         camparam_frozen=True

                if args.difix_l1:
                    l1_flag = True

            else:
                # Unfreeze network_rigid
                img_i_difix = img_i
                if args.difix_freeze_rigid:
                    for p in render_kwargs_train['network_rigid'].parameters():
                        p.requires_grad=True
                    render_kwargs_train['network_rigid'].train()

                # if args.difix_freeze_cam:
                #     if Cam_param is not None:
                #         for param in Cam_param.wb.parameters():
                #             param.requires_grad = True
                #         for param in Cam_param.CRF.parameters():
                #             param.requires_grad = True
                #         camparam_frozen=False
                if args.difix_l1:
                    l1_flag= False

            target = images[img_i_difix].cuda()
            pose = poses[img_i_difix, :3, :4]
            if args.difix_use_depth:
                depth_gt = depths[img_i_difix].cuda()
            else:
                depth_gt = depths[img_i].cuda()
        else:
            target = images[img_i].cuda()
            pose = poses[img_i, :3,:4]
            depth_gt = depths[img_i].cuda()
            
        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()
        mask_gt = masks[img_i].cuda()
        if FT:
            target_enh_hdr = hdr_enh[img_i].cuda()
        else:
            target_enh_hdr = None
        
        if args.sparse:
            step_of_pose = (poses[:-1,:,3] - poses[1:,:,3]).pow(2).sum(1).sqrt()
            step_of_pose = torch.cat((torch.tensor([0.0000]), step_of_pose))
            pose_length = step_of_pose.sum()

        # Cast intrinsics for each image # Shindy modified
        if args.multi_camera == True:
            hwf = HWF[img_i]
            _, _, focal = hwf      
            _, _, focal_prev = HWF[max(0, img_i - 1)]
            _, _, focal_post = HWF[min(int(num_img) - 1, img_i + 1)]
            
        if args.traj_flow:
            
            if img_i == 0:
                flow_fwd, fwd_mask = read_traj_flow(args.datadir, img_i, fwd=True, start_frame=args.start_frame, step=args.step_frame)
                flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
            elif img_i == num_img - 1:
                flow_bwd, bwd_mask = read_traj_flow(args.datadir, img_i, fwd=False, start_frame=args.start_frame, step=args.step_frame)
                flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
            else:
                flow_fwd, fwd_mask = read_traj_flow(args.datadir, img_i, fwd=True, start_frame=args.start_frame, step=args.step_frame)
                flow_bwd, bwd_mask = read_traj_flow(args.datadir, img_i, fwd=False, start_frame=args.start_frame, step=args.step_frame)
            
        elif args.mix_flow:
            
            if img_i == 0:
                tf_fwd, tf_fwd_mask = read_traj_flow(args.datadir, img_i, fwd=True, start_frame=args.start_frame, step=args.step_frame)
                tf_bwd, tf_bwd_mask = np.zeros_like(tf_fwd), np.zeros_like(tf_fwd_mask)
                
                of_fwd, of_fwd_mask = read_optical_flow(args.datadir, img_i,
                                                    args.start_frame, fwd=True, step=args.step_frame)
                of_bwd, of_bwd_mask = np.zeros_like(of_fwd), np.zeros_like(of_fwd_mask)
                
            elif img_i == num_img - 1:
                tf_bwd, tf_bwd_mask = read_traj_flow(args.datadir, img_i, fwd=False, start_frame=args.start_frame, step=args.step_frame)
                tf_fwd, tf_fwd_mask = np.zeros_like(tf_bwd), np.zeros_like(tf_bwd_mask)
                
                of_bwd, of_bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                    args.start_frame, fwd=False, step=args.step_frame)
                of_fwd, of_fwd_mask = np.zeros_like(of_bwd), np.zeros_like(of_bwd_mask)
            else:
                tf_fwd, tf_fwd_mask = read_traj_flow(args.datadir, img_i, fwd=True, start_frame=args.start_frame, step=args.step_frame)
                tf_bwd, tf_bwd_mask = read_traj_flow(args.datadir, img_i, fwd=False, start_frame=args.start_frame, step=args.step_frame)
                
                of_fwd, of_fwd_mask = read_optical_flow(args.datadir, 
                                                    img_i, args.start_frame, 
                                                    fwd=True, step=args.step_frame)
                of_bwd, of_bwd_mask = read_optical_flow(args.datadir, 
                                                    img_i, args.start_frame, 
                                                    fwd=False, step=args.step_frame)
            
            flow_fwd = of_fwd.copy()
            flow_fwd[tf_fwd_mask > 0] = tf_fwd[tf_fwd_mask > 0]
            fwd_mask = np.logical_or(of_fwd_mask, tf_fwd_mask).astype(np.float32)
            
            flow_bwd = of_bwd.copy()
            flow_bwd[tf_bwd_mask > 0] = tf_bwd[tf_bwd_mask > 0]
            bwd_mask = np.logical_or(of_bwd_mask, tf_bwd_mask).astype(np.float32)
                    
        else: # original optical flow
            if img_i == 0:
                flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
                                                    args.start_frame, fwd=True, step=args.step_frame)
                flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
            elif img_i == num_img - 1:
                flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                    args.start_frame, fwd=False, step=args.step_frame)
                flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
            else:
                flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
                                                    img_i, args.start_frame, 
                                                    fwd=True, step=args.step_frame)
                flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
                                                    img_i, args.start_frame, 
                                                    fwd=False, step=args.step_frame)
            

        # # ======================== TEST 
        # if args.mask_warp and is_syn:
        #     import pdb; pdb.set_trace()
        #     mask_syn = reproject_mask_to_view(
        #                     mask_ref=masks[img_i],           # [H,W] or adapt
        #                     depth_ref=depths[img_i],                    # [H,W]
        #                     c2w_ref=poses[img_i,:3,:4],                 # [3,4]
        #                     c2w_syn=poses[img_i_difix,:3,:4],           # [3,4] (합성 뷰의 pose)
        #                     H=H, W=W, focal=focal
        #     )
        #     mask_np = (mask_syn.cpu().numpy() * 255).astype(np.uint8)
        #     save_path = f"mask_syn_{img_i:05d}_{img_i_difix}.png"
        #     cv2.imwrite(save_path, mask_np)

        TEST = False
        if TEST:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            point_x, point_y = 300, 200
            
            depth_value = depth_gt[point_y, point_x].item()  # Assuming depth_gt is a tensor
            depth_value_2 = depth_gt[point_y - 100, point_x].item()
            print(f"Depth at (300, 200): {depth_value}")
            print(f"Depth at (300, 100): {depth_value_2}")

            print('CHECK DEPTH and FLOW and exiting')
            print(images[img_i].shape)
            print(flow_fwd.shape, img_i)

            warped_im2 = warp_flow(images[img_i + 1].cpu().numpy(), flow_fwd)
            warped_im0 = warp_flow(images[img_i - 1].cpu().numpy(), flow_bwd)
            mask_gt = masks[img_i].cpu().numpy()

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 3, 1)
            plt.imshow(target.cpu().numpy())
            plt.plot(point_x, point_y, 'ro')  # Mark the point in the target image
            
            plt.subplot(2, 3, 4)
            plt.imshow(depth_gt.cpu().numpy(), cmap='jet') 
            plt.plot(point_x, point_y, 'ro')  # Mark the point in the target image

            plt.subplot(2, 3, 2)
            plt.imshow(flow_to_image(flow_fwd)/255. * fwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 3)
            plt.imshow(flow_to_image(flow_bwd)/255. * bwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 5)
            plt.imshow(mask_gt, cmap='gray')

            cv2.imwrite('im_%d.jpg'%(img_i),
                        np.uint8(np.clip(target.cpu().numpy()[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i + 1), 
                        np.uint8(np.clip(warped_im2[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i - 1), 
                        np.uint8(np.clip(warped_im0[:, :, ::-1], 0, 1) * 255))
            plt.savefig('depth_flow_%d.jpg'%img_i)
            sys.exit()

        #  END OF TEST
        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
    
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()
        # more correct way for flow loss
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            # if FT:
            #     print('only hard mining for fine_tuning!!')
            #     # select_inds = np.random.choice(hard_coords.shape[0], size=[min(hard_coords.shape[0], num_extra_sample)], replace=False) # choose 1024 or 512
            #     select_inds = np.random.choice(hard_coords.shape[0], size=[N_rand], replace=False)
            #     select_coords = hard_coords[select_inds].long()
            
            # if args.use_percept: # for debugging
            #     select_coords = sample_path_coords(H, W, patch_sz).long().cuda()
            if args.use_difix and args.difix_ver2 and i > difix_start_step and is_syn and args.use_percept:
                if args.difix_weights_map_ray:
                    num_patches=1
                    mix_ratio=1.0
                    coords_np = np.asarray(difix_updater.motion_coords[img_i][which_syn], dtype=np.int64)
                    select_coords = sample_mixed_patches(
                        coords_np, H, W, patch_sz, num_patches, mix_ratio, device
                    )

                else:
                    select_coords = sample_path_coords(H, W, patch_sz).long().cuda()
                
            #FIXME : for weights_map_ray
            elif args.use_difix and args.difix_ver2 and i > difix_start_step and is_syn and args.difix_weights_map_ray:

                difix_motion_coords = torch.Tensor(difix_updater.motion_coords[img_i][which_syn]).cuda()

                select_difix_inds = np.random.choice(difix_motion_coords.shape[0], size=[N_rand], replace=False)
                select_coords = difix_motion_coords[select_difix_inds].long()

                
            elif (args.use_motion_mask and i < decay_iteration * 1000) or FT:
                print('HARD MINING STAGE !')
                num_extra_sample = args.num_extra_sample
                print('num_extra_sample ', num_extra_sample)
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)
                
            # elif args.use_percept and (global_step > args.start_percept):
            #     select_coords = sample_path_coords(H, W, patch_sz).long().cuda()

            else:
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
            
            rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_rgb = target[select_coords[:, 0], 
                                select_coords[:, 1]]  # (N_rand, 3)
            if FT:
                target_enh_hdr_rgb = target_enh_hdr[select_coords[:,0],
                                                    select_coords[:,1]] # (N_rand, 3)
            target_depth = depth_gt[select_coords[:, 0], 
                                select_coords[:, 1]]
            target_mask = mask_gt[select_coords[:, 0], 
                                select_coords[:, 1]].unsqueeze(-1)

            target_of_fwd = flow_fwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

            target_of_bwd = flow_bwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)
        

        if args.sparse:
            img_idx_embed = step_of_pose[:img_i + 1].sum() / pose_length * 2. - 1.0
        else:
            img_idx_embed = img_i/num_img * 2. - 1.0
        #img_idx_embed = img_i/num_img * 2. - 1.0

        #####  Core optimization loop  #####
            
        if args.no_chain_5frames:
            chain_5frames = False
        elif args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
            if args.chain_5frames_flow:
                chain_5frames_flow = True
        else:
            chain_5frames = False
            chain_5frames_flow = False
        chain_5frames_flow = False

        print('chain_5frames ', chain_5frames, ' chain_bwd ', chain_bwd, 'chain_5frames_flow', chain_5frames_flow)
        optimizer.zero_grad()
        ret = render(img_idx_embed, 
                     chain_bwd, 
                     chain_5frames,
                     num_img, H, W, focal,
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)
        
        if args.use_difix and i >= difix_start_step:

            if args.difix_ver2:
                if global_step >= next_refresh:
                    difix_updater.update(global_step, difix_pipe, images, Cam_param, args.difix_gt_ref, args.difix_gt_ref_ratio)
                    next_refresh += args.difix_interval
                    images_syn = difix_updater.synthetic_imgs.to(images.device)
                    poses_syn = difix_updater.synthetic_poses.to(poses.device)
                    images = torch.cat([images_orig, images_syn], 0)
                    poses = torch.cat([poses_orig, poses_syn], 0)


        if args.use_tone_mapping == True:
            rgb_map_ref = Cam_param.RAD2LDR(ret['rgb_map_ref'], img_i, is_train=True)
            rgb_map_rig = Cam_param.RAD2LDR(ret['rgb_map_rig'], img_i, is_train=True)
            rgb_map_ref_dy = Cam_param.RAD2LDR(ret['rgb_map_ref_dy'], img_i, is_train=True)
            rgb_map_prev_dy = Cam_param.RAD2LDR(ret['rgb_map_prev_dy'], img_i, is_train=True)
            rgb_map_post_dy = Cam_param.RAD2LDR(ret['rgb_map_post_dy'], img_i, is_train=True)
            if chain_5frames==True:
                rgb_map_pp_dy = Cam_param.RAD2LDR(ret['rgb_map_pp_dy'], img_i, is_train=True)
        
        else: # original nsff
            rgb_map_ref = ret['rgb_map_ref']
            rgb_map_rig = ret['rgb_map_rig']
            rgb_map_ref_dy = ret['rgb_map_ref_dy']
            rgb_map_prev_dy = ret['rgb_map_prev_dy']
            rgb_map_post_dy = ret['rgb_map_post_dy']
            if chain_5frames==True:
                rgb_map_pp_dy = ret['rgb_map_pp_dy']
            
        print("img_i :", img_i)

        # FIXME: difix_ver_2 -> maybe not a problem?
        pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
        pose_prev = poses[max(img_i - 1, 0), :3,:4]
        if args.use_difix and args.difix_ver2 and i > difix_start_step:
            pose = poses[img_i]
        ####################
        
        if args.multi_camera == True:
            render_of_fwd, render_of_bwd = compute_optical_flow_multi_camera(pose_post, pose, pose_prev, H, W, focal_post, focal, focal_prev, ret)
            
        else:
            render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                                pose, pose_prev, 
                                                                H, W, focal, 
                                                                ret)
            if chain_5frames_flow:
                pose_postpost = poses[min(img_i + 2, int(num_img) - 1), :3,:4]
                pose_prevprev = poses[max(img_i - 2, 0), :3,:4]

                render_of_i2_fwd, render_of_i2_bwd = compute_optical_flow_i2(pose_postpost, pose, pose_prevprev, H, W, focal, ret)
        
        
        # # FIXME
        # if args.use_tone_mapping == True and global_step <= crf_stop_step:
        #     # optim_cam.zero_grad()
        #     optim_cam_wb.zero_grad()
        #     if optim_cam_crf is not None:
        #         optim_cam_crf.zero_grad()

        if args.use_tone_mapping:
            optim_cam_wb.zero_grad(set_to_none=True)
            if optim_cam_crf is not None:
                optim_cam_crf.zero_grad(set_to_none=True)
        
        if (args.use_tone_mapping and (not Cam_param.is_frozen_forever()) and (global_step > crf_stop_step) and (not args.difix_freeze_cam)):
            print(f'[Step {global_step}] Freezing Cam_param FOREVER...')
            Cam_param.freeze_forever()
                
        # if args.use_tone_mapping and not camparam_frozen and global_step > crf_stop_step:
        #     print(f'[Step {global_step}] Freezing Cam_param gradients...')
        #     for param in Cam_param.wb.parameters():
        #         param.requires_grad = False
        #     if Cam_param.CRF is not None:
        #         for param in Cam_param.CRF.parameters():
        #             param.requires_grad = False
        #     camparam_frozen = True

        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        
        if not args.skip_prob_reg_loss:
            prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))
        else:
            prob_reg_loss = torch.zeros(1).to(device)
        
        if args.use_sat_mask :

            if args.sat_mask_0:
                sat_mask = saturation_mask_0(target_rgb, threshold_low=args.leaky_th_l, threshold_high=args.leaky_th_h)
            elif args.sat_mask_3:
                sat_mask = saturation_mask_3(target_rgb, threshold_low=args.leaky_th_l, threshold_high=args.leaky_th_h)
            else:
                sat_mask = saturation_mask(target_rgb, threshold_low=args.leaky_th_l, threshold_high=args.leaky_th_h)       

        # dynamic rendering loss
        if i <= decay_iteration * 1000:
            if args.use_sat_mask :
                if args.render_l1_l2:
                    # import pdb; pdb.set_trace()
                    alpha = getattr(args, 'w_l1_l2', 0.01)
                    w_ones = torch.ones_like(target_rgb[..., :1])
                    render_loss  = mixed_loss_sat(rgb_map_ref_dy,  target_rgb, w_ones,                      sat_mask, alpha)
                    render_loss += mixed_loss_sat(rgb_map_post_dy, target_rgb, weight_map_post.unsqueeze(-1), sat_mask, alpha)
                    render_loss += mixed_loss_sat(rgb_map_prev_dy, target_rgb, weight_map_prev.unsqueeze(-1), sat_mask, alpha)
                elif args.use_render_l1 or l1_flag:
                    render_loss = torch.mean(sat_mask * torch.abs(rgb_map_ref_dy - target_rgb))
                    render_loss += compute_mae_sat(rgb_map_post_dy,
                                                   target_rgb,
                                                   weight_map_post.unsqueeze(-1),
                                                   sat_mask)
                    render_loss += compute_mae_sat(rgb_map_prev_dy,
                                                   target_rgb,
                                                   weight_map_prev.unsqueeze(-1),
                                                   sat_mask)

                else:
                    render_loss = torch.mean(sat_mask * ((rgb_map_ref_dy - target_rgb) ** 2))
                    render_loss += compute_mse_sat(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1),
                                            sat_mask)
                    render_loss += compute_mse_sat(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1),
                                            sat_mask)
            else :
                if args.render_l1_l2:
                    alpha = getattr(args, 'w_l1_l2', 0.01)
                    render_loss  = mixed_loss_basic(rgb_map_ref_dy,  target_rgb, None,                      alpha)
                    render_loss += mixed_loss_basic(rgb_map_post_dy, target_rgb, weight_map_post.unsqueeze(-1), alpha)
                    render_loss += mixed_loss_basic(rgb_map_prev_dy, target_rgb, weight_map_prev.unsqueeze(-1), alpha)

                elif args.use_render_l1 or l1_flag:
                    render_loss = img2mae(rgb_map_ref_dy, target_rgb)
                    render_loss += compute_mae(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1))
                    render_loss += compute_mae(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1))
                else:

                    render_loss = img2mse(rgb_map_ref_dy, target_rgb)
                    render_loss += compute_mse(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1))
                    render_loss += compute_mse(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1))
            if FT:  # Fine Tuning
                motion_mask_FT = 1 - target_mask
                render_enh_loss = compute_mse(ret['rgb_map_ref'], target_enh_hdr_rgb, motion_mask_FT)
                render_enh_loss += compute_mse(ret['rgb_map_post_dy'], target_enh_hdr_rgb, weight_map_post.unsqueeze(-1)*motion_mask_FT)
                render_enh_loss += compute_mse(ret['rgb_map_prev_dy'], target_enh_hdr_rgb, weight_map_prev.unsqueeze(-1)*motion_mask_FT)
                
                # render_enh_loss = img2mse(ret['rgb_map_ref'], target_enh_hdr_rgb)
                # render_enh_loss += compute_mse(ret['rgb_map_post_dy'], target_enh_hdr_rgb, weight_map_post.unsqueeze(-1))
                # render_enh_loss += compute_mse(ret['rgb_map_prev_dy'], target_enh_hdr_rgb, weight_map_prev.unsqueeze(-1))
            
        else:
            print('only compute dynamic render loss in masked region')
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()
            if args.use_motion_mask_weights:
                weights_map_dd = torch.maximum(weights_map_dd, target_mask)

            # dynamic rendering loss
            if args.use_sat_mask :
                if args.render_l1_l2:
                    alpha = getattr(args, 'w_l1_l2', 0.01)
                    render_loss  = mixed_loss_sat(rgb_map_ref_dy,  target_rgb, weights_map_dd,                         sat_mask, alpha)
                    render_loss += mixed_loss_sat(rgb_map_post_dy, target_rgb, weight_map_post.unsqueeze(-1)*weights_map_dd, sat_mask, alpha)
                    render_loss += mixed_loss_sat(rgb_map_prev_dy, target_rgb, weight_map_prev.unsqueeze(-1)*weights_map_dd, sat_mask, alpha)

                elif args.use_render_l1 or l1_flag:
                    render_loss = compute_mae_sat(rgb_map_ref_dy, 
                                            target_rgb, 
                                            weights_map_dd,
                                            sat_mask)
                    render_loss += compute_mae_sat(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1) * weights_map_dd,
                                            sat_mask)
                    render_loss += compute_mae_sat(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1) * weights_map_dd,
                                            sat_mask)                    
                
                else:

                
                    render_loss = compute_mse_sat(rgb_map_ref_dy, 
                                            target_rgb, 
                                            weights_map_dd,
                                            sat_mask)
                    render_loss += compute_mse_sat(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1) * weights_map_dd,
                                            sat_mask)
                    render_loss += compute_mse_sat(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1) * weights_map_dd,
                                            sat_mask)
                
                
            else:
                if args.render_l1_l2:
                    alpha = getattr(args, 'w_l1_l2', 0.01)
                    render_loss  = mixed_loss_basic(rgb_map_ref_dy,  target_rgb, weights_map_dd,                         alpha)
                    render_loss += mixed_loss_basic(rgb_map_post_dy, target_rgb, weight_map_post.unsqueeze(-1)*weights_map_dd, alpha)
                    render_loss += mixed_loss_basic(rgb_map_prev_dy, target_rgb, weight_map_prev.unsqueeze(-1)*weights_map_dd, alpha)
                elif args.use_render_l1 or l1_flag:
                    render_loss = compute_mae(rgb_map_ref_dy, 
                                            target_rgb, 
                                            weights_map_dd)
                    render_loss += compute_mae(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1) * weights_map_dd)
                    render_loss += compute_mae(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1) * weights_map_dd)                    
                
                else:
                    render_loss = compute_mse(rgb_map_ref_dy, 
                                            target_rgb, 
                                            weights_map_dd)
                    render_loss += compute_mse(rgb_map_post_dy, 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1) * weights_map_dd)
                    render_loss += compute_mse(rgb_map_prev_dy, 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1) * weights_map_dd)
                
            if FT:  # Fine Tuning
                motion_mask_FT = 1 - target_mask
                render_enh_loss = compute_mse(ret['rgb_map_ref_dy'], target_enh_hdr_rgb, weights_map_dd * motion_mask_FT)
                render_enh_loss += compute_mse(ret['rgb_map_post_dy'], target_enh_hdr_rgb, weight_map_post.unsqueeze(-1) * weights_map_dd * motion_mask_FT)
                render_enh_loss += compute_mse(ret['rgb_map_prev_dy'], target_enh_hdr_rgb, weight_map_prev.unsqueeze(-1) * weights_map_dd * motion_mask_FT)

        # union rendering loss
        if args.render_l1_l2:
            alpha = getattr(args, 'w_l1_l2', 0.01)
            render_loss += F.mse_loss(rgb_map_ref[:N_rand, ...], target_rgb[:N_rand, ...]) + \
                        alpha * F.l1_loss(rgb_map_ref[:N_rand, ...], target_rgb[:N_rand, ...])
        elif args.use_render_l1 or l1_flag:
            render_loss += img2mae(rgb_map_ref[:N_rand, ...],
                                   target_rgb[:N_rand, ...])
        else:

            render_loss += img2mse(rgb_map_ref[:N_rand, ...], 
                                target_rgb[:N_rand, ...])
           
        # if args.use_percept and (global_step > args.start_percept):
        if args.use_difix and args.difix_ver2 and i > difix_start_step and is_syn and args.use_percept:
            rgb_pred_patch = rgb_map_ref.reshape(patch_sz, patch_sz, 3).permute(2,0,1).unsqueeze(0)
            tgt_patch = target_rgb.reshape(patch_sz, patch_sz, 3).permute(2,0,1).unsqueeze(0)

            with torch.no_grad():
                feat_tgt = perceptual_net(tgt_patch)
            feat_pred = perceptual_net(rgb_pred_patch)
            # feat_pred = perceptual_net(rgb_pred_patch)
            # feat_tgt = perceptual_net(tgt_patch).detach()

            feat_pred_p = maxpool_feats(feat_pred)
            feat_tgt_p = maxpool_feats(feat_tgt)
            percept_loss = args.w_percept * F.l1_loss(feat_pred_p, feat_tgt_p)

            # percept_loss = args.w_percept * F.mse_loss(feat_pred, feat_tgt)
            
            rgb_pred_post_dy_patch = rgb_map_post_dy.reshape(patch_sz, patch_sz, 3).permute(2,0,1).unsqueeze(0)
            rgb_pred_prev_dy_patch = rgb_map_prev_dy.reshape(patch_sz, patch_sz, 3).permute(2,0,1).unsqueeze(0)
            feat_pred_post_dy = perceptual_net(rgb_pred_post_dy_patch)
            feat_pred_prev_dy = perceptual_net(rgb_pred_prev_dy_patch)

            feat_pred_post_dy_p = maxpool_feats(feat_pred_post_dy)
            feat_pred_prev_dy_p = maxpool_feats(feat_pred_prev_dy)

            percept_loss += args.w_percept * F.l1_loss(feat_pred_post_dy_p, feat_tgt_p)
            percept_loss += args.w_percept * F.l1_loss(feat_pred_prev_dy_p, feat_tgt_p)            

            # percept_loss += args.w_percept * F.l1_loss(feat_pred_post_dy, feat_tgt)
            # percept_loss += args.w_percept * F.l1_loss(feat_pred_prev_dy, feat_tgt)

            percept_loss = percept_loss / 3.
            render_loss = 0. # Do not use render loss in difix step
            
        else:
            percept_loss = None
        
        if FT:
            motion_mask_FT = 1 - target_mask
            render_enh_loss += compute_mse(ret['rgb_map_ref'], target_enh_hdr_rgb, motion_mask_FT)
            # render_enh_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], target_enh_hdr_rgb[:N_rand, ...])

        if args.unit_sf_bidirection:

            sf_cycle_loss = args.w_cycle * compute_mae_unit(ret['raw_sf_ref2post'],  
                                                    -ret['raw_sf_post2ref'], 
                                                    weight_post.unsqueeze(-1), dim=3) 
            sf_cycle_loss += args.w_cycle * compute_mae_unit(ret['raw_sf_ref2prev'], 
                                                        -ret['raw_sf_prev2ref'], 
                                                        weight_prev.unsqueeze(-1), dim=3)
        else:
            sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'],  
                                                    -ret['raw_sf_post2ref'], 
                                                    weight_post.unsqueeze(-1), dim=3) 
            sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                        -ret['raw_sf_prev2ref'], 
                                                        weight_prev.unsqueeze(-1), dim=3)
        
        # regularization loss
        if not args.skip_sf_reg_loss:
            
            render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
            render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)

            sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                        + torch.mean(torch.abs(render_sf_ref2post))) 
        else:
            sf_reg_loss = torch.zeros(1).to(device)

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        if args.decay_depth_w:
            w_depth = args.w_depth/(decay_rate ** divsor)
        else:
            w_depth = args.w_depth

        if args.decay_optical_flow_w:
            w_of = args.w_optical_flow/(decay_rate ** divsor)
        else:
            w_of = args.w_optical_flow
        
        if args.use_difix and args.difix_ver2 and i > difix_start_step and is_syn:

            if args.difix_use_depth == False:
                w_depth = 0.
            elif args.difix_use_depth_weight != 1.:
                w_depth = w_depth * args.difix_use_depth_weight
                print(f'difix_use_depth_weight: {args.difix_use_depth_weight}')
            elif args.difix_use_depth_weight_explicit != 0.:
                w_depth = args.difix_use_depth_weight_explicit
                print(f'difix_use_depth_weight_explicit: {args.difix_use_depth_weight_explicit}')
            w_of = 0.
            
        if args.decay_crf_w:
            w_crf_smooth = args.w_crf_smooth / (decay_rate ** divsor)
            w_crf_monotonic = args.w_crf_monotonic / (decay_rate ** divsor)
        else:
            w_crf_smooth = args.w_crf_smooth
            w_crf_monotonic = args.w_crf_monotonic
        
        if args.decay_traj_flow_w:
            w_tf = args.w_traj_flow/(decay_rate ** divsor)
        else:
            w_tf = args.w_traj_flow

        if args.aligned_depth:
            depth_loss = w_depth * compute_aligned_depth_loss(ret['depth_map_ref_dy'], target_depth)
        elif args.disp_loss == 'depth-anything-o1':
            target_depth = 1./ (target_depth + 1e-5)
            depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'], -target_depth)
        elif args.disp_loss == 'depth-anything-o2':
            depth_loss = w_depth * compute_depth_loss_o2(ret['depth_map_ref_dy'], target_depth)
        else:
            depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'], -target_depth)

        print('w_depth ', w_depth, 'w_of ', w_of)

        if img_i == 0:
            print('only fwd flow')
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
        elif img_i == num_img - 1:
            print('only bwd flow')
            flow_loss = w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)
        else:
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
            flow_loss += w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)

        
        if not args.skip_sf_sm_loss:
            
            if args.multi_camera == True:
                sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            H, W, focal) \
                                        + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal))

                # scene flow least kinectic loss
                sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal)
                sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal)
                
                        # # ======================================  two-frames chain loss ===============================
                if chain_bwd:
                    sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                                ret['raw_pts_ref'], 
                                                                ret['raw_pts_pp'], 
                                                                H, W, focal)

                else:
                    sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                                ret['raw_pts_pp'], 
                                                                ret['raw_pts_ref'], 
                                                                H, W, focal)
            else:
                # scene flow smoothness loss
                sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            H, W, focal) \
                                        + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal))

                # scene flow least kinectic loss
                sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal)
                sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                            ret['raw_pts_post'], 
                                                            ret['raw_pts_prev'], 
                                                            H, W, focal)
                
                        # # ======================================  two-frames chain loss ===============================
                if chain_bwd:
                    sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                                ret['raw_pts_ref'], 
                                                                ret['raw_pts_pp'], 
                                                                H, W, focal)

                else:
                    sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                                ret['raw_pts_pp'], 
                                                                ret['raw_pts_ref'], 
                                                                H, W, focal)
        else:
            sf_sm_loss = torch.zeros(1).to(device)
            
        entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        if chain_5frames:
            # print('5 FRAME RENDER LOSS ADDED') 
            if args.render_l1_l2:
                alpha = getattr(args, 'w_l1_l2', 0.01)
                if args.use_sat_mask:
                    render_loss += mixed_loss_sat(rgb_map_pp_dy, target_rgb, weights_map_dd, sat_mask, alpha)
                else:
                    render_loss += mixed_loss_basic(rgb_map_pp_dy, target_rgb, weights_map_dd, alpha)
            elif args.use_render_l1:
                render_loss += compute_mae(rgb_map_pp_dy, 
                                        target_rgb, 
                                        weights_map_dd)

            else:
                render_loss += compute_mse(rgb_map_pp_dy, 
                                        target_rgb, 
                                        weights_map_dd)
            # if chain_5frames_flow:
            #     if img_i < 2:
            #         print('only fwd traj_i2 flow')
            #         traj_flow_loss = w_tf * compute_mae(render_of_i2_fwd, 
            #                                     target_traj_flow_i2_fwd, 
            #                                     target_seg_mask_i2_fwd)
            #     elif img_i > num_img - 3:
            #         print('only bwd traj_i2 flow')
            #         traj_flow_loss = w_tf * compute_mae(render_of_i2_bwd, 
            #                                     target_traj_flow_i2_bwd, 
            #                                     target_seg_mask_i2_bwd)
            #     else:
            #         traj_flow_loss = w_tf * compute_mae(render_of_i2_fwd, 
            #                                     target_traj_flow_i2_fwd, 
            #                                     target_seg_mask_i2_fwd)
            #         traj_flow_loss += w_tf * compute_mae(render_of_i2_bwd, 
            #                                     target_traj_flow_i2_bwd, 
            #                                     target_seg_mask_i2_bwd)

        if args.use_difix and args.difix_ver2 and i > difix_start_step and is_syn and args.difix_w_render!=1.:
            # if difix iteration, reduce weight of photometric loss
            print(f'difix_w_render: {args.difix_w_render}')
            render_loss = args.difix_w_render * render_loss

        if args.w_render != 1.:
            print(f'w_render: {args.difix_w_render}')
            render_loss = args.w_render * render_loss

        loss = sf_reg_loss + sf_cycle_loss + \
               render_loss + flow_loss + \
               sf_sm_loss + prob_reg_loss + \
               depth_loss + entropy_loss 
               
        if FT:
            loss += render_enh_loss
            print('render_enh_loss', render_enh_loss.item())
        
        print('render_loss ', render_loss.item(), 
              ' bidirection_loss ', sf_cycle_loss.item(), 
              ' sf_reg_loss ', sf_reg_loss.item())
        print('depth_loss ', depth_loss.item(), 
              ' flow_loss ', flow_loss.item(), 
              ' sf_sm_loss ', sf_sm_loss.item())
        print('prob_reg_loss ', prob_reg_loss.item(),
              ' entropy_loss ', entropy_loss.item())
        
        
        if args.use_tone_mapping and global_step <= crf_stop_step and not camparam_frozen:
            if optim_cam_crf is not None:
                crf_smooth_loss = w_crf_smooth * Cam_param.crf_smoothness_loss()
                loss = loss + crf_smooth_loss
            else:
                crf_smooth_loss = None    
            
            if args.unit_loss :
                unit_loss = Cam_param.point_constraint()
                if unit_loss is not None :
                    loss = loss + unit_loss * args.w_unit
                    
            if args.crf_monotonic:
                crf_monotonic_loss = w_crf_monotonic * Cam_param.crf_monotonicity_loss()
                loss = loss + crf_monotonic_loss
            else:
                crf_monotonic_loss = None
        else:
            crf_smooth_loss = None
            crf_monotonic_loss = None
            
        if crf_smooth_loss is not None:
            print('crf_smooth_loss ', crf_smooth_loss.item())
        if crf_monotonic_loss is not None:
            print('crf_monotonic_loss ', crf_monotonic_loss.item())
        
        if percept_loss is not None:
            loss = loss + percept_loss
                
        loss.backward()
        optimizer.step()

        if args.use_tone_mapping and (not freeze_cam_this_iter) and (not Cam_param.is_frozen_forever()):
            optim_cam_wb.step()
            if optim_cam_crf is not None:
                optim_cam_crf.step()
        # if args.use_tone_mapping and not camparam_frozen: # FIXME
        #     # optim_cam.step()
        #     optim_cam_wb.step()
        #     if optim_cam_crf is not None:
        #         optim_cam_crf.step()

        # Check Training by wandb
        
        if not debug_mode:
            wandb.log({
                'loss': loss,
                'render_loss': render_loss.item(),
                'bidirection_loss': sf_cycle_loss.item(),
                'sf_reg_loss': sf_reg_loss.item(),
                'depth_loss': depth_loss.item(),
                'traj_loss' if args.traj_flow else 'flow_loss': flow_loss.item(),
                'sf_sm_loss': sf_sm_loss.item(),
                'prob_reg_loss': prob_reg_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'crf_smoothness_loss': crf_smooth_loss.item() if crf_smooth_loss is not None else None,
                'crf_monotonicity_loss': crf_monotonic_loss.item() if crf_monotonic_loss is not None else None,
                'render_enh_loss': render_enh_loss.item() if FT else None,
                'perceptual_loss': percept_loss.item() if percept_loss is not None else None,
            }, step=i)
        
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        if args.stage == 2:
            new_lrate = 1e-3
        else:
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            
            if args.use_tone_mapping and (not Cam_param.is_frozen_forever()):
                if optim_cam_crf is not None:
                    for pg in optim_cam_crf.param_groups:
                        if args.share_crf:
                            pg['lr'] = new_lrate * args.share_crf_lr_ratio
                        else:
                            pg['lr'] = new_lrate
                for pg in optim_cam_wb.param_groups:
                    pg['lr'] = new_lrate
            # if args.use_tone_mapping and not camparam_frozen:
            #     if optim_cam_crf is not None:
            #         for param_group in optim_cam_crf.param_groups:
            #             param_group['lr'] = new_lrate
            #     for param_group in optim_cam_wb.param_groups:
            #         param_group['lr'] = new_lrate
        
        if args.use_tone_mapping and args.tone_mapping == 'piece_wise':
            if optim_cam_crf is None and global_step > args.crf_start_step:
                print(f'[Step {global_step}] Adding CRF params to optimizer.')
                if args.share_crf:
                    optim_cam_crf = torch.optim.Adam(params=Cam_param.crf_params,
                                                     lr=new_lrate / num_img , betas=(0.9, 0.999))
                else:     
                    optim_cam_crf = torch.optim.Adam(params=Cam_param.crf_params,
                                                    lr = new_lrate, betas=(0.9, 0.999))
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            
            if args.use_tone_mapping_camparam == True:
                
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'optim_cam_state_dict': optim_cam.state_dict(),
                    'optim_cam_crf_state_dict': optim_cam_crf.state_dict() if optim_cam_crf is not None else None,
                    'optim_cam_wb_state_dict': optim_cam_wb.state_dict() if optim_cam_wb is not None else None,
                    'white_balance': Cam_param.wb.state_dict(),
                    'CRF': Cam_param.CRF.state_dict() if Cam_param.CRF is not None else None,
                    'rgb_model': Cam_param.rgb_model.state_dict() if args.tone_mapping == 'nn' else None,
                    'color_refiner' : render_kwargs_train['color_refiner'].state_dict() if args.stage == 2 else None,
                }, path)
            
            else:
                if args.N_importance > 0:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)

            print('Saved checkpoints at', path)


        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/flow_loss", flow_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)

            writer.add_scalar("train/sf_reg_loss", sf_reg_loss.item(), i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss", sf_sm_loss.item(), i)
        
        if i % 100 == 0 and args.use_tone_mapping and Cam_param is not None:
            print(f'[Iter {i}] is_syn={int((args.use_difix and args.difix_ver2 and i>difix_start_step) and (is_syn!=0))} '
                  f'WB_grad={grad_on(Cam_param.wb)} '
                  f'CRF_grad={grad_on(Cam_param.CRF)} '
                  f'freeze_iter={Cam_param.is_frozen_this_iter()} '
                  f'freeze_forever={Cam_param.is_frozen_forever()}')

        if i%args.i_img== 0 and not debug_mode:
            # img_i = np.random.choice(i_val)
            # if i % 3 == 0:
            #     img_i = 10
            # elif i % 3 == 1:
            #     img_i = 20
            # else:
            #     img_i = N-3
            img_idx_embed = img_i/num_img * 2. - 1.0
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            target_depth = depths[img_i] - torch.min(depths[img_i])

            with torch.no_grad():
                ret = render(img_idx_embed, 
                             chain_bwd, False,
                             num_img, H, W, focal, 
                             chunk=1024*16, 
                             c2w=pose,
                             **render_kwargs_test)

                if args.use_tone_mapping == True:
                    wandb.log({"rgb_map_ref_ldr": wandb.Image((torch.clamp(Cam_param.RAD2LDR_img(ret['rgb_map_ref'], img_i), 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                    wandb.log({"rgb_map_ref": wandb.Image((torch.clamp(torch.log1p(ret['rgb_map_ref'] * 50) / torch.log1p(torch.tensor(50, device=device)), 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                    wandb.log({"wb": wandb.Image(Cam_param.get_wb_plot())}, step=i)
                    if args.share_crf == True:
                        wandb.log({"share_crf": wandb.Image(Cam_param.get_visualize_crf(0))}, step=i)
                    elif args.tone_mapping == 'piece_wise':
                        wandb.log({"crf": wandb.Image(Cam_param.get_visualize_crf(img_i))}, step=i)
                    elif args.tone_mapping == 'hdr_hexplane':
                        wandb.log({"crf": wandb.Image(Cam_param.get_visualize_gamma())}, step=i)
                    elif args.tone_mapping == 'nn':
                        wandb.log({"crf": wandb.Image(Cam_param.get_visualize_crf_nn())}, step=i)

                    mse   = torch.mean((torch.clamp(Cam_param.RAD2LDR_img(ret['rgb_map_ref'], img_i), 0., 1.) - target.to(ret['rgb_map_ref'].device)) ** 2)
                    psnr  = -10. * torch.log10(mse + 1e-8) 
                    wandb.log({"psnr": psnr.item()}, step=i)
                        
                else:
                    # Clamp and log rgb_map_ref
                    wandb.log({"rgb_map_ref": wandb.Image((torch.clamp(ret['rgb_map_ref'], 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                    mse   = torch.mean((ret['rgb_map_ref'] - target.to(ret['rgb_map_ref'].device)) ** 2)
                    psnr  = -10. * torch.log10(mse + 1e-8) 
                    wandb.log({"psnr": psnr.item()}, step=i)
                
                # Normalize and log depth_map_ref
                wandb.log({'depth_map_ref': wandb.Image(cv2.applyColorMap((normalize_depth(ret['depth_map_ref']).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_PLASMA))}, step=i)
                
                # Clamp and log rgb_map_rig
                wandb.log({"rgb_map_rig": wandb.Image((torch.clamp(ret['rgb_map_rig'], 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                
                # Normalize and log depth_map_rig
                wandb.log({"depth_map_rig": wandb.Image(cv2.applyColorMap((normalize_depth(ret['depth_map_rig']).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_PLASMA))}, step=i)
                
                # Clamp and log rgb_map_ref_dy
                wandb.log({"rgb_map_ref_dy": wandb.Image((torch.clamp(ret['rgb_map_ref_dy'], 0., 1.)*255).cpu().numpy().astype(np.uint8))}, step=i)
                
                # Normalize and log depth_map_ref_dy
                wandb.log({"depth_map_ref_dy": wandb.Image(cv2.applyColorMap((normalize_depth(ret['depth_map_ref_dy']).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_PLASMA))}, step=i)
                
                # Log target ground truth image
                wandb.log({"gt_rgb": wandb.Image((torch.clamp(target, 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                
                # Clamp and log monocular_disp
                monocular_disp = cv2.applyColorMap((torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                wandb.log({"monocular_disp": wandb.Image(monocular_disp, caption="Monocular Disp")}, step=i)
                
                # Log weights_map_dd
                # wandb.log({"weights_map_dd": wandb.Image(ret['weights_map_dd'], caption="Weights Map DD")}, step=i)
                wandb.log({"weights_map_dd": wandb.Image(ret['weights_map_dd'].cpu().numpy(), caption="Weights Map DD")}, step=i)
                                
                if args.use_motion_mask_weights: 
                    wandb.log({"motion_mask": wandb.Image((torch.clamp(masks[img_i], 0., 1.) * 255).cpu().numpy().astype(np.uint8))}, step=i)
                
                writer.add_image("val/rgb_map_ref", torch.clamp(ret['rgb_map_ref'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref", normalize_depth(ret['depth_map_ref']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_rig", torch.clamp(ret['rgb_map_rig'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_rig", normalize_depth(ret['depth_map_rig']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_ref_dy", torch.clamp(ret['rgb_map_ref_dy'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref_dy", normalize_depth(ret['depth_map_ref_dy']), 
                                global_step=i, dataformats='HW')

                # writer.add_image("val/rgb_map_pp_dy", torch.clamp(ret['rgb_map_pp_dy'], 0., 1.), 
                                # global_step=i, dataformats='HWC')

                writer.add_image("val/gt_rgb", target, 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/monocular_disp", 
                                torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/weights_map_dd", 
                                 ret['weights_map_dd'], 
                                 global_step=i, 
                                 dataformats='HW')

            # torch.cuda.empty_cache()

        global_step += 1
    if not debug_mode:
        wandb.finish()


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()