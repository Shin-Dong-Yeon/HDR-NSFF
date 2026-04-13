
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
import imageio.v3 as iio
import cam_param
import os
import numpy as np
import torch.nn.functional as F

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



def saturation_mask_(color, threshold_low=0.15, threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color <= threshold_high) & (color >= threshold_low)] = 1
    mask[(color < threshold_low)] = torch.pow((color[(color < threshold_low)] + threshold_low) / (2 * threshold_low), 2)
    mask[(color > threshold_high)] = torch.pow((color[(color > threshold_high)] - 2 + threshold_high) / (2 * (1 - threshold_high)), 2)
    return mask

def saturation_mask(color, threshold_low=0.15, threshold_high=0.9):
    
    gray = color.mean(dim=0)
    mask = torch.zeros_like(gray)
    mask[(gray <= threshold_high) & (gray >= threshold_low)] = 1
    mask[(gray < threshold_low)] = torch.pow((gray[(gray < threshold_low)] + threshold_low) / (2 * threshold_low), 2)
    mask[(gray > threshold_high)] = torch.pow((gray[(gray > threshold_high)] - 2 + threshold_high) / (2 * (1 - threshold_high)), 2)
    
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def compute_masked_content_loss(features_generated, features_target, mask):
    loss = 0.0    
    # import pdb; pdb.set_trace()
    for key in features_target.keys():
        # feature map 해상도에 맞게 mask를 리사이즈합니다.
        mask_resized = F.interpolate(mask, size=features_target[key].shape[-2:], mode='bilinear', align_corners=False)
        loss += F.mse_loss(features_generated[key] * mask_resized,
                           features_target[key] * mask_resized)
    return loss

def load_image(path, transform=None, max_size=None, shape=None):
    """
    이미지를 PIL 형식으로 로드한 후, transform을 적용하여 텐서로 변환합니다.
    max_size 또는 shape가 주어지면 해당 크기로 리사이즈합니다.
    """
    image = Image.open(path).convert('RGB')
    if max_size is not None:
        size = max(image.size)
        if size > max_size:
            image = transforms.Resize(max_size)(image)
    if shape is not None:
        image = transforms.Resize(shape)(image)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def load_hdr_exr(filepath):
    """OpenCV를 사용해 .exr HDR 이미지를 로드하는 함수"""
    # hdr_image = iio.imread(filepath, format="exr")  # OpenEXR 파일 읽기
    hdr_image = iio.imread(filepath)
    # return cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB) 
    return hdr_image

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

def optimize_image(H_init, Cam_param, index, L, feature_extractor, num_steps=300, lr=0.01):
    # import pdb; pdb.set_trace()
    H_opt = H_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([H_opt], lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        H_tm=Cam_param.RAD2LDR_img(H_opt, index)       
        features_generated = feature_extractor(H_tm.permute(2, 0, 1).unsqueeze(0))
        features_target = feature_extractor(L)
        loss = compute_content_loss(features_generated, features_target)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f'step {step}: loss={loss.item():.4f}')
            
    return H_opt

def optimize_masked_image(H_init, Cam_param, index, L, feature_extractor, num_steps=300, lr=0.01):
    H_opt = H_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([H_opt], lr=lr)
    
    # L은 (1, C, H, W) 형태의 텐서여야 하며, [0,1] 범위로 normalize 되어있다고 가정합니다.
    # 제공된 saturation_mask 함수를 사용해 L 이미지의 saturation mask를 계산합니다.
    sat_mask = saturation_mask(L)  # shape: (1, C, H, W)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        H_tm = Cam_param.RAD2LDR_img(H_opt, index)       
        # VGG의 입력 형식에 맞게 (H, W, C) -> (1, C, H, W)
        features_generated = feature_extractor(H_tm.permute(2, 0, 1).unsqueeze(0))
        features_target = feature_extractor(L)
        
        loss = compute_masked_content_loss(features_generated, features_target, sat_mask)
        loss.backward()
        optimizer.step()
        
        # [0,1] 범위 유지 (옵션)
        with torch.no_grad():
            H_opt.clamp_(0, 1)
        
        if step % 50 == 0:
            print(f'step {step}: loss={loss.item():.4f}')
            
    return H_opt

def optimize_image_sigmoid(H_init, Cam_param, index, L, feature_extractor, num_steps=300, lr=0.01):
    # import pdb; pdb.set_trace()
    # H_img가 [0,1] 범위라고 가정하면, inverse sigmoid를 취해 theta를 초기화합니다.
    theta_init = torch.log(H_init / (1 - H_init))
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([theta], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        # 항상 [0,1] 범위의 이미지 생성
        H_opt = torch.sigmoid(theta)
        H_tm = Cam_param.RAD2LDR_img(H_opt, index)
        features_generated = feature_extractor(H_tm.permute(2, 0, 1).unsqueeze(0))
        features_target = feature_extractor(L)
        loss = compute_content_loss(features_generated, features_target)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f'step {step}: loss={loss.item():.4f}')

    # 최종적으로 H_opt를 생성합니다.
    H_opt = torch.sigmoid(theta)
            
    return H_opt

def optimize_masked_image_sigmoid(H_init, Cam_param, index, L, feature_extractor, num_steps=300, lr=0.01):
    # import pdb; pdb.set_trace()
    # H_img가 [0,1] 범위라고 가정하면, inverse sigmoid를 취해 theta를 초기화합니다.
    theta_init = torch.log(H_init / (1 - H_init))
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([theta], lr=lr)
    import pdb; pdb.set_trace()
    
    sat_mask = saturation_mask(L)  # shape: (1, 1, H, W)
    iio.imwrite('sat_mask.png', (sat_mask.squeeze(0).squeeze(0).detach().cpu().numpy()*255).astype(np.uint8))

    # iio.imwrite((sat_mask.squeeze(0).squeeze(0).detach().cpu().numpy()*255).astype(np.uint8), 'sat_mask.png')
    

    for step in range(num_steps):
        optimizer.zero_grad()
        # 항상 [0,1] 범위의 이미지 생성
        H_opt = torch.sigmoid(theta)
        H_opt.register_hook(lambda grad: grad*sat_mask.squeeze(0).squeeze(0).unsqueeze(-1))
        H_tm = Cam_param.RAD2LDR_img(H_opt, index)
        features_generated = feature_extractor(H_tm.permute(2, 0, 1).unsqueeze(0))
        features_target = feature_extractor(L)
        loss = compute_masked_content_loss(features_generated, features_target, sat_mask)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f'step {step}: loss={loss.item():.4f}')

    # 최종적으로 H_opt를 생성합니다.
    H_opt = torch.sigmoid(theta)
            
    return H_opt


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path_hdr = '/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/logs/big_jump_1_36_tf_hdr_250217_F00-36/render-train_250001/hdr_exr/00008.exr'
    path_ldr = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/big_jump_1_36/dense/images_641x360/00008.png'
    
    # 

    # #FIXME: 이 부분 고쳐야함!
    N, H, W, C = 36, 360, 641, 3
    images = torch.zeros(N, H, W, C).to(device)
    Cam_param = cam_param.CamParam(N, H, W, device=device, gts=images, initialize=False)
    # optim_cam = Cam_param.optimizer(l_rate = args.lrate)
    optim_cam_wb, optim_cam_crf = Cam_param.optimizer(l_rate = 0.01)

    basedir = '/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/logs/'
    expname = 'big_jump_1_36_tf_hdr_250217_F00-36'
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        
        print('Reloading CamParam from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        Cam_param.load_ckpt(ckpt)
        
    for param in Cam_param.wb.parameters():
        param.requires_grad = False
    for param in Cam_param.CRF.parameters():
        param.requires_grad = False

    H_img = load_hdr_exr(path_hdr)
    H_img = torch.from_numpy(H_img).float().to(device)
    L_img = load_image(path_ldr)
    
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    L_img = transforms(L_img).to(device)
    
    selected_layers_all = ['3', '8', '17', '26', '35']
    selected_layers_high = ['17', '26', '35']
    # selected_layers_low = ['3', '8', '17']
    selected_layers_low = ['3', '8']
    selected_layers_low_2 = ['3']
    selected_layers_low_3 = ['1', '3', '6', '8']
    
    percept_dir_name = 'perceptual_sigmoid_masked_3_50'
    os.makedirs(os.path.join(basedir, expname, percept_dir_name), exist_ok=True)
    
    # feature_extractor_all = VGGFeatures(selected_layers_all).to(device)
    # print(">> Optimizing with all layers (Variant 1)")
    # H_opt_all = optimize_image(H_img, Cam_param, 8, L_img, feature_extractor_all, num_steps=300, lr=0.01)
    # filename_all = os.path.join(basedir, expname, percept_dir_name, 'H_opt_all.png')
    # iio.imwrite(filename_all, (Cam_param.RAD2LDR_img(H_opt_all, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # feature_extractor_high = VGGFeatures(selected_layers_high).to(device)
    # print(">> Optimizing with high layers (Variant 2)")
    # H_opt_high = optimize_image(H_img, Cam_param, 8, L_img, feature_extractor_high, num_steps=300, lr=0.01)
    # filename_high = os.path.join(basedir, expname, percept_dir_name, 'H_opt_high.png')
    # iio.imwrite(filename_high, (Cam_param.RAD2LDR_img(H_opt_high, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # feature_extractor_low = VGGFeatures(selected_layers_low).to(device)
    # print(">> Optimizing with high layers (Variant 3)")
    # H_opt_low = optimize_image(H_img, Cam_param, 8, L_img, feature_extractor_low, num_steps=300, lr=0.01)
    # filename_low = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low.png')
    # iio.imwrite(filename_low, (Cam_param.RAD2LDR_img(H_opt_low, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
    # feature_extractor_low_2 = VGGFeatures(selected_layers_low_2).to(device)
    # H_opt_low_2 = optimize_image(H_img, Cam_param, 8, L_img, feature_extractor_low_2, num_steps=300, lr=0.01)

    
    # filename_low_2 = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_2.png')
    # iio.imwrite(filename_low_2, (Cam_param.RAD2LDR_img(H_opt_low_2, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    # filename_low_2_mid = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_2_mid.png')
    # iio.imwrite(filename_low_2_mid, (Cam_param.RAD2LDR_img(H_opt_low_2, 9).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # filename_low_2_low = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_2_low.png')
    # iio.imwrite(filename_low_2_low, (Cam_param.RAD2LDR_img(H_opt_low_2, 7).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # filename_origin = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin_h.png')
    # iio.imwrite(filename_origin, (Cam_param.RAD2LDR_img(H_img, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # filename_origin_m = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin_m.png')
    # iio.imwrite(filename_origin_m, (Cam_param.RAD2LDR_img(H_img, 9).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # filename_origin_l = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin_l.png')
    # iio.imwrite(filename_origin_l, (Cam_param.RAD2LDR_img(H_img, 7).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    feature_extractor_low_3 = VGGFeatures(selected_layers_low_3).to(device)
    H_opt_low_3 = optimize_masked_image_sigmoid(H_img, Cam_param, 8, L_img, feature_extractor_low_3, num_steps=300, lr=0.01)
    filename_low_3 = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3.png')
    iio.imwrite(filename_low_3, (Cam_param.RAD2LDR_img(H_opt_low_3, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    filename_low_3_mid = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_mid.png')
    iio.imwrite(filename_low_3_mid, (Cam_param.RAD2LDR_img(H_opt_low_3, 9).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    filename_low_3_low = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_low.png')
    iio.imwrite(filename_low_3_low, (Cam_param.RAD2LDR_img(H_opt_low_3, 7).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    filename_low_3_tm = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_tm.png')
    mu = torch.tensor(50, dtype=H_opt_low_3.dtype, device=H_opt_low_3.device)
    H_opt_low_3_tm = torch.log(1 + mu * H_opt_low_3) / torch.log(1 + mu)
    iio.imwrite(filename_low_3_tm, (H_opt_low_3_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    filename_origin = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin.png')
    iio.imwrite(filename_origin, (Cam_param.RAD2LDR_img(H_img, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    filename_origin_tm = os.path.join(basedir, expname, percept_dir_name, 'H_origin_tm.png')
    mu = torch.tensor(50, dtype=H_img.dtype, device=H_img.device)
    H_img_tm = torch.log(1 + mu * H_img) / torch.log(1 + mu)
    iio.imwrite(filename_origin_tm, (H_img_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
    
    # filename_opt_hdr = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3.exr')
    # iio.imwrite(filename_opt_hdr, H_opt_low_3.detach().cpu().numpy())
    

    # save_image(Cam_param.RAD2LDR_img(H_opt_all, 8), filename_all)
    # save_image(Cam_param.RAD2LDR_img(H_opt_high, 8), filename_high)
    