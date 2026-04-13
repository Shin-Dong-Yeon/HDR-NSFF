
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
import glob

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
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
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

    
    
    selected_layers_all = ['3', '8', '17', '26', '35']
    selected_layers_high = ['17', '26', '35']
    selected_layers_high_2 = ['26', '35']
    selected_layers_high_17_26 = ['17','26']
    # selected_layers_low = ['3', '8', '17']
    selected_layers_low = ['3', '8']
    selected_layers_low_2 = ['3']
    selected_layers_low_3 = ['1', '3', '6', '8']
    
    transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    
    
    
    ALL = True
    if ALL:       
        
        path_hdr = sorted(glob.glob(os.path.join(basedir, expname, 'render-train_250001', 'hdr_exr' , '*exr')))
        path_ldr = sorted(glob.glob(os.path.join('/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/big_jump_1_36/dense/images_641x360/', '*png')))
        
        k = 0
        
        for hdr_file, ldr_file in zip(path_hdr, path_ldr):
            print(hdr_file)
            print(ldr_file)
            
            H_img = load_hdr_exr(hdr_file)
            H_img = torch.from_numpy(H_img).float().to(device)
            L_img = load_image(ldr_file)
            
            L_img = transforms(L_img).to(device)
            
            save_dir = os.path.join(basedir, expname, 'perceptual_sigmoid_high_17_16_norm_all')
            # os.makedirs(save_dir, exist_ok=True)
            
            opt_dir = os.path.join(save_dir, 'opt')
            os.makedirs(opt_dir, exist_ok=True)
            origin_dir = os.path.join(save_dir, 'origin')
            os.makedirs(origin_dir, exist_ok=True)

            
            feature_extractor = VGGFeatures(selected_layers_high_17_26).to(device)
            H_opt = optimize_image_sigmoid(H_img, Cam_param, k, L_img, feature_extractor, num_steps=300, lr=0.01)
            filename = os.path.join(opt_dir, f'{k:05d}.png')
            mu = torch.tensor(50, dtype=H_opt.dtype, device=H_opt.device)
            H_opt_tm = torch.log(1 + mu * H_opt) / torch.log(1 + mu)
            iio.imwrite(filename, (H_opt_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            filename_origin = os.path.join(origin_dir, f'{k:05d}.png')
            mu = torch.tensor(50, dtype=H_img.dtype, device=H_img.device)
            H_img_tm = torch.log(1 + mu * H_img) / torch.log(1 + mu)
            iio.imwrite(filename_origin, (H_img_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            k += 1
                        
            # filename_low_3_mid = os.path.join(save_dir, 'H_opt_high_3_mid.png')
            # iio.imwrite(filename_low_3_mid, (Cam_param.RAD2LDR_img(H_opt_low_3, k+1).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            # filename_low_3_low = os.path.join(save_dir, 'H_opt_high_3_low.png')
            # iio.imwrite(filename_low_3_low, (Cam_param.RAD2LDR_img(H_opt_low_3, k-1).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            # filename_low_3_tm = os.path.join(save_dir, 'H_opt_high_3_tm.png')
            # mu = torch.tensor(50, dtype=H_opt_low_3.dtype, device=H_opt_low_3.device)
            # H_opt_low_3_tm = torch.log(1 + mu * H_opt_low_3) / torch.log(1 + mu)
            # iio.imwrite(filename_low_3_tm, (H_opt_low_3_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            # filename_origin = os.path.join(save_dir, 'H_opt_origin.png')
            # iio.imwrite(filename_origin, (Cam_param.RAD2LDR_img(H_img, 23).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            
            # filename_origin_tm = os.path.join(save_dir, 'H_origin_tm.png')
            # mu = torch.tensor(50, dtype=H_img.dtype, device=H_img.device)
            # H_img_tm = torch.log(1 + mu * H_img) / torch.log(1 + mu)
            # iio.imwrite(filename_origin_tm, (H_img_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
    
    else:
        path_hdr = '/home/shindy/projects/hdr-4d/hdr-nsff/nsff_exp/logs/big_jump_1_36_tf_hdr_250217_F00-36/render-train_250001/hdr_exr/00023.exr'
        path_ldr = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset/gopro/250213/big_jump_1_36/dense/images_641x360/00023.png'
        
        H_img = load_hdr_exr(path_hdr)
        H_img = torch.from_numpy(H_img).float().to(device)
        L_img = load_image(path_ldr)
        
        transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        L_img = transforms(L_img).to(device)
        
    
        percept_dir_name = '00023_perceptual_sigmoid_high_17_26_norm'
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
        
        ### importatnt!! ####################################################################
        
        feature_extractor_high = VGGFeatures(selected_layers_high_17_26).to(device)
        H_opt_low_3 = optimize_image_sigmoid(H_img, Cam_param, 23, L_img, feature_extractor_high, num_steps=300, lr=0.01)
        filename_low_3 = os.path.join(basedir, expname, percept_dir_name, 'H_opt_high_3.png')
        iio.imwrite(filename_low_3, (Cam_param.RAD2LDR_img(H_opt_low_3, 23).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        filename_low_3_mid = os.path.join(basedir, expname, percept_dir_name, 'H_opt_high_3_mid.png')
        iio.imwrite(filename_low_3_mid, (Cam_param.RAD2LDR_img(H_opt_low_3, 24).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        filename_low_3_low = os.path.join(basedir, expname, percept_dir_name, 'H_opt_high_3_low.png')
        iio.imwrite(filename_low_3_low, (Cam_param.RAD2LDR_img(H_opt_low_3, 22).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        filename_low_3_tm = os.path.join(basedir, expname, percept_dir_name, 'H_opt_high_3_tm.png')
        mu = torch.tensor(50, dtype=H_opt_low_3.dtype, device=H_opt_low_3.device)
        H_opt_low_3_tm = torch.log(1 + mu * H_opt_low_3) / torch.log(1 + mu)
        iio.imwrite(filename_low_3_tm, (H_opt_low_3_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        filename_origin = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin.png')
        iio.imwrite(filename_origin, (Cam_param.RAD2LDR_img(H_img, 23).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        filename_origin_tm = os.path.join(basedir, expname, percept_dir_name, 'H_origin_tm.png')
        mu = torch.tensor(50, dtype=H_img.dtype, device=H_img.device)
        H_img_tm = torch.log(1 + mu * H_img) / torch.log(1 + mu)
        iio.imwrite(filename_origin_tm, (H_img_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        ##########################################################################################
        
        # feature_extractor_low_3 = VGGFeatures(selected_layers_low_3).to(device)
        # H_opt_low_3 = optimize_masked_image_sigmoid(H_img, Cam_param, 8, L_img, feature_extractor_low_3, num_steps=300, lr=0.01)
        # filename_low_3 = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3.png')
        # iio.imwrite(filename_low_3, (Cam_param.RAD2LDR_img(H_opt_low_3, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_low_3_mid = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_mid.png')
        # iio.imwrite(filename_low_3_mid, (Cam_param.RAD2LDR_img(H_opt_low_3, 9).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_low_3_low = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_low.png')
        # iio.imwrite(filename_low_3_low, (Cam_param.RAD2LDR_img(H_opt_low_3, 7).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_low_3_tm = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3_tm.png')
        # mu = torch.tensor(50, dtype=H_opt_low_3.dtype, device=H_opt_low_3.device)
        # H_opt_low_3_tm = torch.log(1 + mu * H_opt_low_3) / torch.log(1 + mu)
        # iio.imwrite(filename_low_3_tm, (H_opt_low_3_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_origin = os.path.join(basedir, expname, percept_dir_name, 'H_opt_origin.png')
        # iio.imwrite(filename_origin, (Cam_param.RAD2LDR_img(H_img, 8).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_origin_tm = os.path.join(basedir, expname, percept_dir_name, 'H_origin_tm.png')
        # mu = torch.tensor(50, dtype=H_img.dtype, device=H_img.device)
        # H_img_tm = torch.log(1 + mu * H_img) / torch.log(1 + mu)
        # iio.imwrite(filename_origin_tm, (H_img_tm.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
        
        # filename_opt_hdr = os.path.join(basedir, expname, percept_dir_name, 'H_opt_low_3.exr')
        # iio.imwrite(filename_opt_hdr, H_opt_low_3.detach().cpu().numpy())
        

        # save_image(Cam_param.RAD2LDR_img(H_opt_all, 8), filename_all)
        # save_image(Cam_param.RAD2LDR_img(H_opt_high, 8), filename_high)
        