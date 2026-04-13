import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

gt_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/logs/_gopro/250913/bag_swing_3_1_i2_tf_hdr_250914_F00-36/render-train_250001/hdr_mertens'
pred_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/hdr_video/lanhdr_bag_swing_3_1_i2/tm'

loss_fn_vgg = lpips.LPIPS(net='alex').cuda() # GPU 사용 시 .cuda()

def im2tensor(image):
    # 이미지를 [0, 255] -> [-1, 1] 범위의 Torch Tensor로 변환 (LPIPS 입력 규격)
    image = torch.Tensor(image).permute(2, 0, 1) # HWC to CHW
    image = (image / 127.5) - 1.0
    return image.unsqueeze(0).cuda()

# 결과 저장을 위한 리스트
psnr_values = []
ssim_values = []
lpips_values = []

print(f"{'Step':<10} | {'PSNR':<10} | {'SSIM':<10} | {'LPIPS':<10}")
print("-" * 50)

# 3. 루프 실행 (00000.png ~ 00035.png)
for i in range(2,33):
    img_name = f'{i:05d}.png'
    
    gt_path = os.path.join(gt_dir, img_name)
    pred_path = os.path.join(pred_dir, img_name)
    
    # 이미지 로드 (OpenCV: BGR -> RGB 변환)
    img_gt = cv2.imread(gt_path)
    img_pred = cv2.imread(pred_path)
    
    if img_gt is None or img_pred is None:
        print(f"Skip {img_name}: File not found.")
        continue

    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

    # PSNR 계산
    cur_psnr = psnr(img_gt, img_pred, data_range=255)
    psnr_values.append(cur_psnr)

    # SSIM 계산 (채널 축 지정 필수)
    cur_ssim = ssim(img_gt, img_pred, channel_axis=2, data_range=255)
    ssim_values.append(cur_ssim)

    # LPIPS 계산
    with torch.no_grad():
        t_gt = im2tensor(img_gt)
        t_pred = im2tensor(img_pred)
        cur_lpips = loss_fn_vgg(t_gt, t_pred).item()
        lpips_values.append(cur_lpips)

    print(f"{img_name:<10} | {cur_psnr:>10.4f} | {cur_ssim:>10.4f} | {cur_lpips:>10.4f}")

# 4. 최종 결과 출력
print("-" * 50)
print(f"AVERAGE    | {np.mean(psnr_values):>10.4f} | {np.mean(ssim_values):>10.4f} | {np.mean(lpips_values):>10.4f}")