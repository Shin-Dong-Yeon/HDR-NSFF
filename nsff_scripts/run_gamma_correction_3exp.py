import cv2
import numpy as np
import os
from glob import glob

def get_gamma_amount(source, reference):
    """source 이미지를 reference 이미지 밝기에 맞추기 위한 gamma 값 계산"""
    src_n = source.astype(np.float32) / 255.0 + 1e-6
    ref_n = reference.astype(np.float32) / 255.0 + 1e-6
    
    # ln(Ref) = (1/gamma) * ln(Src) => 1/gamma = mean(ln(Ref)) / mean(ln(Src))
    gamma_inv = np.mean(np.log(ref_n)) / np.mean(np.log(src_n))
    return gamma_inv

def apply_gamma(image, gamma_inv):
    """계산된 gamma_inv를 이미지에 적용"""
    out = np.power(image.astype(np.float32) / 255.0, gamma_inv)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

# 설정: 경로 지정
base_path = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/i2/big_jump_1/dense_gamma/images'
output_path = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/i2/big_jump_1/dense_gamma/images_cor'
os.makedirs(output_path, exist_ok=True)

# png 파일만 가져와서 정렬 (00000, 00001, ...)
file_list = sorted(glob(os.path.join(base_path, '*.png')))

# "_g_"가 붙은 결과 파일들이 이미 있다면 리스트에서 제외 (중복 처리 방지)
file_list = [f for f in file_list if not os.path.basename(f).startswith('_g_')]

print(f"총 {len(file_list)}개의 파일을 찾았습니다. {len(file_list)//3}개 그룹을 처리합니다.")

# 3개씩 묶어서 처리
for i in range(0, len(file_list) - 2, 3):
    # 파일 경로 매칭 (0: Mid, 1: Low, 2: High)
    mid_path  = file_list[i]
    low_path  = file_list[i+1]
    high_path = file_list[i+2]
    
    print(f"Processing group: {os.path.basename(mid_path)}, {os.path.basename(low_path)}, {os.path.basename(high_path)}")
    
    # 이미지 로드
    img_mid  = cv2.imread(mid_path)
    img_low  = cv2.imread(low_path)
    img_high = cv2.imread(high_path)
    
    # 1. Low -> Mid 보정
    g_inv_low = get_gamma_amount(img_low, img_mid)
    corrected_low = apply_gamma(img_low, g_inv_low)
    
    # 2. High -> Mid 보정
    g_inv_high = get_gamma_amount(img_high, img_mid)
    corrected_high = apply_gamma(img_high, g_inv_high)
    
    # 3. 결과 저장 (Mid는 보정이 필요 없으므로 원본 복사 혹은 _g_ 이름으로 저장)
    # 원본 파일명 앞에 _g_를 붙여 저장합니다.
    cv2.imwrite(os.path.join(output_path, os.path.basename(mid_path)), img_mid)
    cv2.imwrite(os.path.join(output_path, os.path.basename(low_path)), corrected_low)
    cv2.imwrite(os.path.join(output_path, os.path.basename(high_path)), corrected_high)

print("모든 프레임의 감마 보정이 완료되었습니다.")