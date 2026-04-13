import cv2
import numpy as np
import os

# 1. 경로 설정
input_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset_gpu/gopro/250914/bear_thread/i4/dense_gamma/images'
output_dir = os.path.join(input_dir, '..', 'gamma_corrected')

# 출력 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 감마 값 설정 (보통 2.2를 많이 사용하며, 1보다 크면 밝아지고 1보다 작으면 어두워집니다)
gamma = 2.2

# 3. Look-Up Table (LUT) 생성 (속도 최적화를 위해 미리 계산)
# 공식: Out = ((In / 255) ^ (1/gamma)) * 255
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

print(f"변환 시작: {input_dir}")

# 4. 00000.png ~ 00059.png 루프 처리
for i in range(60):
    filename = f"{i:05d}.png"  # 5자리 숫자 포맷 (00000, 00001...)
    file_path = os.path.join(input_dir, filename)
    
    # 이미지 읽기
    img = cv2.imread(file_path)
    
    if img is not None:
        # 감마 보정 적용 (LUT 사용으로 매우 빠름)
        corrected_img = cv2.LUT(img, table)
        
        # 저장
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, corrected_img)
        
        if i % 10 == 0:
            print(f"{filename} 처리 중...")
    else:
        print(f"경고: {filename}을 찾을 수 없습니다.")

print("모든 작업이 완료되었습니다!")