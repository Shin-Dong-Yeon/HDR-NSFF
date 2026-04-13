import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rgb_to_LuvL(img_rgb_uint8):
    luv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Luv)
    L = luv[..., 0].astype(np.float32)
    uv = luv[..., 1:].astype(np.float32)
    return L, uv

def LuvL_to_rgb(L, uv):
    Luv = np.dstack([L, uv]).astype(np.float32)
    Luv = np.clip(Luv, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(Luv, cv2.COLOR_Luv2RGB)
    return rgb

def estimate_affine_L(L_src, L_tgt, sat_thr_low=2, sat_thr_high=253):
    # saturation/암부 클리핑 영역 제외
    mask = (L_src > sat_thr_low) & (L_src < sat_thr_high) & \
           (L_tgt > sat_thr_low) & (L_tgt < sat_thr_high)
    x = L_src[mask].reshape(-1,1).astype(np.float64)
    y = L_tgt[mask].reshape(-1,1).astype(np.float64)
    if len(x) < 100:  # 유효 픽셀 너무 적으면 안전장치
        return 1.0, 0.0
    # 최소제곱: [a b]^T = (A^T A)^{-1} A^T y,  A=[x 1]
    A = np.hstack([x, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol.flatten()
    return float(a), float(b)

def match_exposure_L(L_src, L_tgt):
    a, b = estimate_affine_L(L_src, L_tgt)
    L_adj = a * L_src + b
    return np.clip(L_adj, 0, 255).astype(np.float32), (a,b)

img_path = "/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/big_jump_1_36/dense/images_641x360/0009.png"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

L_ref, uv_ref = rgb_to_LuvL(img_rgb)

L_uint8 = np.clip(L_ref, 0, 255).astype(np.uint8)
save_path = os.path.basename(img_path) + "_L_visualize.png"
cv2.imwrite(save_path, L_uint8)
print(f"Saved visualization to {save_path}")

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.imshow(img_rgb)
# plt.title("Original RGB")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(L_ref, cmap="gray")
# plt.title("L-channel (Lightness)")
# plt.axis("off")


# plt.savefig(save_path, dpi=200, bbox_inches="tight")
# plt.close()

# print(f"Saved visualization to {save_path}")

