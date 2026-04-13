# File: nsff_scripts/luv/luv_optical_flow.py
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---- Try to import RAFT & flow_viz from your tree ----
# flow_viz = None
# try:
#     from core.raft import RAFT
#     from core.utils import flow_viz as _flow_viz
#     flow_viz = _flow_viz
# except Exception:
#     try:
#         from raft import RAFT  # fallback if your RAFT is elsewhere
#     except Exception as e:
#         raise ImportError("RAFT import 실패. core/raft.py 또는 raft 모듈 경로를 확인하세요.") from e

# --- add this block at the very top ---
# ---- at the very top of luv_optical_flow.py ----
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(THIS_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from raft import RAFT          # core/raft.py
from utils import flow_viz     # core/utils/flow_viz.py


def load_L_3ch_tensor(path):
    """L PNG(0~255, uint8)을 3채널로 복제해 1x3xHxW float 텐서 반환(0~255 스케일)."""
    L = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if L is None:
        raise FileNotFoundError(f"Not found: {path}")
    L = L.astype(np.float32)
    img = np.stack([L, L, L], axis=0)         # 3xHxW
    return torch.from_numpy(img).unsqueeze(0)  # 1x3xHxW

def resize_flow(flow_hw2, H, W):
    """flow (Hf,Wf,2) -> (H,W,2) 보간 + 좌표 스케일 보정"""
    h, w = flow_hw2.shape[:2]
    fx, fy = W / float(w), H / float(h)
    flow = cv2.resize(flow_hw2, (W, H), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= fx
    flow[..., 1] *= fy
    return flow

def warp_flow(img_hw3_uint8, flow_hw2):
    """img(H,W,3) uint8을 flow(H,W,2)로 워핑."""
    H, W = img_hw3_uint8.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + flow_hw2[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_hw2[..., 1]).astype(np.float32)
    return cv2.remap(img_hw3_uint8, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def flow_to_color_safe(flow):
    if flow_viz is not None:
        return flow_viz.flow_to_image(flow)  # (H,W,3) RGB uint8
    # fallback: magnitude heatmap
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mag = (mag / (np.percentile(mag, 99) + 1e-6) * 255.0).clip(0,255).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_TURBO)[:, :, ::-1]  # RGB

def compute_fwdbwd_mask(fwd, bwd, alpha=0.01, beta=0.5):
    """Forward-Backward consistency mask."""
    H, W = fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + fwd[..., 0]).astype(np.float32)
    map_y = (grid_y + fwd[..., 1]).astype(np.float32)
    bwd_x = cv2.remap(bwd[..., 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bwd_y = cv2.remap(bwd[..., 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bwd_warp = np.stack([bwd_x, bwd_y], axis=-1)

    fb = fwd + bwd_warp
    fb_norm = np.linalg.norm(fb, axis=-1)
    f_norm  = np.linalg.norm(fwd, axis=-1)
    thr = alpha * f_norm + beta
    return (fb_norm < thr).astype(np.uint8)

class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplet_dir", default="./ref_00009", help="L PNG 3장 위치 폴더")
    ap.add_argument("--model", required=True, help="RAFT 가중치(.pth)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--save_tag", default="i1")
    args = ap.parse_args()

    # RAFT 로드
    raft_args = AttrDict(
    small=False,
    mixed_precision=False,
    # RAFT가 내부에서 참조하는 기본 하이퍼파라미터들
    corr_levels=4,
    corr_radius=4,
    dropout=0.0,
    alternate_corr=False,
    )

    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model = model.module.to(DEVICE).eval()

    # 입력 파일들
    names = ["00009_L.png", "00010_L.png", "00011_L.png"]
    paths = [os.path.join(args.triplet_dir, n) for n in names]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # 결과 폴더
    of_dir       = os.path.join(args.triplet_dir, f"flow_{args.save_tag}")
    viz_flow_dir = os.path.join(args.triplet_dir, f"viz_flow_{args.save_tag}")
    viz_warp_dir = os.path.join(args.triplet_dir, f"viz_warp_{args.save_tag}")
    os.makedirs(of_dir, exist_ok=True)
    os.makedirs(viz_flow_dir, exist_ok=True)
    os.makedirs(viz_warp_dir, exist_ok=True)

    def Lpng_to_rgb(path):
        L = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return np.stack([L, L, L], axis=-1)

    with torch.no_grad():
        # (00009→00010), (00010→00011)
        pairs = [(0,1), (1,2)]
        for idx_i, idx_j in pairs:
            p1, p2 = paths[idx_i], paths[idx_j]
            I1 = load_L_3ch_tensor(p1).to(DEVICE).float()
            I2 = load_L_3ch_tensor(p2).to(DEVICE).float()

            _, flow_fwd_t = model(I1, I2, iters=args.iters, test_mode=True)
            _, flow_bwd_t = model(I2, I1, iters=args.iters, test_mode=True)

            flow_fwd = flow_fwd_t[0].detach().cpu().numpy().transpose(1,2,0)
            flow_bwd = flow_bwd_t[0].detach().cpu().numpy().transpose(1,2,0)

            img1_rgb = Lpng_to_rgb(p1)  # H,W,3 uint8
            img2_rgb = Lpng_to_rgb(p2)
            H, W = img1_rgb.shape[:2]

            flow_fwd = resize_flow(flow_fwd, H, W)
            flow_bwd = resize_flow(flow_bwd, H, W)

            fwd_mask = compute_fwdbwd_mask(flow_fwd, flow_bwd)
            bwd_mask = compute_fwdbwd_mask(flow_bwd, flow_fwd)

            # 저장명: 앞 프레임 번호 기준
            base_i = int(os.path.splitext(names[idx_i])[0][:5])  # '00009' -> 9
            # npz 저장
            np.savez(os.path.join(of_dir, f"{base_i:05d}_fwd.npz"), flow=flow_fwd, mask=fwd_mask)
            np.savez(os.path.join(of_dir, f"{base_i+1:05d}_bwd.npz"), flow=flow_bwd, mask=bwd_mask)

            # 시각화 저장
            fwd_color = flow_to_color_safe(flow_fwd)
            bwd_color = flow_to_color_safe(flow_bwd)
            fwd_color_m = (fwd_color.astype(np.float32) * fwd_mask[..., None]).astype(np.uint8)
            bwd_color_m = (bwd_color.astype(np.float32) * bwd_mask[..., None]).astype(np.uint8)

            cv2.imwrite(os.path.join(viz_flow_dir, f"{base_i:05d}_fwd.jpg"), fwd_color[..., ::-1])
            cv2.imwrite(os.path.join(viz_flow_dir, f"{base_i:05d}_fwd_masked.jpg"), fwd_color_m[..., ::-1])
            cv2.imwrite(os.path.join(viz_flow_dir, f"{base_i:05d}_bwd.jpg"), bwd_color[..., ::-1])
            cv2.imwrite(os.path.join(viz_flow_dir, f"{base_i:05d}_bwd_masked.jpg"), bwd_color_m[..., ::-1])

            warped_2 = warp_flow(img2_rgb, flow_fwd)
            warped_1 = warp_flow(img1_rgb, flow_bwd)
            cv2.imwrite(os.path.join(viz_warp_dir, f"im_{base_i:05d}.jpg"), img1_rgb[..., ::-1])
            cv2.imwrite(os.path.join(viz_warp_dir, f"im_{base_i:05d}_fwd.jpg"), warped_2[..., ::-1])
            cv2.imwrite(os.path.join(viz_warp_dir, f"im_{base_i+1:05d}_bwd.jpg"), warped_1[..., ::-1])

            print(f"[OK] {names[idx_i]} → {names[idx_j]} saved (base {base_i})")

if __name__ == "__main__":
    main()
