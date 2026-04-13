import os, sys
import numpy as np
import imageio
import struct
# import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from run_nerf_helpers import *
from posenc_utils import *
from cam_param import *
from glob import glob
import ffmpeg
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from PIL import Image
to_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_ROOT = os.path.join(THIS_DIR, "depth_anything_v2")
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2

def neighbor_indices_(t, n_frames, k):
    """
    current t, [t-k/2, t] & [t, t+k/2]
    """
    half = k // 2
    idx = [ max(0, min(n_frames-1, t+i)) for i in range(-half, half+1) if i!= 0]

    return idx[:k]

def neighbor_indices__(t, n_frames, k, step=1):
    """
    current t: [t - step*k:t:step] & [t:t+step*k:step]
    k: number of render images
    step: how far do you want to render
    """
    half = (k // 2) * step
    idx = [ max(0, min(n_frames-1, t+i)) for i in range(-half, half+1, step) if i!= 0]

    return idx[:k]

def neighbor_indices(t: int, n_frames: int, k: int, step: int = 1):
    """
    Return up to `k` neighbour indices around `t`, expanding outward by `step`.
    Always excludes `t` itself and skips indices that fall outside [0, n_frames-1].

    Examples
    --------
    >>> neighbor_indices(0, n_frames=36, k=2, step=1)
    [1, 2]
    >>> neighbor_indices(10, n_frames=15, k=4, step=2)
    [8, 12, 6, 14]
    """
    neighbours = []
    offset = step
    while len(neighbours) < k:
        # negative direction
        neg = t - offset
        if 0 <= neg < n_frames:
            neighbours.append(neg)
            if len(neighbours) == k:
                break
        # positive direction
        pos = t + offset
        if 0 <= pos < n_frames:
            neighbours.append(pos)
            if len(neighbours) == k:
                break
        offset += step            # increase search radius
    return neighbours

def weights_map_to_mask(weights_map_dd, mode="quantile", q=0.80, thr=0.5):
    """
    weights_map_dd: torch.Tensor [H,W] or [H,W,1]
    return: np.uint8 mask [H,W] (0/255)
    """
    if weights_map_dd.ndim == 3 and weights_map_dd.shape[-1] == 1:
        w = weights_map_dd[..., 0]
    else:
        w = weights_map_dd
    w = w.detach().float().cpu().numpy()
    w01 = (w - w.min()) / (w.max() - w.min() + 1e-8)

    if mode == "quantile":
        cut = np.quantile(w01, q)
        m = (w01 >= cut).astype(np.uint8) * 255
    else:  # "threshold"
        m = (w01 >= thr).astype(np.uint8) * 255

    return m

def save_weights_mask_png(base_dir, t, mask_png):
    """
    base_dir: dataset root (basedir)
    save_dir: {basedir}/motion_masks_difix/{t:05d}.png
    """
    outdir = os.path.join(base_dir, "motion_masks_difix")
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, f"{t:05d}.png"), mask_png)

def save_depth_pngs(depth_t: torch.Tensor, out_stem: str, valid_mask: torch.Tensor=None,
                    do_jet=True, do_gray=True, do_u16=False, u16_scale_mm=1000.0,
                    clip_percentile=2.0):
    """
    depth_t: [H,W] torch float (meters or any consistent scale)
    out_stem: 저장 경로 prefix (확장자/후미는 함수가 붙임) 예) ".../n_007_depth"
    valid_mask: [H,W] bool/float. None이면 자동(Z>0 & finite)
    do_u16: True면 16-bit PNG도 저장 (정밀도 유지; u16_scale_mm 배 스케일, 예: m*1000=mm)
    clip_percentile: 2.0이면 [2%, 98%] 구간으로 min-max 정규화 (outlier 안정화)
    """
    d = depth_t.detach().float().cpu().numpy()

    # 유효영역 마스크
    if valid_mask is None:
        valid = np.isfinite(d) & (d > 0)
    else:
        valid = (valid_mask.detach().cpu().numpy() > 0) & np.isfinite(d) & (d > 0)

    if not np.any(valid):
        # 비정상 상황 방어
        d[:] = 0
        valid = np.zeros_like(d, dtype=bool)

    # --- 8-bit 시각화용 범위 결정 (percentile clip) ---
    v = d[valid]
    lo = np.percentile(v, clip_percentile)
    hi = np.percentile(v, 100.0 - clip_percentile)
    if hi <= lo:  # 수치 불안정 방어
        lo, hi = v.min(), v.max()

    d_norm = np.clip((d - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    d_norm[~valid] = 0.0
    d_u8 = (d_norm * 255.0 + 0.5).astype(np.uint8)

    # --- 저장 ---
    if do_gray:
        cv2.imwrite(f"{out_stem}_gray.png", d_u8)

    if do_jet:
        # OpenCV 컬러맵은 BGR 반환
        jet = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        # invalid 영역은 검정으로
        jet[~valid] = (0, 0, 0)
        cv2.imwrite(f"{out_stem}_jet.png", jet)

    if do_u16:
        # 16-bit: 물리 스케일 유지용 (예: m * 1000 -> mm)
        d_scaled = d * u16_scale_mm
        d_scaled[~valid] = 0.0
        # 16-bit 범위를 초과할 수 있으니 클램프 (필요시 스케일 조정)
        d_u16 = np.clip(d_scaled, 0, 65535).astype(np.uint16)
        cv2.imwrite(f"{out_stem}_metric_u16.png", d_u16)

@torch.no_grad()
def extract_ref_camZ_depth(ret_ref, H, W, focal, c2w_ref):
    """
    ret_ref: dict from render(retraw=True), must contain
            'raw_pts_ref'  [H,W,S,3] in NDC coords
            'weights_ref_dy' or 'weights_mix_ref'  [H,W,S]
    H, W, focal: intrinsics (pixel unit)
    c2w_ref: [3,4] camera-to-world of the ref view

    return: depth_ref_camZ [H,W]  (expected Z in ref-cam coords)
    """
    device = c2w_ref.device

    # 1. 샘플링된 NDC 점들 불러오기
    pts_ndc = ret_ref['raw_pts_ref'].to(device)       # [H,W,S,3]
    weights = ret_ref.get('weights_mix_ref', None)
    if weights is None:
        weights = ret_ref['weights_ref_dy']
    weights = weights.to(device)                      # [H,W,S]

    # 2. NDC -> Euclidean (ref 카메라 좌표계)
    pts_euc = NDC2Euclidean(pts_ndc, H, W, focal)     # [H,W,S,3]

    # 3. Euclidean → World
    R_ref = c2w_ref[:,:3]
    t_ref = c2w_ref[:,3:].reshape(3,1)
    pts_world = (R_ref @ pts_euc.reshape(-1,3).T + t_ref).T
    pts_world = pts_world.view(H,W,-1,3)              # [H,W,S,3]

    # 4. World → Ref-Cam (다시 변환해서 cam-Z 얻기)
    R_w2c = R_ref.T
    t_w2c = -(R_ref.T @ t_ref).reshape(3)
    pts_cam = (R_w2c @ pts_world.reshape(-1,3).T + t_w2c[:,None]).T
    pts_cam = pts_cam.view(H,W,-1,3)                  # [H,W,S,3]
    Zs = pts_cam[...,2].clamp_min(1e-6)               # [H,W,S]

    # 5. 가중합으로 기대 cam-Z 뽑기
    depth_ref_camZ = torch.sum(weights * Zs, dim=-1)  # [H,W]

    return depth_ref_camZ

import numpy as np
import torch

@torch.no_grad()
def save_depth_as_pointcloud_ply_v2(
    depth_ref: torch.Tensor,      # [H,W]
    c2w_ref: torch.Tensor,        # [3,4] (R|t), camera->world
    H: int, W: int, focal: float,
    out_path: str,
    color_img=None,               # [H,W,3] uint8 or [3,H,W] tensor (optional)
    valid_mask: torch.Tensor=None,# [H,W] bool/0-1 (optional)
    stride: int=1,
    z_min: float=1e-6,
    z_max: float=None,            # 예: 너무 먼 점 제거용 (단위는 depth 단위)
    max_range: float=None,        # 예: 월드좌표 |X|<max_range 컷 (m)
    verbose: bool=True,
):
    import pdb; pdb.set_trace()
    device = depth_ref.device
    c2w = c2w_ref.to(device)

    # 1) 원시 깊이와 유효성
    z_raw = depth_ref[::stride, ::stride]
    valid = torch.isfinite(z_raw) & (z_raw > z_min)
    if z_max is not None:
        valid = valid & (z_raw < z_max)

    if valid_mask is not None:
        vm = valid_mask[::stride, ::stride]
        if vm.dtype != torch.bool:
            vm = vm > 0.5
        valid = valid & vm

    if verbose:
        total = z_raw.numel()
        bad = int((~torch.isfinite(z_raw)).sum().item() + (z_raw <= z_min).sum().item())
        print(f"[depth] total={total}, invalid(z<=min or nonfinite)~={bad}, keep={(valid.sum().item())}")

    if not torch.any(valid):
        print(f"[warn] no valid points -> skip writing {out_path}")
        return

    # 2) 픽셀 좌표 -> 카메라좌표
    ys, xs = torch.meshgrid(
        torch.arange(0, H, stride, device=device),
        torch.arange(0, W, stride, device=device),
        indexing='ij'
    )
    ys = ys[valid].float()
    xs = xs[valid].float()
    z  = z_raw[valid].float()

    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    fx = fy = float(focal)

    Xc = torch.stack([
        (xs - cx) / fx * z,
        (ys - cy) / fy * z,
        z
    ], dim=-1)  # [N,3]

    # 3) 카메라->월드
    R = c2w[:, :3]
    t = c2w[:, 3]
    Xw = (Xc @ R.T) + t  # [N,3]

    pts = Xw.detach().cpu().numpy().astype(np.float32)  # [N,3]

    # 4) 최종 안전장치: NaN/inf 제거 + (선택)거리 컷
    good = np.isfinite(pts).all(axis=1)
    if max_range is not None:
        good &= (np.linalg.norm(pts, axis=1) < float(max_range))

    if not np.any(good):
        print(f"[warn] all points filtered -> skip writing {out_path}")
        return

    pts = pts[good]

    # --- color fix: handle float [0,1] properly ---
    if color_img is not None:
        if isinstance(color_img, torch.Tensor):
            img = color_img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1,2,0))  # [H,W,3]
        else:
            img = np.asarray(color_img)

        if img.dtype != np.uint8:
            # float → uint8로 안전 변환 (0~1 이면 255 곱, 그 외는 클립)
            maxv = float(np.nanmax(img))
            if maxv <= 1.01:
                img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
        color_np = img

        yy, xx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij')
        vv = valid.detach().cpu().numpy()
        cols_all = color_np[yy[vv], xx[vv]]   # [N,3]
        cols = cols_all[good]
    else:
        cols = np.full((pts.shape[0], 3), 255, dtype=np.uint8)

# --- clean PLY header (no leading spaces) ---
    N = pts.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    data = np.empty(N, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                              ('red','u1'),('green','u1'),('blue','u1')])
    data['x']=pts[:,0]; data['y']=pts[:,1]; data['z']=pts[:,2]
    data['red']=cols[:,0]; data['green']=cols[:,1]; data['blue']=cols[:,2]

    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    if verbose:
        print(f"[ok] wrote {out_path} with {N} points (filtered NaN/inf)")

@torch.no_grad()
def save_depth_as_pointcloud_ply_v3(
    depth_ref: torch.Tensor,      # [H,W]  (cam-Z / ray / NDC-t 중 하나)
    c2w_ref: torch.Tensor,        # [3,4] (R|t), camera->world
    H: int, W: int,
    fx: float, fy: float = None,  # fx!=fy 가능
    cx: float = None, cy: float = None,
    out_path: str = "cloud.ply",
    color_img=None,
    valid_mask: torch.Tensor=None,
    stride: int=1,
    depth_mode: str="ndc_t",      # 'cam_z' | 'ray' | 'ndc_t'
    near: float=1.0,              # ndc_t일 때 사용 (보통 1.0)
    max_range: float=None,        # 월드 거리 컷
    z_min: float=1e-6, z_max: float=None,
    verbose: bool=True,
):
    import numpy as np, torch, cv2
    device = depth_ref.device
    if fy is None: fy = fx
    if cx is None: cx = (W-1)*0.5
    if cy is None: cy = (H-1)*0.5
    R = c2w_ref[:,:3].to(device); t = c2w_ref[:,3].to(device)

    # 샘플링 & 유효 z/t
    d_raw = depth_ref[::stride, ::stride]
    valid = torch.isfinite(d_raw)
    if depth_mode == "cam_z":
        valid &= (d_raw > z_min)
        if z_max is not None: valid &= (d_raw < z_max)
    elif depth_mode == "ray":
        valid &= (d_raw > 0)   # 거리이므로 양수
    elif depth_mode == "ndc_t":
        # t∈[0,1), t→1이면 z→∞. t==1은 무효 취급
        valid &= (d_raw >= 0) & (d_raw < 1.0)
    else:
        raise ValueError("depth_mode must be one of {'cam_z','ray','ndc_t'}")

    if valid_mask is not None:
        vm = valid_mask[::stride, ::stride]
        if vm.dtype != torch.bool: vm = vm > 0.5
        valid &= vm

    if not torch.any(valid):
        if verbose: print(f"[warn] no valid points -> skip {out_path}")
        return

    ys, xs = torch.meshgrid(
        torch.arange(0, H, stride, device=device),
        torch.arange(0, W, stride, device=device),
        indexing='ij'
    )
    xs = xs[valid].float(); ys = ys[valid].float()
    d  = d_raw[valid].float()

    # --- cam 좌표 Xc 만들기 ---
    if depth_mode == "cam_z":
        z = d.clamp(min=z_min)
        Xc = torch.stack([(xs - cx)/fx * z, (ys - cy)/fy * z, z], dim=-1)
    elif depth_mode == "ray":
        # 광선 단위벡터
        dirs = torch.stack([(xs - cx)/fx, (ys - cy)/fy, torch.ones_like(xs)], dim=-1)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
        Xc = dirs * d.unsqueeze(-1)
    else:  # 'ndc_t'
        # t → cam-Z: z = near / (1 - t)   (z-forward 관례)
        z = (near / (1.0 - d)).clamp(min=z_min)
        Xc = torch.stack([(xs - cx)/fx * z, (ys - cy)/fy * z, z], dim=-1)

    # cam->world
    Xw = (Xc @ R.T) + t
    pts = Xw.detach().cpu().numpy().astype(np.float32)

    # 안전 필터
    good = np.isfinite(pts).all(axis=1)
    if max_range is not None:
        good &= (np.linalg.norm(pts, axis=1) < float(max_range))
    if not np.any(good):
        if verbose: print(f"[warn] all filtered -> skip {out_path}")
        return
    pts = pts[good]

    # 색상
    if color_img is not None:
        if isinstance(color_img, torch.Tensor):
            img = color_img.detach().cpu().numpy()
            if img.ndim==3 and img.shape[0]==3: img = np.transpose(img,(1,2,0))
        else:
            img = np.asarray(color_img)
        if img.dtype != np.uint8:
            m = float(np.nanmax(img))
            img = ((np.clip(img,0,1) * 255.0) if m<=1.01 else np.clip(img,0,255)).astype(np.uint8)
        yy, xx = np.meshgrid(np.arange(0,H,stride), np.arange(0,W,stride), indexing='ij')
        vv = valid.detach().cpu().numpy()
        cols_all = img[yy[vv], xx[vv]]
        cols = cols_all[good]
    else:
        cols = np.full((pts.shape[0],3), 255, dtype=np.uint8)

    # PLY 저장
    N = pts.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    arr = np.empty(N, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                             ('red','u1'),('green','u1'),('blue','u1')])
    arr['x']=pts[:,0]; arr['y']=pts[:,1]; arr['z']=pts[:,2]
    arr['red']=cols[:,0]; arr['green']=cols[:,1]; arr['blue']=cols[:,2]
    with open(out_path,'wb') as f:
        f.write(header.encode('ascii')); f.write(arr.tobytes())
    if verbose: print(f"[ok] wrote {out_path} with {N} pts (mode={depth_mode})")


@torch.no_grad()
def compute_optical_flow_from_ret(
    ret_ref: dict,
    c2w_post: torch.Tensor,     # [3,4]
    c2w_ref: torch.Tensor,      # [3,4]
    c2w_prev: torch.Tensor,     # [3,4]
    H: int, W: int, f: float,
    n_dim: int = 1,
):
    device = c2w_ref.device

    # 1) weights (H,W,S)
    weights = ret_ref.get('weights_mix_ref', None)
    if weights is None:
        weights = ret_ref['weights_ref_dy']
    weights = weights.to(device)

    # 2) NDC points for post/prev (H,W,S,3)
    #    ※ 모델이 NDC에서 SF를 예측한다는 가정: raw_pts_ref + raw_sf_*
    pts_ndc_ref  = ret_ref['raw_pts_ref'        ].to(device)  # [H,W,S,3]
    sf_ref2post  = ret_ref['raw_sf_ref2post'    ].to(device)  # [H,W,S,3]
    sf_ref2prev  = ret_ref['raw_sf_ref2prev'    ].to(device)  # [H,W,S,3]
    pts_ndc_post = pts_ndc_ref + sf_ref2post
    pts_ndc_prev = pts_ndc_ref + sf_ref2prev

    # 3) NDC → Euclidean
    pts_e_world_post = NDC2Euclidean(pts_ndc_post, H, W, f)  # [H,W,S,3]
    pts_e_world_prev = NDC2Euclidean(pts_ndc_prev, H, W, f)  # [H,W,S,3]

    # (만약 NDC2Euclidean이 'ref-카메라 좌표'를 돌려주는 구현이라면,
    #  아래 두 줄을 주석 해제해서 ref c2w로 월드로 올린 뒤 쓰세요.)
    # R_ref, t_ref = c2w_ref[:3,:3], c2w_ref[:3,3]
    # pts_e_world_post = (pts_e_world_post @ R_ref.T) + t_ref
    # pts_e_world_prev = (pts_e_world_prev @ R_ref.T) + t_ref

    # 4) 월드→(post/prev) 카메라 투영 (가중합은 함수 내부에서)
    pts2d_post = projection_from_world(pts_e_world_post, c2w_post, H, W, f,
                                       n_dim=n_dim, weights=weights)  # [H,W,2]
    pts2d_prev = projection_from_world(pts_e_world_prev, c2w_prev, H, W, f,
                                       n_dim=n_dim, weights=weights)  # [H,W,2]

    # 5) ref 픽셀 그리드
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    xs = xs.float(); ys = ys.float()

    # 6) flow = target_uv - ref_uv
    flow_fwd = torch.stack([pts2d_post[...,0] - xs, pts2d_post[...,1] - ys], dim=-1)  # [H,W,2]
    flow_bwd = torch.stack([pts2d_prev[...,0] - xs, pts2d_prev[...,1] - ys], dim=-1)  # [H,W,2]

    # 7) 유효성 (in-bounds & finite)
    valid_fwd = (
        torch.isfinite(pts2d_post[...,0]) & torch.isfinite(pts2d_post[...,1]) &
        (pts2d_post[...,0] >= 0) & (pts2d_post[...,0] <= (W-1)) &
        (pts2d_post[...,1] >= 0) & (pts2d_post[...,1] <= (H-1))
    )
    valid_bwd = (
        torch.isfinite(pts2d_prev[...,0]) & torch.isfinite(pts2d_prev[...,1]) &
        (pts2d_prev[...,0] >= 0) & (pts2d_prev[...,0] <= (W-1)) &
        (pts2d_prev[...,1] >= 0) & (pts2d_prev[...,1] <= (H-1))
    )

    return flow_fwd, flow_bwd, valid_fwd, valid_bwd


@torch.no_grad()
def projection_from_world(pts_e_world: torch.Tensor, c2w: torch.Tensor,
                          H: int, W: int, f: float, n_dim: int = 1,
                          weights: torch.Tensor = None):
    """
    pts_e_world: [H,W,S,3] (Euclidean 3D in *world*)
    c2w:         [3,4]  (camera->world)
    반환: pts2d [H,W,2]
    내부에서 W2C로 바꾼 후, Z>0인 점들에 대해 가중합으로 3D 한 점을 만들고 투영.
    """
    device = pts_e_world.device
    R_cw = c2w[:3,:3]
    t_cw = c2w[:3, 3:4]           # [3,1]
    R_wc = R_cw.T
    t_wc = -(R_wc @ t_cw)         # [3,1]

    H_, W_, S, _ = pts_e_world.shape
    assert H_==H and W_==W, "shape mismatch"

    # [H,W,S,3] -> [H,W,S,3] (world->cam)
    Xw = pts_e_world.reshape(-1, S, 3)        # [HW,S,3]
    Xs = (Xw @ R_wc.T) + t_wc.T               # [HW,S,3]
    Xs = Xs.view(H, W, S, 3)

    # 가중합 3D (cam 좌표)
    if weights is None:
        raise ValueError("weights (H,W,S) required")
    w = weights.to(device)
    wsum = (w.sum(dim=-1, keepdim=True) + 1e-12)      # [H,W,1]
    Xc = (w[...,None] * Xs).sum(dim=-2) / wsum        # [H,W,3]

    # 투영 (fx=fy=f)
    z = Xc[...,2].clamp(min=1e-6)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    u = (Xc[...,0] / z) * f + cx
    v = (Xc[...,1] / z) * f + cy
    pts2d = torch.stack([u, v], dim=-1)               # [H,W,2]
    return pts2d


def flow_to_color(flow: np.ndarray, valid: np.ndarray=None, clip_percentile: float=95.0):
    """
    flow: [H,W,2] (du,dv), numpy float32
    valid: [H,W] bool or None
    Middlebury 스타일 간단 HSV 시각화 (각도=Hue, 크기=Value)
    """
    H, W, _ = flow.shape
    fx = flow[:,:,0]; fy = flow[:,:,1]
    mag = np.sqrt(fx*fx + fy*fy)
    ang = np.arctan2(fy, fx)  # [-pi, pi]
    ang = (ang + np.pi) / (2*np.pi)  # [0,1] → Hue

    vmax = np.percentile(mag[np.isfinite(mag)], clip_percentile) if np.any(np.isfinite(mag)) else 1.0
    vmax = max(vmax, 1e-6)
    val = np.clip(mag / vmax, 0.0, 1.0)

    hsv = np.zeros((H,W,3), dtype=np.float32)
    hsv[...,0] = ang        # H in [0,1]
    hsv[...,1] = 1.0        # full saturation
    hsv[...,2] = val        # value = normalized magnitude

    hsv_8u = (hsv * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    if valid is not None:
        v = valid.astype(bool)
        bgr[~v] = 0
    return bgr


def write_flo(path: str, flow: np.ndarray):
    """
    .flo (Middlebury) 저장: header 'PIEH' + width + height + float32 data (u then v)
    flow: [H,W,2] float32
    """
    H, W, _ = flow.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4s', b'PIEH'))
        f.write(struct.pack('i', W))
        f.write(struct.pack('i', H))
        f.write(flow.astype(np.float32).tobytes())


def save_flow_bundle(flow: torch.Tensor, valid: torch.Tensor, out_stem: str):
    """
    flow: [H,W,2] torch
    valid: [H,W] torch/bool
    저장: *_flow.png, *_flow.npy, *_flow.flo, *_mag.png
    """
    f = flow.detach().cpu().numpy().astype(np.float32)
    v = valid.detach().cpu().numpy().astype(bool)

    # color
    vis = flow_to_color(f, v)               # BGR
    cv2.imwrite(out_stem + "_flow.png", vis)

    # npy
    np.save(out_stem + "_flow.npy", f)

    # flo
    try:
        write_flo(out_stem + ".flo", f)
    except Exception as e:
        print("[warn] write_flo failed:", e)

    # magnitude viz
    mag = np.sqrt(f[...,0]**2 + f[...,1]**2)
    mag_u8 = (np.clip(mag / (np.percentile(mag[v], 95.0)+1e-6), 0, 1) * 255).astype(np.uint8)
    mag_u8[~v] = 0
    cv2.imwrite(out_stem + "_mag.png", mag_u8)



class DifixUpdater:
    def __init__(self, H, W, focal, render_fn, render_kwargs_test,
                 base_images, base_poses, k, step, save_root, data_dir, use_depth=False,
                 mask_warp=False, motion_masks=None, weights_map_ray=False, use_mv_mask=False,
                 inter_pose=False):
        self.H, self.W, self.focal = H, W, focal
        self.render = render_fn
        self.kw_test = render_kwargs_test
        self.base_imgs = base_images
        self.base_poses = base_poses
        self.k = k # number of synthetic images
        self.step = step
        self.save_root = save_root
        self.synthetic_imgs = None
        self.synthetic_poses = None
        self.data_dir = data_dir

        # Depth
        self.use_depth = use_depth
        self.synthetic_depths = None

        # Mask warp
        self.mask_warp = mask_warp
        self.base_motion_masks = motion_masks

        # Weights map ray
        self.weights_map_ray = weights_map_ray
        self.motion_coords = None
        self.use_mv_mask = use_mv_mask

        # Synthetic pose option
        self.inter_pose = inter_pose


    def _load_gt_mv_mask(self, t, n, dilate_ks=3, dilate_iter=2):
        """
        GT multi-view motion mask 로드 후 소량 dilation 적용.
        반환: (mask_uint8, coords_int64) 또는 (None, None)  # 없으면 None
        """
        # self.data_dir가 .../dense 를 가리킨다고 가정
        gt_path = os.path.join(self.data_dir, "mv_masks", f"{t:05d}", f"{n%9+1:05d}.png")
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None

        # 이진화(혹시 회색값이면 >0을 255로)
        mask = (mask > 0).astype(np.uint8) * 255

        # 소량 dilation
        kernel = np.ones((dilate_ks, dilate_ks), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

        # resize to (self.W, self.H) — note: cv2 takes (width, height)
        mask_resized = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # re-binarize just in case
        mask_resized = (mask_resized > 0).astype(np.uint8) * 255

        # 좌표 추출 (y, x)
        ys, xs = np.where(mask_resized > 0)
        if ys.size == 0:
            coords = np.empty((0, 2), dtype=np.int64)
        else:
            coords = np.stack((ys, xs), axis=-1).astype(np.int64)

        return mask_resized, coords



    @torch.no_grad()
    def update(self, step, pipe, ref_imgs, cam_param, gt=True, gt_ratio=1.):
        """
            For difix ver 2
        """
        import pdb; pdb.set_trace()
        N = self.base_imgs.shape[0]
        syn_imgs, syn_poses = [], []
        if self.weights_map_ray:
            self.motion_coords = {t: [] for t in range(N)}
        for t in range(N):
            neigh = neighbor_indices(t, N, self.k, self.step)
            for n in neigh:
                outdir = f"{self.save_root}/step_{step:06d}/t_{t:03d}"
                img_path = f"{outdir}/n_{n:03d}.png"

                mask_path = os.path.join(outdir, f"n_{n:03d}_weights_map_mask.png")
                coords_path = os.path.join(outdir, f"n_{n:03d}_weights_coords.npy")

                if self.inter_pose:
                    pose = (self.base_poses[n] + self.base_poses[t]) / 2
                pose = self.base_poses[n].to(device)

                needs_ret = self.mask_warp or (
                    self.weights_map_ray and not (os.path.exists(coords_path) or os.path.exists(mask_path))
                )

                # needs_ret = self.mask_warp or self.weights_map_ray

                ret = None
                rgb = None

                print(f'Time: {t}, n:{n}')

                if needs_ret:
                    img_idx_embed = t / float(N) * 2. - 1.0
                    ret = self.render(img_idx_embed, 0, False, float(N),
                                    self.H, self.W, self.focal,
                                    chunk=1024*16, c2w=pose,
                                    **self.kw_test)
                    
                    rgb = cam_param.RAD2LDR_img(ret['rgb_map_ref'], t).permute(2,0,1)


                if os.path.exists(img_path): # for debugging
                    pil_img = Image.open(img_path).convert("RGB")
                    enh_tensor = (pil_to_tensor(pil_img).float() / 255.0).permute(1,2,0) # H,W,C
                    # import pdb; pdb.set_trace()
                    if self.weights_map_ray:
                        if os.path.exists(coords_path):
                            coords = np.load(coords_path)
                            # 안전장치: dtype/shape 보정
                            if coords.ndim != 2 or coords.shape[-1] != 2:
                                coords = np.empty((0,2), dtype=np.int64)
                            else:
                                coords = coords.astype(np.int64, copy=False)
                            self.motion_coords[t].append(coords)
                            
                            print('Load coords_path')
                        elif os.path.exists(mask_path):
                            mask_png = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask_png is None:
                                self.motion_coords[t].append(np.empty((0,2), dtype=np.int64))
                            else:
                                ys, xs = np.where(mask_png > 0)
                                coords = np.stack((ys, xs), axis=-1).astype(np.int64) if ys.size > 0 else np.empty((0,2), np.int64)
                                self.motion_coords[t].append(coords)
                                # 선택: 좌표가 없으면 만들어두기
                                np.save(coords_path, coords)
                                print('Load mask_path')
                        else:
                            # 저장본이 전혀 없는 경우: 일관성 위해 빈 좌표 push (또는 needs_ret를 True로 강제해 다시 계산하도록 설계 변경 가능)
                            # print()
                            self.motion_coords[t].append(np.empty((0,2), dtype=np.int64))
                    
                
                else:
                    if rgb is None:
                        print("Difix no RGB")
                        return

                    if gt:
                        gt_img_path = os.path.join(self.data_dir, "images", f"{t:05d}.png")
                        gt_img = Image.open(gt_img_path).convert("RGB")
                        gt_tensor = (pil_to_tensor(gt_img).float() / 255.).to(rgb.device)
                        

                        if gt_ratio != 1.:
                            # import pdb; pdb.set_trace()
                            new_H = int(gt_tensor.shape[1] * gt_ratio)
                            new_W = int(gt_tensor.shape[2] * gt_ratio)


                            gt_tensor = F.interpolate(
                                gt_tensor.unsqueeze(0),
                                size=(new_H, new_W),
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0).clamp(0,1)

                        rgb_resized = F.interpolate(
                            rgb.unsqueeze(0),
                            size=(gt_tensor.shape[1], gt_tensor.shape[2]),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0).clamp(0,1)

                        enh = pipe(
                            "remove degradation",
                            image=rgb_resized,
                            ref_image=gt_tensor,
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0,
                        ).images[0]

                    else:
                        enh = pipe(
                            "remove degradation",
                            image=rgb,
                            ref_image=ref_imgs[t].permute(2,0,1),
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0,
                        ).images[0]
                    
                    
                    os.makedirs(outdir, exist_ok=True)
                    enh_resized = enh.resize((rgb.shape[2], rgb.shape[1]), Image.BILINEAR)
                    enh_resized.save(img_path)                    
                    
                    rgb_pil = to_pil_image(rgb.cpu().clamp(0,1))
                    rgb_pil.save(f"{outdir}/n_{n:03d}_o.png")

                    enh_tensor = pil_to_tensor(enh_resized).to(rgb.device).permute(1,2,0) / 255.


                # size reshape
                h,w,_ = enh_tensor.shape
                if (h,w) != (self.H, self.W):
                    if h == self.H and w == self.W - 1:
                        enh_tensor = F.pad(enh_tensor.permute(2,0,1),
                                           (0,1,0,0), mode='replicate').permute(1,2,0)
                    else:
                        enh_tensor=F.interpolate(enh_tensor.permute(2,0,1).unsqueeze(0),
                                                 size=(self.H, self.W),
                                                 mode='bilinear',
                                                 align_corners=False)[0].permute(1,2,0)
                
                syn_imgs.append(enh_tensor.cpu())
                syn_poses.append(pose.cpu())

                if self.mask_warp:

                    # ref view (t)의 depth를 얻기 위해 렌더링 필요
                    if ret is None or ret['depth_map_ref'] is None or pose is not self.base_poses[t]:
                        # t번째 pose에서 다시 렌더링
                        pose_ref = self.base_poses[t].to(device)
                        img_idx_embed = t / float(N) * 2. - 1.0
                        ret_ref = self.render(img_idx_embed, 0, False, float(N),
                                            self.H, self.W, self.focal,
                                            chunk=1024*16, c2w=pose_ref,
                                            **self.kw_test)
                        depth_ref = ret_ref['depth_map_ref'].detach()
                    else:
                        depth_ref = ret['depth_map_ref'].detach()


                    # Flow visualize #########################################################
                    # pose_ref  = self.base_poses[t,:3,:4].to(device)
                    # idx_prev  = max(0, t - self.step)
                    # idx_post  = min(N-1, t + self.step)
                    # pose_prev = self.base_poses[idx_prev,:3,:4].to(device)
                    # pose_post = self.base_poses[idx_post,:3,:4].to(device)

                    # # ref pose에서 raw_pts_post/prev와 weights가 필요함
                    # ret_flow = ret_ref  # 위에서 ref로 렌더한 딕셔너리. 키가 없으면 다시 렌더(retraw=True) 필요.

                    # # optical flow 계산
                    # flow_fwd, flow_bwd, valid_fwd, valid_bwd = compute_optical_flow_from_ret(
                    #     ret_flow, pose_post, pose_ref, pose_prev, self.H, self.W, self.focal, n_dim=1
                    # )

                    # # 저장
                    # flow_dir = outdir  # f"{self.save_root}/step_{step:06d}/t_{t:03d}"
                    # save_flow_bundle(flow_fwd, valid_fwd, os.path.join(flow_dir, f"t_{t:03d}_fwd"))
                    # save_flow_bundle(flow_bwd, valid_bwd, os.path.join(flow_dir, f"t_{t:03d}_bwd"))
                    ###########################################################################

                    

                    ply_path =f"{outdir}/t_{t:03d}_depth_cloud.ply"
                    color_img = None
                    ci = self.base_imgs[t]
                    color_img = ci
                    # if hasattr(self, "base_imgs") and self.base_imgs is not None:
                    #     ci = self.base_imgs[t]
                    #     # tensor면 [3,H,W]일 가능성 높음 -> 함수에서 처리함
                    #     color_img = ci

                    # save_depth_as_pointcloud_ply_v2(
                    #     depth_ref=depth_ref.squeeze(),
                    #     c2w_ref=self.base_poses[t,:3,:4],
                    #     H=self.H, W=self.W, focal=self.focal,
                    #     out_path=ply_path,
                    #     color_img=self.base_imgs[t] if hasattr(self, "base_imgs") else None,
                    #     stride=1,
                    #     z_min=1e-6,
                    #     z_max=None,        # 예: 50.0 (깊이 50m 초과 컷)
                    #     max_range=None,    # 예: 200.0 (월드 200m 밖 컷)
                    #     verbose=True,
                    # )

                    save_depth_as_pointcloud_ply_v3(
                        depth_ref=depth_ref.squeeze(),
                        c2w_ref=self.base_poses[t,:3,:4],
                        H=self.H, W=self.W,
                        fx=self.focal, fy=self.focal,
                        out_path=ply_path,
                        color_img=self.base_imgs[t],
                        depth_mode="ndc_t"
                    )

                    # mask_syn = reproject_mask_to_view(
                    #                 mask_ref=self.base_motion_masks[t],    # ref view mask
                    #                 depth_ref=depth_ref,                   # ref view depth
                    #                 c2w_ref=self.base_poses[t,:3,:4],
                    #                 c2w_syn=self.base_poses[n,:3,:4],      # target pose
                    #                 H=self.H, W=self.W, focal=self.focal
                    # )
                    # us, vs, idxs, mask_syn = project_mask_points_v2(
                    #     self.base_motion_masks[t], depth_ref, self.base_poses[t,:3,:4], self.base_poses[n,:3,:4], self.H, self.W, self.focal, return_sparse_mask=True
                    # )
                    us, vs, idxs, mask_syn = project_mask_points_v2(
                        self.base_motion_masks[t],
                        depth_ref,
                        self.base_poses[t,:3,:4],
                        self.base_poses[n,:3,:4],
                        self.H, self.W, self.focal,
                        depth_mode='ndc_t',  # ★ 추가: NDC t 사용
                        near=1.0,            # ★ 보통 LLFF/NeRF에서 near=1.0
                        return_sparse_mask=True
                    )

                    mask_np = (mask_syn.cpu().numpy() * 255).astype(np.uint8)
                    kernel = np.ones((3,3), np.uint8)
                    dilated = cv2.dilate(mask_np, kernel, iterations=2)
                    save_path = f"{outdir}/n_{n:03d}_mask.png"
                    cv2.imwrite(save_path, dilated)

                    # depth 저장도 ref depth 기준으로
                    depth_out_stem = f"{outdir}/t_{t:03d}_depth"
                    save_depth_pngs(depth_ref.squeeze(), depth_out_stem,
                                    valid_mask=None,
                                    do_jet=True, do_gray=True,
                                    do_u16=False, u16_scale_mm=1000.0)
                    
                    depth_ref_npy_path = f"{outdir}/t_{t:03d}_depth.npy"
                    np.save(depth_ref_npy_path, depth_ref.squeeze().cpu().numpy())

                    depth_n = ret['depth_map_ref'].detach()
                    depth_n_npy_path = f"{outdir}/n_{t:03d}_depth.npy"
                    np.save(depth_n_npy_path, depth_n.squeeze().cpu().numpy())

                elif self.weights_map_ray:
                    wrote_coords_this_iter = False

                     # 2-1) GT 모션 마스크 우선 시도 (있으면 바로 사용)
                    
                    if self.use_mv_mask: 
                        gt_mask, gt_coords = self._load_gt_mv_mask(t, n, dilate_ks=3, dilate_iter=2)
                        if gt_mask is not None:
                            # 캐시 및 로그 저장 경로 재사용
                            cv2.imwrite(mask_path, gt_mask)  # n_{n:03d}_weights_map_mask.png 에 저장
                            np.save(coords_path, gt_coords)

                            # motion_coords에 push
                            self.motion_coords[t].append(gt_coords)
                            wrote_coords_this_iter = True
                            print(f"[MV-GT] used GT mask for t={t}, n={n} -> {gt_coords.shape[0]} coords")

                    # ret 기반 생성 (ret 있고, 키가 있을 때만)
                    if (not wrote_coords_this_iter) and (ret is not None) and ('weights_map_dd' in ret):
                        mask_png = weights_map_to_mask(
                            ret['weights_map_dd'],
                            mode='threshold',  # 또는 'quantile'
                            q=0.80,
                            thr=0.99
                        )

                        # kernel = np.ones((3,3), np.uint8)     # 3x3 structuring element
                        # mask_png = cv2.dilate(mask_png, kernel, iterations=2)

                        cv2.imwrite(mask_path, mask_png)
                        ys, xs = np.where(mask_png > 0)
                        coords = np.stack((ys, xs), axis=-1).astype(np.int64) if ys.size > 0 else np.empty((0,2), np.int64)
                        self.motion_coords[t].append(coords)
                        np.save(coords_path, coords)
                        wrote_coords_this_iter = True

                    # 캐시 히트 분기에서 이미 append 했다면 여기서 또 append 하지 않음
                    if not wrote_coords_this_iter and os.path.exists(coords_path) and len(self.motion_coords[t]) < len(neigh):
                        coords = np.load(coords_path)
                        if coords.ndim != 2 or coords.shape[-1] != 2:
                            coords = np.empty((0,2), dtype=np.int64)
                        else:
                            coords = coords.astype(np.int64, copy=False)
                        self.motion_coords[t].append(coords)

        
        self.synthetic_imgs = torch.stack(syn_imgs)
        self.synthetic_poses = torch.stack(syn_poses)


        if self.use_depth:
            viz_root = f"{self.save_root}/step_{step:06d}"
            self.compute_synthetic_depths(
                encoder='vitl',
                ckpt_path='./depth_anything_v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
                input_size=518,
                max_depth=40.0,
                save_numpy=True,
                save_viz=True,
                viz_grayscale=False,
                pred_only=True,
                viz_root=viz_root,
            )

    @torch.no_grad()
    def update_3(self, step, pipe, ref_imgs, cam_param, gt=True):
        """
            For difix ver 3
        """
        N = self.base_imgs.shape[0]
        syn_imgs, syn_poses = [], []
        for t in range(N):
            neigh = neighbor_indices(t, N, self.k, self.step)
            for n in neigh:
                outdir = f"{self.save_root}/step_{step:06d}/t_{t:03d}"
                img_path = f"{outdir}/n_{n:03d}.png"

                print(f'Time: {t}, n:{n}')
                if os.path.exists(img_path): # for debugging
                    pil_img = Image.open(img_path).convert("RGB")
                    enh_tensor = (pil_to_tensor(pil_img).float() / 255.0).permute(1,2,0) # H,W,C
                    pose = self.base_poses[n].to(device)
                elif gt:
                    pose = self.base_poses[n].to(device)
                    DELTA=0.5
                    forward = -pose[:3, 2]
                    forward /= torch.linalg.norm(forward)
                    pose[:3, 3] += DELTA * forward
                    img_idx_embed = t / float(N) * 2. - 1.0
                    ret = self.render(img_idx_embed, 0, False, float(N),
                                    self.H, self.W, self.focal,
                                    chunk=1024*16, c2w=pose,
                                    **self.kw_test)
                    
                    rgb = cam_param.RAD2LDR_img(ret['rgb_map_ref'], t).permute(2,0,1)
                    # import pdb; pdb.set_trace()
                    gt_img_path = os.path.join(self.data_dir, "images", f"{t:05d}.png")
                    gt_img = Image.open(gt_img_path).convert("RGB")
                    gt_tensor = (pil_to_tensor(gt_img).float() / 255.).to(rgb.device)
                    
                    # original size
                    rgb_resized = F.interpolate(
                                    rgb.unsqueeze(0),                   # [1, 3, 360, 641]
                                    size=(gt_tensor.shape[1], gt_tensor.shape[2]),                  # 목표 크기 (H, W)
                                    mode="bilinear",                    # 보간법 (bilinear/cbicubic 등 가능)
                                    align_corners=False
                                ).squeeze(0).clamp(0,1)

                    enh = pipe(
                        "remove degradation",
                        image=rgb_resized,
                        ref_image=gt_tensor,
                        num_inference_steps=1,
                        timesteps=[199],
                        guidance_scale=0.0,
                    ).images[0]

                    enh_resized = enh.resize((rgb.shape[2], rgb.shape[1]), Image.BILINEAR)
                    enh_tensor = pil_to_tensor(enh_resized).to(rgb.device).permute(1,2,0) / 255.

                    os.makedirs(outdir, exist_ok=True)
                    
                    enh_resized.save(f"{outdir}/n_{n:03d}.png")
                    rgb_pil = to_pil_image(rgb.cpu().clamp(0,1))
                    rgb_pil.save(f"{outdir}/n_{n:03d}_o.png")

                else:
                    pose = self.base_poses[n].to(device)
                    img_idx_embed = t / float(N) * 2. - 1.0
                    ret = self.render(img_idx_embed, 0, False, float(N),
                                    self.H, self.W, self.focal,
                                    chunk=1024*16, c2w=pose,
                                    **self.kw_test)
                    
                    rgb = cam_param.RAD2LDR_img(ret['rgb_map_ref'], t).permute(2,0,1).clamp(0,1)
                    enh = pipe(
                        "remove degradation",
                        image=rgb,
                        ref_image=ref_imgs[t].permute(2,0,1),
                        num_inference_steps=1,
                        timesteps=[199],
                        guidance_scale=0.0,
                    ).images[0]
                    enh_tensor = pil_to_tensor(enh).to(rgb.device).permute(1,2,0) / 255.

                    os.makedirs(outdir, exist_ok=True)
                    enh.save(f"{outdir}/n_{n:03d}.png")
                    rgb_pil = to_pil_image(rgb.cpu().clamp(0,1))
                    rgb_pil.save(f"{outdir}/n_{n:03d}_o.png")

                h,w,_ = enh_tensor.shape
                if (h,w) != (self.H, self.W):
                    if h == self.H and w == self.W - 1:
                        enh_tensor = F.pad(enh_tensor.permute(2,0,1),
                                           (0,1,0,0), mode='replicate').permute(1,2,0)
                    else:
                        enh_tensor=F.interpolate(enh_tensor.permute(2,0,1).unsqueeze(0),
                                                 size=(self.H, self.W),
                                                 mode='bilinear',
                                                 align_corners=False)[0].permute(1,2,0)
                
                syn_imgs.append(enh_tensor.cpu())
                syn_poses.append(pose.cpu())

                if self.mask_warp:
                    mask_syn = reproject_mask_to_view(
                                    mask_ref=self.base_motion_masks[t],           # [H,W] or adapt
                                    depth_ref=ret['depth_map_ref'].detach(),                    # [H,W]
                                    c2w_ref=self.base_poses[t,:3,:4],                 # [3,4]
                                    c2w_syn=self.base_poses[n,:3,:4],           # [3,4] (합성 뷰의 pose)
                                    H=self.H, W=self.W, focal=self.focal
                    )
                    mask_np = (mask_syn.cpu().numpy() * 255).astype(np.uint8)
                    # cv2.imwrite(save_path, mask_np)

                    kernel = np.ones((3,3), np.uint8)
                    dilated = cv2.dilate(mask_np, kernel, iterations=2)
                    save_path = f"{outdir}/n_{n:03d}_mask.png"
                    cv2.imwrite(save_path, dilated)
        
        self.synthetic_imgs = torch.stack(syn_imgs)
        self.synthetic_poses = torch.stack(syn_poses)


        if self.use_depth:
            viz_root = f"{self.save_root}/step_{step:06d}"
            self.compute_synthetic_depths(
                encoder='vitl',
                ckpt_path='./depth_anything_v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
                input_size=518,
                max_depth=40.0,
                save_numpy=True,
                save_viz=True,
                viz_grayscale=False,
                pred_only=True,
                viz_root=viz_root,
            )

    def init_depth_model(self, encoder: str, ckpt_path: str, max_depth: float, device: str):
        """
        Load Depth-Anything V2 model once.
        """

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError("Cannot import depth-anything-v2") from e
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in model_configs:
            raise ValueError(f"encoder must be one of {list(model_configs.keys())}")
        
        depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        state = torch.load(ckpt_path, map_location='cpu')
        depth_anything.load_state_dict(state)
        depth_anything = depth_anything.to(device).eval()

        return depth_anything

    @torch.no_grad()
    def compute_synthetic_depths(
        self,
        encoder: str = 'vitl',
        ckpt_path: str = './depth_anything_v2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
        input_size: int = 518,
        max_depth: float = 40.0,
        save_numpy: bool = False,
        save_viz: bool = False,
        viz_grayscale: bool = False,
        pred_only: bool = True,
        viz_root: str = None,   # 예: f"{self.save_root}/depth-anything"
    ):
        """
        Run Depth-Anything V2 on self.synthetic_imgs (list/torch.Tensor stack) and
        save results into self.synthetic_depths (torch.FloatTensor, shape [M,H,W]).
        - self.synthetic_imgs: Tensor [M,H,W,3] or list of HxWx3 float (0~1)
        """

        # 준비 체크
        if self.synthetic_imgs is None:
            raise RuntimeError("self.synthetic_imgs가 비어 있습니다. 먼저 update() 등으로 이미지를 채워주세요.")

        DEVICE = 'cuda' if torch.cuda.is_available() else \
                ('mps' if torch.backends.mps.is_available() else 'cpu')

        depth_anything = self.init_depth_model(encoder, ckpt_path, max_depth, DEVICE)

        # synthetic_imgs는 update()에서 torch.stack([...])로 저장됨: [M, H, W, 3], float, cpu
        syn = self.synthetic_imgs
        if isinstance(syn, list):
            syn_list = syn
        else:
            # Tensor -> list로 순회
            syn_list = [syn[i] for i in range(syn.shape[0])]

        depths = []
        # 시각화/저장 디렉토리 준비
        if (save_numpy or save_viz) and viz_root is not None:
            import matplotlib
            os.makedirs(viz_root, exist_ok=True)
            npy_dir = os.path.join(viz_root, 'depth-anything')
            viz_dir = os.path.join(viz_root, 'depth-anything-viz')
            if save_numpy:
                os.makedirs(npy_dir, exist_ok=True)
            if save_viz:
                os.makedirs(viz_dir, exist_ok=True)
            cmap = matplotlib.colormaps.get_cmap('Spectral')

        for idx, img in enumerate(syn_list):
            # img: torch.FloatTensor, [H,W,3], 0~1
            if torch.is_tensor(img):
                img_np = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # RGB
            else:
                # numpy일 경우 0~1 가정
                img_np = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)

            # OpenCV는 BGR을 기대
            bgr = img_np[:, :, ::-1]

            # 추론
            depth = depth_anything.infer_image(bgr, input_size)  # float32 [H,W], metric-ish up to max_depth
            depths.append(torch.from_numpy(depth).float())       # 저장용

            # 옵션: 저장
            if (save_numpy or save_viz) and viz_root is not None:
                base = f"syn_{idx:05d}"
                if save_numpy:
                    np.save(os.path.join(npy_dir, base + ".npy"), depth)

                if save_viz:
                    # 0-255 normalize for visualization
                    dmin, dmax = depth.min(), depth.max()
                    if dmax > dmin:
                        dvis = (depth - dmin) / (dmax - dmin) * 255.0
                    else:
                        dvis = np.zeros_like(depth)
                    dvis = dvis.astype(np.uint8)

                    if viz_grayscale:
                        dvis_rgb = np.repeat(dvis[..., None], 3, axis=-1)
                    else:
                        # Spectral colormap, convert to BGR for cv2.imwrite
                        dvis_rgb = (cmap(dvis)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]

                    if pred_only:
                        cv2.imwrite(os.path.join(viz_dir, base + ".png"), dvis_rgb)
                    else:
                        split_region = np.ones((bgr.shape[0], 50, 3), dtype=np.uint8) * 255
                        combined = cv2.hconcat([bgr, split_region, dvis_rgb])
                        cv2.imwrite(os.path.join(viz_dir, base + ".png"), combined)

        # [M,H,W] 텐서로 저장
        self.synthetic_depths = torch.stack(depths, dim=0)
