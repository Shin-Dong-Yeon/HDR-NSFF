
# import os
# import glob
# import cv2
# import numpy as np

# def rgb_to_LuvL(img_rgb_uint8):
#     luv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Luv)
#     L = luv[..., 0].astype(np.float32)
#     uv = luv[..., 1:].astype(np.float32)
#     return L, uv

# def estimate_affine_L(L_src, L_tgt, sat_thr_low=2, sat_thr_high=253):
#     # 포화/암부 제외하고 a,b 최소제곱 추정 (L' = aL + b)
#     mask = (L_src > sat_thr_low) & (L_src < sat_thr_high) & \
#            (L_tgt > sat_thr_low) & (L_tgt < sat_thr_high)
#     x = L_src[mask].reshape(-1,1).astype(np.float64)
#     y = L_tgt[mask].reshape(-1,1).astype(np.float64)
#     if len(x) < 100:
#         return 1.0, 0.0
#     A = np.hstack([x, np.ones_like(x)])
#     sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
#     a, b = sol.flatten()
#     return float(a), float(b)

# def match_exposure_L(L_src, L_tgt):
#     a, b = estimate_affine_L(L_src, L_tgt)
#     L_adj = a * L_src + b
#     return np.clip(L_adj, 0, 255).astype(np.float32), (a, b)

# # ===== 설정 =====
# dir_path = "/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/250213/big_jump_1_36/dense/images"
# ref_name = "00000.png"   # 기준 프레임(없으면 자동으로 첫 이미지 사용)

# # ===== 입력 이미지 목록 수집/정렬 =====
# all_pngs = glob.glob(os.path.join(dir_path, "*.png"))
# if not all_pngs:
#     raise FileNotFoundError(f"No PNGs in: {dir_path}")

# def numeric_key(p):
#     bn = os.path.basename(p)
#     stem = os.path.splitext(bn)[0]
#     try:
#         return int(stem)
#     except ValueError:
#         return bn  # 숫자가 아니면 문자열로 정렬

# all_pngs = sorted(all_pngs, key=numeric_key)

# # 기준 프레임 결정
# ref_path = os.path.join(dir_path, ref_name)
# if ref_path not in all_pngs:
#     ref_path = all_pngs[0]
#     ref_name = os.path.basename(ref_path)
#     print(f"[WARN] Provided ref not found. Using first image as reference: {ref_name}")

# # 저장 디렉토리 (기준 프레임 이름 반영)
# save_dir = os.path.join(os.getcwd(), f"full_res_ref_{os.path.splitext(ref_name)[0]}")
# os.makedirs(save_dir, exist_ok=True)

# # ===== 레퍼런스 L 계산 =====
# ref_bgr = cv2.imread(ref_path)
# if ref_bgr is None:
#     raise FileNotFoundError(f"Reference image not readable: {ref_path}")
# ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
# L_ref, _ = rgb_to_LuvL(ref_rgb)

# print(f"[INFO] Reference frame: {ref_name} ({ref_bgr.shape[1]}x{ref_bgr.shape[0]})")
# print(f"[INFO] Total images: {len(all_pngs)}  → Output dir: {save_dir}")

# # ===== 전체 처리 =====
# for img_path in all_pngs:
#     nm = os.path.basename(img_path)

#     bgr = cv2.imread(img_path)
#     if bgr is None:
#         print(f"[WARN] Cannot read: {img_path} (skip)")
#         continue
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     L_cur, _ = rgb_to_LuvL(rgb)

#     if img_path == ref_path:
#         L_out = L_cur
#         print(f"[INFO] {nm}: using reference L as-is")
#     else:
#         L_out, (a, b) = match_exposure_L(L_cur, L_ref)
#         print(f"[INFO] {nm}: matched to {ref_name} with a={a:.4f}, b={b:.2f}")

#     # 3채널로 복제하여 저장 (HxWx3, uint8)
#     L_uint8 = np.clip(L_out, 0, 255).astype(np.uint8)
#     L_3ch = np.repeat(L_uint8[..., None], 3, axis=2)  # (H,W,3)

#     save_path = os.path.join(save_dir, os.path.splitext(nm)[0] + "_L.png")
#     ok = cv2.imwrite(save_path, L_3ch)
#     if not ok:
#         print(f"[WARN] Failed to save: {save_path}")
#     else:
#         print(f"[SAVE] {save_path}")

import os
import glob
import cv2
import numpy as np
import argparse

def rgb_to_LuvL(img_rgb_uint8):
    luv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2Luv)
    L = luv[..., 0].astype(np.float32)
    uv = luv[..., 1:].astype(np.float32)
    return L, uv

def estimate_affine_L(L_src, L_tgt, sat_thr_low=2, sat_thr_high=253):
    # 포화/암부 제외하고 a,b 최소제곱 추정 (L' = aL + b)
    mask = (L_src > sat_thr_low) & (L_src < sat_thr_high) & \
           (L_tgt > sat_thr_low) & (L_tgt < sat_thr_high)
    x = L_src[mask].reshape(-1,1).astype(np.float64)
    y = L_tgt[mask].reshape(-1,1).astype(np.float64)
    if len(x) < 100:
        return 1.0, 0.0
    A = np.hstack([x, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol.flatten()
    return float(a), float(b)

def match_exposure_L(L_src, L_tgt):
    a, b = estimate_affine_L(L_src, L_tgt)
    L_adj = a * L_src + b
    return np.clip(L_adj, 0, 255).astype(np.float32), (a, b)

def numeric_key(p):
    bn = os.path.basename(p)
    stem = os.path.splitext(bn)[0]
    try:
        return int(stem)
    except ValueError:
        return bn  # 숫자가 아니면 문자열 정렬

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_path", required=True,
                    help="입력 이미지 폴더 (예: .../dense/images)")
    ap.add_argument("--ref_name", default="00000.png",
                    help="기준 프레임 파일명 (폴더에 없으면 첫 이미지 사용)")
    ap.add_argument("--save_dir", default="",
                    help="결과 저장 폴더 (미지정 시 full_res_ref_<ref> 생성)")
    # normalize on/off 옵션
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="정규화(레퍼런스 매칭) 없이 raw L만 저장")
    ap.set_defaults(normalize=True)
    args = ap.parse_args()

    # 입력 수집
    all_pngs = glob.glob(os.path.join(args.dir_path, "*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No PNGs in: {args.dir_path}")
    all_pngs = sorted(all_pngs, key=numeric_key)

    # 기준 프레임 결정
    ref_path = os.path.join(args.dir_path, args.ref_name)
    if ref_path not in all_pngs:
        ref_path = all_pngs[0]
        args.ref_name = os.path.basename(ref_path)
        print(f"[WARN] Provided ref not found. Using first image as reference: {args.ref_name}")

    # 저장 폴더
    save_dir = args.save_dir.strip()
    if save_dir == "":
        save_dir = os.path.join(os.getcwd(), f"full_res_ref_{os.path.splitext(args.ref_name)[0]}")
    os.makedirs(save_dir, exist_ok=True)

    # 레퍼런스 L
    ref_bgr = cv2.imread(ref_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference image not readable: {ref_path}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    L_ref, _ = rgb_to_LuvL(ref_rgb)

    print(f"[INFO] Reference frame: {args.ref_name} ({ref_bgr.shape[1]}x{ref_bgr.shape[0]})")
    print(f"[INFO] Total images: {len(all_pngs)}  → Output dir: {save_dir}")
    print(f"[INFO] Normalize exposure: {args.normalize}")

    # 전체 처리
    for img_path in all_pngs:
        nm = os.path.basename(img_path)
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path} (skip)")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L_cur, _ = rgb_to_LuvL(rgb)

        if not args.normalize:
            # 정규화 없이 raw L 저장
            L_out = L_cur
            suffix = "_L_raw.png"
            if img_path == ref_path:
                print(f"[INFO] {nm}: raw L (reference as raw)")
            else:
                print(f"[INFO] {nm}: raw L (no matching)")
        else:
            # 정규화 수행
            if img_path == ref_path:
                L_out = L_cur
                print(f"[INFO] {nm}: using reference L as-is")
            else:
                L_out, (a, b) = match_exposure_L(L_cur, L_ref)
                print(f"[INFO] {nm}: matched to {args.ref_name} with a={a:.4f}, b={b:.2f}")
            suffix = "_L.png"

        # 3채널로 저장
        L_uint8 = np.clip(L_out, 0, 255).astype(np.uint8)
        L_3ch = np.repeat(L_uint8[..., None], 3, axis=2)
        save_path = os.path.join(save_dir, os.path.splitext(nm)[0] + suffix)
        ok = cv2.imwrite(save_path, L_3ch)
        if not ok:
            print(f"[WARN] Failed to save: {save_path}")
        else:
            print(f"[SAVE] {save_path}")

if __name__ == "__main__":
    main()
