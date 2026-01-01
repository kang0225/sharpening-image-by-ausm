import os
import cv2
import joblib
import numpy as np
import pandas as pd

import params as pr
import preprocessing as pre 

MODEL_NAME = "xgb_model.joblib"

def _fft_high_freq_ratio(gray: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * cutoff_ratio)
    Y, X = np.ogrid[:h, :w]
    mask_low = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    low_e = mag[mask_low].sum()
    high_e = mag[~mask_low].sum()
    return float(high_e / (low_e + high_e + 1e-8))

def dispatcher_extract_features(img_bgr: np.ndarray) -> dict:
    luma = pre.rgb_to_luma(pre.bgr_to_rgb(img_bgr))
    return pre.extract_features(luma) 

def build_k_map(img_bgr: np.ndarray, model, patch=64, stride=None, k_clip=(0.0, 5.0)):
    if stride is None:
        stride = patch
    h, w = img_bgr.shape[:2]
    kh = (h + stride - 1) // stride
    kw = (w + stride - 1) // stride
    k_grid = np.zeros((kh, kw), dtype=np.float32)

    rows = []
    locs = []
    for gy, y in enumerate(range(0, h, stride)):
        for gx, x in enumerate(range(0, w, stride)):
            patch_img = img_bgr[y:min(y+patch, h), x:min(x+patch, w)]
            feats = dispatcher_extract_features(patch_img)
            rows.append([feats["mean"], feats["std"], feats["high_freq_ratio"]])
            locs.append((gy, gx))

    df = pd.DataFrame(rows, columns=["mean", "std", "high_freq_ratio"])
    k_pred = model.predict(df).astype(np.float32)
    k_pred = np.clip(k_pred, k_clip[0], k_clip[1])

    for (gy, gx), k in zip(locs, k_pred):
        k_grid[gy, gx] = k
    k_map = cv2.resize(k_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return k_map

def unsharp_with_kmap(img_bgr: np.ndarray, k_map: np.ndarray, sigma=2.0, kernel_size=0):
    img = img_bgr.astype(np.float32)
    if kernel_size and kernel_size % 2 == 1:
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma, sigma)
    else:
        blur = cv2.GaussianBlur(img, (0, 0), sigma, sigma)
    mask = img - blur
    out = img + (k_map[..., None]) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

def enhance_image_patchwise(input_path, patch, stride):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    sigma = pre.get_sigma(img)
    print("[INFO] 사용된 시그마 : ", float(sigma))
    if img is None:
        raise FileNotFoundError(f"이미지 없음.")

    model_path = os.path.join(pr.output_dir, MODEL_NAME)
    model = joblib.load(model_path)
    
    k_map = build_k_map(img, model, patch, stride, k_clip=(0, 100))

    k_map_amplified = k_map * 125.0
    k_map_final = np.clip(k_map_amplified, 0.0, 5.0)

    out = unsharp_with_kmap(img, k_map_final, sigma)
    
    result_path = "./result/sharpened_image.png"
    cv2.imwrite(result_path, out)
    
    return result_path