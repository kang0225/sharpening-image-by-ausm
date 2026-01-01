import os, cv2, glob, random
import pandas as pd
import numpy as np
import params
from tqdm import tqdm
from typing import Tuple, List, Dict

def list_images(path: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(path, f"*{e}")))
    return sorted(files)

def get_sigma(img: np.ndarray) -> float:
    if img is None:
        return 1.0
        
    height, width = img.shape[:2]
    long_side = max(height, width)
    calculated_sigma = long_side / 250.0
    final_sigma = np.clip(calculated_sigma, 1.2, 4.8)
    return float(final_sigma)

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    #흑백 이미지와 같이 2채널 형태일 경우, 3채널(BGR)로 변형
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #RGBA나 BGRA와 같이 투명도(알파 채널)도 들어간 4채널 형태일 경우, 3채널(BGR)로 변형
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = (np.flip(img_bgr, axis=-1)).astype(np.float32) / 255
    return img_rgb

def make_lq_from_hq(hq_bgr: np.ndarray, scale: float, blur_sigma: float) -> np.ndarray:
    img_height, img_width = hq_bgr.shape[:2]

    # Down-Scaling & Up-Scaling
    to_width = max(1, int(img_width * scale))
    to_height = max(1, int(img_height * scale))

    down = cv2.resize(hq_bgr, (to_width, to_height), interpolation=cv2.INTER_AREA)
    up   = cv2.resize(down, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    # Gaussian Blur for removing noise, aliasing
    if blur_sigma > 0:
        blur_img = cv2.GaussianBlur(up, (0,0), sigmaX=blur_sigma, sigmaY=0)
    else:
        blur_img = up
    return blur_img

def rgb_to_luma(rgb: np.ndarray) -> np.ndarray: # BT.601 근사 
    return (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32) 

def grid_search_for_image(lq_rgb: np.ndarray, hq_rgb: np.ndarray) -> np.ndarray:
    lq_rgb_float32 = lq_rgb.astype(np.float32)
    hq_rgb_float32 = hq_rgb.astype(np.float32)
    sigma = 2.0
    patch = params.patch
    stride = params.stride
    k_values = params.k_values
    blur = cv2.GaussianBlur(lq_rgb_float32, (0, 0), sigmaX=2.0, sigmaY=0)
    detail = lq_rgb_float32 - blur
    height, width = lq_rgb_float32.shape[:2]
    r = patch // 2
    kmap = np.zeros((height, width), dtype=np.float32)

    def psnr_float(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
        mse = float(np.mean((a - b) ** 2))
        return 100.0 if mse <= eps else 10.0 * np.log10(1.0 / mse)

    for y in range(r, height - r, stride):
        for x in range(r, width - r, stride):
            ys, ye, xs, xe = y - r, y + r + 1, x - r, x + r + 1
            I_p, GT_p, det_p = lq_rgb_float32[ys:ye, xs:xe], hq_rgb_float32[ys:ye, xs:xe], detail[ys:ye, xs:xe]
            best_k, best_ps = 0.0, -1e9
            for k in k_values:
                Sh_p = np.clip(I_p + k * det_p, 0.0, 1.0)
                ps = psnr_float(Sh_p, GT_p)
                if ps > best_ps:
                    best_ps, best_k = ps, k
            kmap[y, x] = best_k
            
    kmap = cv2.GaussianBlur(kmap, (0, 0), sigmaX=float(stride))
    kmin, kmax = float(min(k_values)), float(max(k_values))
    kmap = np.clip(kmap, kmin, kmax)
    return kmap

def extract_features(patch_luma: np.ndarray) -> Dict[str, float]:
    mean, std = float(np.mean(patch_luma)), float(np.std(patch_luma))
    f = np.fft.fft2(patch_luma)
    fshift = np.fft.fftshift(f)
    mag_spec = np.abs(fshift)

    rows, cols = patch_luma.shape
    crow, ccol = rows // 2, cols // 2

    p = 0.25
    lr, lc = int(rows * p), int(cols * p)

    total_energy = float(np.sum(mag_spec))
    high_freq_energy = total_energy - float(np.sum(mag_spec[crow-lr:crow+lr, ccol-lc:ccol+lc]))
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 1e-6 else 0.0
    return {"mean": mean, "std": std, "high_freq_ratio": high_freq_ratio}


def generate_dataset() -> str:
    os.makedirs(params.output_dir, exist_ok=True)

    hq_paths = list_images(params.train_dir, params.img_extensions)
    if not hq_paths:
        print(f"Can't find image: '{params.train_dir}'")
        return None
        
    all_patch_data = []
    print("Creating Data Set...")
    for hq_path in tqdm(hq_paths, desc="Processing Images"):
        hq_bgr = ensure_3ch(cv2.imread(hq_path))
        lq_bgr = make_lq_from_hq(hq_bgr, params.DOWNSCALE, params.LQ_BLUR_SIGMA)
        hq_rgb, lq_rgb = bgr_to_rgb(hq_bgr), bgr_to_rgb(lq_bgr)
        k_map = grid_search_for_image(lq_rgb, hq_rgb)

        height, width = lq_rgb.shape[:2]
        stride, r = params.stride, params.patch // 2
        lq_luma = rgb_to_luma(lq_rgb)
        
        for y in range(r, height - r, stride):
            for x in range(r, width - r, stride):
                patched = lq_luma[y-r:y+r+1, x-r:x+r+1]
                features = extract_features(patched)

                features['target_k'] = k_map[y, x]
                all_patch_data.append(features)

    df = pd.DataFrame(all_patch_data)
    output_path = os.path.join(params.output_dir, "training_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Successfully created. : {output_path}")
    return output_path