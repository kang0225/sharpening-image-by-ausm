import random

train_dir = "./images"
input_dir = "./image_input/new_img.png"
output_dir = "./result"

img_extensions = (".png", ".jpg", ".jpeg")

# LQ Creation Parameters for Downsampling HQ
DOWNSCALE = 0.5
LQ_BLUR_SIGMA = random.uniform(0.5, 3.5)

# Unsharp Mask
k_values = [x * 0.1 for x in range(0, 51)]
patch = 16
stride = 16

# SSIM vs PSNR Weights
SSIM_WEIGHT = 0.7
PSNR_WEIGHT = 0.3 