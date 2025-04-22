import os
import time
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data, color, img_as_float, util
from skimage.transform import resize
from main import svd_compress, adaptive_svd, dct_svd, svd_denoise, jpeg_compress

# Create output directory
os.makedirs("output", exist_ok=True)

# Load test images
sample_images = [data.astronaut(), data.chelsea(), data.coffee()]
sample_images = [(resize(img, (256, 256), anti_aliasing=True) * 255).astype(np.uint8) for img in sample_images]
sample_names = ["astronaut", "chelsea", "coffee"]

# Utility
def to_gray(img):
    return color.rgb2gray(img_as_float(img)) if img.ndim == 3 else img_as_float(img)

def save_image(name, img):
    path = os.path.join("output", name)
    cv2.imwrite(path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return path

# Evaluation function
def evaluate(method, image, **kwargs):
    start = time.time()
    if method == "standard_svd":
        result = svd_compress(image, kwargs.get("rank", 50))
    elif method == "adaptive_svd":
        result = adaptive_svd(image, kwargs.get("block_size", 32))
    elif method == "dct_svd":
        result = dct_svd(image)
    elif method == "svd_denoise":
        _, result = svd_denoise(image, rank=kwargs.get("rank", 50), noise_sigma=kwargs.get("sigma", 25))
    elif method == "jpeg":
        result = jpeg_compress(image, quality=kwargs.get("quality", 50))
    else:
        raise ValueError("Unknown method")

    elapsed = time.time() - start
    gray_orig = to_gray(image)
    gray_result = to_gray(np.clip(result, 0, 255))

    return result, {
        "psnr": psnr(gray_orig, gray_result, data_range=1.0),
        "ssim": ssim(gray_orig, gray_result, data_range=1.0),
        "time": elapsed
    }

# Modes and Params
modes = {
    "standard_svd": {"rank": 50},
    "adaptive_svd": {"block_size": 32},
    "dct_svd": {},
    "svd_denoise": {"rank": 50, "sigma": 25},
    "jpeg": {"quality": 50}
}

# Run evaluation
results = []

for name, img in zip(sample_names, sample_images):
    print(f"\n📂 Evaluating on {name}.jpg")
    for mode, params in modes.items():
        print(f"  ➤ {mode}...", end=" ", flush=True)
        output_img, metrics = evaluate(mode, img, **params)
        filename = f"{name}_{mode}.jpg"
        save_image(filename, output_img)
        results.append({
            "image": name,
            "method": mode,
            **metrics
        })
        print(f"Done (PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, Time={metrics['time']:.2f}s)")

# Save summary
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("output/evaluation_summary.csv", index=False)
print("\n✅ Evaluation complete. Results saved in 'output/evaluation_summary.csv'")
