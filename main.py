
#Section 1:libraries
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.cluster import KMeans
import time
import os
from skimage import data, color, img_as_float, util
from skimage.transform import resize
from PIL import Image
import io

# Section 2: Utility Functions

def svd_compress(image, rank):
    if image.ndim == 3:
        h, w, c = image.shape
        compressed_channels = []
        for i in range(c):
            U, S, VT = np.linalg.svd(image[:, :, i], full_matrices=False)
            S = np.diag(S[:rank])
            U = U[:, :rank]
            VT = VT[:rank, :]
            compressed = np.dot(U, np.dot(S, VT))
            compressed_channels.append(compressed)
        return np.clip(np.stack(compressed_channels, axis=2), 0, 255)
    else:
        U, S, VT = np.linalg.svd(image, full_matrices=False)
        S = np.diag(S[:rank])
        U = U[:, :rank]
        VT = VT[:rank, :]
        compressed = np.dot(U, np.dot(S, VT))
        return np.clip(compressed, 0, 255)

def adaptive_svd(image, block_size=32):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    compressed_image = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.size == 0:
                continue
            edge_strength = np.std(block)
            rank = min(max(1, int(edge_strength * 0.5)), block.shape[0])
            comp_block = svd_compress(block, rank)
            compressed_image[i:i+block_size, j:j+block_size] = comp_block[:block.shape[0], :block.shape[1]]
    return compressed_image

def dct_svd(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')
    compressed = svd_compress(dct_image, rank=50)
    idct_image = idct(idct(compressed.T, norm='ortho').T, norm='ortho')
    return idct_image

def svd_denoise(image, rank=50, noise_sigma=25):
    noisy = util.random_noise(image, var=(noise_sigma/255.0)**2)
    noisy = (255 * noisy).astype(np.uint8)
    denoised = svd_compress(noisy, rank)
    return noisy, denoised

def jpeg_compress(image, quality=50):
    img_pil = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    return np.array(compressed_img)

# Section 3:Load Sample Dataset and Test
@st.cache_data
def load_sample_images():
    sample_images = []
    for i in range(3):
        image = data.astronaut() if i == 0 else data.chelsea() if i == 1 else data.coffee()
        image = resize(image, (256, 256), anti_aliasing=True)
        image = (image * 255).astype(np.uint8)
        sample_images.append(image)
    return sample_images

# Section 4: Streamlit UI
st.title("Enhanced SVD Image Processing")

st.markdown("### Upload or Choose a Sample Image")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
sample_images = load_sample_images()
use_sample = st.checkbox("Use Sample Dataset")

if uploaded_file or use_sample:
    if use_sample:
        selected_img = st.selectbox("Select Sample", ["Astronaut", "Chelsea", "Coffee"])
        img = sample_images[["Astronaut", "Chelsea", "Coffee"].index(selected_img)]
    else:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    st.image(img, caption='Original Image', use_container_width=True)

    mode = st.selectbox("Select Mode", ["Standard SVD", "Adaptive SVD", "DCT + SVD", "SVD Denoising", "JPEG Compression"])

    if mode == "Standard SVD":
        rank = st.slider("Select rank", 1, 100, 50)
        start = time.time()
        compressed = svd_compress(img, rank)
        elapsed = time.time() - start
        st.image(compressed.astype(np.uint8), caption=f"Compressed with rank {rank} in {elapsed:.2f}s", use_container_width=True)
        gray_img = img_as_float(img)
        gray_compressed = img_as_float(np.clip(compressed, 0, 255))
        if gray_img.ndim == 3:
            gray_img = color.rgb2gray(gray_img)
        if gray_compressed.ndim == 3:
            gray_compressed = color.rgb2gray(gray_compressed)
        st.write(f"PSNR: {psnr(gray_img, gray_compressed, data_range=1.0):.2f}, SSIM: {ssim(gray_img, gray_compressed, data_range=1.0):.4f}")

    elif mode == "Adaptive SVD":
        block_size = st.slider("Block Size", 8, 64, 32)
        start = time.time()
        compressed = adaptive_svd(img, block_size)
        elapsed = time.time() - start
        st.image(compressed.astype(np.uint8), caption=f"Adaptive Compression in {elapsed:.2f}s", use_container_width=True)
        gray_img = img_as_float(img)
        gray_compressed = img_as_float(np.clip(compressed, 0, 255))
        if gray_img.ndim == 3:
            gray_img = color.rgb2gray(gray_img)
        if gray_compressed.ndim == 3:
            gray_compressed = color.rgb2gray(gray_compressed)
        st.write(f"PSNR: {psnr(gray_img, gray_compressed, data_range=1.0):.2f}, SSIM: {ssim(gray_img, gray_compressed, data_range=1.0):.4f}")

    elif mode == "DCT + SVD":
        start = time.time()
        compressed = dct_svd(img)
        elapsed = time.time() - start
        st.image(compressed.astype(np.uint8), caption=f"DCT + SVD Compression in {elapsed:.2f}s", use_container_width=True)
        gray_img = img_as_float(img)
        gray_compressed = img_as_float(np.clip(compressed, 0, 255))
        if gray_img.ndim == 3:
            gray_img = color.rgb2gray(gray_img)
        if gray_compressed.ndim == 3:
            gray_compressed = color.rgb2gray(gray_compressed)
        st.write(f"PSNR: {psnr(gray_img, gray_compressed, data_range=1.0):.2f}, SSIM: {ssim(gray_img, gray_compressed, data_range=1.0):.4f}")

    elif mode == "SVD Denoising":
        rank = st.slider("Denoising Rank", 1, 100, 50)
        sigma = st.slider("Noise Sigma", 5, 75, 25)
        noisy, denoised = svd_denoise(img, rank=rank, noise_sigma=sigma)
        st.image(noisy.astype(np.uint8), caption="Noisy Image", use_container_width=True)
        st.image(denoised.astype(np.uint8), caption=f"Denoised Image using SVD (rank={rank})", use_container_width=True)

    elif mode == "JPEG Compression":
        quality = st.slider("JPEG Quality", 10, 95, 50)
        compressed = jpeg_compress(img, quality)
        st.image(compressed, caption=f"JPEG Compressed Image (Quality = {quality})", use_container_width=True)
        gray_img = img_as_float(img)
        gray_compressed = img_as_float(np.clip(compressed, 0, 255))
        if gray_img.ndim == 3:
            gray_img = color.rgb2gray(gray_img)
        if gray_compressed.ndim == 3:
            gray_compressed = color.rgb2gray(gray_compressed)
        st.write(f"PSNR: {psnr(gray_img, gray_compressed, data_range=1.0):.2f}, SSIM: {ssim(gray_img, gray_compressed, data_range=1.0):.4f}")

        buffer_png = io.BytesIO()
        buffer_jpg = io.BytesIO()
        out_img = Image.fromarray(np.uint8(compressed))
        out_img.save(buffer_png, format="PNG", optimize=True)
        out_img.save(buffer_jpg, format="JPEG", quality=85)
        size_original = img.size if isinstance(img, np.ndarray) else out_img.size
        size_png = len(buffer_png.getvalue()) / 1024
        size_jpg = len(buffer_jpg.getvalue()) / 1024

        st.download_button("📥 Download as PNG", data=buffer_png.getvalue(), file_name="compressed.png", mime="image/png")
        st.caption(f"PNG Size: {size_png:.2f} KB")

        st.download_button("📥 Download as JPEG", data=buffer_jpg.getvalue(), file_name="compressed.jpg", mime="image/jpeg")
        st.caption(f"JPEG Size: {size_jpg:.2f} KB")

        st.write(f"📊 PNG is {(size_png / size_jpg * 100):.1f}% the size of JPEG")

        st.write(f"📊 JPEG is {(size_jpg / size_png * 100):.1f}% the size of PNG")
        
st.markdown("---")
st.markdown("Built with ❤️ for advanced image processing with SVD")
st.markdown("**Author: Ayush Prajapati**")