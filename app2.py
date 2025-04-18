# import streamlit as st
# import numpy as np
# import cv2
# import os
# import time
# import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
# from sklearn.datasets import fetch_olivetti_faces
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# from io import BytesIO
# from PIL import Image

# # --- Utility Functions ---
# def load_image(uploaded_file):
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# def svd_compress_reconstruct(image, k):
#     if len(image.shape) == 3:
#         channels = []
#         for c in range(3):
#             U, S, Vt = np.linalg.svd(image[:, :, c], full_matrices=False)
#             S_k = np.diag(S[:k])
#             U_k = U[:, :k]
#             Vt_k = Vt[:k, :]
#             recon = U_k @ S_k @ Vt_k
#             channels.append(recon)
#         recon_image = np.stack(channels, axis=-1)
#     else:
#         U, S, Vt = np.linalg.svd(image, full_matrices=False)
#         S_k = np.diag(S[:k])
#         U_k = U[:, :k]
#         Vt_k = Vt[:k, :]
#         recon_image = U_k @ S_k @ Vt_k
#     return np.clip(recon_image, 0, 255).astype(np.uint8)

# def calculate_metrics(original, reconstructed):
#     original = original.astype(np.float32) / 255.0
#     reconstructed = reconstructed.astype(np.float32) / 255.0
#     psnr_val = psnr(original, reconstructed, data_range=1.0)
#     ssim_val = ssim(original, reconstructed, data_range=1.0, channel_axis=-1)
#     return psnr_val, ssim_val

# def image_to_bytes(image, format="PNG"):
#     pil_img = Image.fromarray(image)
#     buf = BytesIO()
#     pil_img.save(buf, format=format, optimize=True, quality=85 if format == "JPEG" else None)
#     size_kb = len(buf.getvalue()) / 1024
#     return buf.getvalue(), format, size_kb

# # --- Streamlit App ---
# st.set_page_config(page_title="SVD Image Compression App", layout="wide")
# st.title("📸 Interactive SVD Image Compression & Denoising")

# uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# # Webcam capture and snapshot
# class VideoProcessor(VideoTransformerBase):
#     def __init__(self):
#         self.frame = None

#     def transform(self, frame):
#         self.frame = frame.to_ndarray(format="bgr24")
#         return self.frame

# # st.subheader("📷 Or Capture from Webcam")
# # ctx = webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)
# # webcam_img = None
# # capture_btn = st.button("📸 Capture Snapshot from Webcam")

# # if ctx.video_processor and capture_btn:
# #     if ctx.video_processor.frame is not None:
# #         webcam_img = cv2.cvtColor(ctx.video_processor.frame, cv2.COLOR_BGR2RGB)
# #         st.image(webcam_img, caption="Captured Image", use_container_width=True)
# #         image_captured = True

# image_source = None
# if uploaded_file is not None:
#     image = load_image(uploaded_file)
#     image_source = image

# if image_source is not None:
#     k_max = min(image.shape[0], image.shape[1])
#     k = st.slider("Select number of singular values (k)", 1, k_max, value=50)

#     # Show original size
#     original_bytes_png, _, original_size = image_to_bytes(image_source, format="PNG")
#     st.caption(f"Original Image Size: {original_size:.2f} KB")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         if st.button("Compress Image"):
#             compressed_image = svd_compress_reconstruct(image, k)
#             psnr_val, ssim_val = calculate_metrics(image, compressed_image)

#             st.image([image, compressed_image], caption=["Original", f"Compressed (k={k})"], width=300)
#             st.success(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

#             image_bytes_png, _, size_png = image_to_bytes(compressed_image, format="PNG")
#             image_bytes_jpeg, _, size_jpeg = image_to_bytes(compressed_image, format="JPEG")

#             st.download_button("📥 Download as PNG",
#                                data=image_bytes_png,
#                                file_name="compressed_image.png")
#             st.caption(f"PNG Size: {size_png:.2f} KB ({100 * size_png/original_size:.1f}% of original)")

#             st.download_button("📥 Download as JPEG (smaller size)",
#                                data=image_bytes_jpeg,
#                                file_name="compressed_image.jpg")
#             st.caption(f"JPEG Size: {size_jpeg:.2f} KB ({100 * size_jpeg/original_size:.1f}% of original)")

#     with col2:
#         if st.button("Animate Compression"):
#             st.write("Animating compression from k=1 to k={}".format(k))
#             img_placeholder = st.empty()
#             for ki in range(1, k + 1, max(1, k // 30)):
#                 recon = svd_compress_reconstruct(image, ki)
#                 img_placeholder.image(recon, caption=f"k={ki}", use_container_width=True)
#                 time.sleep(0.1)

#     with col3:
#         if st.button("Add Gaussian Noise and Denoise"):
#             noise = np.random.normal(0, 25, image.shape)
#             noisy_img = np.clip(image + noise, 0, 255).astype(np.uint8)
#             denoised_img = svd_compress_reconstruct(noisy_img, k)

#             psnr_noisy, ssim_noisy = calculate_metrics(image, noisy_img)
#             psnr_denoised, ssim_denoised = calculate_metrics(image, denoised_img)

#             st.image([image, noisy_img, denoised_img], caption=["Original", "Noisy", f"Denoised (k={k})"], width=300)
#             st.write(f"Noisy Image - PSNR: {psnr_noisy:.2f} | SSIM: {ssim_noisy:.4f}")
#             st.write(f"Denoised Image - PSNR: {psnr_denoised:.2f} | SSIM: {ssim_denoised:.4f}")

#             denoised_bytes_png, _, dsize_png = image_to_bytes(denoised_img, format="PNG")
#             denoised_bytes_jpeg, _, dsize_jpeg = image_to_bytes(denoised_img, format="JPEG")

#             st.download_button("📥 Download Denoised as PNG",
#                                data=denoised_bytes_png,
#                                file_name="denoised_image.png")
#             st.caption(f"PNG Size: {dsize_png:.2f} KB ({100 * dsize_png/original_size:.1f}% of original)")

#             st.download_button("📥 Download Denoised as JPEG (smaller)",
#                                data=denoised_bytes_jpeg,
#                                file_name="denoised_image.jpg")
#             st.caption(f"JPEG Size: {dsize_jpeg:.2f} KB ({100 * dsize_jpeg/original_size:.1f}% of original)")

# # --- Footer ---
# st.markdown("---")
# st.markdown("**Made with 💡 using SVD, Streamlit, and NumPy**")
# st.markdown("**Author: Ayush Prajapati**")

# Enhanced SVD Project: Digital Image Processing Framework

# Section 1: Imports and Setup
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

# Section 3: Groundwork - Load Sample Dataset and Train/Test
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