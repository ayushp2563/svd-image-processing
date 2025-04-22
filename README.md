# Enhanced SVD Image Processing

A complete interactive web application for image compression, denoising, and comparative analysis using Singular Value Decomposition (SVD), built with **Python** and **Streamlit**.

## 🔧 Features
- 📷 Upload or capture images via webcam
- 📉 Compress images using:
  - Standard SVD
  - Adaptive block-wise SVD
  - DCT + SVD hybrid
  - JPEG baseline compression for comparison
- 🔊 Add and denoise Gaussian noise
- 📥 Download results in PNG and JPEG
- 📊 Compare file sizes and compute PSNR/SSIM quality metrics

## 📦 Technologies Used
- **Streamlit** for frontend UI
- **NumPy** & **OpenCV** for image operations
- **scikit-image** for evaluation metrics
- **PIL** for format conversion
- **Plotly/Matplotlib** for optional visualizations

## 🚀 Getting Started
### 1. Clone this repository:
```bash
git clone https://github.com/ayushp2563/svd-image-processing
cd svd-image-processing
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the application:
```bash
streamlit run main.py
```


### 4. Run the Evulation Script:
```bash
python evaluate.py
```


## 📁 Folder Structure
```
.
├── main.py                       # Main Streamlit App
├── requirements.txt              # Python dependencies
├── evaluate.py                   # Evaluation Script
└── README.md                     # This file
```

# Image Compression and Denoising Evaluation

This script processes a batch of images and outputs the following:

- **Compressed or Denoised Images**: Saved in the `output/` directory.
- **Evaluation Summary**: A CSV file named `evaluation_summary.csv` containing performance metrics.

## Output Details

### 1. Processed Images

All input images are processed using various compression or denoising methods. The resulting images are saved in the `output/` folder.

### 2. Evaluation Summary (evaluation_summary.csv)

This CSV file provides a summary of the evaluation for each processed image, including the following information:

| Column Name   | Description                                      |
|---------------|--------------------------------------------------|
| image name    | Name of the original image file                  |
| method used   | Compression or denoising method applied          |
| PSNR          | Peak Signal-to-Noise Ratio of the result         |
| SSIM          | Structural Similarity Index of the result        |
| runtime       | Time taken to process the image (in seconds)     |

## Example

After running the script, you should see:

- Processed image files inside `output/`
- A summary CSV: `evaluation_summary.csv`

## Usage

To run the script, simply execute:

```bash
python evaluate.py
```
## ✨ Screenshots
![UI Screenshot](assets/ui_demo.png)

## 📊 Metrics Used
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

## 🔮 Future Improvements
- Real-time face detection with compression focus
- Batch processing support
- Comparison with WebP and deep learning models

## 👨‍💻 Author
**Ayush Prajapati**  
MSc Computational Sciences – Laurentian University  
[Portfolio](https://prajapatiayush.vercel.app/) | [LinkedIn](https://linkedin.com/in/ayush-p-prajapati)

## 🌐 Live Demo
Check out the live demo here: [https://svd-image-processing.streamlit.app](https://svd-image-processing.streamlit.app/)

## 📜 License
This project is open source under the MIT License.

