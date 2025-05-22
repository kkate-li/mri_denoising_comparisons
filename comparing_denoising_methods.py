import cv2
import numpy as np
import pywt

def wavelet_denoising(img, wavelet='db1', level=2, threshold=0.05):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_float = np.float32(img) / 255.0
    
    # Wavelet decomposition
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)
    
    # Hard thresholding
    coeffs_thresh = [coeffs[0]]  
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(
            tuple(np.where(np.abs(detail) < threshold, 0, detail) 
            for detail in coeffs[i])
        )
    
    # Reconstruct the image
    img_denoised = pywt.waverec2(coeffs_thresh, wavelet)
    img_denoised = np.clip(img_denoised, 0, 1)
    img_denoised = (img_denoised * 255).astype(np.uint8)
    
    return img_denoised

def fourier_denoise(img, cutoff_ratio=0.1):
    """Denoise image using Fourier low-pass filter"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_float = np.float32(img)
    
    # 2D Fourier Transform
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create low-pass mask
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    radius = int(min(rows, cols) * cutoff_ratio)
    cv2.circle(mask, (ccol, crow), radius, (1,1), -1)
    
    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return np.uint8(np.clip(img_back, 0, 255))

# Load noisy image
img_noise = cv2.imread('images/AD/moderateDem4.jpg')  # Replace with your image path

# Convert to grayscale for PSNR comparison
if len(img_noise.shape) == 3:
    img_noise_gray = cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY)
else:
    img_noise_gray = img_noise

# Apply denoising methods
bilateral_denoised = cv2.bilateralFilter(img_noise, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)
wavelet_denoised = wavelet_denoising(img_noise)
nlm_denoised = cv2.fastNlMeansDenoising(img_noise, h=10, templateWindowSize=7, searchWindowSize=21)
fourier_denoised = fourier_denoise(img_noise, cutoff_ratio=0.1)

# Compute PSNR values (all in grayscale)
psnr_values = {
    "Bilateral": cv2.PSNR(img_noise_gray, cv2.cvtColor(bilateral_denoised, cv2.COLOR_BGR2GRAY)),
    "Wavelet": cv2.PSNR(img_noise_gray, wavelet_denoised),
    "Non-Local Means (NLM)": cv2.PSNR(img_noise_gray, cv2.cvtColor(nlm_denoised, cv2.COLOR_BGR2GRAY)),
    "Fourier": cv2.PSNR(img_noise_gray, fourier_denoised)
}

# Find the method with the highest PSNR
best_method = max(psnr_values, key=psnr_values.get)

print("\nPSNR Values:")
for method, psnr in psnr_values.items():
    print(f"{method}: {psnr:.2f} dB")

print(f"\nBest Denoising Method: {best_method} (PSNR: {psnr_values[best_method]:.2f} dB)")