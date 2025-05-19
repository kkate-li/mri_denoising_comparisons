import cv2
import numpy as np

def fourier_denoise(img, cutoff_ratio=0.1):
    """Denoise image using Fourier low-pass filter"""
    # Convert to float32 for FFT
    img_float = np.float32(img)
    
    # Perform 2D Fourier Transform
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
    
    # Convert back to uint8
    return np.uint8(np.clip(img_back, 0, 255))

# Read image (force grayscale)
img_noise = cv2.imread('images/AD/mildDem8.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Fourier denoising
fourier_denoised = fourier_denoise(img_noise, cutoff_ratio=0.1)

# Display and save results
cv2.imshow("Original", img_noise)
cv2.imshow("Fourier Denoised", fourier_denoised)
cv2.imwrite('images/AD/mildDem8_fourier.jpg', fourier_denoised)

# Compute PSNR
print('PSNR:', cv2.PSNR(img_noise, fourier_denoised))
