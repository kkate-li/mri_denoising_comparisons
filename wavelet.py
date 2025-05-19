import cv2
import numpy as np
import pywt

def wavelet_denoising(img, wavelet='db1', level=2, threshold=0.05):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32 for wavelet transform
    img_float = np.float32(img) / 255.0
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)
    
    # Threshold the coefficients (hard thresholding)
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(
            tuple(np.where(np.abs(detail) < threshold, 0, detail) 
            for detail in coeffs[i])
        )
    
    # Reconstruct the image
    img_denoised = pywt.waverec2(coeffs_thresh, wavelet)
    
    # Clip and convert back to uint8
    img_denoised = np.clip(img_denoised, 0, 1)
    img_denoised = (img_denoised * 255).astype(np.uint8)
    
    return img_denoised

# Read the noisy image
img_noise = cv2.imread('images/AD/mildDem8.jpg')

# Apply wavelet denoising
img_denoised = wavelet_denoising(img_noise, wavelet='db1', level=2, threshold=0.05)

# Ensure both images are grayscale for PSNR calculation
if len(img_noise.shape) == 3:
    img_noise_gray = cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY)
else:
    img_noise_gray = img_noise

# Display and save results
cv2.imshow("Original", img_noise_gray)
cv2.imshow("Wavelet Denoised", img_denoised)

cv2.imwrite('images/AD/mildDem8_wavelet.jpg', img_denoised)

# Compute PSNR (both images must be grayscale)
print('PSNR: ', cv2.PSNR(img_noise_gray, img_denoised))
