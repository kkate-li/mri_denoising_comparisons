import cv2

img_noise = cv2.imread('images/AD/mildDem8.jpg', cv2.IMREAD_GRAYSCALE)

nlm_denoised = cv2.fastNlMeansDenoising(
    img_noise, 
    h=10,               # Filter strength (higher = stronger denoising)
    templateWindowSize=7,  # Size of patch used for comparison
    searchWindowSize=21    # Area where similar patches are searched
)

cv2.imshow("Original", img_noise)
cv2.imshow("Non-Local Means Denoised", nlm_denoised)

cv2.imwrite('images/AD/mildDem8_nlm.jpg', nlm_denoised)

print('PSNR:', cv2.PSNR(img_noise, nlm_denoised))

