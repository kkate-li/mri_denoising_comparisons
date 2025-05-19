# -*- coding: utf-8 -*-

# Bilateral for nonDem95

import cv2

img_noise = cv2.imread('images/AD/moderateDem4.jpg')

img = img_noise

bilateral_using_cv2 = cv2.bilateralFilter(img,5,20,100,borderType=cv2.BORDER_CONSTANT)

cv2.imshow("Original",img)
cv2.imshow("bilateral",bilateral_using_cv2)

cv2.imwrite('images/AD/moderateDem4_bilateral.jpg', bilateral_using_cv2)

print('PSNR: ', cv2.PSNR(img, bilateral_using_cv2))



