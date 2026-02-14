"""
Image smoothing (also called blurring) is used to:
- Reduce noise
- Remove small details
- Prepare image for edge detection
- Improve image processing results

Common Smoothing Techniques:
1. Averaging (Mean Blur)
2. Gaussian Blur
3. Median Blur
4. Bilateral Filter
"""

import cv2
import numpy as np

# Read image
img = cv2.imread('bird.jpg')

# Resize image
img = cv2.resize(img, (720, 500))

# 1. Averaging Blur (Mean Filter)
blur_avg = cv2.blur(img, (7, 7))

# 2. Gaussian Blur (More natural blur)
blur_gaussian = cv2.GaussianBlur(img, (7, 7), 0)

# 3. Median Blur (Best for salt-and-pepper noise)
blur_median = cv2.medianBlur(img, 7)

# 4. Bilateral Filter (Keeps edges sharp)
blur_bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Averaging Blur", blur_avg)
cv2.imshow("Gaussian Blur", blur_gaussian)
cv2.imshow("Median Blur", blur_median)
cv2.imshow("Bilateral Filter", blur_bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
