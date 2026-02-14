"""
Image Gradients Tutorial (Advanced Edge Concepts)

Image gradients measure intensity changes in an image.
They are used to detect edges and object boundaries.

In this tutorial we cover:
1. Sobel Operator
2. Laplacian Operator
3. Scharr Operator
4. Gradient Magnitude
"""

import cv2
import numpy as np

# Read image
img = cv2.imread('bird.jpg')

# Resize image
img = cv2.resize(img, (720, 500))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------
# 1. Sobel Operator
# -------------------------

# Sobel X (horizontal edges)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y (vertical edges)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert back to uint8
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# -------------------------
# 2. Laplacian Operator
# -------------------------

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# -------------------------
# 3. Scharr Operator
# -------------------------
# More accurate than Sobel when kernel size = 3

scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.convertScaleAbs(scharr_y)

# -------------------------
# 4. Gradient Magnitude
# -------------------------
# Combine Sobel X and Sobel Y

gradient_magnitude = cv2.magnitude(
    cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
    cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
)

gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# -------------------------
# Show results
# -------------------------

cv2.imshow("Original", img)
cv2.imshow("Sobel X", sobel_x)
cv2.imshow("Sobel Y", sobel_y)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Scharr X", scharr_x)
cv2.imshow("Scharr Y", scharr_y)
cv2.imshow("Gradient Magnitude", gradient_magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()
