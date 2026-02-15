"""
Histogram: A histogram shows the distribution of pixel intensities in an image.

In this tutorial we cover:
1. Grayscale Histogram
2. Color Histogram
3. Histogram Equalization
4. CLAHE (Adaptive Histogram Equalization)

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread('bird.jpg')

# Resize image
img = cv2.resize(img, (720, 500))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------
# 1. Grayscale Histogram
# -------------------------------------------------

hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure("Grayscale Histogram")
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.plot(hist_gray)
plt.xlim([0, 256])

# -------------------------------------------------
# 2. Color Histogram
# -------------------------------------------------

colors = ('b', 'g', 'r')

plt.figure("Color Histogram")
plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

# -------------------------------------------------
# 3. Histogram Equalization (Grayscale only)
# -------------------------------------------------

equalized = cv2.equalizeHist(gray)

# -------------------------------------------------
# 4. CLAHE (Adaptive Equalization)
# -------------------------------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(gray)

# -------------------------------------------------
# Show images
# -------------------------------------------------

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Histogram Equalization", equalized)
cv2.imshow("CLAHE", clahe_result)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
