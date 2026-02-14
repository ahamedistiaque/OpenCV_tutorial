"""
Morphological operations are used to process binary images.
They are very important after thresholding.

Used for:
- Removing noise
- Filling gaps
- Separating connected objects
- Highlighting object boundaries

Basic Operations:
1. Erosion
2. Dilation
3. Opening
4. Closing
5. Morphological Gradient
"""

import cv2
import numpy as np

# Read image
img = cv2.imread('bird.jpg')

# Resize image
img = cv2.resize(img, (720, 500))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold to create binary image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Create structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# 1. Erosion (Removes small white noise)
erosion = cv2.erode(thresh, kernel, iterations=1)

# 2. Dilation (Increases white region)
dilation = cv2.dilate(thresh, kernel, iterations=1)

# 3. Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 4. Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 5. Morphological Gradient (Difference between dilation and erosion)
gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

# Show results

cv2.imshow("Original", img)
cv2.imshow("Threshold", thresh)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Gradient", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()
