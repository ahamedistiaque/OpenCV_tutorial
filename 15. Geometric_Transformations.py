"""
Geometric Transformations  (Advanced)

Geometric transformations change the spatial position of pixels.

In this tutorial we cover:
1. Translation
2. Rotation
3. Affine Transformation
4. Perspective Transformation
5. Basic idea of Homography

These concepts are used in:
- Image alignment
- Document scanning
- Augmented Reality
- Object tracking
"""

import cv2
import numpy as np

# Read image
img = cv2.imread('bird.jpg')

# Resize image
img = cv2.resize(img, (720, 500))

rows, cols = img.shape[:2]

# -------------------------------------------------
# 1. Translation (Shift image)
# -------------------------------------------------

# Move 100 pixels right and 50 pixels down
M_translate = np.float32([[1, 0, 100],
                          [0, 1, 50]])

translated = cv2.warpAffine(img, M_translate, (cols, rows))

# -------------------------------------------------
# 2. Rotation
# -------------------------------------------------

# Rotate around center
center = (cols // 2, rows // 2)

M_rotate = cv2.getRotationMatrix2D(center, 45, 1)  # 45 degrees, scale=1

rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

# -------------------------------------------------
# 3. Affine Transformation
# -------------------------------------------------

# Select 3 points from original image
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

# New positions of those 3 points
pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])

M_affine = cv2.getAffineTransform(pts1, pts2)

affine = cv2.warpAffine(img, M_affine, (cols, rows))

# -------------------------------------------------
# 4. Perspective Transformation
# -------------------------------------------------

# Select 4 points from original image
pts1_p = np.float32([[100, 100],
                     [600, 100],
                     [100, 400],
                     [600, 400]])

# Destination points (rectangle)
pts2_p = np.float32([[0, 0],
                     [500, 0],
                     [0, 300],
                     [500, 300]])

M_perspective = cv2.getPerspectiveTransform(pts1_p, pts2_p)

perspective = cv2.warpPerspective(img, M_perspective, (500, 300))

# -------------------------------------------------
# Show Results
# -------------------------------------------------

cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Affine", affine)
cv2.imshow("Perspective", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()

