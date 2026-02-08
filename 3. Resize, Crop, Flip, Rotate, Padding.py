"""
This code performs basic image manipulations using OpenCV, 
including resizing, cropping, horizontal flipping, and 
rotating an image by 45 degrees. It also adds a black 
border (padding) around the image and displays all the 
processed versions in separate windows.
"""

import cv2

img = cv2.imread('bird2.jpg')

# Resize
resized = cv2.resize(img, (720, 512))

# Crop
crop = resized[100:400, 100:400]

# Flip
flip_h = cv2.flip(resized, 1)

# Rotate
(h, w) = resized.shape[:2]
center = (w//2, h//2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(resized, matrix, (w, h))

# Padding
padded = cv2.copyMakeBorder(resized, 50, 50, 50, 50,
                            cv2.BORDER_CONSTANT, value=(0,0,0))

cv2.imshow("Resized", resized)
cv2.imshow("Cropped", crop)
cv2.imshow("Flipped", flip_h)
cv2.imshow("Rotated", rotated)
cv2.imshow("Padded", padded)

cv2.waitKey(0)
cv2.destroyAllWindows()

