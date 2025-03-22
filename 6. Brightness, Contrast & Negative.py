import cv2
import numpy as np

img = cv2.imread('bird.jpg')

# Brightness & contrast
bright = cv2.convertScaleAbs(img, alpha=1.3, beta=40)

# Negative
negative = 255 - img

cv2.imshow("Original", img)
cv2.imshow("Bright", bright)
cv2.imshow("Negative", negative)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Note: Ensure you have 'bird.jpg' in the same directory as this script to run it successfully.
# Improves visibility and preprocessing for detection.