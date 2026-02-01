import cv2
import numpy as np

# Read image
img = cv2.imread('macaw.jpg')
#img = cv2.resize(img, (500, 500))


# Increase brightness
brightness = cv2.convertScaleAbs(img, alpha=1, beta=50)

# Increase contrast
contrast = cv2.convertScaleAbs(img, alpha=2, beta=0)

# Negative 
negative = 255 - img

# Show results
cv2.imshow("Original", img)
cv2.imshow("Brightness", brightness)
cv2.imshow("Contrast", contrast)
cv2.imshow("Negative", negative)

cv2.waitKey(0)
cv2.destroyAllWindows()
