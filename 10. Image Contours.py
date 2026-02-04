# """mage contours in Python are detected using OpenCV (cv2), typically 
involving loading an image, converting to grayscale, applying thresholding or edge detection (like Canny) to create a binary image, and using cv2.findContours().""" 

import cv2

# Read  image
img = cv2.imread('bird.jpg')


img = cv2.resize(img, (500, 500)) # Resize image

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Show image
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

