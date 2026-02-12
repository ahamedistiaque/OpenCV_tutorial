#how to read, convert, and display an image in different color formats using OpenCV.
import cv2

# Read image
img = cv2.imread('macaw.jpg')

# Resize for easier display
#img = cv2.resize(img, (500, 500))

# Convert to Gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Show images
cv2.imshow("Original (BGR)", img)
cv2.imshow("Gray", gray)
cv2.imshow("HSV", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()

