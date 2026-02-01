import cv2

# Read original in color
img_color = cv2.imread('macaw.jpg')
#img_color = cv2.resize(img_color, (500, 500))

# Convert to grayscale for thresholding
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Apply threshold (binary and inverse)
thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
thresh_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

# Show images
cv2.imshow("Original Color", img_color)
cv2.imshow("Gray Color", img_gray)
cv2.imshow("Binary", thresh)
cv2.imshow("Inverse Binary", thresh_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()
