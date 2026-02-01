import cv2

# Read original image in color
img_color = cv2.imread('bird.jpg')

img_color = cv2.resize(img_color, (720, 500))

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(img_gray, 100, 200)

# Show images
cv2.imshow("Original", img_color)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
