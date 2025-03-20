
##
import cv2

img = cv2.imread('bird.jpg')
img = cv2.imread('bird.jpg')

img = cv2.resize(img, (720, 600))  # (width, height)
# 1. Most popular color space conversions
bgr = img                              
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. Split channels (very useful!)
b, g, r = cv2.split(img)          # BGR channels

# Show interesting comparisons
cv2.imshow('Original BGR', img)
cv2.imshow('RGB (matplotlib style)', rgb)
cv2.imshow('Grayscale', gray)

# Show separate channels (will look strange at first!)
cv2.imshow('Blue channel', b)
cv2.imshow('Green channel', g)
cv2.imshow('Red channel', r)

cv2.waitKey(0)
cv2.destroyAllWindows()