import cv2

img1 = cv2.resize(cv2.imread('bird.jpg'), (720,512))
img2 = cv2.resize(cv2.imread('macaw.jpg'), (720,512))

blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()


