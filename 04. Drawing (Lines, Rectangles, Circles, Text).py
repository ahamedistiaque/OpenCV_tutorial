## Used in visualization, annotations, object detection results.
import cv2

img = cv2.imread('macaw.jpg')
img = cv2.resize(img, (600, 600))

cv2.line(img, (50,50), (550,50), (0,255,0), 2)
cv2.rectangle(img, (100,100), (300,300), (255,0,0), 2)
cv2.circle(img, (400,400), 50, (0,0,255), -1)

cv2.putText(img, "Macaw",
            (200,550),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2)

cv2.imshow("Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
