import cv2

cam=cv2.VideoCapture(0)

# Object detection from Stable camera

object_detector= cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=80)

while True:
    ret, frame=cam.read()
    resized_frame = cv2.resize(frame, (640, 360))
    #Extract region 
    height, width,_=frame.shape
    print(width, height)
    roi= resized_frame[100:360,80:500]

    #object detection
    mask= object_detector.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    countours ,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countours:
        area=cv2.contourArea(cnt)
        if 50 <= area <= 2500:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),1)
            #cv2.drawContours(resized_frame, [cnt],-1,(0,255,0),1)


    cv2.imshow('Frame',resized_frame)
    cv2.imshow('Frame2',roi)
    #cv2.imshow('Frame3',mask)
    key=cv2.waitKey(50)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
 
