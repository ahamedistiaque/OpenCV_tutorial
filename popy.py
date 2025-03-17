import cv2

cam = cv2.VideoCapture("traffic2.mp4")

# Object detection from a stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 360))

    # Adjust region of interest (expand if needed)
    roi = resized_frame[50:360, 50:600]

    # Object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 <= area <= 1000:  # Adjust area range for better detection
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display frames
    cv2.imshow('Frame', resized_frame)
    cv2.imshow('ROI', roi)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
