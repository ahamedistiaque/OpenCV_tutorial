"""
Real-Time Color Detection

Topics Covered:
1. HSV Trackbars for live color selection
2. Mask creation
3. Bitwise operations
4. Live color detection on webcam feed

This is your first mini-project using OpenCV.
"""

import cv2
import numpy as np

# -------------------------------
# 1. Create Trackbars Window
# -------------------------------
def nothing(x):
    pass

cv2.namedWindow("Trackbars")

# Create trackbars for HSV min/max values
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# -------------------------------
# 2. Access Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

# -------------------------------
# 3. Real-Time Processing Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (720, 500))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Define HSV range
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Detected Color", result)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if any window closed manually
    if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1:
        break

# -------------------------------
# 4. Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()

