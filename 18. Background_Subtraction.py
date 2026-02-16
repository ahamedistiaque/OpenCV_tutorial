"""
Background Subtraction & Motion Detection Tutorial

Topics Covered:
1. MOG2 background subtractor
2. KNN background subtractor
3. Motion detection
4. Drawing bounding boxes around moving objects

Video Source: traffic.mp4
"""

import cv2
import numpy as np

# -------------------------------
# 1. Load Video
# -------------------------------
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# -------------------------------
# 2. Create Background Subtractors
# -------------------------------
# MOG2 (Gaussian Mixture-based Background/Foreground Segmentation)
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# KNN (K-Nearest Neighbors Background/Foreground Segmentation)
fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

# -------------------------------
# 3. Processing Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (720, 500))

    # -------------------------------
    # Apply Background Subtraction
    # -------------------------------
    fgmask_mog2 = fgbg_mog2.apply(frame)
    fgmask_knn = fgbg_knn.apply(frame)

    # -------------------------------
    # Motion Detection (Contours)
    # -------------------------------
    contours, _ = cv2.findContours(fgmask_mog2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter out small contours
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Moving Object", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # -------------------------------
    # Display results
    # -------------------------------
    cv2.imshow("Original with Motion", frame)
    cv2.imshow("Foreground Mask MOG2", fgmask_mog2)
    cv2.imshow("Foreground Mask KNN", fgmask_knn)

    # Exit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Stop if window closed manually
    if cv2.getWindowProperty("Original with Motion", cv2.WND_PROP_VISIBLE) < 1:
        break

# -------------------------------
# 4. Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
