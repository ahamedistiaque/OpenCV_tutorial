"""
Video Processing Basics

In this tutorial we cover:

1. Reading video using VideoCapture
2. Accessing Webcam
3. Frame-by-frame processing
4. Saving video to file
5. FPS handling

This is the foundation of:
- Real-time computer vision
- Object tracking
- Face detection
- Surveillance systems
"""

import cv2
import time

# -------------------------------------------------
# 1. Read Video File (Optional)
# -------------------------------------------------

# cap = cv2.VideoCapture("video.mp4")  # Uncomment to use video file

# -------------------------------------------------
# 2. Access Webcam
# -------------------------------------------------

cap = cv2.VideoCapture(0)  # 0 = default webcam
# cap = cv2.VideoCapture("video.mp4")  # Uncomment for video file

if not cap.isOpened():
    print("Error: Cannot open camera/video")
    exit()

# Get frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------------------------------------
# 2. FPS Handling
# -------------------------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # Some webcams return 0
    fps = 30
print("FPS:", fps)

# -------------------------------------------------
# 3. Video Writer
# -------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, fps, (frame_width, frame_height))

# -------------------------------------------------
# 4. Frame-by-frame Processing
# -------------------------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Example processing: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show frames
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)

    # Save original frame
    out.write(frame)

    # Exit cleanly if 'q' pressed OR window manually closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if windows are closed manually
    if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty("Grayscale", cv2.WND_PROP_VISIBLE) < 1:
        break

# -------------------------------------------------
# 5. Release resources
# -------------------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

