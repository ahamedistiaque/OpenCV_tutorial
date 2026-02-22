"""
Face Detection Tutorial using Haar Cascades

Topics Covered:
- Haar features
- Pretrained XML classifiers
- Multi-scale detection
"""

import cv2

# -------------------------------------------------
# 1. Load Pretrained Haar Cascade for Face Detection
# -------------------------------------------------
# Haar cascades are classifiers trained with positive & negative images
# to detect objects by scanning with sliding windows and Haar features.

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------------------------
# 2. Load Image or Video Frame
# -------------------------------------------------
img = cv2.imread("images.png")  # Replace with your image path

if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale - Haar cascade works on grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------
# 3. Detect Faces (Multi-scale detection)
# -------------------------------------------------
# detectMultiScale scans the image at different scales (sizes)
# scaleFactor: image size reduction at each scale step (1.1 means 10% reduction)
# minNeighbors: number of neighbors each candidate rectangle should have to retain it
# minSize: minimum size of detected faces to filter out smaller objects

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Detected {len(faces)} face(s)")

# -------------------------------------------------
# 4. Draw rectangles around detected faces
# -------------------------------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# -------------------------------------------------
# 5. Show Result
# -------------------------------------------------
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
