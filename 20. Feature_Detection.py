"""
Feature Detection & Keypoints Tutorial

Topics Covered:
1. Harris Corner Detection
2. ORB Keypoint Detection
3. ORB Feature Matching

This is a major step toward real object recognition.
"""

import cv2
import numpy as np

# -------------------------------------------------
# 1. Harris Corner Detection
# -------------------------------------------------

img = cv2.imread("bird.jpg")
img = cv2.resize(img, (720, 500))
if img is None:
    print("Error: bird.jpg not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Apply Harris Corner Detector
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate to mark corners
dst = cv2.dilate(dst, None)

# Mark corners in red
img_harris = img.copy()
img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]

# -------------------------------------------------
# 2. ORB Keypoint Detection
# -------------------------------------------------

img1 = cv2.imread("bird.jpg", 0)
img1 = cv2.resize(img1, (720, 500))
img2 = cv2.imread("flower.png", 0)
img2 = cv2.resize(img2, (720, 500))

if img1 is None or img2 is None:
    print("Error: bird.jpg or flower.png not found!")
    exit()

# Create ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Draw keypoints
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)

# -------------------------------------------------
# 3. Feature Matching (ORB)
# -------------------------------------------------

# Use Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# Sort matches by distance (lower = better)
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 30 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

# -------------------------------------------------
# Show Results
# -------------------------------------------------

cv2.imshow("Harris Corners", img_harris)
cv2.imshow("ORB Keypoints", img1_kp)
cv2.imshow("ORB Feature Matching", img_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()
