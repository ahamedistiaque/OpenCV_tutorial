"""
Template Matching - 

Features:
1. Safe image loading
2. Best match detection
3. Multiple object detection
4. Score display
"""

import cv2
import numpy as np

# -------------------------------------------------
# 1. Load Images
# -------------------------------------------------

img = cv2.imread("pounds.png")
template = cv2.imread("coin.jpg")

if img is None:
    print("Error: pounds.png not found!")
    exit()

if template is None:
    print("Error: coin.jpg not found!")
    exit()

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template_gray.shape[::-1]

# -------------------------------------------------
# 2. Apply Template Matching
# -------------------------------------------------

method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(img_gray, template_gray, method)

# Get best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("Best Match Score:", max_val)

# -------------------------------------------------
# 3. Draw Best Match Box (Always Shows)
# -------------------------------------------------

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
cv2.putText(img, f"Best Match: {round(max_val,2)}",
            (top_left[0], top_left[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,255,0), 2)

# -------------------------------------------------
# 4. Multiple Object Detection (Optional)
# -------------------------------------------------

threshold = 0.6   # Lower for better detection

locations = np.where(result >= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

print("Total Matches Found:", len(list(zip(*locations[::-1]))))

# -------------------------------------------------
# 5. Show Result
# -------------------------------------------------

cv2.imshow("Template", template)
cv2.imshow("Detection Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

