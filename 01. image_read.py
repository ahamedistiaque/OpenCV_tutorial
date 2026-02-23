import cv2

# Read the image in color mode
image = cv2.imread('macaw.jpg', cv2.IMREAD_COLOR)

# Check if the image loaded successfully
if image is None:
    print("Error: Could not read the image.")
else:
    print("Image loaded successfully!")

# Display the loaded image
cv2.imshow('My Image', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
