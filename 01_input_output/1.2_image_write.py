import cv2

# Read the image
image = cv2.imread('macaw.jpg')

# Save the image with a new name
success = cv2.imwrite('output.jpg', image)

if success:
    print("Image saved successfully!")
else:
    print("Error: Could not save the image.")