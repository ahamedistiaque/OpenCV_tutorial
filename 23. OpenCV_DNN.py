"""
Deep Learning with OpenCV DNN - 
"""
#problem ase
import cv2
import numpy as np
import os
import urllib.request
import json

# -------------------------------------------------
# 1. Download ONNX model if not exists
# -------------------------------------------------
model_path = "mobilenetv2-7.onnx"
if not os.path.exists(model_path):
    print("Downloading MobileNetV2 ONNX model...")
    url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"

    urllib.request.urlretrieve(url, model_path)
    print("Downloaded successfully!")

# -------------------------------------------------
# 2. Load Model
# -------------------------------------------------
net = cv2.dnn.readNetFromONNX(model_path)
print("Model loaded successfully.")

# -------------------------------------------------
# 3. Load Image
# -------------------------------------------------
img = cv2.imread("macaw.jpg")  # Replace with your image
if img is None:
    print("Image not found.")
    exit()

# Preprocess for MobileNetV2
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(224,224),
                             mean=(0,0,0), swapRB=True, crop=False)
net.setInput(blob)

# Forward pass
outputs = net.forward()

# Get predicted class
class_id = int(np.argmax(outputs))
confidence = float(outputs[0][class_id])
print(f"Predicted class ID: {class_id}, Confidence: {confidence:.4f}")

# Load ImageNet labels
labels_path = "imagenet_labels.json"
if not os.path.exists(labels_path):
    # Download labels automatically
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    urllib.request.urlretrieve(url, labels_path)
    print("Labels downloaded!")

with open(labels_path) as f:
    class_names = json.load(f)

label = class_names[class_id]
print(f"Predicted Label: {label}")

# Display result
cv2.putText(img, f"{label}: {confidence:.2f}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow("Image Classification", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

