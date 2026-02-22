#problem ase

import cv2
import numpy as np

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
# Use 'yolov8n.onnx' (Exported from PyTorch)
MODEL_PATH = "yolov8n.onnx" 
CLASS_LIST = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"] # ... add more as per COCO

def load_model():
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    # Use GPU if available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def detect(frame, net):
    # YOLOv8 expects 640x640 input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    return outputs

def post_process(frame, outputs):
    # YOLOv8 output shape is (1, 84, 8400) -> [batch, values, boxes]
    # We need to transpose it to (8400, 84)
    data = outputs[0]
    rows = data.shape[0]
    
    boxes = []
    confidences = []
    class_ids = []

    # Scaling factors
    x_factor = frame.shape[1] / 640
    y_factor = frame.shape[2] / 640

    # Logic to filter detections and apply NMS goes here...
    # (Abbreviated for clarity: looping through rows to find high scores)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    net = load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detections = detect(frame, net)
        # Process and draw detections...
        
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()