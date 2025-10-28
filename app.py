import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow import keras

# Load YOLO model
yolov8_model = YOLO('yolov8n-seg.pt')

# Load trained CNN model if it exists
try:
    cnn_model = keras.models.load_model('image_classifier.keras')
    cnn_available = True
except:
    cnn_available = False

# CIFAR-10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def detect_objects(image):
    """Perform object detection and classification on the input image."""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # YOLO detection
    results = yolov8_model(img_array)
    
    # Get annotated image from YOLO
    annotated_img = results[0].plot()
    
    # CNN classification (if model is available)
    classification_text = ""
    if cnn_available:
        # Resize for CNN
        img_resized = cv2.resize(img_array, (32, 32))
        img_normalized = img_resized / 255.0
        
        # Predict
        prediction = cnn_model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]
        
        classification_text = f"CNN Classification: {cifar10_classes[predicted_class_idx]} ({confidence:.2%} confidence)"
    
    # Get detection information
    detection_info = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detection_info.append(f"{r.names[cls]}: {conf:.2%}")
    
    info_text = "\n".join(detection_info) if detection_info else "No objects detected"
    if classification_text:
        info_text = f"{classification_text}\n\nDetections:\n{info_text}"
    
    return annotated_img, info_text

# Create Gradio interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Detection Results"),
        gr.Textbox(label="Detection Info", lines=10)
    ],
    title="Multi-Object Detection using YOLO and Custom CNN",
    description="Upload an image to detect objects using YOLOv8 and classify with a custom CNN trained on CIFAR-10.",
    examples=[],
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
