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

def process_frame(frame):
    """Process a single frame for real-time detection."""
    if frame is None:
        return None
    
    # YOLO detection
    results = yolov8_model(frame, verbose=False)
    
    # Get annotated image from YOLO
    annotated_img = results[0].plot()
    
    # CNN classification (if model is available)
    if cnn_available:
        # Resize for CNN
        img_resized = cv2.resize(frame, (32, 32))
        img_normalized = img_resized / 255.0
        
        # Predict
        prediction = cnn_model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]
        
        # Add CNN classification text to image
        cv2.putText(annotated_img, 
                    f"CNN: {cifar10_classes[predicted_class_idx]} ({confidence:.2%})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_img

def detect_objects(image):
    """Perform object detection and classification on static images."""
    if image is None:
        return None, "No image provided"
    
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

# Create Gradio interface with tabs
with gr.Blocks(title="Multi-Object Detection using YOLO and Custom CNN") as demo:
    gr.Markdown("# Multi-Object Detection using YOLO and Custom CNN")
    gr.Markdown("Detect objects using YOLOv8 and classify with a custom CNN trained on CIFAR-10.")
    
    with gr.Tab("📸 Image Upload"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                image_button = gr.Button("Detect Objects")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Detection Results")
                info_output = gr.Textbox(label="Detection Info", lines=10)
        
        image_button.click(
            fn=detect_objects,
            inputs=image_input,
            outputs=[image_output, info_output]
        )
    
    with gr.Tab("🎥 Webcam Stream"):
        gr.Markdown("### Real-time object detection from your webcam")
        webcam_stream = gr.Interface(
            fn=process_frame,
            inputs=gr.Image(source="webcam", streaming=True, type="numpy"),
            outputs=gr.Image(type="numpy", streaming=True),
            live=True,
            css=".gradio-container {max-width: 900px}"
        )

if __name__ == "__main__":
    demo.launch()
