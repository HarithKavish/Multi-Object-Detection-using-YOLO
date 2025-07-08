# Multi-Object Detection using YOLO and Custom CNN

This project performs real-time multi-object detection and image classification using YOLOv8 and a custom-trained CNN on CIFAR-10. It captures webcam video, detects objects, classifies images, and displays results live.

## Video Preview
Original Video Speed: 0.5X

[![Demo Video](https://github.com/user-attachments/assets/22620da8-8508-4100-b455-af382fc313c5)](https://github.com/user-attachments/assets/22620da8-8508-4100-b455-af382fc313c5)

## Features
- Real-time object detection using YOLOv8 segmentation
- Image classification using a custom CNN trained on CIFAR-10
- Live webcam video processing
- Model saving and loading

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository and navigate to the project folder.
2. Ensure you have a webcam connected.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Follow the prompts to train the model and start detection.
5. Press 'q' to quit the video window.

## Files
- `main.py`: Main script for training and detection
- `yolov8n-seg.pt`: YOLOv8 model weights
- `image_classifier.keras`: Saved CNN model
- `requirements.txt`: Python dependencies

## Contributors
- Harith Kavish S
- Sharwan Krishnan P
- Sanjay R