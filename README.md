<<<<<<< HEAD
---
title: Multi-Object Detection using YOLO
emoji: 🦾
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Multi-Object Detection API (YOLO + CNN)

A FastAPI backend for multi-object detection and image classification using YOLOv8 and a custom CNN. Deployable on Hugging Face Spaces.

## API Endpoints

- **POST `/detect-object`**: Upload an image (form-data, key: `file`). Returns detected objects, their counts, and CNN class prediction.
- **GET `/health`**: Health check endpoint. Returns `{ "status": "ok" }`.

## Deployment
- All files (`app.py`, `requirements.txt`, `space.yaml`, `image_classifier.keras`, `yolov8n-seg.pt`) must be in the repo.
- Deploy on Hugging Face Spaces with FastAPI SDK.

## Example (Python)
```python
import requests
files = {'file': open('your_image.jpg', 'rb')}
r = requests.post('https://your-space-url.hf.space/detect-object', files=files)
print(r.json())
```
=======
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
>>>>>>> 1c3894d2065701b19ab2a500433a6d7f7a075d13
