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
