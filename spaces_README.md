---
title: Multi-Object Detection Demo
emoji: 🦾
sdk: fastapi
app_file: app.py
---

# Multi-Object Detection Demo (YOLO + CNN)

This Space demonstrates multi-object detection and image classification using a FastAPI backend. Models are loaded from the Hugging Face Model Hub.

- Upload an image to get object detections and class predictions.
- See [GitHub](https://github.com/harithkavish/Multi-Object-Detection-using-YOLO) for code.
- See [Model Hub](https://huggingface.co/harithkavish/SkinNet-Analyzer) for model files.

## API Endpoints
- `POST /detect-object`: Upload an image, get detections and class.
- `GET /health`: Health check.

---
