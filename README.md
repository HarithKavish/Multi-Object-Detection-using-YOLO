---
title: Multi-Object Detection using YOLO
emoji: 🦾
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Multi-Object Detection using YOLO and Custom CNN

A robust, cloud-ready multi-object detection system using FastAPI, YOLOv8, and a custom CNN. Supports MLOps, edge (browser) inference, and cloud deployment.

- **Backend:** FastAPI (`app.py`)
- **Frontend:** TensorFlow.js demo (`tfjs_demo/index.html`)
- **Model Hub:** [Hugging Face Model Repo](https://huggingface.co/harithkavish/SkinNet-Analyzer)
- **Spaces Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/harithkavish/Multi-Object-Detection-using-YOLO)

## Usage
- Run the backend: `uvicorn app:app`
- Try the browser demo: open `tfjs_demo/index.html`
- See `.github/workflows/train-and-upload.yml` for CI/CD

---
