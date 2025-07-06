---
license: mit
tags:
  - object-detection
  - image-classification
  - yolo
  - tensorflowjs
---

# Multi-Object Detection Models

This repo hosts the trained models for the multi-object detection project:

- `image_classifier.keras`: Custom CNN for CIFAR-10
- `yolov8n-seg.pt`: YOLOv8 segmentation model
- `tfjs_model/`: TensorFlow.js model for browser inference

## Intended Use
- **API/Backend:** Use `.keras` and `.pt` models for FastAPI inference.
- **Edge/Browser:** Use `tfjs_model/model.json` with TensorFlow.js.

## Training & Conversion
- Models are trained via GitHub Actions and uploaded automatically.
- The Keras model is converted to TensorFlow.js using `tensorflowjs_converter`.

## Example (TensorFlow.js)
```js
const model = await tf.loadLayersModel('https://huggingface.co/harithkavish/SkinNet-Analyzer/resolve/main/tfjs_model/model.json');
```

## Citation
See main repo for full details and usage.
