import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ultralytics import YOLO
import torch

def start():
    print("=" * 80)
    print("YOLO Model Export Pipeline")
    print("=" * 80)
    
    # Load YOLOv8 nano model (smallest, fastest)
    print("\n[1/3] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Export to ONNX format
    print("\n[2/3] Exporting to ONNX format...")
    model.export(
        format='onnx',
        simplify=True,
        dynamic=False,
        imgsz=640
    )
    print("✓ ONNX export complete: yolov8n.onnx")
    
    print("\n[3/3] ONNX model ready for TensorFlow conversion")
    print("=" * 80)
    print("Export complete! Next step: ONNX → TensorFlow → TensorFlow.js")
    print("=" * 80)

def main():
    start()

if __name__ == "__main__":
    main()


# Contributors:
# 1) Harith Kavish S
# 2) Sharwan Krishnan P
# 3) Sanjay R