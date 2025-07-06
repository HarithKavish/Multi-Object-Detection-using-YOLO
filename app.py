from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
from tensorflow import keras
from ultralytics import YOLO
import io
from PIL import Image
from huggingface_hub import hf_hub_download

app = FastAPI()

# Allow CORS for specific origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://harithkavish.github.io/",
        "https://harithkavish.github.io/Multi-Object-Detection-using-YOLO"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CNN model
try:
    keras_model_path = hf_hub_download(repo_id="harithkavish/multi-object-detection-models", filename="image_classifier.keras")
    cnn_model = keras.models.load_model(keras_model_path)
except Exception as e:
    from tensorflow import keras as keras_build
    from tensorflow.keras import layers
    # Define a simple CNN model structure matching the original
    def build_cnn_model():
        model = keras_build.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu', name='dense_layer_2'),
            layers.Dense(10, activation='softmax', name='dense_1')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    cnn_model = build_cnn_model()
    print('Warning: image_classifier.keras not found or incompatible. Created a new model instead.')

def load_yolo_model(model_path, max_dynamic_patches=5):
    """
    Robust YOLOv8 loader for PyTorch >=2.6 with dynamic safe globals patching and fallback.
    Attempts to patch required classes dynamically from error messages.
    As a last resort, tries weights_only=False (unsafe, but local model is trusted).
    """
    import importlib
    yolo_model = None
    patched_classes = set([
        'ultralytics.nn.modules.Conv',
        'ultralytics.nn.tasks.SegmentationModel'
    ])
    try:
        from torch.serialization import add_safe_globals
        # Initial patch
        for cls_path in patched_classes:
            module_name, class_name = cls_path.rsplit('.', 1)
            mod = importlib.import_module(module_name)
            add_safe_globals([getattr(mod, class_name)])
        from ultralytics import YOLO
        for _ in range(max_dynamic_patches):
            try:
                return YOLO(model_path)
            except Exception as e:
                msg = str(e)
                # Look for 'Unsupported global: GLOBAL ...' in error
                import re
                m = re.search(r"Unsupported global: GLOBAL ([\w\.]+)", msg)
                if m:
                    cls_path = m.group(1)
                    if cls_path not in patched_classes:
                        try:
                            module_name, class_name = cls_path.rsplit('.', 1)
                            mod = importlib.import_module(module_name)
                            add_safe_globals([getattr(mod, class_name)])
                            patched_classes.add(cls_path)
                            print(f"Patched safe global: {cls_path}")
                            continue  # Try loading again
                        except Exception as patch_e:
                            print(f"Failed to patch {cls_path}: {patch_e}")
                # If not a safe globals error, break
                print(f"YOLO model load failed: {e}")
                break
        # Last resort: try weights_only=False if available
        try:
            import torch
            print("Attempting to load YOLO with weights_only=False (unsafe, but local model is trusted)...")
            orig_torch_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return orig_torch_load(*args, **kwargs)
            torch.load = patched_load
            yolo = YOLO(model_path)
            torch.load = orig_torch_load  # Restore
            print("YOLO loaded with weights_only=False.")
            return yolo
        except Exception as e2:
            print(f"YOLO model load with weights_only=False also failed: {e2}")
    except Exception as e:
        print(f"YOLO loader setup failed: {e}")
    print("YOLO model could not be loaded after all attempts.")
    return None

# Load YOLOv8 model robustly
try:
    yolo_model_path = hf_hub_download(repo_id="harithkavish/multi-object-detection-models", filename="yolov8n-seg.pt")
    yolo_model = load_yolo_model(yolo_model_path)
except Exception as e:
    print('Error in YOLO model loading logic:', e)
    yolo_model = None

# Helper functions
def resize_for_cnn(frame, target_size=(32, 32)):
    return cv.resize(frame, target_size)

def preprocess_for_yolo(frame):
    return cv.resize(frame, (620, 620))

@app.post("/detect-object")
async def detect_object(file: UploadFile = File(...)):
    # Read image from frontend
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    frame = np.array(image)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # CNN prediction
    preprocessed_frame_cnn = resize_for_cnn(frame)
    prediction = cnn_model.predict(np.expand_dims(preprocessed_frame_cnn, axis=0))
    predicted_class = int(np.argmax(prediction))

    # YOLO detection
    detected_objects = []
    object_counts = {}
    if yolo_model is not None:
        preprocessed_frame_yolo = preprocess_for_yolo(frame)
        results = yolo_model(preprocessed_frame_yolo)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.model.names[cls] if hasattr(yolo_model.model, 'names') else str(cls)
                detected_objects.append(label)
                object_counts[label] = object_counts.get(label, 0) + 1
    else:
        print('YOLO model not loaded, skipping detection.')

    return JSONResponse({
        "cnn_class": predicted_class,
        "detected_objects": detected_objects,
        "object_counts": object_counts
    })

@app.get("/health")
def health_check():
    return {"status": "ok"}
