from huggingface_hub import HfApi, HfFolder, upload_file
import os

# Set these variables
REPO_ID = "harithkavish/multi-object-detection-models"
MODEL_FILES = [
    "image_classifier.keras",
    "yolov8n-seg.pt"
]
TFJS_DIR = "tfjs_model"

# Authenticate
api = HfApi()
api.set_access_token(os.environ["HF_TOKEN"])

# Upload model files
for file in MODEL_FILES:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=REPO_ID,
            repo_type="model"
        )

# Upload TensorFlow.js model directory
if os.path.exists(TFJS_DIR):
    for root, dirs, files in os.walk(TFJS_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            repo_path = os.path.relpath(local_path, ".")
            print(f"Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model"
            )
print("Upload complete.")
