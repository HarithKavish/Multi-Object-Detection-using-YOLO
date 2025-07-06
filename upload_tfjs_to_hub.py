# Upload only TensorFlow.js model files to Hugging Face Hub
# Usage: python upload_tfjs_to_hub.py

from huggingface_hub import HfApi
import os

# Set these variables
REPO_ID = "harithkavish/SkinNet-Analyzer"  # Your model repo
TFJS_MODEL_DIR = "tfjs_model"  # Path to your tfjs_model directory (containing model.json)

api = HfApi()

# Only upload the tfjs_model directory and its contents
api.upload_folder(
    folder_path=TFJS_MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    path_in_repo="tfjs_model"  # Uploads to tfjs_model/ in the repo
)

print(f"Uploaded TensorFlow.js model files from '{TFJS_MODEL_DIR}' to '{REPO_ID}/tfjs_model/' on Hugging Face Hub.")
