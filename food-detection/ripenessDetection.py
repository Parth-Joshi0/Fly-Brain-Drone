import os
import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata
import os
from dotenv import load_dotenv

# Load the .env file from the current directory
load_dotenv()

# 1. Import the library
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ROBOFLOW_API_KEY"
).configure(InferenceConfiguration(
    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="amalchalayilsreekumar",
    workflow_id="banana-ripeness-classification-vbanana-ripeness-classification-6ph5z-1-resnet18-t1-logic",
    images={
        "image": "testImage.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)

