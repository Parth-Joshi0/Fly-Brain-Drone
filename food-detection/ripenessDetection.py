import os
import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata
import os
from dotenv import load_dotenv

# Load the .env file from the current directory
load_dotenv()


# Initialize client (reads key from environment variable)
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    processing_timeout=3600,
    requested_plan="webrtc-gpu-medium",
    requested_region="us"
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="banana-ripeness-classification-vbanana-ripeness-classification-6ph5z-1-resnet18-t1-logic",
    workspace="amalchalayilsreekumar",
    image_input="image",
    config=config
)

@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

session.run()