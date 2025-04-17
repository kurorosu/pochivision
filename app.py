# cap_app.py
import cv2
import json
from capture_runner import LivePreviewRunner
from core import PipelineExecutor
from capturelib.capture_manager import CaptureManager

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

with open("config.json", "r") as f:
    config = json.load(f)

capture_manager = CaptureManager()
pipeline = PipelineExecutor.from_config(
    config, capture_manager=capture_manager)

app = LivePreviewRunner(cap, pipeline)
app.run()
