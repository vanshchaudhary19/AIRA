MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.45
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CLASSES = [
    "bed", "bottle", "cell phone", "chair",
    "door", "Electric pole", "laptop",
    "person", "stairs", "vehicle"
]

TTS_RATE = 160
DETECTION_INTERVAL = 0.08   # run detection every 80ms (~12fps)
SPEAK_INTERVAL = 5           # repeat same message every 5 seconds