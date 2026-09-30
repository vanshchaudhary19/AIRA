from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE, FRAME_WIDTH
import torch

class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        # use MPS on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"✅ AIRA model loaded on {self.device}")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=CONFIDENCE,
            verbose=False,
            device=self.device,
            imgsz=640
        )[0]

        detections = []
        w = FRAME_WIDTH

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            zone = (
                "left"  if cx < w // 3 else
                "right" if cx > (w * 2) // 3 else
                "ahead"
            )

            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 2),
                "box":        [x1, y1, x2, y2],
                "zone":       zone,
                "cx":         cx,
                "area":       (x2 - x1) * (y2 - y1)
            })

        # sort by area descending so closest/largest come first
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections