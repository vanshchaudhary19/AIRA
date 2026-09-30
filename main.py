import cv2
import threading
import time
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO
from core.detector import Detector
from core.audio import AudioFeedback
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, DETECTION_INTERVAL

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

detector  = Detector()
audio     = AudioFeedback()

latest_frame      = None
latest_detections = []
scanning_active   = False
lock              = threading.Lock()

# ── drawing ──────────────────────────────────────────────────────────────────
def draw_frame(frame, detections):
    h, w = frame.shape[:2]
    # zone dividers
    cv2.line(frame, (w // 3, 0),       (w // 3, h),       (180, 180, 180), 1)
    cv2.line(frame, ((w * 2) // 3, 0), ((w * 2) // 3, h), (180, 180, 180), 1)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = (220, 80, 40) if d["zone"] == "ahead" else (40, 160, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['confidence']}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# ── camera loop ───────────────────────────────────────────────────────────────
def camera_loop():
    global latest_frame, latest_detections, scanning_active
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep buffer minimal to avoid stale frames

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if not scanning_active:
            with lock:
                latest_frame = frame.copy()
                latest_detections = []
            time.sleep(0.1)
            continue

        detections = detector.detect(frame)
        annotated  = draw_frame(frame.copy(), detections)

        with lock:
            latest_frame      = annotated
            latest_detections = detections

        audio.speak_detections(detections, enabled=scanning_active)

        socketio.emit("detections", {
            "objects": [
                {"class": d["class"], "zone": d["zone"], "confidence": d["confidence"]}
                for d in detections
            ]
        })

        time.sleep(DETECTION_INTERVAL)

    cap.release()

# ── video stream ──────────────────────────────────────────────────────────────
def generate_feed():
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            frame = latest_frame.copy()

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.03)

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start():
    global scanning_active
    scanning_active = True
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global scanning_active
    scanning_active = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def status():
    return jsonify({"scanning": scanning_active})

# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("🚀 AIRA running → http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, use_reloader=False)