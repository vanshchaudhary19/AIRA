import pyttsx3
import threading
import time
from config import TTS_RATE, SPEAK_INTERVAL

class AudioFeedback:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", TTS_RATE)
        self.engine.setProperty("volume", 1.0)

        voices = self.engine.getProperty("voices")
        for v in voices:
            if any(n in v.name.lower() for n in ["samantha", "karen", "victoria"]):
                self.engine.setProperty("voice", v.id)
                break

        self._last_message   = None
        self._last_time      = 0
        self._lock           = threading.Lock()
        self._queue          = []
        self._speaking       = False

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        while True:
            msg = None
            with self._lock:
                if self._queue:
                    msg = self._queue.pop(0)
            if msg:
                self._speaking = True
                self.engine.say(msg)
                self.engine.runAndWait()
                self._speaking = False
            time.sleep(0.05)

    def speak(self, text):
        now = time.time()
        with self._lock:
            same    = (text == self._last_message)
            too_soon = (now - self._last_time) < SPEAK_INTERVAL
            if same and too_soon:
                return
            self._last_message = text
            self._last_time    = now
            self._queue        = [text]   # replace queue, don't pile up

    def speak_detections(self, detections, enabled=True):
        if not enabled:
            return

        if not detections:
            self.speak("path is clear")
            return

        ahead = [d for d in detections if d["zone"] == "ahead"]
        left  = [d for d in detections if d["zone"] == "left"]
        right = [d for d in detections if d["zone"] == "right"]

        def summarise(group):
            counts = {}
            for d in group:
                counts[d["class"]] = counts.get(d["class"], 0) + 1
            parts = []
            for cls, cnt in counts.items():
                parts.append(f"{cnt} {cls}s" if cnt > 1 else cls)
            return " and ".join(parts)

        if ahead:
            desc = summarise(ahead)
            if not left and not right:
                msg = f"{desc} ahead. move left"
            elif left and not right:
                msg = f"{desc} ahead. move right"
            elif right and not left:
                msg = f"{desc} ahead. move left"
            else:
                msg = f"{desc} directly ahead. stop"
            self.speak(msg)

        else:
            parts = []
            if left:
                parts.append(f"{summarise(left)} on your left")
            if right:
                parts.append(f"{summarise(right)} on your right")
            if parts:
                self.speak(". ".join(parts))