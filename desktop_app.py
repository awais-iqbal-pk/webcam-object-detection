"""
JARVIS Vision - Real-Time Object Detection with Voice Announcements
---------------------------------------------------------------------
Model   : YOLOv8n (Ultralytics) - modern, fast, accurate, runs on CPU
Voice   : pyttsx3 (fully offline text-to-speech, no internet needed)
Camera  : OpenCV (your laptop webcam)

Run with:  python main.py
Quit with: press 'q' in the video window

Controls:
  q  -> quit
  p  -> pause / resume detection
  v  -> toggle voice on/off
  s  -> save a screenshot
  +  -> increase confidence threshold
  -  -> decrease confidence threshold
"""

import cv2
import time
import threading
import queue
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MODEL_NAME = "yolov8n.pt"      # nano = smallest/fastest YOLOv8 model, auto-downloads
CONF_THRESHOLD = 0.40          # starting confidence threshold (0.0 - 1.0)
SPEAK_COOLDOWN = 3.0           # seconds before the same object is announced again
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "detection_log.csv"
CAMERA_INDEX = 0               # change to 1 if you have multiple cameras

# ---------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "object", "confidence"])

print("[INFO] Loading YOLOv8 model (first run downloads ~6MB weights)...")
model = YOLO(MODEL_NAME)
print("[INFO] Model loaded successfully.")

# ---------------------------------------------------------------------
# OFFLINE VOICE ENGINE (runs in its own thread so it never freezes video)
# ---------------------------------------------------------------------
import pyttsx3

speech_queue = queue.Queue()
voice_enabled = True

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        if voice_enabled:
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    if voice_enabled:
        speech_queue.put(text)

# ---------------------------------------------------------------------
# COLOR PER CLASS (consistent color for each object type)
# ---------------------------------------------------------------------
def color_for(label):
    h = abs(hash(label)) % 255
    color = cv2.cvtColor(
        __import__("numpy").uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR
    )[0][0]
    return int(color[0]), int(color[1]), int(color[2])

# ---------------------------------------------------------------------
# DRAW IRON-MAN STYLE BRACKET BOX
# ---------------------------------------------------------------------
def draw_bracket_box(frame, x1, y1, x2, y2, color, thickness=2):
    w, h = x2 - x1, y2 - y1
    l = int(min(w, h) * 0.2)
    # top-left
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, thickness)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, thickness)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, thickness)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - l), color, thickness)
    # faint full box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

# ---------------------------------------------------------------------
# HUD PANEL (semi-transparent background box for text)
# ---------------------------------------------------------------------
def draw_panel(frame, x, y, w, h, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (247, 255, 0), 1)

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
def log_detection(label, confidence):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, f"{confidence:.2f}"]
        )

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
def main():
    global voice_enabled, CONF_THRESHOLD

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check CAMERA_INDEX or camera permissions.")
        return

    last_spoken = {}
    detection_history = []   # for the on-screen scrolling log
    total_scans = 0
    session_start = time.time()
    prev_time = time.time()
    paused = False

    print("[INFO] Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural
        h_frame, w_frame = frame.shape[:2]

        if not paused:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            now = time.time()
            object_count = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = color_for(label)
                draw_bracket_box(frame, x1, y1, x2, y2, color)

                text = f"{label.upper()} {int(confidence * 100)}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                object_count += 1

                if label not in last_spoken or now - last_spoken[label] > SPEAK_COOLDOWN:
                    speak(label)
                    log_detection(label, confidence)
                    detection_history.insert(0, f"{datetime.now().strftime('%H:%M:%S')}  {label} ({int(confidence*100)}%)")
                    detection_history = detection_history[:8]
                    last_spoken[label] = now
                    total_scans += 1

        # ---------------- FPS ----------------
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # ---------------- HUD: top-left status ----------------
        draw_panel(frame, 10, 10, 230, 90)
        dot_color = (0, 255, 0) if not paused else (0, 0, 255)
        cv2.circle(frame, (28, 32), 5, dot_color, -1)
        cv2.putText(frame, "VISION CORE", (40, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elapsed = int(time.time() - session_start)
        cv2.putText(frame, f"SESSION: {elapsed//60:02d}:{elapsed%60:02d}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---------------- HUD: top-right stats ----------------
        draw_panel(frame, w_frame - 220, 10, 210, 70)
        obj_in_frame = len(results.boxes) if not paused else 0
        cv2.putText(frame, f"OBJECTS: {obj_in_frame}", (w_frame - 205, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"TOTAL SCANS: {total_scans}", (w_frame - 205, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---------------- HUD: bottom-left detection log ----------------
        log_h = 30 + len(detection_history) * 18
        draw_panel(frame, 10, h_frame - log_h - 10, 300, log_h)
        cv2.putText(frame, "DETECTION LOG", (20, h_frame - log_h + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        for i, entry in enumerate(detection_history):
            cv2.putText(frame, entry, (20, h_frame - log_h + 28 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # ---------------- HUD: bottom-right controls ----------------
        draw_panel(frame, w_frame - 260, h_frame - 110, 250, 100)
        voice_status = "ON" if voice_enabled else "OFF"
        cv2.putText(frame, f"[v] Voice: {voice_status}", (w_frame - 245, h_frame - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"[p] Pause: {'YES' if paused else 'NO'}", (w_frame - 245, h_frame - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"[+/-] Confidence: {int(CONF_THRESHOLD*100)}%", (w_frame - 245, h_frame - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "[s] Screenshot  [q] Quit", (w_frame - 245, h_frame - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cv2.imshow("JARVIS Vision - Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('v'):
            voice_enabled = not voice_enabled
        elif key == ord('s'):
            filename = os.path.join(SCREENSHOT_DIR, f"scan_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")
        elif key == ord('+'):
            CONF_THRESHOLD = min(0.95, CONF_THRESHOLD + 0.05)
        elif key == ord('-'):
            CONF_THRESHOLD = max(0.10, CONF_THRESHOLD - 0.05)

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    print(f"[INFO] Session ended. Total scans: {total_scans}. Log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
