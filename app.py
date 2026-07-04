import cv2
import time
import numpy as np
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# 1. Page Configuration and Initialization
st.set_page_config(page_title="JARVIS Vision Cloud Core", layout="wide")
st.title("🧠 JARVIS Vision - Web Control Center")
st.caption("Cloud Deployment Protocol // Powered by YOLOv8 & Streamlit WebRTC")

# Initialize session state tracking variables for the web environment
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "prev_time" not in st.session_state:
    st.session_state.prev_time = time.time()

# 2. Interactive Web Sidebar UI (Replaces cv2.waitKey and keyboard commands)
st.sidebar.header("🎛️ JARVIS Main Core Controls")

paused = st.sidebar.checkbox("⏸️ Pause System Pipeline", value=False)
CONF_THRESHOLD = st.sidebar.slider("🎯 Target Confidence Limit (%)", 10, 95, 40, 5) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Statistics")
st.sidebar.write(f"📈 Total Unique Scans: `{st.session_state.total_scans}`")
elapsed = int(time.time() - st.session_state.session_start)
st.sidebar.write(f"⏱️ Session Active Time: `{elapsed//60:02d}:{elapsed%60:02d}`")

# 3. Model Optimization Cache
@st.cache_resource
def load_yolo_core():
    return YOLO("yolov8n.pt")

model = load_yolo_core()

# 4. Custom Architectural Graphic Engines
def color_for(label):
    h = abs(hash(label)) % 255
    color = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(color[0]), int(color[1]), int(color[2])

def draw_bracket_box(frame, x1, y1, x2, y2, color, thickness=2):
    w, h = x2 - x1, y2 - y1
    l = int(min(w, h) * 0.2)
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - l, y2), color, thickness)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def draw_panel(frame, x, y, w, h, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (247, 255, 0), 1)

# 5. Core WebRTC Live Video Processing Loop
class JarvisVideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Natural mirror framing matrix
        h_frame, w_frame = img.shape[:2]
        
        # Calculate Web FPS Metrics
        curr_time = time.time()
        fps = 1.0 / (curr_time - st.session_state.prev_time) if curr_time != st.session_state.prev_time else 0
        st.session_state.prev_time = curr_time
        
        obj_in_frame = 0
        
        if not paused:
            results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
            obj_in_frame = len(results.boxes)
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                color = color_for(label)
                draw_bracket_box(img, x1, y1, x2, y2, color)
                
                text = f"{label.upper()} {int(confidence * 100)}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
                cv2.putText(img, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Manage scrolling logs within state threads safely
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_entry = f"{timestamp}  {label} ({int(confidence*100)}%)"
                if not st.session_state.detection_history or log_entry not in st.session_state.detection_history[:1]:
                    st.session_state.detection_history.insert(0, log_entry)
                    st.session_state.detection_history = st.session_state.detection_history[:6]
                    st.session_state.total_scans += 1

        # ---------------- HUD Layer Parsing ----------------
        # Top-Left Dashboard System Box
        draw_panel(img, 10, 10, 230, 90)
        dot_color = (0, 255, 0) if not paused else (0, 0, 255)
        cv2.circle(img, (28, 32), 5, dot_color, -1)
        cv2.putText(img, "VISION CORE", (40, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        session_elapsed = int(time.time() - st.session_state.session_start)
        cv2.putText(img, f"SESSION: {session_elapsed//60:02d}:{session_elapsed%60:02d}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Top-Right Data Infrastructure Box
        draw_panel(img, w_frame - 220, 10, 210, 70)
        cv2.putText(img, f"OBJECTS: {obj_in_frame}", (w_frame - 205, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f"TOTAL SCANS: {st.session_state.total_scans}", (w_frame - 205, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom-Left Scrolling Real-Time Verification History Log
        log_h = 30 + len(st.session_state.detection_history) * 18
        draw_panel(img, 10, h_frame - log_h - 10, 300, log_h)
        cv2.putText(img, "DETECTION LOG (LIVE)", (20, h_frame - log_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        for idx, entry in enumerate(st.session_state.detection_history):
            cv2.putText(img, entry, (20, h_frame - log_h + 28 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return img

# 6. Boot Streamlit Render Element with Correct Network Protocols
webrtc_streamer(
    key="jarvis-web-core",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=JarvisVideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:://google.com"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# 7. Web Output Log Area Layout Component Display
if st.session_state.detection_history:
    st.markdown("### 📋 Captured Session Log Stream")
    for entry in st.session_state.detection_history:
        st.text(entry)
