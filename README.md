# JARVIS Vision — Real-Time Object Detection with Voice

A semester computer vision project: detects objects from your webcam in real time
using **YOLOv8** (a modern, fast object-detection model) and announces each detected
object out loud using **fully offline text-to-speech**.

---

## 1. What's inside

```
object-detection-project/
├── main.py              <- the whole program (run this)
├── requirements.txt      <- libraries to install
├── detection_log.csv     <- auto-created log of every detection (for your report)
└── screenshots/           <- auto-created folder for saved snapshots (press 's')
```

## 2. Requirements

- Python 3.9–3.11 (recommended)
- A webcam
- ~2 minutes for setup, no GPU needed

## 3. Setup (run these in your terminal / VS Code terminal)

```bash
# 1. Go into the project folder
cd object-detection-project

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

That's it. `ultralytics` will automatically download the YOLOv8n model weights
(~6 MB) the first time you run the script — no manual downloading needed.

## 4. Run it

```bash
python main.py
```

A window will open showing your webcam feed with detection boxes, a HUD, and
voice announcements of each new object detected.

## 5. Controls (press while the video window is focused)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / resume detection |
| `v` | Toggle voice on/off |
| `s` | Save a screenshot to `screenshots/` |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

## 6. Why YOLOv8 instead of the browser model?

YOLOv8 (2023, Ultralytics) is a newer, more accurate architecture than older
models like SSD/MobileNet. It handles **multiple overlapping objects much
better** (e.g., a person holding a phone and a toothbrush at the same time)
because of its anchor-free detection head and improved non-max suppression.
It also runs comfortably on CPU in real time, so no expensive hardware or
long setup is needed — ideal for a semester project demo.

## 7. Important limitation to mention in your report

YOLOv8n (like most general object detectors) is trained on the **COCO
dataset's 80 object classes**. It can only recognize objects from that
fixed list — it cannot detect *every* object that exists. For example:

- ✅ Detected: person, cell phone, toothbrush, bottle, laptop, book, cup, etc.
- ❌ Not detected: toothpaste, chargers, and other items with no matching
  COCO category — no general-purpose detector trained on COCO will catch these.

You can list the full 80-class list in your report (run `print(model.names)`
in Python, or look up "COCO 80 classes") to show you understand the model's
scope rather than treating it as a bug.

### Optional stretch goal (only if you want to go further)
If your project requires detecting *arbitrary* objects by name (not limited
to COCO's 80 classes), look into **YOLO-World** (open-vocabulary detection)
or **Grounding DINO**. These are heavier and slower to set up, so they are
not included here to keep the project simple and fast to run — but worth
mentioning as "future work" in your report.

## 8. Talking points for your viva / demo

- Explain the real-time pipeline: webcam frame → YOLOv8 inference → bounding
  boxes + confidence scores → HUD overlay → offline TTS announcement.
- Show the `detection_log.csv` as evidence of a working logging system.
- Explain the cooldown logic (`SPEAK_COOLDOWN`) — prevents the same object
  being announced repeatedly every frame.
- Explain the confidence threshold and what raising/lowering it does
  (precision vs. recall trade-off).
- Mention model is CPU-friendly and lightweight — good engineering choice
  for a real-time demo without GPU.
