# Delhi Metro Rail Corporation (DMRC) PPE Compliance Tracking System

A real-time Personal Protective Equipment (PPE) compliance detection and monitoring dashboard designed for worker safety tracking in Delhi Metro construction and maintenance sites. The system integrates a custom-trained **YOLOv8** computer vision model, a high-performance **FastAPI** backend featuring REST and WebSockets, and a modern **React + Vite** dashboard.

---

## Technical Architecture

```
                                  +-----------------------+
                                  |     Video Streams     |
                                  | (RTSP/USB/Video Files)|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |   Video Processor     |
                                  |    (OpenCV Capture)   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |      YOLOv8 Engine    |
                                  |   (Object Detection)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |     IoU-based PPE     |
                                  |  Association & Tracker|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |    FastAPI Backend    |
                                  |    (REST + WS Server) |
                                  +-----+-----------+-----+
                                        |           |
                                        v           v
                        +---------------+---+   +---+---------------+
                        |   SQLite Database |   |   React Frontend  |
                        | (Metrics & Alerts)|   |  (Vite Dashboard) |
                        +-------------------+   +-------------------+
```

---

## Quick Start & Setup

### Prerequisites
- **Python 3.10** or **3.11**
- **Node.js 18** or newer
- **FFmpeg** (installed on your system PATH, or resolved via `imageio-ffmpeg` package)

### 1. Backend Installation & Launch

1. Navigate to the root directory and create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI API server:
   ```bash
   python backend/core/main.py
   ```
   *The server will start at `http://localhost:8000`. Swagger API documentation is available at `http://localhost:8000/docs`.*

### 2. Frontend Installation & Launch

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install NPM packages:
   ```bash
   npm install
   ```

3. Start the Vite React development server:
   ```bash
   npm run dev
   ```
   *The dashboard will be available at `http://localhost:5173`. Frontend API calls are proxied to the backend via `vite.config.js` settings.*

---

## Machine Learning Model & Training

### 1. Target Safety Classes
The system detects 6 critical classes required for compliance assessment:
- **Person** (Class ID `0`): Baseline object to which accessories are bound.
- **Helmet** (Class ID `1`): Head protection gear.
- **Hands** (Class ID `2`): Protective safety gloves.
- **Shoes** (Class ID `3`): Protective safety shoes.
- **Safety Suit** (Class ID `4`): High-visibility safety jackets or body coveralls.
- **Tools** (Class ID `5`): Maintenance tools or machinery carried by workers.

### 2. Dataset Preparation
The system includes utility scripts under `model/training/prepare_dataset.py` to ingest XML annotation files in Pascal VOC format and normalize the data into standard YOLO format. 

All coordinates are clamped within the $[0, 1]$ range. The script sets up an 80/20 train/validation split dynamically and creates the `data.yaml` configuration.

### 3. Model Training
We fine-tune the **YOLOv8** model architecture utilizing pretrained base weights.
- **Base Architecture:** YOLOv8 Nano (`yolov8n`)
- **Input Dimensions:** $640 \times 640$ pixels
- **Training Config:**
  - **Optimizer:** Adam
  - **Learning Rate:** $0.001$ with cosine scheduler
  - **Epochs:** $8$
  - **Batch Size:** $16$
  - **Augmentations:** Mosaic ($0.5$), MixUp ($0.1$), color jittering in HSV space, random translations, and horizontal flips.
  - **Early Stopping:** Patience of 20 epochs based on validation box loss.

---

## Core Algorithm: Proximity & Overlap-Based PPE Association

Instead of using complex multi-task model heads, compliance is assessed via a spatial intersection algorithm. Accessories are associated with a corresponding person bounding box by calculating their Intersection over Union (IoU) overlay. 

The overlap formula between the worker bounding box ($B_{\text{worker}}$) and the safety accessory bounding box ($B_{\text{gear}}$) is:

$$\text{IoU}(B_{\text{worker}}, B_{\text{gear}}) = \frac{\text{Area}(B_{\text{worker}} \cap B_{\text{gear}})}{\text{Area}(B_{\text{worker}} \cup B_{\text{gear}})}$$

To avoid an $O(N \times M)$ computational bottleneck, the pipeline first performs a cheap bounding box boundary collision test. If the boxes do not intersect, the IoU computation is skipped.

Here is the exact implementation of the association logic in `model/inference/detector.py`:

```python
def get_ppe_violations(self) -> Dict[str, List[Detection]]:
    """
    Check for PPE violations per person.
    Returns dict of violation_type -> list of violating persons.
    """
    violations = {}
    
    persons = [d for d in self.detections if d.class_name == 'person']
    if not persons:
        return violations
    
    # Pre-slice non-person detections once to avoid repeated filtering cost
    non_person_dets = [d for d in self.detections if d.class_name != 'person']

    for person in persons:
        has_helmet = False
        has_gloves = False
        has_shoes = False
        has_suit = False

        px1, py1, px2, py2 = person.bbox
        for det in non_person_dets:
            # Early exit once we have found everything for the current worker
            if has_helmet and has_gloves and has_shoes and has_suit:
                break

            dx1, dy1, dx2, dy2 = det.bbox

            # Coarse overlap check: if there is no intersection, IoU is 0
            if dx2 <= px1 or dx1 >= px2 or dy2 <= py1 or dy1 >= py2:
                continue

            # Calculate IoU with person bbox to check association
            iou = self._calculate_iou(person.bbox, det.bbox)
            if iou < 0.01:
                continue

            if det.class_name == 'helmet':
                has_helmet = True
            elif det.class_name == 'hands':
                has_gloves = True
            elif det.class_name == 'shoes':
                has_shoes = True
            elif det.class_name == 'safety_suit':
                has_suit = True

        if not has_helmet:
            violations.setdefault('no_helmet', []).append(person)
        if not has_gloves:
            violations.setdefault('no_gloves', []).append(person)
        if not has_shoes:
            violations.setdefault('no_shoes', []).append(person)
        if not has_suit:
            violations.setdefault('no_safety_suit', []).append(person)

    return violations
```

---

## FastAPI Video Processing Service

When a video is uploaded via the dashboard, the backend triggers an asynchronous worker thread that reads the video frames, runs the YOLOv8 model, tracks worker targets across frames, annotates bounding boxes/labels, re-encodes the stream using FFmpeg with H.264 compression for browser compatibility, and exposes the file for streaming.

Below is the frame-skipping pipeline implementation from `backend/services/detection/detection_service.py` that accelerates inference on long video uploads:

```python
# Speed up processing for uploads by reducing YOLO inference frequency
skip_frames = 8
frame_idx = 0
processed_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    should_infer = frame_idx % (skip_frames + 1) == 0

    if not should_infer:
        if last_annotated is not None:
            out.write(last_annotated)
        else:
            out.write(frame)
        continue

    # Execute YOLO inference, track items, and assess violations
    detection_result = self.detector.detect(frame)
    violations = detection_result.get_ppe_violations()
    
    # Annotate frame
    annotated = frame.copy()
    annotated = self.detector.draw_detections(annotated, detection_result, show_conf=False, show_track=True)
    annotated = self.detector.draw_violations(annotated, violations)
    annotated = self.detector.draw_ppe_status(annotated, violations, len(persons))
    
    last_annotated = annotated
    out.write(annotated)
    processed_idx += 1
```

---

## React Frontend Pages & Features

The frontend is a dark-mode theme SPA utilizing Tailwind CSS and React Router. Below are the functional details of each page layout:

### 1. Main Dashboard
Provides site managers with a high-level real-time overview of safety compliance. It features KPI cards tracking the current shift compliance rate, total active safety alerts, cumulative violations, and the current headcount of workers in frame. It also lists real-time alert updates and displays breakdown bars for safety violation categories (e.g. Missing Helmet vs Missing Shoes).

<img width="1680" height="935" alt="Screenshot 2026-06-27 at 10 50 48 PM" src="https://github.com/user-attachments/assets/fe3ad273-7400-4f16-b355-73467b178230" />


*Figure 1: Main Dashboard displaying metrics summaries, active alert streams, and compliance trends.*

### 2. Live Video Monitor
Enables live streaming of active cameras (or simulated RTSP feeds). It displays the output frames with colored boxes indicating detected workers, helmets, gloves, and tools, alongside a red highlighted alert box for any worker missing safety equipment.

<img width="1679" height="936" alt="Screenshot 2026-06-29 at 11 40 30 AM" src="https://github.com/user-attachments/assets/20cd71dd-c96b-43d9-966c-e888f4293ed7" />


*Figure 2: Real-time visual monitoring feed with custom annotations and safety warning highlights.*

### 3. Alerts Management
A detailed table sorting active, acknowledged, and resolved alerts. Security operators can check the exact timestamp, camera ID, violation type, and track ID of the violating person. It includes action buttons to **Acknowledge** or **Resolve** warnings as they are handled in the field.

<img width="1679" height="471" alt="Screenshot 2026-06-29 at 11 40 45 AM" src="https://github.com/user-attachments/assets/ed1687d5-3ba3-4df2-818b-cdf8fe1095f2" />


*Figure 3: Alerts manager table with filter states and operator action controls.*

### 4. Safety Analytics
Visualizes long-term trends and safety metrics. It includes line graphs detailing compliance rates over time, histograms of violations by hour/shift, and heatmaps that highlight safety breach hotspots within different zones.

<img width="1680" height="935" alt="Screenshot 2026-06-29 at 11 42 06 AM" src="https://github.com/user-attachments/assets/bed24721-098e-4a3f-9ea3-958d8272a24e" />


*Figure 4: Historical analytics charts detailing worker safety violation trends.*

### 5. Offline Video Upload
Allows operators to drag-and-drop or select local video recordings (MP4, AVI, MOV, MKV) to be analyzed by the server. It displays a live progress bar, a current frame processing counter, and details on active violation detections in the uploaded video. Once complete, it renders an embedded HTML5 video player to stream and download the re-encoded annotated result.

<img width="1679" height="937" alt="Screenshot 2026-06-29 at 11 30 33 AM" src="https://github.com/user-attachments/assets/aec28fe0-c5ca-49d0-a30a-58b4a009a430" />  
<img width="1680" height="935" alt="Screenshot 2026-06-29 at 11 32 12 AM" src="https://github.com/user-attachments/assets/c300775b-ae49-44b3-a698-fe1425d521b0" />  
<img width="1680" height="995" alt="Screenshot 2026-06-29 at 11 32 24 AM" src="https://github.com/user-attachments/assets/ff880a17-5555-4529-ad9b-2dc1aa49d83d" />





*Figure 5: Drag-and-drop video upload queue showing process tracking and output stream previews.*

### 6. Camera Settings
Enables operator management of video inputs. Users can add new RTSP URLs, USB camera indexes, or video test files, toggle the camera status (Active/Inactive), define monitored locations, and name camera zones.

<img width="1677" height="980" alt="Screenshot 2026-06-27 at 10 52 02 PM" src="https://github.com/user-attachments/assets/2608dd91-2bc2-48f5-b75d-a6524d3181c2" />


*Figure 6: System settings panel for configuring, testing, and editing camera streams.*
