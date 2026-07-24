# 🚦 AI-Based Traffic Vehicle Detection & Congestion Monitoring

> **Computer Vision-Based Vehicle Movement Detection for Intelligent Traffic Monitoring and Smart Transportation Systems**

## 🚀 Project Overview

**Traffic Signals Car Detection** is a Computer Vision project designed to detect moving vehicles and traffic activity from road-camera video feeds.

The system uses **Python and OpenCV** to process traffic footage frame by frame, separate moving foreground objects from the background, identify significant moving regions, and generate bounding boxes around detected traffic objects.

The project explores how **Computer Vision can automate traffic monitoring** and establishes a foundation for more advanced systems involving:

* 🚗 Vehicle Detection
* 🚦 Traffic Congestion Analysis
* 📹 CCTV-Based Traffic Monitoring
* 🧠 AI-Based Vehicle Classification
* 🚑 Emergency Vehicle Prioritization
* ⏱️ Adaptive Traffic Signal Control
* 🏙️ Smart City Traffic Management

---

# 🎯 Problem Statement

Traditional traffic signals generally operate using predefined timing cycles.

However, real-world traffic conditions constantly change.

One road might contain:

```text
🚗 🚗 🚗 🚗 🚗 🚗 🚗 🚗
```

while another contains:

```text
🚗 🚗
```

Using the same signal duration for both roads can result in unnecessary waiting time and inefficient traffic flow.

A smarter system should be capable of **observing traffic conditions automatically** and using those observations to support traffic-management decisions.

This project develops the Computer Vision foundation required for such a system.

---

# 💡 Core Concept

The current implementation analyzes a traffic video and identifies significant moving objects.

```text
Traffic Video / CCTV Feed
            │
            ▼
      OpenCV Capture
            │
            ▼
      Frame Processing
            │
            ▼
 Background Subtraction
            │
            ▼
      Binary Mask
            │
            ▼
    Contour Detection
            │
            ▼
 Small Object Filtering
            │
            ▼
  Moving Object Detection
            │
            ▼
     Bounding Boxes
            │
            ▼
 Traffic Monitoring Output
```

This converts raw traffic footage into structured visual information that can later be used for traffic-density estimation and intelligent signal management.

---

# 🧠 Computer Vision Pipeline

## 1. Video Capture

Traffic footage is loaded using OpenCV:

```python
cap = cv2.VideoCapture("traffic_video.mp4")
```

The architecture can also be adapted to receive frames from a compatible camera source.

---

## 2. Background Subtraction

The project uses OpenCV's **MOG2 Background Subtractor**:

```python
fgbg = cv2.createBackgroundSubtractorMOG2()
```

MOG2 attempts to distinguish moving foreground objects from the relatively static road background.

Conceptually:

```text
Original Traffic Frame
          +
Background Estimation
          ↓
Foreground Difference
          ↓
Moving Objects
```

This makes it possible to isolate moving traffic without manually analyzing every frame.

---

# 🎥 Frame-by-Frame Processing

Each frame is continuously extracted from the video:

```python
ret, frame = cap.read()
```

The frame is then passed through the foreground detector:

```python
fgmask = fgbg.apply(frame)
```

The result is a foreground mask representing regions containing movement.

---

# ⚫ Binary Thresholding

The foreground mask is converted into a binary representation using:

```python
_, thresh = cv2.threshold(
    fgmask,
    25,
    255,
    cv2.THRESH_BINARY
)
```

This simplifies subsequent object detection.

Conceptually:

```text
Traffic Frame
     ↓
Foreground Mask
     ↓
Thresholding
     ↓
┌──────────────────────┐
│ BLACK → Background   │
│ WHITE → Movement     │
└──────────────────────┘
```

---

# 🔍 Contour Detection

Once moving regions are isolated, OpenCV searches for their boundaries:

```python
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

Contours represent potential moving objects within the traffic scene.

However, not every contour corresponds to a meaningful vehicle.

Small movements, shadows, visual noise, and environmental changes may also produce contours.

---

# 🧹 Noise Filtering

To reduce insignificant detections, the system applies a minimum contour-area threshold:

```python
min_contour_area = 500
```

Only contours larger than the threshold are retained:

```python
valid_contours = [
    cnt for cnt in contours
    if cv2.contourArea(cnt) > min_contour_area
]
```

This helps remove smaller visual artifacts and focuses the detection pipeline on larger moving objects.

---

# 🚗 Object Localization

For each valid contour, the system calculates a rectangular region:

```python
x, y, w, h = cv2.boundingRect(contour)
```

A bounding box is then displayed around the detected object:

```python
cv2.rectangle(
    frame_with_boxes,
    (x, y),
    (x+w, y+h),
    (0, 255, 0),
    2
)
```

The result is a visual traffic-monitoring system where significant moving objects are highlighted in real time.

---

# 🖥️ Detection Output

The processed traffic footage is displayed through:

```python
cv2.imshow(
    "Traffic Congestion Detection",
    frame_with_boxes
)
```

Conceptually, the output resembles:

```text
┌──────────────────────────────────┐
│                                  │
│       ┌─────────┐                │
│       │ Vehicle │                │
│       └─────────┘                │
│                   ┌──────────┐   │
│                   │ Vehicle  │   │
│                   └──────────┘   │
│                                  │
│    Traffic Camera Footage        │
└──────────────────────────────────┘
```

---

# 🔄 Complete Detection Workflow

```text
            START
              │
              ▼
       Load Traffic Video
              │
              ▼
         Read Frame
              │
              ▼
     MOG2 Background Model
              │
              ▼
       Foreground Mask
              │
              ▼
          Threshold
              │
              ▼
       Detect Contours
              │
              ▼
      Calculate Area
              │
       ┌──────┴──────┐
       │             │
   Area > 500?      NO
       │             │
      YES          Ignore
       │
       ▼
Create Bounding Rectangle
       │
       ▼
 Display Detection
       │
       ▼
    Next Frame
```

---

# 🤖 YOLOv3 & COCO Resources

The repository additionally contains:

```text
yolov3.cfg
coco.names
```

The included **YOLOv3 configuration** is configured for **80 object classes**, while `coco.names` contains the corresponding COCO class names.

Relevant transportation-related classes include:

```text
🚲 Bicycle
🚗 Car
🏍️ Motorbike
🚌 Bus
🚚 Truck
🚦 Traffic Light
```

### Important Implementation Note

The current `main.py` does **not yet perform YOLOv3 inference**.

The implemented detector currently uses:

> **MOG2 Background Subtraction + Thresholding + Contour Detection**

The YOLOv3/COCO resources provide a strong direction for extending the system from generic **moving-object detection** to actual **vehicle recognition and classification**.

---

# ⚖️ Current Detection vs Future YOLO Detection

| Current System                          | YOLO-Based Extension                      |
| --------------------------------------- | ----------------------------------------- |
| Detects moving regions                  | Detects specific objects                  |
| Background subtraction                  | Deep Neural Network                       |
| Contour-based                           | Bounding-box predictions                  |
| Motion focused                          | Object focused                            |
| Lightweight                             | More computationally intensive            |
| Cannot inherently identify vehicle type | Can distinguish cars, buses, trucks, etc. |
| Useful for traffic movement             | Useful for detailed traffic analytics     |

Combining both approaches could create a much more capable traffic-analysis pipeline.

---

# 🛠️ Technology Stack

| Technology        | Purpose                                    |
| ----------------- | ------------------------------------------ |
| **Python**        | Core development                           |
| **OpenCV**        | Computer Vision and video processing       |
| **NumPy**         | Numerical operations                       |
| **MOG2**          | Foreground/background separation           |
| **Contours**      | Moving-object detection                    |
| **YOLOv3 Config** | Future deep-learning detection integration |
| **COCO Classes**  | Object-class definitions                   |

---

# 📁 Project Structure

```text
traffic_signals_car-detection/
│
├── main.py
│   └── Traffic video processing
│       Background subtraction
│       Thresholding
│       Contour detection
│       Bounding-box visualization
│
├── coco.names
│   └── COCO object class names
│
├── yolov3.cfg
│   └── YOLOv3 network configuration
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd traffic_signals_car-detection
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install opencv-python numpy
```

---

# 🎥 Add Traffic Footage

The current script expects:

```text
traffic_video.mp4
```

Place the video in the project directory:

```text
traffic_signals_car-detection/
│
├── main.py
├── traffic_video.mp4
├── coco.names
└── yolov3.cfg
```

Alternatively, modify:

```python
cap = cv2.VideoCapture("traffic_video.mp4")
```

to point to your own traffic video.

---

# ▶️ Run the Project

Execute:

```bash
python main.py
```

A window titled:

```text
Traffic Congestion Detection
```

will display the processed traffic footage.

Press:

```text
Q
```

to stop the program.

---

# 🚦 Application to Intelligent Traffic Signals

The current detection system can act as the **perception layer** of a larger intelligent traffic-signal architecture.

```text
             CCTV Camera
                  │
                  ▼
        Computer Vision Model
                  │
                  ▼
          Vehicle Detection
                  │
                  ▼
         Traffic Estimation
                  │
                  ▼
        Congestion Analysis
                  │
                  ▼
       Traffic Signal Engine
            ┌─────┴─────┐
            ▼           ▼
      More Traffic   Less Traffic
            │           │
            ▼           ▼
      Longer Green   Shorter Green
```

Instead of operating entirely on predefined timings, traffic signals could eventually respond to observed traffic conditions.

---

# 🌆 Potential Smart City Applications

This project can serve as the foundation for several intelligent transportation applications:

* **Adaptive traffic signal control**
* **Automated vehicle counting**
* **Traffic-density estimation**
* **Intersection congestion monitoring**
* **CCTV-based traffic analytics**
* **Emergency vehicle detection**
* **Traffic violation analysis**
* **Historical traffic analysis**
* **Smart city dashboards**

---

# 🔮 Future Scope

## 🚘 YOLO-Based Vehicle Classification

The next major improvement would be integrating YOLO inference into the existing pipeline.

The system could then distinguish:

```text
Car ─────────► 🚗
Motorcycle ──► 🏍️
Bus ─────────► 🚌
Truck ───────► 🚚
```

This would transform the system from movement detection into **AI-based vehicle detection and classification**.

---

## 🔢 Automatic Vehicle Counting

Detected vehicles could be tracked across frames to calculate:

```text
Cars        → 35
Motorcycles → 21
Buses       → 4
Trucks      → 7

Total       → 67 vehicles
```

Vehicle tracking would be required to avoid repeatedly counting the same vehicle across consecutive frames.

---

## 📊 Traffic Density Classification

Vehicle counts and occupied road area could be converted into congestion categories:

```text
LOW TRAFFIC
████░░░░░░

MEDIUM TRAFFIC
███████░░░

HIGH TRAFFIC
██████████
```

This would provide a direct traffic-density metric for signal-control algorithms.

---

# 🚦 Adaptive Traffic Signal Timing

A major extension of this project is dynamically calculating green-light duration.

For example:

```text
Lane A → 52 vehicles → 60 sec green
Lane B → 31 vehicles → 40 sec green
Lane C → 12 vehicles → 20 sec green
Lane D →  8 vehicles → 15 sec green
```

Signal timing could therefore respond to actual traffic demand rather than remaining completely fixed.

---

# 🚑 Emergency Vehicle Detection

A future deep-learning model could recognize emergency vehicles such as:

```text
🚑 Ambulance
🚒 Fire Engine
🚓 Police Vehicle
```

When detected, the traffic-control system could potentially prioritize the corresponding route.

```text
Emergency Vehicle Detected
           ↓
Identify Direction
           ↓
Traffic Controller
           ↓
Prioritize Green Signal
           ↓
Clear Traffic Corridor
```

---

# 📹 Live CCTV Integration

The current implementation processes a video file.

Future versions could integrate:

```text
CCTV Cameras
      ↓
RTSP / Live Video
      ↓
Computer Vision
      ↓
Traffic Analytics
      ↓
Signal Controller
```

This would enable continuous real-world intersection monitoring.

---

# 🧠 Advanced Computer Vision

Future versions could incorporate:

* YOLOv8 / newer object detectors
* Object tracking
* ByteTrack
* DeepSORT
* Lane detection
* Vehicle-speed estimation
* Number-plate detection
* Traffic-flow estimation
* Accident detection
* Queue-length estimation

This would significantly increase the intelligence available to the traffic-management system.

---

# 🏗️ Long-Term Architecture

```text
              SMART INTERSECTION
                      │
                CCTV Cameras
                      │
                      ▼
            Computer Vision AI
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      Vehicle Detection    Vehicle Tracking
            │                   │
            └─────────┬─────────┘
                      ▼
               Traffic Analytics
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      Vehicle Count       Traffic Density
            │                   │
            └─────────┬─────────┘
                      ▼
             Decision Algorithm
                      │
                      ▼
          Dynamic Signal Timing
                      │
                      ▼
              Traffic Signals
```

---

# 🌍 Why This Project Matters

Traffic congestion contributes to:

* Increased travel time
* Higher fuel consumption
* Vehicle emissions
* Delayed emergency response
* Reduced road efficiency
* Driver frustration

Computer Vision provides the ability to transform ordinary traffic-camera footage into actionable information.

This project demonstrates the intersection of:

> **🤖 Artificial Intelligence × 👁️ Computer Vision × 🚦 Traffic Management × 🏙️ Smart Cities**

The current implementation establishes the first layer — **detecting traffic movement from video** — while providing a foundation for intelligent vehicle classification and adaptive traffic management.

---

# 👨‍💻 Developed By

**Alok Roy**

Computer Science & Engineering

**Artificial Intelligence • Computer Vision • Data Analytics • IoT • Intelligent Transportation Systems**

---

# ⭐ Support the Project

If you find this project useful, consider giving the repository a **⭐ Star**.

Contributions and ideas related to Computer Vision, Intelligent Transportation Systems, traffic optimization, and Smart City technologies are welcome.

> ### 🚦 Turning traffic cameras into intelligent traffic infrastructure.
>
> **Detect • Analyze • Optimize**
