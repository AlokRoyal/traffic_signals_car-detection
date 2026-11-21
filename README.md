# Traffic Signal Vehicle Detection

![Python](https://img.shields.io/badge/Python-computer%20vision-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-video%20processing-5C3EE8?logo=opencv&logoColor=white)

A computer-vision prototype that marks moving regions in traffic footage. The current main.py uses OpenCV background subtraction and contour filtering to draw bounding boxes over a video stream.

## What it does

- Reads video frames from a local video file or a camera source
- Builds a foreground mask with the MOG2 background subtractor
- Thresholds the mask and extracts external contours
- Discards small contours below a configurable area
- Draws rectangles around the remaining moving regions
- Shows the annotated video until the user presses q

The current implementation detects motion regions; it does not classify vehicle types or control traffic lights.

## Processing flow

Video capture → frame-by-frame MOG2 foreground mask → threshold → contour filtering → bounding boxes → OpenCV display window

## Start locally

Install Python 3 and the direct dependencies, then provide a video named traffic_video.mp4 in the project directory or update the capture source in main.py:

    python -m pip install opencv-python numpy
    python main.py

The video file is not listed in the repository root. You can also change the VideoCapture argument to a camera index, such as 0, if a local camera is available. Press q in the video window to exit.

## Project layout

- main.py — video capture and moving-object detection loop
- yolov3.cfg and coco.names — detector-related files included in the repository; they are not loaded by the current main.py implementation
- README.md — project overview and run instructions

## Limitations and next steps

Background subtraction can produce false detections from shadows, lighting changes, camera motion, or moving background objects. Bounding boxes represent moving regions, not verified cars. The current script has no counting, lane modeling, congestion metric, tracking, signal timing, or automated evaluation. No requirements.txt or test suite is included.

## License

No license file is listed. Check reuse terms for the included configuration and class-name files before redistribution.