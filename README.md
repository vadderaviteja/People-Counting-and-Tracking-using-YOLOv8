# PROJECT TITLE
### People Counting and Tracking using YOLOv8


# Overview

This project detects, tracks, and counts people crossing a virtual line in a video using YOLOv8 (a state-of-the-art object detection model) and a Centroid Tracker.

It can analyze security footage, office entrances, event videos, or any public area to estimate how many people entered or exited a particular zone.

# Features

Real-time multi-person detection using YOLOv8

Automatic tracking of each person using a Centroid Tracker

Counts people who enter or exit across a virtual line

Works on any video file or live camera feed

Supports multiple YOLOv8 model sizes (yolov8n.pt, yolov8s.pt)

Saves processed output as a new video

# Technologies Used

Language: Python

Libraries:

ultralytics (for YOLOv8 detection)

OpenCV (for video handling and drawing)

NumPy (for mathematical operations)

argparse (for command-line arguments)

# Installation

Clone or download this project.

git clone https://github.com/vadderaviteja/People-Counting-and-Tracking-using-YOLOv8.git

cd People-Counting-and-Tracking-using-YOLOv8


# Install dependencies:

pip install ultralytics opencv-python numpy

Download YOLOv8 pretrained weights (automatically handled):
When you run the script for the first time, YOLOv8 will automatically download the required model weights (yolov8n.pt or yolov8s.pt) from the internet.
You don’t need to upload or include these files.

# Usage
Run the program:
python personcount.py --input /path/to/input.mp4 --output /path/to/output.mp4

Optional arguments:
Argument	Description	Default

--input	Path to input video file	Required

--output	Path to save processed output	Required

--k	Process every k-th frame (to control speed)	1

--model	YOLOv8 model name (yolov8n.pt or yolov8s.pt)	yolov8n.pt
