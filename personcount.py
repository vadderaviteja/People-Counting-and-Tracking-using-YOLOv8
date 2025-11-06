"""
people_counter_yolov8.py

Usage:
    python people_counter_yolov8.py --input /path/to/input.mp4 --output /path/to/output.mp4

Notes:
- Uses YOLOv8 (pretrained) for robust multi-person detection.
- Tracking handled by CentroidTracker.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import math
from ultralytics import YOLO


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=90):
        self.nextObjectID = 0
        self.objects = {}  # id -> (x,y)
        self.disappeared = {}  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_history = {}  # id -> [(x,y), ...]
        self.counted = {}  # id -> bool

    def register(self, centroid):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.track_history[oid] = [centroid]
        self.counted[oid] = False
        self.nextObjectID += 1

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.disappeared[oid]
            del self.track_history[oid]
            del self.counted[oid]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assignedRows, assignedCols = set(), set()
        for r, c in zip(rows, cols):
            if r in assignedRows or c in assignedCols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = objectIDs[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0
            self.track_history[oid].append(input_centroids[c])
            assignedRows.add(r)
            assignedCols.add(c)

        for i in range(len(objectCentroids)):
            if i not in assignedRows:
                oid = objectIDs[i]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for j in range(len(input_centroids)):
            if j not in assignedCols:
                self.register(input_centroids[j])

        return self.objects


def main(input_path, output_path, process_every_k=2, model_name="yolov8n.pt"):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Load YOLOv8 pretrained model
    model = YOLO(model_name)  # "yolov8n.pt" or "yolov8s.pt"

    tracker = CentroidTracker(max_disappeared=40, max_distance=90)
    frame_no = 0
    entered_count = 0
    exited_count = 0
    line_y = height // 2
    line_offset = 12

    print("Starting YOLOv8 processing... (press Ctrl+C to stop)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if frame_no % process_every_k != 0:
            cv2.line(frame, (0, line_y), (width, line_y), (0,255,255), 2)
            out.write(frame)
            continue

        # Run YOLOv8 inference
        results = model.predict(source=frame, conf=0.4, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # Class 0 = person in COCO
                    x1, y1, x2, y2 = box[:4].astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append((cx, cy, x1, y1, x2-x1, y2-y1))

        input_centroids = [(d[0], d[1]) for d in detections]
        _ = tracker.update(input_centroids)

        # Check line crossing
        for oid, centroid in list(tracker.objects.items()):
            history = tracker.track_history.get(oid, [])
            if len(history) >= 2 and not tracker.counted.get(oid, False):
                prev_y = history[-2][1]
                cur_y = history[-1][1]
                if prev_y < line_y - line_offset and cur_y >= line_y - line_offset:
                    entered_count += 1
                    tracker.counted[oid] = True
                elif prev_y > line_y + line_offset and cur_y <= line_y + line_offset:
                    exited_count += 1
                    tracker.counted[oid] = True

        # Draw overlays
        cv2.line(frame, (0, line_y), (width, line_y), (0,255,255), 2)
        cv2.putText(frame, f"Entered: {entered_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"Exited: {exited_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(frame, f"Tracked: {len(tracker.objects)}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Draw detection boxes and IDs
        for (cx, cy, x, y, w, h) in detections:
            assigned_id = None
            for oid, cent in tracker.objects.items():
                if abs(cent[0] - cx) < 40 and abs(cent[1] - cy) < 60:
                    assigned_id = oid
                    break
            color = (0,255,0) if assigned_id is not None else (0,128,255)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            if assigned_id is not None:
                cv2.putText(frame, f"ID {assigned_id}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Finished. Entered:", entered_count, "Exited:", exited_count)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video")
    ap.add_argument("--output", required=True, help="Path to output video")
    ap.add_argument("--k", type=int, default=1, help="Process every k-th frame (for speed)")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model file (yolov8n.pt or yolov8s.pt)")
    args = ap.parse_args()

    main(args.input, args.output, process_every_k=args.k, model_name=args.model)





