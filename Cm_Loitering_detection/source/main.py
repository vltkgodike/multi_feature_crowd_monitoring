import cv2
import os
import sys
import time
import numpy as np
import argparse
from loit_detect import LoiteringDetector

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from danger_zone_monitor.person_tracker import PersonTracker
from danger_zone_monitor.video_recorder import VideoRecorder, extract_camera_id

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("output/logs", exist_ok=True)

# ==========================================
# CLI ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Loitering Detection System")
parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g., 0) or path to video file")
parser.add_argument("--camera-id", type=str, default=None, help="Camera identifier (auto-extracted from source if not provided)")
parser.add_argument("--loiter-threshold", type=int, default=30, help="Loitering threshold in seconds")
args = parser.parse_args()

source = args.source
if source.isdigit():
    source = int(source)

# ==========================================
# SETUP LOGGING
# ==========================================

log_file = f"output/logs/loitering_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    """Save message to log file and print to console"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_entry + "\n")
    except Exception as e:
        # Fallback if UTF-8 fails
        with open(log_file, "a", encoding='utf-8', errors='ignore') as f:
            f.write(log_entry + "\n")

log_message("[SYSTEM] Loitering Detection System Started")

# ==========================================
# GLOBAL VARIABLES FOR MANUAL BOX DRAWING
# ==========================================

drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
roi_selected = False
polygon_points = None

# ==========================================
# MOUSE CALLBACK FOR DRAWING BOX
# ==========================================

def draw_box(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        roi_selected = True
        log_message("[BOX] Loitering box selected!")

# ==========================================
# LOAD MODEL
# ==========================================

log_message("[MODEL] Loading YOLOv8 TensorRT engine model...")
model = PersonTracker("models/yolov8n.engine", async_inference=True)
log_message("[MODEL] Model loaded successfully")

# ==========================================
# VIDEO SOURCE - USB ARDUCAM
# ==========================================

log_message(f"[CAMERA] Initializing camera source: {source}...")
cap = cv2.VideoCapture(source)

# ==========================================
# SET USB CAMERA PROPERTIES
# ==========================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Validate frame dimensions
if frame_width == 0 or frame_height == 0:
    log_message("[ERROR] Invalid frame dimensions, using defaults")
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Ensure dimensions are even (required for many codecs)
if frame_width % 2 != 0:
    frame_width -= 1
if frame_height % 2 != 0:
    frame_height -= 1

log_message(f"[CAMERA] Resolution: {frame_width}x{frame_height} @ {fps}fps")
print(f"[INFO] Camera ready - {frame_width}x{frame_height} @ {fps}fps")

# ==========================================
# SETUP WINDOW AND MOUSE CALLBACK
# ==========================================

cv2.namedWindow("Loitering Detection")
cv2.setMouseCallback("Loitering Detection", draw_box)

log_message("[UI] Waiting for user to draw loitering box...")
print("[INFO] Draw loitering box on screen...")
print("[INFO] Click and drag to draw rectangle, then press Q to confirm")

# ==========================================
# WAIT FOR USER TO DRAW BOX
# ==========================================

while not roi_selected:
    ret, frame = cap.read()
    
    if not ret:
        log_message("[ERROR] Failed to read frame from camera")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    display_frame = frame.copy()
    
    # Draw box while dragging
    if drawing or roi_selected:
        cv2.rectangle(
            display_frame,
            (ix, iy),
            (fx, fy),
            (0, 0, 255),
            2
        )
    
    cv2.putText(
        display_frame,
        "Draw box and press Q to confirm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    cv2.imshow("Loitering Detection", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and roi_selected:
        break

# ==========================================
# CREATE POLYGON FROM DRAWN BOX
# ==========================================

polygon_points = [
    (ix, iy),
    (fx, iy),
    (fx, fy),
    (ix, fy)
]

log_message(f"[BOX] Loitering box coordinates: {polygon_points}")

# ==========================================
# SETUP VIDEO RECORDING WITH VIDEORECORDER
# ==========================================

camera_id = args.camera_id or extract_camera_id(args.source)
recordings_dir = os.path.join(REPO_ROOT, "recordings")
snapshots_dir = os.path.join(REPO_ROOT, "snapshots")

video_recorder = VideoRecorder(
    recordings_dir=recordings_dir,
    snapshots_dir=snapshots_dir,
    camera_id=camera_id
)

log_message(f"[VIDEO] Initializing VideoRecorder for camera_id='{camera_id}'...")
video_writer, video_output_path = video_recorder.start_recording(
    event_id=0,
    fps=fps,
    frame_size=(frame_width, frame_height),
    subdir="loitering"
)

if video_writer is not None and video_writer.isOpened():
    log_message(f"[VIDEO] Recording started: {video_output_path}")
else:
    log_message("[ERROR] Failed to start video recording!")

# ==========================================
# LOITERING DETECTOR INITIALIZATION
# ==========================================

loiter_detector = LoiteringDetector(
    polygon_points=polygon_points,
    loitering_threshold=args.loiter_threshold,
    alert_cooldown=2
)

log_message(f"[DETECTOR] Loitering detector initialized with {args.loiter_threshold} second threshold")

# ==========================================
# TRACKING VARIABLES FOR SAVING
# ==========================================

last_detection_save = {}
last_loitering_save = {}
detection_interval = 1  # Save detection every 1 second
loitering_interval = 1  # Save loitering every 1 second
tracked_people = {}  # Track all people seen

# ==========================================
# MAIN LOOP
# ==========================================

log_message("[MAIN] Starting detection loop...")
print("[INFO] Starting detection... Press Q to stop")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        log_message("[ERROR] Failed to read frame")
        break

    frame_count += 1
    current_time = time.time()
    clean_frame = frame.copy()

    # ======================================
    # YOLO TRACKING
    # ======================================

    tracks = []

    for person in model.track(frame):

        x1, y1, x2, y2 = map(int, person.bbox)
        track_id = int(person.track_id)

        tracks.append({
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2]
        })
        
        # Track people
        if track_id not in tracked_people:
            tracked_people[track_id] = current_time
            log_message(f"[DETECT] New person detected - ID: {track_id}")

    # ======================================
    # LOITERING PROCESS
    # ======================================

    frame, alerts = loiter_detector.process_tracks(
        frame,
        tracks
    )

    # ======================================
    # DISPLAY ZONE STATUS
    # ======================================

    people_in_zone = len(loiter_detector.entry_times)
    
    status_text = f"People in zone: {people_in_zone}"
    cv2.putText(
        frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show entry times for all people in zone
    y_offset = 60
    for person_id, entry_time in loiter_detector.entry_times.items():
        dwell = int(current_time - entry_time)
        dwell_text = f"ID {person_id}: {dwell}s in zone"
        cv2.putText(
            frame,
            dwell_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1
        )
        y_offset += 25

    # ======================================
    # SAVE DETECTIONS IN LOITERING FOLDER
    # ======================================

    if tracks:
        for track in tracks:
            track_id = track["track_id"]
            
            # Save detection image periodically
            if track_id not in last_detection_save:
                last_detection_save[track_id] = 0
            
            if (current_time - last_detection_save[track_id]) > detection_interval:
                # Save snapshot using unified VideoRecorder (full frame + cropped person)
                saved_path = video_recorder.save_snapshot(
                    frame=clean_frame,
                    event_id=0,
                    suffix=f"_detection",
                    subdir="loitering",
                    track_id=track_id,
                    bbox=tuple(track["bbox"])
                )
                last_detection_save[track_id] = current_time
                log_message(f"[SAVE] Detection snapshot saved to {saved_path} for Person ID {track_id}")

    # ======================================
    # SAVE LOITERING ALERTS
    # ======================================

    if alerts:
        for alert in alerts:
            track_id = alert["track_id"]
            dwell_time = alert["dwell_time"]
            
            # Save loitering image when alert occurs
            if track_id not in last_loitering_save:
                last_loitering_save[track_id] = 0
            
            if (current_time - last_loitering_save[track_id]) > loitering_interval:
                # Find bounding box for cropping
                bbox = None
                for t in tracks:
                    if t["track_id"] == track_id:
                        bbox = tuple(t["bbox"])
                        break

                # Draw alert info on frame before saving
                alert_frame = clean_frame.copy()
                x1_alert, y1_alert, x2_alert, y2_alert = 100, 100, 540, 200
                cv2.rectangle(alert_frame, (x1_alert, y1_alert), (x2_alert, y2_alert), (0, 0, 255), -1)
                cv2.putText(
                    alert_frame,
                    "LOITERING ALERT!",
                    (x1_alert + 20, y1_alert + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    alert_frame,
                    f"Person ID: {track_id} | Duration: {dwell_time}s",
                    (x1_alert + 20, y1_alert + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # Save snapshot using unified VideoRecorder (full frame + cropped person)
                saved_path = video_recorder.save_snapshot(
                    frame=alert_frame,
                    event_id=0,
                    suffix=f"_loiter_{dwell_time}s",
                    subdir="loitering",
                    track_id=track_id,
                    bbox=bbox
                )

                last_loitering_save[track_id] = current_time
                log_message(f"[LOITER] 🚨 ALERT SAVED - ID: {track_id} | Duration: {dwell_time}s (saved to {saved_path})")
                print(f"[CONSOLE] 🚨🚨🚨 LOITERING DETECTED - Person {track_id} loitered for {dwell_time} seconds!")


    # ======================================
    # SAVE VIDEO RECORDING
    # ======================================

    # Ensure frame dimensions match video writer
    if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
        # Resize frame if dimensions don't match
        frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Ensure frame is BGR format (standard for OpenCV)
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Write frame to video
    if video_writer is not None:
        video_recorder.write_frame(video_writer, frame)

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow("Loitering Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        log_message("[MAIN] User stopped detection")
        break

# ==========================================
# CLEANUP
# ==========================================

log_message("[CLEANUP] Closing video writer and resources...")
model.close()
cap.release()

if video_writer is not None:
    try:
        video_recorder.stop_recording(video_writer)
        log_message(f"[VIDEO] Video finalized successfully: {video_output_path}")
    except Exception as e:
        log_message(f"[ERROR] Error finalizing video: {str(e)}")

cv2.destroyAllWindows()

log_message(f"[SYSTEM] Total frames processed: {frame_count}")
log_message(f"[SYSTEM] Total people detected: {len(tracked_people)}")
log_message("[SYSTEM] Detection stopped - all outputs saved to hierarchical folders under recordings/ and snapshots/")
log_message(f"[SYSTEM] Videos directory: {recordings_dir}")
log_message(f"[SYSTEM] Snapshots directory: {snapshots_dir}")
log_message("[SYSTEM] System shutdown complete")

print("[SUCCESS] Detection stopped successfully!")
print(f"[SUCCESS] Video file: {video_output_path}")
