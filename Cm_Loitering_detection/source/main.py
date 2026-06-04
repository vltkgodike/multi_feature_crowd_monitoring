import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from loit_detect import LoiteringDetector

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("output/detections", exist_ok=True)
os.makedirs("output/loitering", exist_ok=True)
os.makedirs("output/videos", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)

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

log_message("[MODEL] Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
log_message("[MODEL] Model loaded successfully")

# ==========================================
# VIDEO SOURCE - USB ARDUCAM
# ==========================================

log_message("[CAMERA] Initializing USB ArduCAM...")
cap = cv2.VideoCapture(0)

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
# SETUP VIDEO RECORDING
# ==========================================

video_timestamp = time.strftime("%Y%m%d_%H%M%S")

# Try multiple codecs for .mp4 compatibility - ordered by reliability on Windows
codecs_to_try = [
    ('MJPG', 'Motion JPEG', '.avi'),  # Most reliable on Windows
    ('XVID', 'MPEG-4 Part 2', '.mp4'), # Good MP4 support
    ('mp4v', 'MP4V', '.mp4'),          # Standard MP4 (often fails on Windows)
    ('avc1', 'H.264/AVC1', '.mp4'),    # H.264 (often fails on Windows)
    ('X264', 'X264', '.mp4'),          # X264 encoder
]

video_writer = None
selected_codec = None
video_output_path = None

print("[INFO] Initializing video writer...")
log_message("[VIDEO] Attempting to initialize video writer with multiple codecs...")

for codec_code, codec_name, ext in codecs_to_try:
    try:
        # Build the video path with proper extension
        test_path = f"output/videos/detection_stream_{video_timestamp}{ext}"
        
        fourcc = cv2.VideoWriter_fourcc(*codec_code)
        test_writer = cv2.VideoWriter(
            test_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if test_writer.isOpened():
            # Test write a frame to verify codec actually works
            test_frame = cv2.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            test_result = test_writer.write(test_frame)
            test_writer.release()
            
            if test_result:  # Codec actually worked
                video_writer = cv2.VideoWriter(
                    test_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )
                video_output_path = test_path
                selected_codec = codec_name
                log_message(f"[VIDEO]  Codec verified: {codec_name} ({codec_code})")
                print(f"[INFO] Using codec: {codec_name}")
                break
            else:
                # isOpened() returned True but write() failed - skip this codec
                test_writer.release()
                import os as os_module
                if os_module.path.exists(test_path):
                    os_module.remove(test_path)
                log_message(f"[VIDEO]  Codec failed during write test: {codec_name}")
    except Exception as e:
        log_message(f"[VIDEO]  Codec error: {codec_name} - {str(e)}")
        pass

# Fallback if all codecs fail
if video_writer is None or not video_writer.isOpened():
    log_message("[WARNING] All primary codecs failed, creating Motion JPEG fallback")
    video_output_path = f"output/videos/detection_stream_{video_timestamp}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(
            video_output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        if video_writer.isOpened():
            selected_codec = "MJPEG (Fallback)"
            log_message(f"[VIDEO]  Fallback codec activated: {video_output_path}")
        else:
            log_message(f"[ERROR] Fallback codec also failed!")
            video_writer = None
    except Exception as e:
        log_message(f"[ERROR] Failed to initialize fallback codec: {str(e)}")
        video_writer = None

if video_writer is not None and video_writer.isOpened():
    log_message(f"[VIDEO] Recording to: {video_output_path}")
    log_message(f"[VIDEO] Codec: {selected_codec} | FPS: {fps} | Resolution: {frame_width}x{frame_height}")
    print(f"[INFO] Video recording started: {video_output_path}")
else:
    log_message("[ERROR] Failed to initialize video writer with any codec!")
    print("[ERROR] Video recording will NOT be saved")
    video_writer = None

# ==========================================
# LOITERING DETECTOR - TEST WITH 10 SECONDS
# ==========================================

loiter_detector = LoiteringDetector(
    polygon_points=polygon_points,
    loitering_threshold=30,  # 30 seconds for testing (change to 60 for production)
    alert_cooldown=2
)

log_message("[DETECTOR] Loitering detector initialized with 30 second threshold (TESTING MODE)")

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

    # ======================================
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    tracks = []

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

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
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"output/detections/detection_ID{track_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                last_detection_save[track_id] = current_time
                log_message(f"[SAVE] Detection saved - ID: {track_id}")

    # ======================================
    # SAVE LOITERING ALERTS (10+ seconds for testing)
    # ======================================

    if alerts:
        for alert in alerts:
            track_id = alert["track_id"]
            dwell_time = alert["dwell_time"]
            
            # Save loitering image when alert occurs
            if track_id not in last_loitering_save:
                last_loitering_save[track_id] = 0
            
            if (current_time - last_loitering_save[track_id]) > loitering_interval:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Save with dwell time in filename
                filename = f"output/loitering/loitering_ID{track_id}_{dwell_time}s_{timestamp}.jpg"
                
                # Draw alert info on frame before saving
                alert_frame = frame.copy()
                x1, y1, x2, y2 = 100, 100, 540, 200  # Example coordinates
                
                cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.putText(
                    alert_frame,
                    f"LOITERING ALERT!",
                    (x1 + 20, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    alert_frame,
                    f"Person ID: {track_id} | Duration: {dwell_time}s",
                    (x1 + 20, y1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                cv2.imwrite(filename, alert_frame)
                last_loitering_save[track_id] = current_time
                
                log_message(f"[LOITER] 🚨 ALERT SAVED - ID: {track_id} | Duration: {dwell_time}s")
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
    
    # Write frame to video (with error handling)
    if video_writer is not None and video_writer.isOpened():
        try:
            success = video_writer.write(frame)
            if not success:
                log_message(f"[WARNING] Frame write returned False - codec may have failed")
        except Exception as e:
            log_message(f"[ERROR] Failed to write frame: {str(e)}")

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

log_message("[CLEANUP] Closing video writer...")
cap.release()

# Properly flush and release video writer
if video_writer is not None:
    try:
        # Force release to finalize the file
        video_writer.release()
        import time as time_module
        time_module.sleep(1)  # Wait 1 second for file to be written
        
        # Check if file was created and has content
        import os as os_module
        if os_module.path.exists(video_output_path):
            file_size = os_module.path.getsize(video_output_path)
            if file_size > 0:
                log_message(f"[VIDEO]  Video finalized successfully: {video_output_path}")
                log_message(f"[VIDEO] File size: {file_size / (1024*1024):.2f} MB")
            else:
                log_message(f"[ERROR] Video file is empty (0 bytes): {video_output_path}")
        else:
            log_message(f"[ERROR] Video file was not created: {video_output_path}")
    except Exception as e:
        log_message(f"[ERROR] Error finalizing video: {str(e)}")

cv2.destroyAllWindows()

log_message(f"[SYSTEM] Total frames processed: {frame_count}")
log_message(f"[SYSTEM] Total people detected: {len(tracked_people)}")
log_message("[SYSTEM]  Detection stopped - all outputs saved to output/ folder")
log_message("[SYSTEM]  Videos: output/videos/")
log_message("[SYSTEM]  Detections: output/detections/")
log_message("[SYSTEM]  Loitering alerts: output/loitering/")
log_message("[SYSTEM]  Logs: output/logs/")
log_message("[SYSTEM] System shutdown complete")

print("[SUCCESS]  Detection stopped successfully!")
print(f"[SUCCESS] Video file: {video_output_path}")
print("[SUCCESS] All outputs saved to output/ folder")