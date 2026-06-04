import argparse
import cv2
import logging
import os
import sys

# Ensure the parent directory is in python path so we can import danger_zone_monitor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from in_out.find_direction import SingleLineCounter
from danger_zone_monitor import DangerZoneMonitor

# Configure standard logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Danger Zone Monitor - Webcam/Video Stream Example")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g., 0) or path to video file")
    parser.add_argument("--model", type=str, default="models/yolov8n.engine", help="Path to YOLOv8 TensorRT .engine model")
    parser.add_argument("--config", type=str, default="config/zones.json", help="Path to zones JSON configuration")
    parser.add_argument("--fps", type=float, default=30.0, help="Expected video frames per second")
    parser.add_argument("--loiter-threshold", type=float, default=10.0, help="Loitering threshold in seconds")
    parser.add_argument("--loiter-cooldown", type=float, default=5.0, help="Loitering alert cooldown in seconds")
    parser.add_argument("--sync-inference", action="store_true", help="Disable threaded TensorRT inference")
    args = parser.parse_args()

    # Determine source (integer for webcam, string for video file path)
    source = args.source
    if source.isdigit():
        source = int(source)

    logger.info(f"Connecting to video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # Try to fetch FPS dynamically from stream
    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps > 0:
        fps = stream_fps
        logger.info(f"Dynamically detected video stream FPS: {fps:.2f}")
    else:
        fps = args.fps
        logger.info(f"Could not detect stream FPS. Falling back to default: {fps:.2f}")

    # Initialize DangerZoneMonitor
    monitor = DangerZoneMonitor(
        model_path=args.model,
        zone_file=args.config,
        fps=fps,
        loitering_threshold=args.loiter_threshold,
        loitering_alert_cooldown=args.loiter_cooldown,
        async_inference=not args.sync_inference
    )
    counter = SingleLineCounter(
        json_file="line.json"
    )

    window_name = "Danger Zone Monitor HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, counter.mouse_callback)

    logger.info("Starting processing loop. Press 'q' or Esc to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video source ended or failed to read frame. Exiting.")
                break

            # Process frame
            processed_frame = monitor.process_frame(frame, counter)

            # Display output
            cv2.imshow(window_name, processed_frame)

            # Handle user interruption
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: # 'q' or ESC
                logger.info("User requested exit.")
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        monitor.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Released camera resource and closed windows.")

if __name__ == "__main__":
    main()
