import argparse
import cv2
import logging
import os
import sys

# Ensure the parent directory is in python path so we can import danger_zone_monitor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from in_out.find_direction import SingleLineCounter
from danger_zone_monitor import DangerZoneMonitor
from danger_zone_monitor.tools.zone_drawer import ZoneDrawer
from danger_zone_monitor.video_recorder import extract_camera_id

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
    parser.add_argument("--display", action="store_true", help="Display the live video feed with overlays")
    parser.add_argument("--camera-id", type=str, default=None, help="Camera identifier (auto-extracted from source if not provided)")
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

    # Resolve camera_id: user-provided > auto-extracted > fallback
    camera_id = args.camera_id or extract_camera_id(args.source)

    # Initialize DangerZoneMonitor
    monitor = DangerZoneMonitor(
        model_path=args.model,
        zone_file=args.config,
        fps=fps,
        loitering_threshold=args.loiter_threshold,
        loitering_alert_cooldown=args.loiter_cooldown,
        async_inference=not args.sync_inference,
        camera_id=camera_id
    )
    counter = SingleLineCounter(
        json_file="line.json"
    )
    zone_editor = ZoneDrawer(
        config_path=args.config,
        open_source=False
    )
    ui_state = {
        "edit_mode": "line"
    }

    window_name = "Danger Zone Monitor HUD"
    if args.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param):
            if ui_state["edit_mode"] == "zone":
                zone_editor.handle_mouse_event(event, x, y, flags, param)
            else:
                counter.mouse_callback(event, x, y, flags, param)

        cv2.setMouseCallback(window_name, mouse_callback)

    if args.display:
        logger.info("Starting processing loop. Press 'q' or Esc to exit.")
        logger.info("UI modes: press 'L' for in/out line editing, 'Z' for zone editing.")
        logger.info("Zone mode: click to add/select, drag vertices/zones, S save, D/Delete delete, C clear.")
    else:
        logger.info("Starting processing loop in headless mode. Press Ctrl+C to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video source ended or failed to read frame. Exiting.")
                break

            # Process frame
            processed_frame = monitor.process_frame(frame, counter)

            if args.display:
                if ui_state["edit_mode"] == "zone":
                    zone_editor.set_frame(processed_frame)
                    zone_editor.draw_on_frame(processed_frame)

                mode_label = f"EDIT MODE: {ui_state['edit_mode'].upper()}  |  L: line  Z: zones"
                cv2.putText(
                    processed_frame,
                    mode_label,
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Display output
                cv2.imshow(window_name, processed_frame)

                # Handle user interruption
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]: # 'q' or ESC
                    logger.info("User requested exit.")
                    break
                if key in [ord('z'), ord('Z')]:
                    ui_state["edit_mode"] = "zone"
                    logger.info("Switched to zone editing mode.")
                elif key in [ord('l'), ord('L')]:
                    ui_state["edit_mode"] = "line"
                    logger.info("Switched to in/out line editing mode.")
                elif ui_state["edit_mode"] == "zone" and key in [ord('s'), ord('S')]:
                    zone_editor.save_current_zone()
                elif ui_state["edit_mode"] == "zone" and key in [ord('d'), ord('D'), 8, 127]:
                    zone_editor.delete_selected_zone()
                elif ui_state["edit_mode"] == "zone" and key in [ord('c'), ord('C')]:
                    zone_editor.clear_current_polygon()

                if zone_editor.consume_config_changed():
                    try:
                        monitor.zone_manager.load_zones()
                        logger.info("Reloaded active danger zones after UI edit.")
                    except Exception as e:
                        logger.error(f"Failed to reload danger zones after UI edit: {e}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        monitor.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Released camera resource and closed windows.")

if __name__ == "__main__":
    main()
