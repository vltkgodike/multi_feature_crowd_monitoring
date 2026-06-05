import argparse
import cv2
import logging
import os
import sys
import time

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from danger_zone_monitor import DangerZoneMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_monitor(
    rtsp_url: str,
    model_path: str,
    zone_file: str,
    fallback_fps: float,
    retry_interval: int,
    async_inference: bool,
    display: bool,
):
    """Runs the danger zone monitor on an RTSP stream with robust auto-reconnection.
    
    Args:
        rtsp_url: The RTSP stream address.
        model_path: Path to YOLO TensorRT .engine model.
        zone_file: Path to zone configuration file.
        fallback_fps: Fallback FPS if detection fails.
        retry_interval: Time in seconds to wait before trying to reconnect.
    """
    monitor = None
    window_name = f"Danger Zone Monitor - RTSP Stream"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        logger.info(f"Connecting to RTSP stream: {rtsp_url}...")
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            logger.warning(f"Failed to connect to RTSP stream. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
            continue

        # Get stream parameters
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = fallback_fps
        logger.info(f"Stream connected. Processing at {fps:.2f} FPS.")

        # Instantiate monitor only once if possible to preserve event states across reconnections
        if monitor is None:
            monitor = DangerZoneMonitor(
                model_path=model_path,
                zone_file=zone_file,
                fps=fps,
                async_inference=async_inference
            )
        else:
            # Update target FPS in the existing intrusion manager in case camera properties changed
            monitor.intrusion_manager.fps = fps
            monitor.intrusion_manager.max_buffer_size = int(fps * 3.0)

        consecutive_failures = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to grab frame ({consecutive_failures}/5).")
                    if consecutive_failures >= 5:
                        logger.error("Too many frame grab failures. Reconnecting...")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                # Process frame
                processed_frame = monitor.process_frame(frame)
                if display:
                    cv2.imshow(window_name, processed_frame)

                    # Check for exit request
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:
                        logger.info("User exited program.")
                        if monitor is not None:
                            monitor.close()
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        except Exception as e:
            logger.error(f"Error during stream processing: {e}")
        finally:
            cap.release()
            logger.info("RTSP connection closed. Ready to reconnect.")
            
        time.sleep(retry_interval)

def main():
    parser = argparse.ArgumentParser(description="Danger Zone Monitor - RTSP Example")
    parser.add_argument("--url", type=str, required=True, help="RTSP Stream URL (e.g., rtsp://username:password@ip:port/stream)")
    parser.add_argument("--model", type=str, default="models/yolov8n.engine", help="Path to YOLOv8 TensorRT .engine model")
    parser.add_argument("--config", type=str, default="config/zones.json", help="Path to zones JSON configuration")
    parser.add_argument("--fps", type=float, default=25.0, help="Fallback FPS if stream doesn't expose FPS metadata")
    parser.add_argument("--retry", type=int, default=5, help="Seconds to wait before reconnecting after a disconnect")
    parser.add_argument("--sync-inference", action="store_true", help="Disable threaded TensorRT inference")
    parser.add_argument("--display", action="store_true", help="Display the live video feed with overlays")
    args = parser.parse_args()

    run_monitor(
        rtsp_url=args.url,
        model_path=args.model,
        zone_file=args.config,
        fallback_fps=args.fps,
        retry_interval=args.retry,
        async_inference=not args.sync_inference,
        display=args.display
    )

if __name__ == "__main__":
    main()
