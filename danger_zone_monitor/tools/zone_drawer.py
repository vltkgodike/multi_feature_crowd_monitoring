import argparse
import json
import logging
import os
import sys
import cv2
import numpy as np

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for mouse callback
points = []
current_frame = None
display_frame = None

def mouse_callback(event, x, y, flags, param):
    """Callback function for OpenCV mouse events."""
    global points, current_frame, display_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point
        points.append((x, y))
        logger.info(f"Added point: ({x}, {y})")
        redraw()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last point
        if points:
            removed = points.pop()
            logger.info(f"Removed point: {removed}")
            redraw()

def redraw():
    """Redraws the current frame with points and lines connecting them."""
    global points, current_frame, display_frame
    
    if current_frame is None:
        return
        
    display_frame = current_frame.copy()
    
    # Draw instructions overlay
    h, w = display_frame.shape[:2]
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    instruction_text = "Left Click: Add Point | Right Click: Undo | S: Save | Q: Exit"
    cv2.putText(display_frame, instruction_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw existing points and lines
    for i, pt in enumerate(points):
        # Draw vertex point
        cv2.circle(display_frame, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(display_frame, str(i + 1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Connect to previous point
        if i > 0:
            cv2.line(display_frame, points[i - 1], pt, (0, 255, 0), 2, cv2.LINE_AA)
            
    # Connect last point to first if there are at least 3 points to preview the polygon
    if len(points) >= 3:
        cv2.line(display_frame, points[-1], points[0], (255, 0, 0), 1, cv2.LINE_AA)
        
    cv2.imshow("Zone Drawer Tool", display_frame)

def main():
    global current_frame, display_frame, points
    
    parser = argparse.ArgumentParser(description="Interactive Danger Zone Drawing Tool")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or path to video file/image")
    parser.add_argument("--config", type=str, default="config/zones.json", help="Path to save the JSON coordinates")
    args = parser.parse_args()

    # Determine source type
    source = args.source
    if source.isdigit():
        source = int(source)

    logger.info(f"Opening video/image source: {source}")
    
    # Try reading as an image first
    if isinstance(source, str) and os.path.exists(source) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        current_frame = cv2.imread(source)
    else:
        # Try reading as a video stream or camera
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            # Grab one frame for drawing
            ret, frame = cap.read()
            if ret:
                current_frame = frame
            cap.release()
            
    # Fallback to a blank canvas if no frame could be read
    if current_frame is None:
        logger.warning("Could not read frame from source. Creating a blank 640x480 canvas.")
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50 # Grey background
        
    display_frame = current_frame.copy()
    
    cv2.namedWindow("Zone Drawer Tool", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Zone Drawer Tool", mouse_callback)
    
    # Initial redraw to show instructions
    redraw()
    
    logger.info("Interactions:")
    logger.info("  - LEFT click to add vertices")
    logger.info("  - RIGHT click to undo last vertex")
    logger.info("  - Press 'S' to save")
    logger.info("  - Press 'Q' to exit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' key to quit
        if key == ord('q') or key == 27:
            logger.info("Exiting drawer tool without saving.")
            break
            
        # 's' key to save
        elif key == ord('s'):
            if len(points) < 3:
                logger.warning("A polygon must have at least 3 points. Click more points before saving.")
                continue
                
            # Prompt for zone name
            print("\n" + "="*40)
            zone_name = input("Enter a name for this danger zone [DangerZone]: ").strip()
            if not zone_name:
                zone_name = "DangerZone"
                
            # Load existing config to append rather than overwrite
            config_dir = os.path.dirname(args.config)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                
            existing_zones = []
            max_id = 0
            if os.path.exists(args.config):
                try:
                    with open(args.config, 'r') as f:
                        content = json.load(f)
                        existing_zones = content.get("zones", [])
                        for z in existing_zones:
                            max_id = max(max_id, z.get("zone_id", 0))
                except Exception as e:
                    logger.warning(f"Could not read existing config. Creating new file. Error: {e}")
            
            new_id = max_id + 1
            new_zone = {
                "zone_id": new_id,
                "zone_name": zone_name,
                "points": [list(pt) for pt in points]
            }
            
            existing_zones.append(new_zone)
            
            # Write updated zones back to file
            try:
                with open(args.config, 'w') as f:
                    json.dump({"zones": existing_zones}, f, indent=2)
                logger.info(f"Successfully saved zone '{zone_name}' (ID: {new_id}) to '{args.config}'.")
                print(f"Zone '{zone_name}' saved! Coordinates written to '{args.config}'.")
                print("="*40 + "\n")
                
                # Clear points for drawing the next zone if desired
                points = []
                redraw()
            except Exception as e:
                logger.error(f"Failed to write configuration: {e}")
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
