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

# Global variables for mouse callback and existing zones
points = []
current_frame = None
display_frame = None
existing_zones = []

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
    """Redraws the current frame with points, existing zones, and lines connecting the current points."""
    global points, current_frame, display_frame, existing_zones
    
    if current_frame is None:
        return
        
    display_frame = current_frame.copy()
    
    # Draw instructions overlay
    h, w = display_frame.shape[:2]
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    instruction_text = "Left Click: Add Point | Right Click: Undo | S: Save | D: Delete Zone | Q: Exit"
    cv2.putText(display_frame, instruction_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw existing zones first (so new points are drawn on top)
    for zone in existing_zones:
        zone_pts = zone.get("points", [])
        if len(zone_pts) >= 3:
            pts_np = np.array(zone_pts, dtype=np.int32)
            # Draw existing zone polygon
            color = (0, 255, 255)  # Yellow
            cv2.polylines(display_frame, [pts_np], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            
            # Label the existing zone with ID and Name
            rx, ry, rw, rh = cv2.boundingRect(pts_np)
            label = f"ID {zone.get('zone_id')}: {zone.get('zone_name')}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (w_l, h_l), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(ry, h_l + 10)
            cv2.rectangle(display_frame, (rx, label_y - h_l - 6), (rx + w_l + 10, label_y), color, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(display_frame, label, (rx + 5, label_y - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
    # Draw existing points and lines of the zone currently being drawn
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
    global current_frame, display_frame, points, existing_zones
    
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
    
    # Load existing zones at startup
    existing_zones = []
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                content = json.load(f)
                existing_zones = content.get("zones", [])
                logger.info(f"Loaded {len(existing_zones)} existing zones from '{args.config}'.")
        except Exception as e:
            logger.warning(f"Could not read existing config at startup. Error: {e}")
            
    cv2.namedWindow("Zone Drawer Tool", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Zone Drawer Tool", mouse_callback)
    
    # Initial redraw to show instructions and existing zones
    redraw()
    
    logger.info("Interactions:")
    logger.info("  - LEFT click to add vertices")
    logger.info("  - RIGHT click to undo last vertex")
    logger.info("  - Press 'S' to save current zone")
    logger.info("  - Press 'D' to delete an existing zone")
    logger.info("  - Press 'Q' to exit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' key to quit
        if key == ord('q') or key == 27:
            logger.info("Exiting drawer tool.")
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
                
            # Load config directory
            config_dir = os.path.dirname(args.config)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                
            # Recalculate max_id from current existing zones
            max_id = 0
            for z in existing_zones:
                max_id = max(max_id, z.get("zone_id", 0))
            
            new_id = max_id + 1
            new_pts = [list(pt) for pt in points]
            new_zone = {
                "zone_id": new_id,
                "zone_name": zone_name,
                "points": new_pts
            }
            
            existing_zones.append(new_zone)
            
            # Write updated zones back to file
            try:
                with open(args.config, 'w') as f:
                    json.dump({"zones": existing_zones}, f, indent=2)
                logger.info(f"Successfully saved zone '{zone_name}' (ID: {new_id}) to '{args.config}'.")
                
                # Upsert to database
                try:
                    import postgres_db
                    postgres_db.upsert_zone(zone_id=new_id, name=zone_name, points=new_pts)
                except Exception as db_err:
                    logger.warning(f"Could not upsert zone ID {new_id} to database (may not be configured): {db_err}")
                
                print(f"Zone '{zone_name}' saved! Coordinates written to '{args.config}'.")
                print("="*40 + "\n")
                
                # Clear points for drawing the next zone
                points = []
                redraw()
            except Exception as e:
                logger.error(f"Failed to write configuration: {e}")
                
        # 'd' key to delete zone
        elif key == ord('d'):
            if not existing_zones:
                logger.warning("No existing zones to delete.")
                continue
                
            print("\n" + "="*40)
            print("Existing Zones:")
            for z in existing_zones:
                print(f"  ID: {z.get('zone_id')} - Name: {z.get('zone_name')}")
            
            zone_id_str = input("Enter the ID of the zone to delete (or press Enter to cancel): ").strip()
            if not zone_id_str:
                print("Deletion cancelled.")
                print("="*40 + "\n")
                continue
                
            try:
                zone_id_to_delete = int(zone_id_str)
            except ValueError:
                logger.error("Invalid input. Please enter a valid numeric ID.")
                print("="*40 + "\n")
                continue
                
            # Check if zone exists
            zone_to_delete = next((z for z in existing_zones if z.get("zone_id") == zone_id_to_delete), None)
            if zone_to_delete is None:
                logger.error(f"Zone with ID {zone_id_to_delete} not found.")
                print("="*40 + "\n")
                continue
                
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete zone '{zone_to_delete.get('zone_name')}' (ID: {zone_id_to_delete})? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Deletion cancelled.")
                print("="*40 + "\n")
                continue
                
            # Remove from local list
            existing_zones = [z for z in existing_zones if z.get("zone_id") != zone_id_to_delete]
            
            # Save updated list to file
            try:
                with open(args.config, 'w') as f:
                    json.dump({"zones": existing_zones}, f, indent=2)
                logger.info(f"Successfully deleted zone ID {zone_id_to_delete} from '{args.config}'.")
                
                # Delete from PostgreSQL database as well
                try:
                    import postgres_db
                    postgres_db.delete_zone(zone_id_to_delete)
                except Exception as db_err:
                    logger.warning(f"Could not delete zone ID {zone_id_to_delete} from database: {db_err}")
                
                print(f"Zone ID {zone_id_to_delete} deleted!")
                print("="*40 + "\n")
                
                redraw()
            except Exception as e:
                logger.error(f"Failed to write configuration: {e}")
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
