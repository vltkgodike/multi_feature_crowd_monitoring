import json
import logging
from typing import List, Tuple
import cv2
import numpy as np
from danger_zone_monitor.models import Zone

logger = logging.getLogger(__name__)

class ZoneManager:
    """Manages loading, testing, and drawing of polygon danger zones."""

    def __init__(self, zone_file: str):
        """Initializes ZoneManager and loads zones from file.
        
        Args:
            zone_file: Path to the JSON configuration file containing zone coordinates.
        """
        self.zone_file = zone_file
        self.zones: List[Zone] = []
        self.load_zones()

    def load_zones(self) -> None:
        """Loads and parses zones from a JSON file."""
        try:
            with open(self.zone_file, "r") as f:
                data = json.load(f)
            
            self.zones = []
            for zone_data in data.get("zones", []):
                zone_id = int(zone_data["zone_id"])
                zone_name = str(zone_data["zone_name"])
                # Extract points and convert to (x, y) tuples
                points = [tuple(map(int, pt)) for pt in zone_data["points"]]
                
                # Check for minimum points to form a polygon
                if len(points) < 3:
                    logger.warning(f"Zone {zone_name} (ID: {zone_id}) has less than 3 points. Skipping.")
                    continue
                    
                zone = Zone(zone_id=zone_id, zone_name=zone_name, points=points)
                self.zones.append(zone)
                logger.info(f"Loaded zone '{zone_name}' (ID: {zone_id}) with {len(points)} points.")
                try:
                    import postgres_db
                    postgres_db.upsert_zone(zone_id=zone_id, name=zone_name, points=points)
                except Exception as e:
                    logger.error(f"Failed to upsert zone {zone_id} in database: {e}")
        except Exception as e:
            logger.error(f"Failed to load zones from '{self.zone_file}': {e}")
            raise e

    def check_point_in_zones(self, point: Tuple[int, int]) -> List[Zone]:
        """Checks which zones contain the given point.
        
        Args:
            point: The point (cx, cy) to test.
            
        Returns:
            A list of Zone objects containing the point.
        """
        inside_zones = []
        pt = (float(point[0]), float(point[1]))
        for zone in self.zones:
            # pointPolygonTest returns >= 0 if point is inside or on the contour boundary
            dist = cv2.pointPolygonTest(zone.polygon_np, pt, False)
            if dist >= 0:
                inside_zones.append(zone)
        return inside_zones

    def draw_zones(self, frame: np.ndarray, occupied_zone_ids: List[int]) -> np.ndarray:
        """Draws all danger zones onto the frame.
        
        Args:
            frame: The frame to draw on (OpenCV numpy array).
            occupied_zone_ids: List of zone IDs that are currently occupied.
            
        Returns:
            The processed frame with overlays.
        """
        overlay = frame.copy()
        
        for zone in self.zones:
            is_occupied = zone.zone_id in occupied_zone_ids
            # Dynamic colors: crimson red for occupied, yellow-orange for empty
            color = (0, 0, 220) if is_occupied else (0, 165, 255)
            
            # Fill polygon semi-transparently on the overlay
            cv2.fillPoly(overlay, [zone.polygon_np], color)
            
            # Draw polygon border on the actual frame
            cv2.polylines(frame, [zone.polygon_np], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            
            # Get bounding rect to place label neatly at the top-left of the polygon
            rx, ry, rw, rh = cv2.boundingRect(zone.polygon_np)
            label = f"{zone.zone_name}"
            if is_occupied:
                label += " [OCCUPIED]"
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background container and text
            # Ensure ry is not off-screen
            label_y = max(ry, h + 15)
            cv2.rectangle(frame, (rx, label_y - h - 10), (rx + w + 10, label_y), color, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(frame, label, (rx + 5, label_y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
        # Blend the filled polygon overlay with the original frame (25% opacity)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame
