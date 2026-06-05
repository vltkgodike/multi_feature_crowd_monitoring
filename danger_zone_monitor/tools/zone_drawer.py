import argparse
import json
import logging
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)

WINDOW_NAME = "Zone Drawer Tool"
HANDLE_RADIUS = 10


class ZoneDrawer:
    """Interactive live-feed polygon editor for danger zone configuration."""

    def __init__(self, source=None, config_path: str = "config/zones.json", open_source: bool = True, initial_frame=None):
        self.source = source
        self.config_path = config_path
        self.cap = None
        self.current_frame = initial_frame.copy() if initial_frame is not None else None
        self.existing_zones = []
        self.points = []
        self.selected_zone_id = None
        self.drag_mode = None
        self.drag_zone_index = None
        self.drag_point_index = None
        self.last_mouse_pos = None
        self.config_changed = False

        if open_source:
            self._open_source()
        self._load_zones()

    def _open_source(self) -> None:
        logger.info(f"Opening video/image source: {self.source}")

        if (
            isinstance(self.source, str)
            and os.path.exists(self.source)
            and self.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ):
            self.current_frame = cv2.imread(self.source)
            return

        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame

        if self.current_frame is None:
            logger.warning("Could not read frame from source. Creating a blank 640x480 canvas.")
            self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50

    def _load_zones(self) -> None:
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r") as f:
                content = json.load(f)
            self.existing_zones = content.get("zones", [])
            for zone in self.existing_zones:
                zone["points"] = [self._clamp_point(tuple(point)) for point in zone.get("points", [])]
            logger.info(f"Loaded {len(self.existing_zones)} existing zones from '{self.config_path}'.")
        except Exception as e:
            logger.warning(f"Could not read existing config at startup. Error: {e}")

    def update_frame(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            return

        if not isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_frame(self, frame) -> None:
        self.current_frame = frame

    def render(self):
        if self.current_frame is None:
            self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50
        frame = self.current_frame.copy()
        self.draw_on_frame(frame)
        return frame

    def draw_on_frame(self, frame) -> None:
        self.current_frame = frame
        self._draw_existing_zones(frame)
        self._draw_current_polygon(frame)
        self._draw_instructions(frame)

    def mouse_callback(self, event, x, y, flags, param) -> None:
        self.handle_mouse_event(event, x, y, flags, param)

    def handle_mouse_event(self, event, x, y, flags=None, param=None) -> bool:
        point = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_vertex = self._find_drawing_vertex(point)
            if drawing_vertex is not None:
                self.drag_mode = "drawing_vertex"
                self.drag_point_index = drawing_vertex
                self.last_mouse_pos = point
                return True

            zone_vertex = self._find_zone_vertex(point)
            if zone_vertex is not None:
                self.drag_mode = "zone_vertex"
                self.drag_zone_index, self.drag_point_index = zone_vertex
                self.selected_zone_id = self.existing_zones[self.drag_zone_index].get("zone_id")
                self.last_mouse_pos = point
                return True

            zone_index = self._find_zone(point)
            if zone_index is not None:
                self.drag_mode = "zone"
                self.drag_zone_index = zone_index
                self.selected_zone_id = self.existing_zones[zone_index].get("zone_id")
                self.last_mouse_pos = point
                return True

            self.points.append(self._clamp_point(point))
            self.selected_zone_id = None
            logger.info(f"Added point: ({x}, {y})")
            return True

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_mode:
            self._handle_drag(point)
            return True

        elif event == cv2.EVENT_LBUTTONUP:
            changed_existing_zone = self.drag_mode in {"zone", "zone_vertex"}
            self.drag_mode = None
            self.drag_zone_index = None
            self.drag_point_index = None
            self.last_mouse_pos = None
            if changed_existing_zone:
                self._save_config()
                self._upsert_selected_zone()
            return changed_existing_zone

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                logger.info(f"Removed point: {removed}")
                return True

            zone_index = self._find_zone(point)
            if zone_index is not None:
                self.selected_zone_id = self.existing_zones[zone_index].get("zone_id")
                logger.info(f"Selected zone ID {self.selected_zone_id}. Press D or Delete to remove it.")
                return True

        return False

    def consume_config_changed(self) -> bool:
        changed = self.config_changed
        self.config_changed = False
        return changed

    def save_current_zone(self) -> None:
        if len(self.points) < 3:
            logger.warning("A polygon must have at least 3 points. Click more points before saving.")
            return

        print("\n" + "=" * 40)
        zone_name = input("Enter a name for this danger zone [DangerZone]: ").strip()
        if not zone_name:
            zone_name = "DangerZone"

        max_id = max((zone.get("zone_id", 0) for zone in self.existing_zones), default=0)
        new_id = max_id + 1
        new_pts = [list(point) for point in self.points]
        new_zone = {
            "zone_id": new_id,
            "zone_name": zone_name,
            "points": new_pts,
        }

        self.existing_zones.append(new_zone)
        self.selected_zone_id = new_id

        try:
            self._save_config()
            self._upsert_zone(new_zone)
            logger.info(f"Successfully saved zone '{zone_name}' (ID: {new_id}) to '{self.config_path}'.")
            print(f"Zone '{zone_name}' saved! Coordinates written to '{self.config_path}'.")
            print("=" * 40 + "\n")
            self.points = []
        except Exception as e:
            logger.error(f"Failed to write configuration: {e}")

    def delete_selected_zone(self) -> None:
        if self.selected_zone_id is None:
            logger.warning("No zone selected. Click inside a zone, then press D or Delete.")
            return

        zone_to_delete = next(
            (zone for zone in self.existing_zones if zone.get("zone_id") == self.selected_zone_id),
            None,
        )
        if zone_to_delete is None:
            logger.warning(f"Selected zone ID {self.selected_zone_id} was not found.")
            self.selected_zone_id = None
            return

        self.existing_zones = [
            zone for zone in self.existing_zones if zone.get("zone_id") != self.selected_zone_id
        ]

        try:
            self._save_config()
            try:
                import postgres_db

                postgres_db.delete_zone(self.selected_zone_id)
            except Exception as db_err:
                logger.warning(f"Could not delete zone ID {self.selected_zone_id} from database: {db_err}")

            logger.info(
                f"Deleted zone '{zone_to_delete.get('zone_name')}' "
                f"(ID: {self.selected_zone_id}) from '{self.config_path}'."
            )
            self.selected_zone_id = None
        except Exception as e:
            logger.error(f"Failed to write configuration: {e}")

    def clear_current_polygon(self) -> None:
        self.points = []
        logger.info("Cleared unsaved polygon points.")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def _handle_drag(self, point: Tuple[int, int]) -> None:
        point = self._clamp_point(point)

        if self.drag_mode == "drawing_vertex":
            self.points[self.drag_point_index] = point
            return

        if self.drag_mode == "zone_vertex":
            zone = self.existing_zones[self.drag_zone_index]
            zone["points"][self.drag_point_index] = list(point)
            return

        if self.drag_mode == "zone":
            dx = point[0] - self.last_mouse_pos[0]
            dy = point[1] - self.last_mouse_pos[1]
            zone = self.existing_zones[self.drag_zone_index]
            dx, dy = self._bounded_translation(zone.get("points", []), dx, dy)
            moved_points = []
            for zone_point in zone.get("points", []):
                moved_points.append([int(zone_point[0] + dx), int(zone_point[1] + dy)])
            zone["points"] = moved_points
            self.last_mouse_pos = point

    def _find_drawing_vertex(self, point: Tuple[int, int]) -> Optional[int]:
        for index, vertex in enumerate(self.points):
            if self._distance(point, vertex) <= HANDLE_RADIUS:
                return index
        return None

    def _find_zone_vertex(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        for zone_index in range(len(self.existing_zones) - 1, -1, -1):
            zone = self.existing_zones[zone_index]
            for point_index, vertex in enumerate(zone.get("points", [])):
                if self._distance(point, tuple(vertex)) <= HANDLE_RADIUS:
                    return zone_index, point_index
        return None

    def _find_zone(self, point: Tuple[int, int]) -> Optional[int]:
        for zone_index in range(len(self.existing_zones) - 1, -1, -1):
            zone = self.existing_zones[zone_index]
            zone_points = zone.get("points", [])
            if len(zone_points) < 3:
                continue
            pts_np = np.array(zone_points, dtype=np.int32)
            if cv2.pointPolygonTest(pts_np, (float(point[0]), float(point[1])), False) >= 0:
                return zone_index
        return None

    def _draw_existing_zones(self, frame) -> None:
        overlay = frame.copy()

        for zone in self.existing_zones:
            zone_pts = zone.get("points", [])
            if len(zone_pts) < 3:
                continue

            pts_np = np.array(zone_pts, dtype=np.int32)
            is_selected = zone.get("zone_id") == self.selected_zone_id
            color = (0, 255, 0) if is_selected else (0, 255, 255)

            cv2.fillPoly(overlay, [pts_np], color)
            cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=3 if is_selected else 2, lineType=cv2.LINE_AA)

            for vertex in zone_pts:
                cv2.circle(frame, tuple(vertex), 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(vertex), HANDLE_RADIUS, color, 1, cv2.LINE_AA)

            rx, ry, _, _ = cv2.boundingRect(pts_np)
            label = f"ID {zone.get('zone_id')}: {zone.get('zone_name')}"
            if is_selected:
                label += " [SELECTED]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(ry, label_h + 10)
            cv2.rectangle(frame, (rx, label_y - label_h - 6), (rx + label_w + 10, label_y), color, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(frame, label, (rx + 5, label_y - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    def _draw_current_polygon(self, frame) -> None:
        for index, point in enumerate(self.points):
            cv2.circle(frame, point, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, HANDLE_RADIUS, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(index + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            if index > 0:
                cv2.line(frame, self.points[index - 1], point, (0, 255, 0), 2, cv2.LINE_AA)

        if len(self.points) >= 3:
            cv2.line(frame, self.points[-1], self.points[0], (255, 0, 0), 1, cv2.LINE_AA)

    def _draw_instructions(self, frame) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 58), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        line_1 = "Left: add/select | Drag handles: resize | Drag inside zone: move | Right: undo/select"
        line_2 = "S: save new zone | D/Delete: delete selected | C: clear new points | Q/Esc: exit"
        cv2.putText(frame, line_1, (15, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line_2, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _save_config(self) -> None:
        config_dir = os.path.dirname(self.config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump({"zones": self.existing_zones}, f, indent=2)
        self.config_changed = True

    def _upsert_selected_zone(self) -> None:
        zone = next(
            (item for item in self.existing_zones if item.get("zone_id") == self.selected_zone_id),
            None,
        )
        if zone is not None:
            self._upsert_zone(zone)

    def _upsert_zone(self, zone) -> None:
        try:
            import postgres_db

            postgres_db.upsert_zone(
                zone_id=zone.get("zone_id"),
                name=zone.get("zone_name"),
                points=zone.get("points", []),
            )
        except Exception as db_err:
            logger.warning(f"Could not upsert zone ID {zone.get('zone_id')} to database: {db_err}")

    def _clamp_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        if self.current_frame is None:
            return int(point[0]), int(point[1])

        h, w = self.current_frame.shape[:2]
        x = min(max(int(point[0]), 0), w - 1)
        y = min(max(int(point[1]), 0), h - 1)
        return x, y

    def _bounded_translation(self, points, dx: int, dy: int) -> Tuple[int, int]:
        if not points:
            return 0, 0

        h, w = self.current_frame.shape[:2]
        xs = [int(point[0]) for point in points]
        ys = [int(point[1]) for point in points]
        dx = min(max(dx, -min(xs)), (w - 1) - max(xs))
        dy = min(max(dy, -min(ys)), (h - 1) - max(ys))
        return dx, dy

    @staticmethod
    def _distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Interactive Danger Zone Drawing Tool")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or path to video file/image")
    parser.add_argument("--config", type=str, default="config/zones.json", help="Path to save the JSON coordinates")
    parser.add_argument("--display", action="store_true", help="Open the interactive live-feed zone editor")
    args = parser.parse_args()

    if not args.display:
        logger.info("Headless mode: no zone drawer window opened. Pass --display to edit zones in the UI.")
        return

    source = int(args.source) if args.source.isdigit() else args.source
    drawer = ZoneDrawer(source=source, config_path=args.config)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, drawer.mouse_callback)

    logger.info("Interactions:")
    logger.info("  - LEFT click empty space to add vertices for a new zone")
    logger.info("  - LEFT click inside a saved zone to select and drag it")
    logger.info("  - Drag a vertex handle to resize an unsaved or saved zone")
    logger.info("  - RIGHT click to undo the current unsaved point, or select a saved zone")
    logger.info("  - Press 'S' to save the current new zone")
    logger.info("  - Press 'D', Delete, or Backspace to delete the selected saved zone")
    logger.info("  - Press 'C' to clear unsaved points")
    logger.info("  - Press 'Q' or Esc to exit")

    try:
        while True:
            drawer.update_frame()
            cv2.imshow(WINDOW_NAME, drawer.render())
            key = cv2.waitKey(20) & 0xFF

            if key in [ord("q"), 27]:
                logger.info("Exiting drawer tool.")
                break

            if key in [ord("s"), ord("S")]:
                drawer.save_current_zone()
            elif key in [ord("d"), ord("D"), 8, 127]:
                drawer.delete_selected_zone()
            elif key in [ord("c"), ord("C")]:
                drawer.clear_current_polygon()
    finally:
        drawer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
