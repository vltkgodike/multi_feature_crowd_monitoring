import cv2
import json
import os
import math


class SingleLineCounter:

    def __init__(
        self,
        json_file="line.json",
        default_line=((100, 250), (550, 250)),
        log_file="logs/line_crossing/crossing_log.csv"
    ):

        self.json_file = json_file
        self.log_file = log_file

        self.line = self._load_line(default_line)

        self.previous_positions = {}

        self.in_count = 0
        self.out_count = 0

        self.selected_point = None
        
        self._init_log_file()

    def _init_log_file(self):
        import csv
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CrossingID", "PersonID", "Direction", "CrossingTime"])
            except Exception as e:
                print(f"[ERROR] Failed to initialize crossing log: {e}")

    # =====================================
    # SAVE / LOAD
    # =====================================

    def _save_line(self):

        data = {
            "line": [
                list(self.line[0]),
                list(self.line[1])
            ]
        }

        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=4)

    def _load_line(self, default_line):

        if os.path.exists(self.json_file):

            with open(self.json_file, "r") as f:
                data = json.load(f)

            return tuple(map(tuple, data["line"]))

        return default_line

    # =====================================
    # MATH
    # =====================================

    @staticmethod
    def side_of_line(point, line):

        (x1, y1), (x2, y2) = line
        px, py = point

        return (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

    # =====================================
    # COUNTING
    # =====================================

    def update(
        self,
        track_id,
        center
    ):

        cx, cy = center

        if track_id not in self.previous_positions:

            self.previous_positions[track_id] = center
            return

        prev_x, prev_y = self.previous_positions[track_id]

        prev_side = self.side_of_line(
            (prev_x, prev_y),
            self.line
        )

        curr_side = self.side_of_line(
            (cx, cy),
            self.line
        )

        if prev_side * curr_side < 0:

            if prev_side > 0:
                self.out_count += 1
                direction = "OUT"
            else:
                self.in_count += 1
                direction = "IN"
                
            self._log_crossing(track_id, direction)

        self.previous_positions[track_id] = center

    def _log_crossing(self, track_id, direction):
        from datetime import datetime
        import csv
        
        crossing_id = 1
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    crossing_id = sum(1 for _ in f)
            except Exception:
                pass
                
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    crossing_id,
                    track_id,
                    direction,
                    datetime.now().isoformat()
                ])
        except Exception as e:
            print(f"[ERROR] Failed to log crossing: {e}")
            
        try:
            import postgres_db
            db_crossing_id = postgres_db.create_default_line_crossing()
            if db_crossing_id is not None:
                postgres_db.update_line_crossing(
                    crossing_id=db_crossing_id,
                    person_id=track_id,
                    direction=direction
                )
        except Exception as e:
            print(f"[ERROR] Failed to push line crossing to database: {e}")

    # =====================================
    # DRAW
    # =====================================

    def draw_line(
        self,
        frame,
        color=(255, 255, 0)
    ):

        cv2.line(
            frame,
            self.line[0],
            self.line[1],
            color,
            2,
            cv2.LINE_AA
        )

    def draw_handles(self, frame):

        for pt in self.line:

            cv2.circle(
                frame,
                pt,
                6,
                (235, 145, 30),
                -1,
                cv2.LINE_AA
            )

    def draw_arrow(
        self,
        frame,
        text="IN",
        arrow_length=40
    ):

        (x1, y1), (x2, y2) = self.line

        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length == 0:
            return

        px = -dy / length
        py = dx / length

        start = (mx, my)

        end = (
            int(mx + px * arrow_length),
            int(my + py * arrow_length)
        )

        cv2.arrowedLine(
            frame,
            start,
            end,
            (113, 204, 46),
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.3
        )

        cv2.putText(
            frame,
            text,
            (
                end[0] - 10,
                end[1] - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (113, 204, 46),
            1,
            cv2.LINE_AA
        )

    def draw_counts(self, frame):

        panel_x = 20
        panel_y = 60
        panel_w = 180
        panel_h = 70

        # Translucent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (128, 128, 128), 1, cv2.LINE_AA)

        # Header
        cv2.putText(frame, "PEOPLE COUNTER", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # IN count
        cv2.putText(frame, f"IN : {self.in_count}", (panel_x + 10, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (113, 204, 46), 1, cv2.LINE_AA)

        # OUT count
        cv2.putText(frame, f"OUT: {self.out_count}", (panel_x + 10, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 76, 231), 1, cv2.LINE_AA)

    def draw(self, frame):

        self.draw_line(frame)
        self.draw_arrow(frame)
        self.draw_handles(frame)
        self.draw_counts(frame)

    # =====================================
    # MOUSE
    # =====================================

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            for idx, pt in enumerate(self.line):

                if abs(x - pt[0]) < 15 and abs(y - pt[1]) < 15:

                    self.selected_point = idx

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.selected_point is not None:

                temp = list(self.line)

                temp[self.selected_point] = (
                    x,
                    y
                )

                self.line = tuple(temp)

        elif event == cv2.EVENT_LBUTTONUP:

            self.selected_point = None

            self._save_line()