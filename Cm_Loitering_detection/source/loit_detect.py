import cv2
import time
import numpy as np
from collections import defaultdict


class LoiteringDetector:

    def __init__(
        self,
        polygon_points,
        loitering_threshold=20,
        alert_cooldown=10
    ):

        self.zone = np.array(polygon_points, np.int32)
        self.loitering_threshold = loitering_threshold
        self.alert_cooldown = alert_cooldown

        self.entry_times = {}
        self.last_alert_time = defaultdict(float)
        
        print(f"[DETECTOR] Zone initialized with points: {polygon_points}")
        print(f"[DETECTOR] Loitering threshold: {loitering_threshold} seconds")

    def is_inside_zone(self, point):
        """Check if point is inside the zone polygon"""
        result = cv2.pointPolygonTest(
            self.zone,
            point,
            False
        )
        return result >= 0

    def process_tracks(self, frame, tracks):

        current_time = time.time()
        alerts = []

        # Draw ROI Polygon
        cv2.polylines(
            frame,
            [self.zone],
            True,
            (0, 0, 255),
            2
        )

        for track in tracks:

            track_id = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]

            # Calculate center point (middle of bbox, bottom for better zone detection)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)  # Use middle instead of bottom

            point = (center_x, center_y)

            # Check if inside zone
            inside = self.is_inside_zone(point)

            color = (0, 255, 0)  # Default green (outside)

            if inside:
                color = (0, 0, 255)  # Red (inside zone)

                # Track entry time
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = current_time
                    print(f"[ZONE] Person {track_id} ENTERED zone at {time.strftime('%H:%M:%S')}")

                # Calculate dwell time
                dwell_time = current_time - self.entry_times[track_id]

                # Display dwell time
                cv2.putText(
                    frame,
                    f"In Zone: {int(dwell_time)}s",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # DEBUG: Draw test point to see if detection is correct
                cv2.circle(frame, point, 5, (255, 0, 0), -1)

                # Check for loitering (>= threshold)
                print(f"[CHECK] ID {track_id}: {int(dwell_time)}s / {self.loitering_threshold}s threshold")

                if dwell_time >= self.loitering_threshold:

                    # Cooldown check
                    if (current_time - self.last_alert_time[track_id]) > self.alert_cooldown:

                        alert = {
                            "track_id": track_id,
                            "dwell_time": int(dwell_time)
                        }

                        alerts.append(alert)
                        self.last_alert_time[track_id] = current_time

                        alert_message = (
                            f"[ALERT] "
                            f"ID {track_id} "
                            f"LOITERING "
                            f"{int(dwell_time)} sec"
                        )
                        print(alert_message)

                        # Draw alert on frame
                        cv2.putText(
                            frame,
                            "!!! LOITERING ALERT !!!",
                            (x1, y1 - 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3
                        )

                        # Draw thick red rectangle for alert
                        cv2.rectangle(
                            frame,
                            (x1 - 2, y1 - 2),
                            (x2 + 2, y2 + 2),
                            (0, 0, 255),
                            4
                        )

            else:
                # Person left zone
                if track_id in self.entry_times:
                    exit_time = current_time - self.entry_times[track_id]
                    print(f"[ZONE] Person {track_id} LEFT zone after {int(exit_time)}s")
                    del self.entry_times[track_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Draw track ID
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return frame, alerts