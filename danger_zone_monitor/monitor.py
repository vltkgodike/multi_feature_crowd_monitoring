import logging
from datetime import datetime
from time import time
from typing import List, Tuple
import cv2
import numpy as np

from danger_zone_monitor.person_tracker import PersonTracker
from danger_zone_monitor.zone_manager import ZoneManager
from danger_zone_monitor.video_recorder import VideoRecorder
from danger_zone_monitor.csv_logger import CSVLogger
from danger_zone_monitor.intrusion_manager import IntrusionManager
from danger_zone_monitor.constants import LOITERING_THRESHOLD_SEC, LOITERING_ALERT_COOLDOWN_SEC

logger = logging.getLogger(__name__)

class DangerZoneMonitor:
    """The main orchestration class for the danger zone monitor system."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.engine",
        zone_file: str = "config/zones.json",
        recordings_dir: str = "recordings",
        snapshots_dir: str = "snapshots",
        log_file: str = "logs/intrusion_log.csv",
        fps: float = 30.0,
        loitering_threshold: float = LOITERING_THRESHOLD_SEC,
        loitering_alert_cooldown: float = LOITERING_ALERT_COOLDOWN_SEC,
        async_inference: bool = True,
        camera_id: str = "camera_0",
    ):
        """Initializes the danger zone monitor.
        
        Args:
            model_path: Path to the YOLOv8 model weights file.
            zone_file: Path to the JSON configuration file containing zone coordinates.
            recordings_dir: Directory where video recordings will be saved.
            snapshots_dir: Directory where snapshots will be saved.
            log_file: Path to the CSV file where logs will be stored.
            fps: Expected frame rate of the video feed (important for video writers and buffers).
            loitering_threshold: Duration in seconds to trigger loitering alert.
            loitering_alert_cooldown: Cooldown between successive loitering alerts.
            async_inference: Run TensorRT inference in a worker thread for lower frame-loop latency.
            camera_id: Identifier for the camera (used for output directory structure).
        """
        logger.info("Initializing DangerZoneMonitor package...")
        # Camera-wide loitering tracking
        self.person_first_seen = {}
        self.person_loiter_saved = set()
        self.person_loiter_recorders = {}
        self.person_loiter_event_ids = {}
        self.person_loiter_video_paths = {}
        self.person_loiter_snapshot_paths = {}
        
        # 0. Initialize database connection and tables
        try:
            import postgres_db
            postgres_db.init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
        
        # 1. Initialize core managers and components
        self.zone_manager = ZoneManager(zone_file=zone_file)
        self.tracker = PersonTracker(
            model_path=model_path,
            async_inference=async_inference
        )
        self.video_recorder = VideoRecorder(
            recordings_dir=recordings_dir,
            snapshots_dir=snapshots_dir,
            camera_id=camera_id
        )
        self.csv_logger = CSVLogger(log_path="logs/danger_zone/intrusion_log.csv")
        self.loitering_csv_logger = CSVLogger(log_path="logs/loitering/loitering_log.csv")
        
        # 2. Initialize the intrusion state machine manager
        self.intrusion_manager = IntrusionManager(
            video_recorder=self.video_recorder,
            csv_logger=self.csv_logger,
            loitering_csv_logger=self.loitering_csv_logger,
            fps=fps,
            loitering_threshold=loitering_threshold,
            loitering_alert_cooldown=loitering_alert_cooldown
        )
        
        logger.info("DangerZoneMonitor initialization complete.")

    def close(self) -> None:
        if hasattr(self, "tracker"):
            self.tracker.close()

    def process_frame(self, frame: np.ndarray, counter=None) -> np.ndarray:
        """Processes a single frame: tracks people, checks zones, manages intrusions,
        records videos/snapshots, and overlays a premium visualization dashboard.
        
        Args:
            frame: Input video frame (OpenCV numpy array in BGR format).
            counter: Optional line counter (SingleLineCounter).
            
        Returns:
            The processed frame with UI overlays.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to process_frame.")
            return frame

        # Create a copy to prevent mutating the original frame passed by the caller
        out_frame = frame.copy()
        
        # 1. Track people in the frame
        tracked_people = self.tracker.track(out_frame)
        current_time = time()

        # Camera-wide loitering detection (independent of zones)
        for person in tracked_people:

            track_id = person.track_id

            if track_id not in self.person_first_seen:
                self.person_first_seen[track_id] = current_time

            duration = current_time - self.person_first_seen[track_id]

            # Save once when visible for 10+ seconds
            if duration >= 10 and track_id not in self.person_loiter_saved:

                try:
                    event_id = self.video_recorder.get_next_event_id()
                    # Try to create intrusion event in the database
                    try:
                        import postgres_db
                        db_event_id = postgres_db.create_default_intrusion_event(
                            zone_id=0,
                            entry_time=datetime.fromtimestamp(self.person_first_seen[track_id])
                        )
                        if db_event_id is not None:
                            event_id = db_event_id
                    except Exception as db_err:
                        logger.error(f"Failed to create database intrusion event: {db_err}")

                    self.person_loiter_event_ids[track_id] = event_id
                    person_bbox = tuple(map(int, person.bbox))

                    snapshot_path = self.video_recorder.save_snapshot(
                        frame=out_frame,
                        event_id=event_id,
                        suffix=f"_loiter_ns",
                        subdir="loitering",
                        track_id=track_id,
                        bbox=person_bbox
                    )
                    self.person_loiter_snapshot_paths[track_id] = snapshot_path

                    writer, video_path = self.video_recorder.start_recording(
                        event_id=event_id,
                        fps=self.intrusion_manager.fps,
                        frame_size=(out_frame.shape[1], out_frame.shape[0]),
                        subdir="loitering"
                    )
                    self.person_loiter_recorders[track_id] = writer
                    self.person_loiter_video_paths[track_id] = [video_path]

                    logger.warning(
                        f"LOITERING DETECTED: ID={track_id} visible for {duration:.1f}s"
                    )

                    # Log loitering alert to database
                    try:
                        import postgres_db
                        alert_id = postgres_db.create_default_loitering_alert(
                            event_id=event_id,
                            alert_time=datetime.now()
                        )
                        if alert_id is not None:
                            postgres_db.update_loitering_alert(
                                alert_id=alert_id,
                                event_id=event_id,
                                dwell_time_seconds=duration,
                                snapshot_path=snapshot_path,
                                alert_time=datetime.now()
                            )
                    except Exception as db_err:
                        logger.error(f"Failed to log loitering alert in database: {db_err}")

                    self.person_loiter_saved.add(track_id)

                except Exception as e:
                    logger.error(f"Loitering snapshot failed: {e}")

        # 2. Update intrusion state machine
        occupied_zones = self.intrusion_manager.update_with_frame(
            tracked_people=tracked_people,
            zone_manager=self.zone_manager,
            frame=out_frame
        )

        # Write loitering clip frames for any active recorded people
        for track_id, writer in list(self.person_loiter_recorders.items()):
            try:
                self.video_recorder.write_frame(writer, out_frame)
            except Exception as e:
                logger.error(f"Failed to write loitering frame for ID={track_id}: {e}")

        # Cleanup IDs that disappeared
        active_ids = {p.track_id for p in tracked_people}

        for track_id in list(self.person_first_seen.keys()):
            if track_id not in active_ids:
                entry_time = self.person_first_seen.pop(track_id, None)
                was_saved = track_id in self.person_loiter_saved
                self.person_loiter_saved.discard(track_id)
                
                # Stop recording
                writer = self.person_loiter_recorders.pop(track_id, None)
                if writer is not None:
                    try:
                        self.video_recorder.stop_recording(writer)
                    except Exception as e:
                        logger.error(f"Error stopping loitering recording: {e}")
                
                # Finalize in database if it was a saved loitering event
                if was_saved:
                    event_id = self.person_loiter_event_ids.pop(track_id, None)
                    video_paths = self.person_loiter_video_paths.pop(track_id, [])
                    snapshot_path = self.person_loiter_snapshot_paths.pop(track_id, None)
                    
                    exit_time = datetime.now()
                    duration = (exit_time - datetime.fromtimestamp(entry_time)).total_seconds() if entry_time else 0.0
                    
                    if event_id is not None:
                        try:
                            import postgres_db
                            postgres_db.update_intrusion_event(
                                event_id=event_id,
                                person_id=track_id,
                                zone_id=0,
                                duration_seconds=duration,
                                video_path=";".join(video_paths),
                                snapshot_path=snapshot_path,
                                is_loitering=True,
                                exit_time=exit_time
                            )
                            logger.info(f"Finalized camera-wide loitering event {event_id} in database (duration: {duration:.2f}s).")
                        except Exception as e:
                            logger.error(f"Failed to finalize loitering event {event_id} in database: {e}")

        # 3. Draw monitored danger zones (filled polygons + borders)
        out_frame = self.zone_manager.draw_zones(out_frame, occupied_zones)
        
        # 4. Draw bounding boxes, IDs, and indicators for tracked persons
        for person in tracked_people:
            x1, y1, x2, y2 = map(int, person.bbox)
            cx, cy = person.center
            if counter is not None:
                counter.update(person.track_id, person.center)
            # Determine color state of the person based on active intrusion mappings
            is_in_zone = False
            is_confirmed = False
            is_loitering = False
            loiter_duration = 0.0
            for zone in self.zone_manager.zones:
                key = (person.track_id, zone.zone_id)
                if key in self.intrusion_manager.active_states:
                    is_in_zone = True
                    state = self.intrusion_manager.active_states[key]
                    if state.is_confirmed:
                        is_confirmed = True
                    if getattr(state, "is_loitering", False):
                        is_loitering = True
                        loiter_duration = state.duration

            # Camera-wide loitering detection independent of zone membership
            camera_loiter_duration = 0.0
            if person.track_id in self.person_first_seen:
                camera_loiter_duration = current_time - self.person_first_seen[person.track_id]
                if camera_loiter_duration >= self.intrusion_manager.loitering_threshold:
                    is_loitering = True
                    loiter_duration = camera_loiter_duration

            # Cyan for outside, orange for entering, red for confirmed intrusion/loitering
            box_thickness = 2
            if is_loitering:
                color = (0, 0, 255)
                label_prefix = f"LOITERING - {int(loiter_duration)}s"
                box_thickness = 3
            elif is_confirmed:
                color = (0, 0, 255)
                label_prefix = "WARNING - REC"
            elif is_in_zone:
                color = (0, 165, 255)
                label_prefix = "INTRUDER"
            else:
                color = (255, 255, 0)
                label_prefix = "PERSON"
                
            # Draw bounding box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
            
            # Draw center point used for danger-zone checks
            cv2.circle(out_frame, (cx, cy), 6, (255, 0, 255), -1, cv2.LINE_AA)
            
            # Prepare tag label
            label = f"{label_prefix} ID:{person.track_id} ({person.confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Keep label within screen boundary
            label_y = max(y1, h + 10)
            cv2.rectangle(out_frame, (x1, label_y - h - 6), (x1 + w + 10, label_y), color, cv2.FILLED, cv2.LINE_AA)
            
            # Text color matches background contrast
            text_color = (255, 255, 255) if color != (255, 255, 0) else (0, 0, 0)
            cv2.putText(out_frame, label, (x1 + 5, label_y - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Draw "!!! LOITERING ALERT !!!" above the tag label
            if is_loitering:
                alert_text = "!!! LOITERING ALERT !!!"
                (aw, ah), _ = cv2.getTextSize(alert_text, font, font_scale * 1.5, 2)
                alert_y = max(label_y - h - 12, ah + 5)
                cv2.putText(out_frame, alert_text, (x1, alert_y), font, font_scale * 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw the line counter overlay if provided
        if counter is not None:
            counter.draw(out_frame)

        # 5. Draw System HUD / Status Dashboard
        # Check if any recording is active
        any_recording = any(state.is_confirmed for state in self.intrusion_manager.active_states.values())
        
        # Blinking REC dot on top left
        if any_recording:
            if int(time()) % 2 == 0:
                cv2.circle(out_frame, (35, 35), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(out_frame, "REC ACTIVE", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # HUD overlay at top right
        h_frame, w_frame = out_frame.shape[:2]
        board_x = w_frame - 280
        board_y = 20
        
        # Get active confirmed intrusion events
        active_confirmed = [s for s in self.intrusion_manager.active_states.values() if s.is_confirmed]
        
        # Draw translucent dashboard box
        board_w, board_h = 260, 40 + max(1, len(active_confirmed)) * 20
        board_overlay = out_frame.copy()
        cv2.rectangle(board_overlay, (board_x, board_y), (board_x + board_w, board_y + board_h), (0, 0, 0), -1)
        cv2.addWeighted(board_overlay, 0.5, out_frame, 0.5, 0, out_frame)
        
        # Draw border
        cv2.rectangle(out_frame, (board_x, board_y), (board_x + board_w, board_y + board_h), (128, 128, 128), 1, cv2.LINE_AA)
        
        # Header text
        cv2.putText(out_frame, "DANGER MONITOR STATUS", (board_x + 10, board_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset = board_y + 40
        if not active_confirmed:
            cv2.putText(out_frame, "No Active Intrusions", (board_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            for state in active_confirmed:
                duration = (datetime.now() - state.entry_time).total_seconds()
                is_loit = getattr(state, "is_loitering", False)
                status_lbl = "LOIT" if is_loit else "INTR"
                event_text = f"Evt:{state.event_id:04d} | ID:{state.track_id} | {status_lbl} | {duration:.1f}s"
                hud_color = (255, 0, 255) if is_loit else (0, 0, 255)
                cv2.putText(out_frame, event_text, (board_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1, cv2.LINE_AA)
                y_offset += 20
                
        return out_frame
