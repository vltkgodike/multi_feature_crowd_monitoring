from collections import deque
from datetime import datetime
import logging
import os
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np

from danger_zone_monitor.models import TrackedPerson, Zone
from danger_zone_monitor.constants import (
    MAX_MISSING_FRAMES,
    CONFIRMATION_THRESHOLD_SEC,
    PRE_RECORD_SECONDS,
    RECORDING_SEGMENT_SECONDS,
    LOITERING_THRESHOLD_SEC,
    LOITERING_ALERT_COOLDOWN_SEC
)

logger = logging.getLogger(__name__)

class PersonZoneState:
    """Represents the temporal intrusion state of a single tracked person inside a single zone."""

    def __init__(self, track_id: int, zone_id: int, zone_name: str, max_buffer_size: int):
        self.track_id: int = track_id
        self.zone_id: int = zone_id
        self.zone_name: str = zone_name
        self.entry_time: datetime = datetime.now()
        self.exit_time: Optional[datetime] = None
        self.duration: float = 0.0
        self.is_confirmed: bool = False
        
        # Track disappearance tolerance
        self.missing_frames: int = 0
        
        # Pre-recording frame buffer
        self.frame_buffer: deque = deque(maxlen=max_buffer_size)
        
        # Active recording resources
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_path: Optional[str] = None
        self.video_paths: List[str] = []
        self.snapshot_path: Optional[str] = None
        self.event_id: Optional[int] = None
        self.segment_index: int = 1
        self.segment_frame_count: int = 0
        
        # Loitering states
        self.is_loitering: bool = False
        self.last_loitering_alert_time: Optional[datetime] = None

class IntrusionManager:
    """Manages the state transitions and rule checks for zone intrusions."""

    def __init__(
        self,
        video_recorder,
        csv_logger,
        loitering_csv_logger,
        fps: float = 30.0,
        loitering_threshold: float = LOITERING_THRESHOLD_SEC,
        loitering_alert_cooldown: float = LOITERING_ALERT_COOLDOWN_SEC
    ):
        """Initializes the IntrusionManager.
        
        Args:
            video_recorder: Instance of VideoRecorder.
            csv_logger: Instance of CSVLogger for danger zone intrusions.
            loitering_csv_logger: Instance of CSVLogger for loitering alerts.
            fps: Expected frames per second of the video feed.
            loitering_threshold: Duration in seconds to confirm loitering.
            loitering_alert_cooldown: Cooldown between successive loitering alerts.
        """
        self.video_recorder = video_recorder
        self.csv_logger = csv_logger
        self.loitering_csv_logger = loitering_csv_logger
        self.fps = fps
        self.loitering_threshold = loitering_threshold
        self.loitering_alert_cooldown = loitering_alert_cooldown
        self.active_states: Dict[Tuple[int, int], PersonZoneState] = {}
        self.next_event_id: int = self.video_recorder.get_next_event_id()
        
        # Max buffer size for 3-second pre-recording
        self.max_buffer_size: int = int(PRE_RECORD_SECONDS * self.fps)
        self.max_segment_frames: int = max(1, int(RECORDING_SEGMENT_SECONDS * self.fps))
        logger.info(
            f"IntrusionManager initialized with {self.fps} FPS, "
            f"buffer size={self.max_buffer_size}, "
            f"segment frames={self.max_segment_frames}, next_event_id={self.next_event_id}"
        )

    def _start_recording_segment(self, state: PersonZoneState, frame_size: Tuple[int, int]) -> None:
        """Starts a recording segment for a confirmed intrusion state."""
        segment_index = None if state.segment_index == 1 else state.segment_index
        subdir = "loitering" if getattr(state, "is_loitering", False) else "danger_zone"
        recording_args = {
            "event_id": state.event_id,
            "fps": self.fps,
            "frame_size": frame_size,
            "subdir": subdir
        }
        if segment_index is not None:
            recording_args["segment_index"] = segment_index

        writer, video_path = self.video_recorder.start_recording(**recording_args)
        state.video_writer = writer
        state.video_path = video_path
        state.video_paths.append(video_path)
        state.segment_frame_count = 0

    def _write_recording_frame(
        self,
        state: PersonZoneState,
        frame: np.ndarray,
        frame_size: Tuple[int, int]
    ) -> None:
        """Writes a frame, rotating to a new clip after the segment frame limit."""
        if state.video_writer is not None and state.segment_frame_count >= self.max_segment_frames:
            self.video_recorder.stop_recording(state.video_writer)
            state.segment_index += 1
            self._start_recording_segment(state, frame_size)
            logger.info(
                f"Started recording segment {state.segment_index} for Event ID: {state.event_id}"
            )

        self.video_recorder.write_frame(state.video_writer, frame)
        state.segment_frame_count += 1

    def update(self, tracked_people: List[TrackedPerson], zone_manager) -> List[int]:
        """Updates the intrusion states based on current tracked people.
        
        Args:
            tracked_people: List of TrackedPerson objects detected in the current frame.
            zone_manager: ZoneManager instance.
            
        Returns:
            A list of zone IDs that are currently occupied.
        """
        # Dictionary of current frame active (track_id, zone_id) mappings
        active_this_frame = set()
        
        # Process each person and check which zones they are currently in
        for person in tracked_people:
            cx, cy = person.center
            inside_zones = zone_manager.check_point_in_zones((cx, cy))
            
            for zone in inside_zones:
                key = (person.track_id, zone.zone_id)
                active_this_frame.add(key)
                
                # If this is a new intrusion detection
                if key not in self.active_states:
                    state = PersonZoneState(
                        track_id=person.track_id,
                        zone_id=zone.zone_id,
                        zone_name=zone.zone_name,
                        max_buffer_size=self.max_buffer_size
                    )
                    self.active_states[key] = state
                    logger.info(f"Person {person.track_id} entered Zone {zone.zone_name} (ID: {zone.zone_id}).")

        # Get the current frame from the recorder context if available, otherwise we use standard frame
        # Wait, the manager needs the current frame to buffer it and write it.
        # We will require passing the frame to this update method.
        # Let's adjust the signature to: update(tracked_people, zone_manager, frame)
        return []

    def update_with_frame(self, tracked_people: List[TrackedPerson], zone_manager, frame: np.ndarray) -> List[int]:
        """Updates states and processes frame buffers/recording.
        
        Args:
            tracked_people: Current tracked people in frame.
            zone_manager: ZoneManager instance.
            frame: Current video frame.
            
        Returns:
            A list of zone IDs currently occupied.
        """
        active_this_frame = set()
        height, width = frame.shape[:2]
        
        # 1. Update detections
        for person in tracked_people:
            cx, cy = person.center
            inside_zones = zone_manager.check_point_in_zones((cx, cy))
            
            for zone in inside_zones:
                key = (person.track_id, zone.zone_id)
                active_this_frame.add(key)
                
                if key not in self.active_states:
                    state = PersonZoneState(
                        track_id=person.track_id,
                        zone_id=zone.zone_id,
                        zone_name=zone.zone_name,
                        max_buffer_size=self.max_buffer_size
                    )
                    self.active_states[key] = state
                    logger.info(f"Person {person.track_id} entered {zone.zone_name} (ID: {zone.zone_id}).")
                
                state = self.active_states[key]
                state.missing_frames = 0  # Reset missing frame count
                
                if not state.is_confirmed:
                    # Append copy of the frame to the pre-record buffer
                    state.frame_buffer.append(frame.copy())
                    state.duration = (datetime.now() - state.entry_time).total_seconds()
                    
                    # Check if duration meets threshold to confirm intrusion
                    if state.duration >= CONFIRMATION_THRESHOLD_SEC:
                        state.is_confirmed = True
                        try:
                            import postgres_db
                            db_event_id = postgres_db.create_default_intrusion_event(
                                state.zone_id,
                                entry_time=state.entry_time
                            )
                            if db_event_id is not None:
                                state.event_id = db_event_id
                            else:
                                state.event_id = self.next_event_id
                                self.next_event_id += 1
                        except Exception as e:
                            logger.error(f"Failed to create intrusion event in database: {e}")
                            state.event_id = self.next_event_id
                            self.next_event_id += 1
                        
                        logger.info(f"Intrusion CONFIRMED for Person {state.track_id} in {state.zone_name}. Event ID: {state.event_id}")
                        
                        # Save confirmation snapshot in danger_zone
                        snapshot_path = self.video_recorder.save_snapshot(frame, state.event_id, subdir="danger_zone")
                        state.snapshot_path = snapshot_path
                        
                        # Start video recording
                        self._start_recording_segment(state, (width, height))
                        
                        # Flush the pre-recording buffer to the VideoWriter
                        logger.info(f"Writing {len(state.frame_buffer)} buffered frames to event video...")
                        while state.frame_buffer:
                            buf_frame = state.frame_buffer.popleft()
                            self._write_recording_frame(state, buf_frame, (width, height))
                else:
                    # Write the current frame directly to video
                    self._write_recording_frame(state, frame, (width, height))
                    
                if state.is_confirmed:
                    # Update duration inside zone
                    state.duration = (datetime.now() - state.entry_time).total_seconds()
                    
                    # Check for loitering threshold
                    if state.duration >= self.loitering_threshold:
                        current_time = datetime.now()
                        if not state.is_loitering:
                            state.is_loitering = True
                            logger.warning(f"[LOITER] Loitering detected for Person {state.track_id} in {state.zone_name} (Duration: {state.duration:.2f}s).")
                        
                        if state.last_loitering_alert_time is None or (current_time - state.last_loitering_alert_time).total_seconds() >= self.loitering_alert_cooldown:
                            state.last_loitering_alert_time = current_time
                            logger.warning(f"🚨 LOITERING ALERT: Person {state.track_id} loitering in {state.zone_name} for {int(state.duration)}s!")
                            # Save loitering snapshot in loitering folder
                            loiter_snapshot_path = self.video_recorder.save_snapshot(frame, state.event_id, suffix=f"_loiter_{int(state.duration)}s", subdir="loitering")
                            try:
                                import postgres_db
                                alert_id = postgres_db.create_default_loitering_alert(
                                    state.event_id,
                                    alert_time=current_time
                                )
                                if alert_id is not None:
                                    postgres_db.update_loitering_alert(
                                        alert_id=alert_id,
                                        event_id=state.event_id,
                                        dwell_time_seconds=state.duration,
                                        snapshot_path=loiter_snapshot_path,
                                        alert_time=current_time
                                    )
                            except Exception as e:
                                logger.error(f"Failed to log loitering alert in database: {e}")

        # 2. Handle missing/exited tracks
        exited_keys = []
        for key, state in self.active_states.items():
            if key not in active_this_frame:
                # If they are currently missing, increment counter
                state.missing_frames += 1
                
                # If still buffering or writing, write the current frame to maintain timing continuity
                if state.is_confirmed and state.video_writer is not None:
                    self._write_recording_frame(state, frame, (width, height))
                elif not state.is_confirmed:
                    state.frame_buffer.append(frame.copy())
                
                # If they exceed the maximum missing frames, trigger final exit
                if state.missing_frames > MAX_MISSING_FRAMES:
                    exited_keys.append(key)

        for key in exited_keys:
            state = self.active_states.pop(key)
            track_id, zone_id = key
            
            # Record exit time and final duration
            state.exit_time = datetime.now()
            state.duration = (state.exit_time - state.entry_time).total_seconds()
            
            if state.is_confirmed:
                # Finalize recording
                self.video_recorder.stop_recording(state.video_writer)
                logger.info(f"Intrusion ended for Person {track_id} in Zone {state.zone_name}. Event ID: {state.event_id} (Duration: {state.duration:.2f}s)")
                
                # Relocate files to loitering directory if they loitered
                if state.is_loitering:
                    # Move video path segments to loitering
                    new_video_paths = []
                    for video_path in state.video_paths:
                        if os.path.exists(video_path):
                            filename = os.path.basename(video_path)
                            new_path = os.path.join(self.video_recorder.recordings_dir, "loitering", filename)
                            if os.path.abspath(video_path) != os.path.abspath(new_path):
                                try:
                                    import shutil
                                    shutil.move(video_path, new_path)
                                    new_video_paths.append(new_path)
                                except Exception as e:
                                    logger.error(f"Failed to move video file {video_path} to loitering: {e}")
                                    new_video_paths.append(video_path)
                            else:
                                new_video_paths.append(video_path)
                        else:
                            new_video_paths.append(video_path)
                    state.video_paths = new_video_paths
                    
                    # Move snapshot to loitering
                    if state.snapshot_path and os.path.exists(state.snapshot_path):
                        filename = os.path.basename(state.snapshot_path)
                        new_snapshot_path = os.path.join(self.video_recorder.snapshots_dir, "loitering", filename)
                        if os.path.abspath(state.snapshot_path) != os.path.abspath(new_snapshot_path):
                            try:
                                import shutil
                                shutil.move(state.snapshot_path, new_snapshot_path)
                                state.snapshot_path = new_snapshot_path
                            except Exception as e:
                                logger.error(f"Failed to move snapshot {state.snapshot_path} to loitering: {e}")
                                
                    # Log event to loitering CSV log
                    self.loitering_csv_logger.log_event(
                         event_id=state.event_id,
                         person_id=track_id,
                         zone_id=zone_id,
                         zone_name=state.zone_name,
                         entry_time=state.entry_time,
                         exit_time=state.exit_time,
                         duration=state.duration,
                         video_path=";".join(state.video_paths),
                         snapshot_path=state.snapshot_path
                    )
                else:
                    # Write entry to standard danger zone CSV log
                    self.csv_logger.log_event(
                         event_id=state.event_id,
                         person_id=track_id,
                         zone_id=zone_id,
                         zone_name=state.zone_name,
                         entry_time=state.entry_time,
                         exit_time=state.exit_time,
                         duration=state.duration,
                         video_path=";".join(state.video_paths),
                         snapshot_path=state.snapshot_path
                    )
                
                # Update database record regardless of loitering
                try:
                    import postgres_db
                    postgres_db.update_intrusion_event(
                        event_id=state.event_id,
                        person_id=track_id,
                        zone_id=zone_id,
                        duration_seconds=state.duration,
                        video_path=";".join(state.video_paths),
                        snapshot_path=state.snapshot_path,
                        is_loitering=state.is_loitering,
                        exit_time=state.exit_time
                    )
                except Exception as e:
                    logger.error(f"Failed to update intrusion event {state.event_id} in database: {e}")
            else:
                logger.info(f"Person {track_id} left Zone {state.zone_name} before confirmation threshold. Event discarded.")

        # 3. Calculate currently occupied zones (excluding missing/exited tracks)
        occupied_zones = set()
        for (track_id, zone_id), state in self.active_states.items():
            if state.missing_frames == 0:
                occupied_zones.add(zone_id)
                
        return list(occupied_zones)
