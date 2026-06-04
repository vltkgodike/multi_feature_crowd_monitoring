import logging
import os
import re
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VideoRecorder:
    """Manages the creation, writing, and completion of event video files and snapshots."""

    def __init__(self, recordings_dir: str = "recordings", snapshots_dir: str = "snapshots"):
        """Initializes the VideoRecorder and ensures output folders exist.
        
        Args:
            recordings_dir: Directory where video recordings will be saved.
            snapshots_dir: Directory where snapshots will be saved.
        """
        self.recordings_dir = recordings_dir
        self.snapshots_dir = snapshots_dir
        
        # Ensure subdirectories exist
        os.makedirs(os.path.join(self.recordings_dir, "danger_zone"), exist_ok=True)
        os.makedirs(os.path.join(self.recordings_dir, "loitering"), exist_ok=True)
        os.makedirs(os.path.join(self.snapshots_dir, "danger_zone"), exist_ok=True)
        os.makedirs(os.path.join(self.snapshots_dir, "loitering"), exist_ok=True)
        
        logger.info(f"VideoRecorder initialized. recordings_dir='{self.recordings_dir}', snapshots_dir='{self.snapshots_dir}'")

    def get_next_event_id(self) -> int:
        """Scans recordings and snapshots directories to find the next available event ID.
        
        Returns:
            The next unique event ID as an integer.
        """
        max_id = 0
        pattern = re.compile(r"event_(\d+)(?:_part_\d+)?\.(mp4|jpg)$")
        
        # Scan recordings directory and its subdirectories
        if os.path.exists(self.recordings_dir):
            for root, dirs, files in os.walk(self.recordings_dir):
                for filename in files:
                    match = pattern.match(filename)
                    if match:
                        max_id = max(max_id, int(match.group(1)))
                    
        # Scan snapshots directory and its subdirectories
        if os.path.exists(self.snapshots_dir):
            for root, dirs, files in os.walk(self.snapshots_dir):
                for filename in files:
                    match = pattern.match(filename)
                    if match:
                        max_id = max(max_id, int(match.group(1)))
                    
        next_id = max_id + 1
        logger.info(f"Scanned output directories. Next event ID determined: {next_id}")
        return next_id

    def save_snapshot(self, frame: np.ndarray, event_id: int, suffix: str = "", subdir: str = "danger_zone") -> str:
        """Saves a single snapshot image for a confirmed event.
        
        Args:
            frame: The image frame to save.
            event_id: The unique ID of the event.
            suffix: Optional filename suffix to append (e.g. "_loiter_10s").
            subdir: The subdirectory to save the snapshot (e.g., 'danger_zone' or 'loitering').
            
        Returns:
            The path to the saved snapshot.
        """
        filename = f"event_{event_id:04d}{suffix}.jpg"
        filepath = os.path.join(self.snapshots_dir, subdir, filename)
        try:
            cv2.imwrite(filepath, frame)
            logger.info(f"Saved snapshot to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save snapshot to {filepath}: {e}")
        return filepath

    def start_recording(
        self,
        event_id: int,
        fps: float,
        frame_size: Tuple[int, int],
        segment_index: Optional[int] = None,
        subdir: str = "danger_zone"
    ) -> Tuple[cv2.VideoWriter, str]:
        """Starts a VideoWriter for a confirmed event.
        
        Args:
            event_id: The unique ID of the event.
            fps: Frame rate for saving the video.
            frame_size: (width, height) of the frames.
            segment_index: Optional segment number for long recordings. The first
                segment keeps the original event filename for compatibility.
            subdir: The subdirectory to save the recording (e.g., 'danger_zone' or 'loitering').
            
        Returns:
            A tuple of (cv2.VideoWriter, video_path).
        """
        if segment_index is None or segment_index <= 1:
            filename = f"event_{event_id:04d}.mp4"
        else:
            filename = f"event_{event_id:04d}_part_{segment_index:03d}.mp4"
        filepath = os.path.join(self.recordings_dir, subdir, filename)
        
        # Use 'mp4v' fourcc for portable .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        logger.info(f"Opening VideoWriter for {filepath} with fps={fps}, size={frame_size}")
        writer = cv2.VideoWriter(filepath, fourcc, float(fps), frame_size)
        
        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {filepath}")
            
        return writer, filepath

    def write_frame(self, writer: cv2.VideoWriter, frame: np.ndarray) -> None:
        """Writes a frame to an active VideoWriter.
        
        Args:
            writer: The VideoWriter instance.
            frame: The image frame to write.
        """
        if writer is not None and writer.isOpened():
            writer.write(frame)

    def stop_recording(self, writer: cv2.VideoWriter) -> None:
        """Releases the VideoWriter resource.
        
        Args:
            writer: The VideoWriter instance to release.
        """
        if writer is not None:
            writer.release()
            logger.info("VideoWriter released.")
