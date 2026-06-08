import logging
import os
import re
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_camera_id(source) -> str:
    """Extracts a camera identifier from a video source.

    Attempts to derive a meaningful camera ID from RTSP URLs or other sources.
    Falls back to ``camera_<index>`` when extraction fails.

    Args:
        source: The video source – an RTSP URL string, a file path, or a
            webcam index (int or numeric string).

    Returns:
        A sanitised camera identifier string such as ``camera_0`` or
        ``camera_warehouse_east``.
    """
    if source is None:
        return "camera_0"

    source_str = str(source).strip()

    # Integer webcam index
    if source_str.isdigit():
        return f"camera_{source_str}"

    # RTSP / HTTP URL
    if "://" in source_str:
        try:
            parsed = urlparse(source_str)
            # Try to use the path segment as a meaningful camera name
            path = parsed.path.strip("/")
            if path:
                # Take the last meaningful path segment
                segments = [s for s in path.split("/") if s]
                if segments:
                    # Sanitise: lowercase, replace non-alphanumeric with _
                    raw = segments[-1]
                    sanitised = re.sub(r"[^a-zA-Z0-9]", "_", raw).strip("_").lower()
                    if sanitised:
                        return f"camera_{sanitised}"

            # Fallback: use hostname
            hostname = parsed.hostname or ""
            sanitised = re.sub(r"[^a-zA-Z0-9]", "_", hostname).strip("_").lower()
            if sanitised:
                return f"camera_{sanitised}"
        except Exception:
            pass

    # Video file path – use filename stem
    if os.path.isfile(source_str):
        stem = os.path.splitext(os.path.basename(source_str))[0]
        sanitised = re.sub(r"[^a-zA-Z0-9]", "_", stem).strip("_").lower()
        if sanitised:
            return f"camera_{sanitised}"

    return "camera_0"


class VideoRecorder:
    """Manages the creation, writing, and completion of event video files and snapshots.

    Output is organised hierarchically::

        recordings/<event_type>/<camera_id>/<YYYY-MM-DD>/<HH>/
        snapshots/<event_type>/<camera_id>/<YYYY-MM-DD>/<HH>/full_frame/
        snapshots/<event_type>/<camera_id>/<YYYY-MM-DD>/<HH>/cropped_person/
    """

    def __init__(
        self,
        recordings_dir: str = "recordings",
        snapshots_dir: str = "snapshots",
        camera_id: str = "camera_0",
    ):
        """Initializes the VideoRecorder.

        Args:
            recordings_dir: Root directory for video recordings.
            snapshots_dir: Root directory for snapshot images.
            camera_id: Identifier for the camera producing outputs.
        """
        self.recordings_dir = recordings_dir
        self.snapshots_dir = snapshots_dir
        self.camera_id = camera_id

        logger.info(
            f"VideoRecorder initialized. camera_id='{self.camera_id}', "
            f"recordings_dir='{self.recordings_dir}', snapshots_dir='{self.snapshots_dir}'"
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _build_recording_dir(self, subdir: str = "danger_zone", timestamp: Optional[datetime] = None) -> str:
        """Builds and creates the recording output directory.

        Structure: ``<recordings_dir>/<subdir>/<camera_id>/<YYYY-MM-DD>/<HH>/``

        Args:
            subdir: Event type subdirectory (``danger_zone`` or ``loitering``).
            timestamp: Optional timestamp; defaults to now.

        Returns:
            The absolute directory path.
        """
        ts = timestamp or datetime.now()
        dir_path = os.path.join(
            self.recordings_dir,
            subdir,
            self.camera_id,
            ts.strftime("%Y-%m-%d"),
            ts.strftime("%H"),
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _build_snapshot_dir(self, subdir: str = "danger_zone", snapshot_type: str = "full_frame", timestamp: Optional[datetime] = None) -> str:
        """Builds and creates the snapshot output directory.

        Structure: ``<snapshots_dir>/<subdir>/<camera_id>/<YYYY-MM-DD>/<HH>/<snapshot_type>/``

        Args:
            subdir: Event type subdirectory (``danger_zone`` or ``loitering``).
            snapshot_type: Either ``full_frame`` or ``cropped_person``.
            timestamp: Optional timestamp; defaults to now.

        Returns:
            The absolute directory path.
        """
        ts = timestamp or datetime.now()
        dir_path = os.path.join(
            self.snapshots_dir,
            subdir,
            self.camera_id,
            ts.strftime("%Y-%m-%d"),
            ts.strftime("%H"),
            snapshot_type,
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    # ------------------------------------------------------------------
    # Event ID scanning (legacy compat)
    # ------------------------------------------------------------------

    def get_next_event_id(self) -> int:
        """Scans recordings and snapshots directories to find the next available event ID.

        Returns:
            The next unique event ID as an integer.
        """
        max_id = 0
        pattern = re.compile(r"event_(\d+)(?:_part_\d+)?\.(mp4|jpg)$")
        # Also match new naming convention that embeds event IDs
        pattern_new = re.compile(r"(?:danger_zone|loitering)_(\d+)_\d{8}_\d{6}(?:_part_\d+)?\.(mp4|jpg)$")

        for base_dir in (self.recordings_dir, self.snapshots_dir):
            if os.path.exists(base_dir):
                for root, dirs, files in os.walk(base_dir):
                    for filename in files:
                        match = pattern.match(filename) or pattern_new.match(filename)
                        if match:
                            max_id = max(max_id, int(match.group(1)))

        next_id = max_id + 1
        logger.info(f"Scanned output directories. Next event ID determined: {next_id}")
        return next_id

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        frame: np.ndarray,
        event_id: int,
        suffix: str = "",
        subdir: str = "danger_zone",
        track_id: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        """Saves a full-frame snapshot and optionally a cropped person snapshot.

        Full-frame snapshots are saved to the ``full_frame/`` subdirectory.
        If ``bbox`` is provided, a cropped image of the person is also saved
        to the ``cropped_person/`` subdirectory.

        Args:
            frame: The image frame to save.
            event_id: The unique ID of the event.
            suffix: Optional filename suffix (e.g. ``_loiter_10s``).
            subdir: Event type subdirectory (``danger_zone`` or ``loitering``).
            track_id: Optional tracking ID for the person (used in cropped filename).
            bbox: Optional bounding box ``(x1, y1, x2, y2)`` for cropping.

        Returns:
            The path to the saved full-frame snapshot.
        """
        now = datetime.now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")

        # --- Full-frame snapshot ---
        full_frame_dir = self._build_snapshot_dir(subdir=subdir, snapshot_type="full_frame", timestamp=now)
        full_filename = f"{subdir}_{ts_str}{suffix}.jpg"
        full_filepath = os.path.join(full_frame_dir, full_filename)
        try:
            cv2.imwrite(full_filepath, frame)
            logger.info(f"Saved full-frame snapshot to {full_filepath}")
        except Exception as e:
            logger.error(f"Failed to save full-frame snapshot to {full_filepath}: {e}")

        # --- Cropped person snapshot ---
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            # Clamp bbox to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                cropped = frame[y1:y2, x1:x2]
                crop_dir = self._build_snapshot_dir(subdir=subdir, snapshot_type="cropped_person", timestamp=now)
                person_label = f"_person_{track_id}" if track_id is not None else ""
                crop_filename = f"{subdir}{person_label}_{ts_str}{suffix}.jpg"
                crop_filepath = os.path.join(crop_dir, crop_filename)
                try:
                    cv2.imwrite(crop_filepath, cropped)
                    logger.info(f"Saved cropped person snapshot to {crop_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save cropped person snapshot to {crop_filepath}: {e}")

        return full_filepath

    def _build_full_recording_dir(self, timestamp: Optional[datetime] = None) -> str:
        """Builds and creates the full recording output directory.

        Structure: ``full_recordings/<camera_id>/<YYYY-MM-DD>/<HH>/``
        """
        ts = timestamp or datetime.now()
        parent_dir = os.path.dirname(os.path.abspath(self.recordings_dir))
        dir_path = os.path.join(
            parent_dir,
            "full_recordings",
            self.camera_id,
            ts.strftime("%Y-%m-%d"),
            ts.strftime("%H"),
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def start_full_recording(
        self,
        fps: float,
        frame_size: Tuple[int, int],
    ) -> Tuple[cv2.VideoWriter, str]:
        """Starts a VideoWriter for a full recording.

        Returns:
            A tuple of ``(cv2.VideoWriter, video_path)``.
        """
        now = datetime.now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")

        rec_dir = self._build_full_recording_dir(timestamp=now)
        filename = f"record_{ts_str}.mp4"
        filepath = os.path.join(rec_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        logger.info(f"Opening VideoWriter for full recording: {filepath} with fps={fps}, size={frame_size}")
        writer = cv2.VideoWriter(filepath, fourcc, float(fps), frame_size)

        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for full recording: {filepath}")

        return writer, filepath

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(
        self,
        event_id: int,
        fps: float,
        frame_size: Tuple[int, int],
        segment_index: Optional[int] = None,
        subdir: str = "danger_zone",
    ) -> Tuple[cv2.VideoWriter, str]:
        """Starts a VideoWriter for a confirmed event.

        Args:
            event_id: The unique ID of the event.
            fps: Frame rate for saving the video.
            frame_size: ``(width, height)`` of the frames.
            segment_index: Optional segment number for long recordings.
            subdir: Event type subdirectory (``danger_zone`` or ``loitering``).

        Returns:
            A tuple of ``(cv2.VideoWriter, video_path)``.
        """
        now = datetime.now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")

        rec_dir = self._build_recording_dir(subdir=subdir, timestamp=now)

        if segment_index is not None and segment_index > 1:
            filename = f"{subdir}_{ts_str}_part_{segment_index:03d}.mp4"
        else:
            filename = f"{subdir}_{ts_str}.mp4"

        filepath = os.path.join(rec_dir, filename)

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
