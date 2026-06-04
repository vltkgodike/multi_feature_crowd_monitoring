import csv
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CSVLogger:
    """Handles logging of danger zone intrusion events into a CSV file."""

    def __init__(self, log_path: str = "logs/intrusion_log.csv"):
        """Initializes the CSV logger and creates the log file with headers if missing.
        
        Args:
            log_path: Path to the CSV file where logs should be stored.
        """
        self.log_path = log_path
        
        # Ensure the directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        self._init_file()

    def _init_file(self) -> None:
        """Writes headers to the CSV file if it doesn't already exist."""
        if not os.path.exists(self.log_path):
            try:
                with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "EventID",
                        "PersonID",
                        "ZoneID",
                        "ZoneName",
                        "EntryTime",
                        "ExitTime",
                        "DurationSeconds",
                        "VideoPath",
                        "SnapshotPath"
                    ])
                logger.info(f"Initialized CSV log file at '{self.log_path}' with headers.")
            except Exception as e:
                logger.error(f"Failed to initialize CSV log file: {e}")

    def _format_paths(self, path_value: str) -> str:
        """Formats one or more semicolon-separated paths as absolute paths."""
        if not path_value:
            return ""
        paths = [path for path in str(path_value).split(";") if path]
        return ";".join(os.path.abspath(path) for path in paths)

    def log_event(
        self,
        event_id: int,
        person_id: int,
        zone_id: int,
        zone_name: str,
        entry_time: datetime,
        exit_time: datetime,
        duration: float,
        video_path: str,
        snapshot_path: str
    ) -> None:
        """Appends a new intrusion event row to the CSV file.
        
        Args:
            event_id: Unique event ID.
            person_id: The tracking ID of the person.
            zone_id: The ID of the danger zone.
            zone_name: The name of the danger zone.
            entry_time: Timestamp when they entered.
            exit_time: Timestamp when they exited.
            duration: Time elapsed inside the zone (seconds).
            video_path: Path to the saved video file.
            snapshot_path: Path to the saved confirmation snapshot.
        """
        try:
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event_id,
                    person_id,
                    zone_id,
                    zone_name,
                    entry_time.isoformat(),
                    exit_time.isoformat(),
                    f"{duration:.2f}",
                    self._format_paths(video_path),
                    self._format_paths(snapshot_path)
                ])
            logger.info(f"Successfully logged Event {event_id} to CSV.")
        except Exception as e:
            logger.error(f"Failed to log event {event_id} to CSV: {e}")
