from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Zone:
    """Represents a monitored danger zone polygon."""
    zone_id: int
    zone_name: str
    points: List[Tuple[int, int]]
    polygon_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Convert the list of points to a NumPy array for OpenCV compatibility."""
        self.polygon_np = np.array(self.points, dtype=np.int32)

@dataclass
class TrackedPerson:
    """Represents a tracked person in the frame."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate and return the center coordinate of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Calculate and return the bottom-center coordinate of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        return cx, cy

@dataclass
class IntrusionEvent:
    """Represents a finalized danger zone intrusion event."""
    event_id: int
    track_id: int
    zone_id: int
    zone_name: str
    entry_time: datetime
    exit_time: datetime
    duration: float
    video_path: Optional[str]
    snapshot_path: Optional[str]
