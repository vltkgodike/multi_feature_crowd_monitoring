import logging
import os
from typing import List
import numpy as np
from ultralytics import YOLO
from danger_zone_monitor.models import TrackedPerson

logger = logging.getLogger(__name__)

class PersonTracker:
    """Uses YOLOv8 and ByteTrack to detect and track people in video frames."""

    def __init__(self, model_path: str = "models/yolov8n.engine"):
        """Initializes the YOLOv8 model for tracking.
        
        Args:
            model_path: Path to a YOLO model file. TensorRT .engine files are
                supported, and .pt files can still be passed explicitly.
        """
        self.model_path = model_path
        model_ext = os.path.splitext(self.model_path)[1].lower()
        
        # Ensure parent directory of model_path exists.
        dir_name = os.path.dirname(model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if model_ext == ".engine" and not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"TensorRT engine model not found: {self.model_path}. "
                "Export or copy the .engine file to this path, or pass a valid "
                ".engine path with --model."
            )
            
        logger.info(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(self.model_path)
        logger.info("YOLOv8 model loaded successfully.")

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        """Tracks people in the given frame.
        
        Args:
            frame: The input image/frame (OpenCV numpy array).
            
        Returns:
            A list of TrackedPerson dataclass instances containing bounding boxes,
            confidence scores, and tracking IDs.
        """
        # Run YOLOv8 tracking with persist=True to keep track of state across frames.
        # classes=[0] filters detections to only class 0 (person).
        # tracker="bytetrack.yaml" specifies ByteTrack.
        # verbose=False reduces console noise.
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        tracked_people: List[TrackedPerson] = []
        if not results or results[0].boxes is None:
            return tracked_people

        boxes = results[0].boxes
        for box in boxes:
            # If the box does not have a tracking ID yet, skip it
            if box.id is not None:
                try:
                    track_id = int(box.id[0].item())
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    
                    person = TrackedPerson(
                        track_id=track_id,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf
                    )
                    tracked_people.append(person)
                except Exception as e:
                    logger.warning(f"Failed to parse tracking box data: {e}")
                    
        return tracked_people
