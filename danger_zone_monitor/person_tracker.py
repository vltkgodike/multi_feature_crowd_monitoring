import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from danger_zone_monitor.models import TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float


@dataclass
class _Binding:
    name: str
    index: int
    host: np.ndarray
    device: object
    shape: Tuple[int, ...]
    dtype: object


class TensorRTYOLOEngine:
    """Runs a YOLO TensorRT engine without Ultralytics."""

    def __init__(
        self,
        engine_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
    ):
        self.engine_path = engine_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.trt, self.cuda = self._load_runtime_modules()
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.cuda_context = None
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings: List[int] = []
        self.input_binding: Optional[_Binding] = None
        self.output_bindings: List[_Binding] = []
        self._load_engine()

    @staticmethod
    def _load_runtime_modules():
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except Exception as exc:
            raise RuntimeError(
                "TensorRT tracker requires Jetson TensorRT Python bindings and PyCUDA. "
                "Install them from JetPack/system packages before running this app."
            ) from exc
        return trt, cuda

    def _load_engine(self) -> None:
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine model not found: {self.engine_path}")

        self.cuda.init()
        self.cuda_context = self.cuda.Device(0).make_context()

        try:
            with open(self.engine_path, "rb") as f, self.trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

            self.context = self.engine.create_execution_context()
            self._configure_input_shape()
            self.stream = self.cuda.Stream()
            self._allocate_bindings()
        finally:
            self.cuda_context.pop()

        logger.info("TensorRT YOLO engine loaded: %s", self.engine_path)

    def _configure_input_shape(self) -> None:
        for index in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(index):
                continue

            shape = tuple(self.engine.get_binding_shape(index))
            if len(shape) != 4:
                raise RuntimeError(f"Expected NCHW input binding, got shape={shape}")

            if any(dim < 0 for dim in shape):
                shape = (1, 3, self.input_size, self.input_size)
                self.context.set_binding_shape(index, shape)
            else:
                self.input_size = int(shape[2])
            return

        raise RuntimeError("TensorRT engine has no input binding.")

    def _allocate_bindings(self) -> None:
        self.bindings = [0] * self.engine.num_bindings
        self.output_bindings = []

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            shape = tuple(self.context.get_binding_shape(index))
            dtype = self.trt.nptype(self.engine.get_binding_dtype(index))
            size = int(self.trt.volume(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings[index] = int(device_mem)

            binding = _Binding(
                name=name,
                index=index,
                host=host_mem,
                device=device_mem,
                shape=shape,
                dtype=dtype,
            )
            if self.engine.binding_is_input(index):
                self.input_binding = binding
            else:
                self.output_bindings.append(binding)

        if self.input_binding is None:
            raise RuntimeError("TensorRT input binding was not allocated.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)

        self.cuda_context.push()
        try:
            np.copyto(self.input_binding.host, input_tensor.ravel())
            self.cuda.memcpy_htod_async(
                self.input_binding.device,
                self.input_binding.host,
                self.stream,
            )
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )
            for binding in self.output_bindings:
                self.cuda.memcpy_dtoh_async(binding.host, binding.device, self.stream)
            self.stream.synchronize()
        finally:
            self.cuda_context.pop()

        outputs = [
            binding.host.reshape(binding.shape).copy()
            for binding in self.output_bindings
        ]
        return self._postprocess(outputs, frame.shape[:2], scale, pad_x, pad_y)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        pad_x = (self.input_size - resized_w) / 2
        pad_y = (self.input_size - resized_h) / 2

        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0), scale, pad_x, pad_y

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        frame_shape: Tuple[int, int],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Detection]:
        candidates = self._extract_candidates(outputs)
        if candidates.size == 0:
            return []

        boxes = candidates[:, :4].astype(np.float32)
        scores = candidates[:, 4].astype(np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        frame_h, frame_w = frame_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h - 1)

        keep = self._nms(boxes, scores)
        detections: List[Detection] = []
        for idx in keep[:self.max_detections]:
            x1, y1, x2, y2 = boxes[idx]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(scores[idx]),
                )
            )
        return detections

    def _extract_candidates(self, outputs: List[np.ndarray]) -> np.ndarray:
        rows = []
        for output in outputs:
            arr = np.squeeze(output)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 3:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])

            if arr.ndim != 2:
                continue

            if arr.shape[0] in (84, 85) and arr.shape[1] > arr.shape[0]:
                arr = arr.T

            if arr.shape[1] >= 84:
                boxes_xywh = arr[:, :4]
                if arr.shape[1] == 85:
                    scores = arr[:, 4] * arr[:, 5]
                else:
                    scores = arr[:, 4]
                mask = scores >= self.confidence_threshold
                if not np.any(mask):
                    continue
                boxes = self._xywh_to_xyxy(boxes_xywh[mask])
                rows.append(np.column_stack((boxes, scores[mask])))
            elif arr.shape[1] >= 6:
                class_ids = arr[:, 5].astype(np.int32)
                scores = arr[:, 4]
                mask = (class_ids == 0) & (scores >= self.confidence_threshold)
                if not np.any(mask):
                    continue
                rows.append(np.column_stack((arr[mask, :4], scores[mask])))

        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        converted = np.empty_like(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        if boxes.size == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < self.max_detections:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)
            order = order[1:][iou <= self.nms_threshold]

        return keep

    def close(self) -> None:
        if self.cuda_context is not None:
            try:
                self.cuda_context.push()
                self.cuda_context.pop()
                self.cuda_context.detach()
            except Exception:
                pass
            self.cuda_context = None


class KalmanBoxTrack:
    """Constant-velocity bbox track used by the BoT-SORT-lite tracker."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ):
        self.track_id = track_id
        self.confidence = detection.confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = np.zeros((8, 1), dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state[:4, 0] = self._bbox_to_measurement(detection.bbox)

    def predict(self) -> None:
        motion = np.eye(8, dtype=np.float32)
        motion[0, 4] = 1.0
        motion[1, 5] = 1.0
        motion[2, 6] = 1.0
        motion[3, 7] = 1.0

        process = np.eye(8, dtype=np.float32) * self.process_noise
        self.state = motion @ self.state
        self.covariance = motion @ self.covariance @ motion.T + process
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection) -> None:
        measurement = self._bbox_to_measurement(detection.bbox).reshape(4, 1)
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        measurement_cov = np.eye(4, dtype=np.float32) * self.measurement_noise

        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + measurement_cov
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + gain @ innovation
        identity = np.eye(8, dtype=np.float32)
        self.covariance = (identity - gain @ observation) @ self.covariance
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

    def to_person(self) -> TrackedPerson:
        return TrackedPerson(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=float(self.confidence),
        )

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        cx, cy, width, height = self.state[:4, 0]
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        x1 = float(cx - width / 2)
        y1 = float(cy - height / 2)
        x2 = float(cx + width / 2)
        y2 = float(cy + height / 2)
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_to_measurement(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        cx = float(x1 + width / 2)
        cy = float(y1 + height / 2)
        return np.array([cx, cy, width, height], dtype=np.float32)


class BoTSORTLiteTracker:
    """Fast BoT-SORT-style tracker without ReID, tuned for Jetson Nano."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 15,
        min_hits: int = 1,
        high_confidence_threshold: float = 0.45,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.high_confidence_threshold = high_confidence_threshold
        self.next_id = 1
        self.tracks: Dict[int, KalmanBoxTrack] = {}

    def update(self, detections: List[Detection]) -> List[TrackedPerson]:
        for track in self.tracks.values():
            track.predict()

        high_indices = [
            idx for idx, det in enumerate(detections)
            if det.confidence >= self.high_confidence_threshold
        ]
        low_indices = [
            idx for idx, det in enumerate(detections)
            if det.confidence < self.high_confidence_threshold
        ]

        unmatched_tracks = set(self.tracks.keys())
        unmatched_high = self._match_stage(high_indices, detections, unmatched_tracks)
        self._match_stage(low_indices, detections, unmatched_tracks)

        for det_idx in unmatched_high:
            self._start_track(detections[det_idx])

        stale_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.time_since_update > self.max_missed
        ]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

        return [
            track.to_person()
            for track in self.tracks.values()
            if track.time_since_update == 0 and track.hits >= self.min_hits
        ]

    def _match_stage(
        self,
        detection_indices: List[int],
        detections: List[Detection],
        unmatched_tracks: set,
    ) -> List[int]:
        unmatched_detections = set(detection_indices)
        candidates = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            for det_idx in detection_indices:
                if det_idx in unmatched_detections:
                    candidates.append((
                        self._iou(track.bbox, detections[det_idx].bbox),
                        track_id,
                        det_idx,
                    ))

        for iou, track_id, det_idx in sorted(candidates, reverse=True):
            if iou < self.iou_threshold:
                break
            if track_id not in unmatched_tracks or det_idx not in unmatched_detections:
                continue

            self.tracks[track_id].update(detections[det_idx])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_idx)

        return list(unmatched_detections)

    def _start_track(self, detection: Detection) -> None:
        self.tracks[self.next_id] = KalmanBoxTrack(self.next_id, detection)
        self.next_id += 1

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)


class PersonTracker:
    """TensorRT-backed person detector/tracker with optional async inference."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.engine",
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        tracker_iou_threshold: float = 0.3,
        max_missed: int = 15,
        async_inference: bool = True,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.tracker_iou_threshold = tracker_iou_threshold
        self.max_missed = max_missed
        self.async_inference = async_inference
        self._latest_tracks: List[TrackedPerson] = []
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=1)
        self._ready_event = threading.Event()
        self._worker_error: Optional[BaseException] = None
        self.engine: Optional[TensorRTYOLOEngine] = None
        self.tracker: Optional[BoTSORTLiteTracker] = None
        self.worker: Optional[threading.Thread] = None

        if os.path.splitext(model_path)[1].lower() != ".engine":
            raise ValueError(
                "Ultralytics has been removed. Pass a TensorRT .engine model path, "
                f"got: {model_path}"
            )

        if async_inference:
            self.worker = threading.Thread(
                target=self._worker_loop,
                name="TensorRTInferenceWorker",
                daemon=True,
            )
            self.worker.start()
            self._ready_event.wait()
            if self._worker_error is not None:
                raise RuntimeError("TensorRT inference worker failed to start.") from self._worker_error
        else:
            self.engine = self._create_engine()
            self.tracker = BoTSORTLiteTracker(tracker_iou_threshold, max_missed)

    def _create_engine(self) -> TensorRTYOLOEngine:
        return TensorRTYOLOEngine(
            engine_path=self.model_path,
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        if self.async_inference:
            self._submit_latest_frame(frame)
            with self._latest_lock:
                return list(self._latest_tracks)

        detections = self.engine.detect(frame)
        return self.tracker.update(detections)

    def _submit_latest_frame(self, frame: np.ndarray) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _worker_loop(self) -> None:
        try:
            engine = self._create_engine()
            tracker = BoTSORTLiteTracker(self.tracker_iou_threshold, self.max_missed)
            self._ready_event.set()
        except BaseException as exc:
            self._worker_error = exc
            self._ready_event.set()
            return

        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                detections = engine.detect(frame)
                tracks = tracker.update(detections)
                with self._latest_lock:
                    self._latest_tracks = tracks
            except Exception as exc:
                logger.error("TensorRT inference failed: %s", exc)

        engine.close()

    def close(self) -> None:
        self._stop_event.set()
        if self.worker is not None:
            try:
                while True:
                    self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass
            self.worker.join(timeout=2.0)
            self.worker = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None
