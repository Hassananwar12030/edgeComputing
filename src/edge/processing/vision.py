"""
Vision Processing Module (YOLOv8)
=================================

Handles object detection using YOLOv8 for auto-labeling audio samples.

Purpose:
- Detect objects in camera frames (car, person, dog, etc.)
- Provide labels for the audio classifier training
- This is "weak supervision" - vision labels audio

YOLOv8 on Raspberry Pi:
- Use YOLOv8n (nano) or YOLOv8s (small) for speed
- Export to TFLite for better Pi performance
- Expected: ~5-10 FPS on Pi 4 with TFLite

Labeling strategy:
1. Run detection on frame
2. Filter by confidence threshold (>0.5)
3. If multiple objects, use highest confidence or dominant class
4. Handle temporal consistency (same object across frames)

Usage:
    processor = VisionProcessor(model_path="models/yolov8n.tflite")
    detections = processor.detect(frame)
    # detections = [{'class': 'car', 'confidence': 0.85, 'bbox': [x,y,w,h]}, ...]
    label = processor.get_dominant_label(detections)
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

# Ultralytics is the official YOLOv8 package
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


class VisionProcessor:
    """
    YOLOv8-based object detector for auto-labeling.

    Provides object detection on camera frames and methods
    to extract labels for audio sample training.

    Supports:
    - Standard YOLOv8 models (.pt)
    - TFLite exported models (.tflite)
    - ONNX models (.onnx)

    Attributes:
        model: YOLOv8 model instance
        confidence_threshold: Minimum confidence for detections
        classes_of_interest: List of class names to detect
    """

    # Default COCO classes relevant for audio detection
    DEFAULT_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'dog', 'cat', 'bird'
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        classes_of_interest: Optional[List[str]] = None,
        device: str = "cpu",
        img_size: int = 640
    ):
        """
        Initialize the vision processor.

        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for valid detections
            classes_of_interest: List of class names to detect (None = all)
            device: Device to run on ('cpu', 'cuda', etc.)
            img_size: Input image size for model
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes_of_interest = classes_of_interest or self.DEFAULT_CLASSES
        self.device = device
        self.img_size = img_size

        self.model = None
        self.class_names = []
        self.logger = logging.getLogger(__name__)

        # Mock mode for development
        self._mock_mode = not YOLO_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "ultralytics not available - running in mock mode. "
                "Install with: pip install ultralytics"
            )
            self.class_names = self.DEFAULT_CLASSES
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            self.class_names = list(self.model.names.values())
            self.logger.info(
                f"Loaded YOLOv8 model: {self.model_path}, "
                f"{len(self.class_names)} classes"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self._mock_mode = True

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run object detection on a frame.

        Args:
            frame: RGB image as numpy array, shape (H, W, 3)

        Returns:
            List of detections, each containing:
            - 'class': Class name (str)
            - 'confidence': Detection confidence (float)
            - 'bbox': Bounding box [x1, y1, x2, y2] (list)
            - 'class_id': Class index (int)
        """
        if self._mock_mode:
            return self._mock_detect(frame)

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            imgsz=self.img_size,
            verbose=False
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Filter by classes of interest
                if class_name in self.classes_of_interest:
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'class_id': class_id
                    })

        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Generate mock detections for development.

        Returns random detections from classes of interest.
        """
        import random

        # Randomly return 0-2 detections
        n_detections = random.randint(0, 2)
        detections = []

        for _ in range(n_detections):
            class_name = random.choice(self.classes_of_interest)
            detections.append({
                'class': class_name,
                'confidence': random.uniform(0.5, 0.95),
                'bbox': [
                    random.randint(0, 320),
                    random.randint(0, 240),
                    random.randint(320, 640),
                    random.randint(240, 480)
                ],
                'class_id': self.classes_of_interest.index(class_name)
            })

        return detections

    def get_dominant_label(
        self,
        detections: List[Dict],
        strategy: str = "highest_confidence"
    ) -> Optional[Tuple[str, float]]:
        """
        Get the dominant label from detections.

        Args:
            detections: List of detection dicts
            strategy: How to select dominant label:
                - "highest_confidence": Pick detection with highest confidence
                - "largest_area": Pick detection with largest bounding box
                - "center": Pick detection closest to frame center

        Returns:
            Tuple of (class_name, confidence) or None if no detections
        """
        if not detections:
            return None

        if strategy == "highest_confidence":
            best = max(detections, key=lambda d: d['confidence'])
        elif strategy == "largest_area":
            def area(d):
                x1, y1, x2, y2 = d['bbox']
                return (x2 - x1) * (y2 - y1)
            best = max(detections, key=area)
        else:  # center (not implemented yet)
            best = max(detections, key=lambda d: d['confidence'])

        return (best['class'], best['confidence'])

    def detect_and_label(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[str], float, List[Dict]]:
        """
        Convenience method: detect and get label in one call.

        Args:
            frame: RGB image

        Returns:
            Tuple of (label, confidence, all_detections)
            label is None if no detections
        """
        detections = self.detect(frame)
        result = self.get_dominant_label(detections)

        if result:
            label, confidence = result
            return label, confidence, detections
        else:
            return None, 0.0, detections

    def export_to_tflite(self, output_path: str) -> str:
        """
        Export model to TFLite format for Raspberry Pi.

        Args:
            output_path: Path for exported model

        Returns:
            Path to exported model
        """
        if self._mock_mode:
            self.logger.warning("Cannot export in mock mode")
            return ""

        try:
            export_path = self.model.export(format='tflite')
            self.logger.info(f"Exported model to: {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return ""

    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        return self.class_names

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold."""
        self.confidence_threshold = threshold


class TemporalConsistencyFilter:
    """
    Filters detections for temporal consistency.

    Helps reduce noise by requiring objects to be detected
    across multiple frames before accepting the label.

    Example: A "car" must be detected for 3 consecutive frames
    before we label audio as "car".
    """

    def __init__(
        self,
        min_frames: int = 3,
        max_gap: int = 2
    ):
        """
        Initialize the filter.

        Args:
            min_frames: Minimum consecutive frames for valid detection
            max_gap: Maximum frames an object can disappear
        """
        self.min_frames = min_frames
        self.max_gap = max_gap

        # Track detections: class_name -> consecutive count
        self._class_counts = {}
        self._gap_counts = {}

    def update(self, detections: List[Dict]) -> Optional[str]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts from current frame

        Returns:
            Stable label if available, None otherwise
        """
        current_classes = set(d['class'] for d in detections)

        # Update counts
        for class_name in list(self._class_counts.keys()):
            if class_name in current_classes:
                self._class_counts[class_name] += 1
                self._gap_counts[class_name] = 0
            else:
                self._gap_counts[class_name] += 1
                if self._gap_counts[class_name] > self.max_gap:
                    del self._class_counts[class_name]
                    del self._gap_counts[class_name]

        # Add new classes
        for class_name in current_classes:
            if class_name not in self._class_counts:
                self._class_counts[class_name] = 1
                self._gap_counts[class_name] = 0

        # Find stable label
        for class_name, count in self._class_counts.items():
            if count >= self.min_frames:
                return class_name

        return None

    def reset(self) -> None:
        """Reset all tracking state."""
        self._class_counts.clear()
        self._gap_counts.clear()
