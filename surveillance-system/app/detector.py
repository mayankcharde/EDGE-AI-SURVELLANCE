import logging
from ultralytics import YOLO
from config import YOLO_WEIGHTS, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Detect ALL objects, not just class 0 (person).
# Set DETECT_ALL_CLASSES = False if you only want person detection.
DETECT_ALL_CLASSES = False


class PersonDetector:
    def __init__(self):
        logger.info(f"Loading YOLO model: {YOLO_WEIGHTS}")
        self.model = YOLO(YOLO_WEIGHTS)

    def track(self, frame):
        """
        Run YOLOv8 tracking on a frame.
        Detects ALL classes (person, animal, object…) so that enrolled
        subjects that are not human (e.g. tigers) are still tracked.

        Returns list of dicts: {box, track_id, confidence, class_id}
        """
        if DETECT_ALL_CLASSES:
            # No `classes` filter → detect everything
            results = self.model.track(
                frame, persist=True, imgsz=480,
                conf=CONFIDENCE_THRESHOLD, verbose=False
            )
        else:
            results = self.model.track(
                frame, persist=True, classes=[0], imgsz=480,  # person only
                conf=CONFIDENCE_THRESHOLD, verbose=False
            )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:
            return detections

        xyxy    = boxes.xyxy.cpu().numpy()
        ids     = boxes.id.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cid in zip(xyxy, ids, confs, cls_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            detections.append({
                "box":        (x1, y1, x2, y2),
                "track_id":   int(tid),
                "confidence": float(conf),
                "class_id":   int(cid),
            })
        return detections