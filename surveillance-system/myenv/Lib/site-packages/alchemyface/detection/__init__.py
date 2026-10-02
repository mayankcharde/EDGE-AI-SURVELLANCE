"""Face detection. See ``base`` for the protocol these implement."""

from alchemyface.detection.base import Detector
from alchemyface.detection.yunet import YuNetDetector, face_from_row, row_from_face

__all__ = ["Detector", "YuNetDetector", "face_from_row", "row_from_face"]
