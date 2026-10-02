"""The library's own error hierarchy.

Everything AlchemyFace raises *deliberately* descends from AlchemyFaceError:
model resolution, downloads, database schemas, and "no face here".

It is not a blanket guarantee, and was documented as one until an audit proved
otherwise. Constructing a detector or embedder over a file that is not loadable
ONNX raises ``cv2.error`` — including the Git-LFS-pointer case ``models.py``
warns about, since a file already on disk is trusted rather than checksummed.
Bad arguments raise ``ValueError``; an unopenable camera raises ``RuntimeError``.
A caller wanting total isolation needs ``except Exception`` around
construction, not ``except AlchemyFaceError``."""

from __future__ import annotations


class AlchemyFaceError(Exception):
    """Base class for every error AlchemyFace raises."""


class ModelNotFoundError(AlchemyFaceError):
    """Weights were not on disk and downloading them was not permitted."""


class ModelDownloadError(AlchemyFaceError):
    """A download failed, or what arrived did not match its checksum."""


class NoFaceDetectedError(AlchemyFaceError):
    """An operation required a face and the image had none."""


class PickleSchemaError(AlchemyFaceError):
    """A pickle did not hold a recognisable face database."""
