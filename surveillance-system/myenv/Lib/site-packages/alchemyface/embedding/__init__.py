"""Face embedding. See ``base`` for the protocol these implement."""

from alchemyface.embedding.base import Embedder
from alchemyface.embedding.sface import SFaceEmbedder

__all__ = ["Embedder", "SFaceEmbedder"]
