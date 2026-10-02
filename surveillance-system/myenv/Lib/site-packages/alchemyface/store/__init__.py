"""Gallery backends. See ``base`` for the protocol they implement."""

from alchemyface.store.base import FaceStore
from alchemyface.store.memory import InMemoryStore
from alchemyface.store.pickle import PickleStore

__all__ = ["FaceStore", "InMemoryStore", "PickleStore"]
