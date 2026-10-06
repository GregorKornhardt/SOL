"""SOL: double-sliced Wasserstein distance between token measures."""

from ._typing import Encoder
from ._version import __version__
from .features import Features
from .metric import SOL, SOLConfig
from .text import TextEncoder

__all__ = ["SOL", "SOLConfig", "Features", "TextEncoder", "Encoder", "__version__"]
