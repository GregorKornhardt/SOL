"""Public input contracts without importing optional model dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

CachePath: TypeAlias = str | PathLike[str]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
Offsets: TypeAlias = Sequence[int] | NDArray[np.integer[Any]]
Precision: TypeAlias = Literal["float32", "float64", "float16", "bfloat16"]


class TokenMatrix(Protocol):
    """Shape and indexing shared by NumPy arrays and floating Torch tensors."""

    @property
    def shape(self) -> Sequence[int]: ...

    def __len__(self) -> int: ...

    def __getitem__(self, key: Any) -> Any: ...


class Encoder(Protocol):
    """A streaming token encoder accepted by SOL's text methods.

    Yield one finite floating NumPy matrix of shape ``(tokens, dimension)``
    per input text, preserving order. Dimensions and dtype must stay constant.
    ``provenance`` must be JSON-serializable and identify the encoder settings.
    """

    @property
    def provenance(self) -> dict[str, Any]: ...

    def __call__(self, texts: Iterable[str]) -> Iterable[FloatArray]: ...
