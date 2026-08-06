"""Point-in-time data context for evaluating namespaced DSL fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.dsl.fields import FieldSpec, get_field
from src.dsl.namespaces import get_namespace


@dataclass(frozen=True)
class FieldWindow:
    """A point-in-time field slice returned to the DSL evaluator."""

    field: FieldSpec
    timestamp: pd.Timestamp
    index: pd.Index
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.values.size)


class PointInTimeContext:
    """Expose namespace field history up to a single evaluation timestamp.

    The context is intentionally single-asset. A cross-sectional evaluator can
    create one context per asset/date and then rank factor outputs across assets.
    """

    def __init__(self, data_by_namespace: Mapping[str, pd.DataFrame], timestamp):
        self.timestamp = pd.Timestamp(timestamp)
        self._data: dict[str, pd.DataFrame] = {}

        for namespace, frame in data_by_namespace.items():
            spec = get_namespace(namespace)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"namespace '{namespace}' data must be a pandas DataFrame")
            if frame.empty:
                raise ValueError(f"namespace '{namespace}' data is empty")
            if not frame.index.is_monotonic_increasing:
                frame = frame.sort_index()

            self._data[spec.name] = frame

    def namespaces(self) -> tuple[str, ...]:
        """Return namespaces available in this context."""

        return tuple(sorted(self._data))

    def has_namespace(self, namespace: str) -> bool:
        """Return whether namespace data is available."""

        return get_namespace(namespace).name in self._data

    def available_columns(self, namespace: str) -> tuple[str, ...]:
        """Return available raw columns for a namespace."""

        ns = get_namespace(namespace).name
        if ns not in self._data:
            raise ValueError(f"namespace '{namespace}' is not loaded in this context")
        return tuple(str(c) for c in self._data[ns].columns)

    def series(self, full_field: str) -> pd.Series:
        """Return a point-in-time Series for a namespaced field."""

        field = get_field(full_field)
        if field.namespace not in self._data:
            raise ValueError(f"namespace '{field.namespace}' is not loaded in this context")

        frame = self._data[field.namespace]
        column = field.source_column or field.name
        if column not in frame.columns:
            raise ValueError(
                f"field '{full_field}' maps to missing column '{column}' "
                f"in namespace '{field.namespace}'"
            )

        return pd.to_numeric(frame.loc[: self.timestamp, column], errors="coerce")

    def window(self, full_field: str, window: int) -> FieldWindow:
        """Return the last ``window`` observations for a namespaced field.

        The returned slice includes observations up to and including
        ``self.timestamp`` and never includes future rows.
        """

        n = int(window)
        if n <= 0:
            raise ValueError("window must be positive")

        field = get_field(full_field)
        values = self.series(full_field).tail(n)

        return FieldWindow(
            field=field,
            timestamp=self.timestamp,
            index=values.index,
            values=values.to_numpy(dtype=float),
        )

    def latest(self, full_field: str) -> float:
        """Return the latest available field value at or before timestamp."""

        values = self.window(full_field, 1).values
        if values.size == 0:
            return float("nan")
        return float(values[-1])

    def history_length(self, full_field: str) -> int:
        """Return number of observations available up to timestamp."""

        return int(self.series(full_field).shape[0])
