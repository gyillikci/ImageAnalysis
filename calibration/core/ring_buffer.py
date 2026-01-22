"""
Thread-safe ring buffers for time-series data.
"""

import threading
import numpy as np
from typing import TypeVar, Generic, Optional, List, Callable
from collections import deque
from dataclasses import dataclass

T = TypeVar('T')


class RingBuffer(Generic[T]):
    """
    Thread-safe generic ring buffer.

    Supports push, pop, peek operations with optional timestamp-based queries.
    """

    def __init__(self, maxsize: int):
        """
        Initialize ring buffer.

        Args:
            maxsize: Maximum number of elements
        """
        self._buffer: deque = deque(maxlen=maxsize)
        self._lock = threading.RLock()
        self._maxsize = maxsize

    def push(self, item: T) -> None:
        """Add item to buffer (oldest removed if full)."""
        with self._lock:
            self._buffer.append(item)

    def pop(self) -> Optional[T]:
        """Remove and return oldest item."""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer.popleft()
            return None

    def peek(self) -> Optional[T]:
        """Return oldest item without removing."""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[0]
            return None

    def peek_newest(self) -> Optional[T]:
        """Return newest item without removing."""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[-1]
            return None

    def get_all(self) -> List[T]:
        """Return copy of all items (oldest first)."""
        with self._lock:
            return list(self._buffer)

    def get_last_n(self, n: int) -> List[T]:
        """Return last n items (oldest first among the n)."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def fill_ratio(self) -> float:
        """Return fill ratio [0, 1]."""
        with self._lock:
            return len(self._buffer) / self._maxsize

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self._maxsize

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0


class TimestampedRingBuffer(RingBuffer[T]):
    """
    Ring buffer with timestamp-based queries.

    Items must have a 'timestamp' attribute.
    """

    def __init__(self, maxsize: int, timestamp_attr: str = 'timestamp'):
        """
        Initialize timestamped ring buffer.

        Args:
            maxsize: Maximum number of elements
            timestamp_attr: Name of timestamp attribute on items
        """
        super().__init__(maxsize)
        self._timestamp_attr = timestamp_attr

    def _get_timestamp(self, item: T) -> float:
        """Extract timestamp from item."""
        return getattr(item, self._timestamp_attr)

    def get_time_range(self) -> tuple:
        """Return (oldest_timestamp, newest_timestamp) or (None, None) if empty."""
        with self._lock:
            if len(self._buffer) == 0:
                return (None, None)
            oldest = self._get_timestamp(self._buffer[0])
            newest = self._get_timestamp(self._buffer[-1])
            return (oldest, newest)

    def get_in_time_range(self, t_start: float, t_end: float) -> List[T]:
        """Return items within time range [t_start, t_end]."""
        with self._lock:
            result = []
            for item in self._buffer:
                t = self._get_timestamp(item)
                if t_start <= t <= t_end:
                    result.append(item)
            return result

    def get_nearest(self, timestamp: float) -> Optional[T]:
        """Return item nearest to given timestamp."""
        with self._lock:
            if len(self._buffer) == 0:
                return None

            best_item = None
            best_diff = float('inf')

            for item in self._buffer:
                t = self._get_timestamp(item)
                diff = abs(t - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_item = item

            return best_item

    def get_interpolated(self, timestamp: float,
                         interpolator: Callable[[T, T, float], T]) -> Optional[T]:
        """
        Return interpolated item at given timestamp.

        Args:
            timestamp: Target timestamp
            interpolator: Function(item1, item2, alpha) -> interpolated_item
                         where alpha is interpolation factor [0, 1]

        Returns:
            Interpolated item or None if not possible
        """
        with self._lock:
            if len(self._buffer) < 2:
                return None

            # Find bracketing items
            prev_item = None
            next_item = None

            for item in self._buffer:
                t = self._get_timestamp(item)
                if t <= timestamp:
                    prev_item = item
                elif t > timestamp and next_item is None:
                    next_item = item
                    break

            if prev_item is None or next_item is None:
                return self.get_nearest(timestamp)

            t1 = self._get_timestamp(prev_item)
            t2 = self._get_timestamp(next_item)

            if t2 == t1:
                return prev_item

            alpha = (timestamp - t1) / (t2 - t1)
            return interpolator(prev_item, next_item, alpha)

    def get_window(self, center_time: float, window_size: float) -> List[T]:
        """Return items within window_size seconds of center_time."""
        half_window = window_size / 2.0
        return self.get_in_time_range(center_time - half_window,
                                       center_time + half_window)

    def get_duration(self) -> float:
        """Return time span of data in buffer (seconds)."""
        t_start, t_end = self.get_time_range()
        if t_start is None or t_end is None:
            return 0.0
        return t_end - t_start


class SyncedBufferPair:
    """
    Pair of timestamped buffers with synchronization helpers.

    Useful for correlating two time series (e.g., IMU and visual motion).
    """

    def __init__(self, maxsize_a: int, maxsize_b: int,
                 timestamp_attr: str = 'timestamp'):
        """
        Initialize synced buffer pair.

        Args:
            maxsize_a: Max size of buffer A
            maxsize_b: Max size of buffer B
            timestamp_attr: Timestamp attribute name
        """
        self.buffer_a = TimestampedRingBuffer[any](maxsize_a, timestamp_attr)
        self.buffer_b = TimestampedRingBuffer[any](maxsize_b, timestamp_attr)
        self._lock = threading.RLock()

    def push_a(self, item) -> None:
        """Add item to buffer A."""
        self.buffer_a.push(item)

    def push_b(self, item) -> None:
        """Add item to buffer B."""
        self.buffer_b.push(item)

    def get_overlap_range(self) -> tuple:
        """Return time range where both buffers have data."""
        with self._lock:
            t_start_a, t_end_a = self.buffer_a.get_time_range()
            t_start_b, t_end_b = self.buffer_b.get_time_range()

            if None in (t_start_a, t_end_a, t_start_b, t_end_b):
                return (None, None)

            t_start = max(t_start_a, t_start_b)
            t_end = min(t_end_a, t_end_b)

            if t_start >= t_end:
                return (None, None)

            return (t_start, t_end)

    def get_aligned_samples(self, time_offset: float = 0.0,
                            sample_rate: float = 100.0) -> tuple:
        """
        Get time-aligned samples from both buffers.

        Args:
            time_offset: Offset to apply to buffer B timestamps
                        (t_a = t_b + offset)
            sample_rate: Desired sample rate for output

        Returns:
            (timestamps, samples_a, samples_b) or (None, None, None)
        """
        t_start, t_end = self.get_overlap_range()

        if t_start is None:
            return (None, None, None)

        # Adjust for time offset
        t_start = max(t_start, t_start - time_offset)
        t_end = min(t_end, t_end - time_offset)

        if t_end <= t_start:
            return (None, None, None)

        # Generate sample timestamps
        num_samples = int((t_end - t_start) * sample_rate)
        if num_samples < 2:
            return (None, None, None)

        timestamps = np.linspace(t_start, t_end, num_samples)

        # Get samples from buffer A
        samples_a = self.buffer_a.get_in_time_range(t_start, t_end)

        # Get samples from buffer B (with offset)
        samples_b = self.buffer_b.get_in_time_range(
            t_start - time_offset, t_end - time_offset)

        return (timestamps, samples_a, samples_b)
