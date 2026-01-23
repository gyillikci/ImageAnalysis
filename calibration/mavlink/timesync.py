"""
MAVLink Time Synchronization Module

Implements the TIMESYNC protocol for synchronizing clocks between
the companion computer and flight controller.

This is critical for:
- Converting FC timestamps to local time
- Compensating for MAVLink burst delivery
- Accurate temporal calibration

References:
- https://mavlink.io/en/services/timesync.html
- https://ardupilot.org/dev/docs/ros-timesync.html
"""

import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import monotonic_time


@dataclass
class TimeSyncConfig:
    """Configuration for time synchronization."""
    # Sync interval
    sync_interval: float = 1.0  # seconds between TIMESYNC requests

    # Filter parameters
    filter_alpha: float = 0.1  # EMA alpha for offset smoothing
    outlier_threshold: float = 0.1  # seconds - reject outliers

    # Convergence
    min_samples: int = 10  # Minimum samples before considered stable
    convergence_threshold: float = 0.001  # seconds - offset std dev

    # History
    history_size: int = 100  # Keep last N offset estimates


@dataclass
class TimeSyncState:
    """Time synchronization state."""
    # Current estimates
    offset: float = 0.0  # FC_time - local_time (seconds)
    round_trip_time: float = 0.0  # RTT estimate

    # Statistics
    offset_std: float = float('inf')
    num_samples: int = 0
    converged: bool = False

    # History
    offset_history: List[float] = field(default_factory=list)
    rtt_history: List[float] = field(default_factory=list)


class ExponentialMovingAverage:
    """Exponential moving average filter."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class TimeSynchronizer:
    """
    MAVLink time synchronization using TIMESYNC protocol.

    Establishes and maintains clock offset between companion computer
    and flight controller using NTP-like algorithm.

    Usage:
        sync = TimeSynchronizer(config)
        sync.set_connection(mavlink_connection)
        sync.start()

        # Convert FC timestamp to local time
        local_time = sync.fc_to_local(fc_timestamp_usec)

        # Or convert local to FC time
        fc_time = sync.local_to_fc(local_time)
    """

    def __init__(self, config: Optional[TimeSyncConfig] = None):
        self.config = config or TimeSyncConfig()
        self._connection = None
        self._state = TimeSyncState()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Filters
        self._offset_filter = ExponentialMovingAverage(self.config.filter_alpha)
        self._rtt_filter = ExponentialMovingAverage(self.config.filter_alpha)

        # Pending request tracking
        self._pending_ts1: Optional[int] = None
        self._pending_local_time: Optional[float] = None

        # Callbacks
        self._callbacks: List[Callable[[TimeSyncState], None]] = []

    def set_connection(self, connection) -> None:
        """Set MAVLink connection."""
        self._connection = connection

    def add_callback(self, callback: Callable[[TimeSyncState], None]) -> None:
        """Add callback for sync updates."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start synchronization thread."""
        if self._running:
            return

        if self._connection is None:
            raise RuntimeError("MAVLink connection not set")

        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        print("Time synchronizer started")

    def stop(self) -> None:
        """Stop synchronization thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Time synchronizer stopped")

    def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self._running:
            try:
                self._send_timesync_request()
                time.sleep(self.config.sync_interval)
            except Exception as e:
                print(f"Timesync error: {e}")
                time.sleep(1.0)

    def _send_timesync_request(self) -> None:
        """Send TIMESYNC request to flight controller."""
        if not self._connection:
            return

        # Get current local time in nanoseconds
        local_time = monotonic_time()
        ts1 = int(local_time * 1e9)  # Convert to nanoseconds

        # Store for response matching
        self._pending_ts1 = ts1
        self._pending_local_time = local_time

        # Send TIMESYNC request (tc1=0 indicates request)
        try:
            self._connection.mav.timesync_send(
                tc1=0,  # 0 = request
                ts1=ts1
            )
        except Exception as e:
            print(f"Failed to send TIMESYNC: {e}")

    def handle_timesync_response(self, msg) -> None:
        """
        Handle TIMESYNC response from flight controller.

        Args:
            msg: TIMESYNC MAVLink message
        """
        # Get local receive time immediately
        t4 = monotonic_time()

        # Check if this is a response (tc1 != 0)
        if msg.tc1 == 0:
            # This is a request from FC, not a response
            return

        # Verify this matches our pending request
        if self._pending_ts1 is None or msg.ts1 != self._pending_ts1:
            return

        with self._lock:
            # NTP-like offset calculation
            # t1 = local send time (stored as pending_local_time)
            # t2 = FC receive time (msg.tc1 in nanoseconds)
            # t3 = FC send time (approximately same as t2 for quick response)
            # t4 = local receive time

            t1 = self._pending_local_time
            t2 = msg.tc1 / 1e9  # Convert from nanoseconds
            t3 = t2  # Assume FC responds immediately

            # Round-trip time
            rtt = (t4 - t1) - (t3 - t2)

            # Clock offset: FC_time - local_time
            # offset = ((t2 - t1) + (t3 - t4)) / 2
            offset = ((t2 - t1) + (t3 - t4)) / 2

            # Reject outliers
            if self._state.num_samples > 5:
                if abs(offset - self._state.offset) > self.config.outlier_threshold:
                    print(f"Timesync outlier rejected: {offset:.6f}s (current: {self._state.offset:.6f}s)")
                    return

            # Update filters
            filtered_offset = self._offset_filter.update(offset)
            filtered_rtt = self._rtt_filter.update(rtt)

            # Update state
            self._state.offset = filtered_offset
            self._state.round_trip_time = filtered_rtt
            self._state.num_samples += 1

            # Update history
            self._state.offset_history.append(offset)
            self._state.rtt_history.append(rtt)
            if len(self._state.offset_history) > self.config.history_size:
                self._state.offset_history.pop(0)
                self._state.rtt_history.pop(0)

            # Check convergence
            if len(self._state.offset_history) >= self.config.min_samples:
                self._state.offset_std = np.std(self._state.offset_history[-self.config.min_samples:])
                self._state.converged = self._state.offset_std < self.config.convergence_threshold

            # Fire callbacks
            for cb in self._callbacks:
                try:
                    cb(self._state)
                except Exception as e:
                    print(f"Timesync callback error: {e}")

        # Clear pending
        self._pending_ts1 = None
        self._pending_local_time = None

    def fc_to_local(self, fc_timestamp_usec: int) -> float:
        """
        Convert flight controller timestamp to local time.

        Args:
            fc_timestamp_usec: FC timestamp in microseconds (time_usec field)

        Returns:
            Local monotonic time in seconds
        """
        fc_time_sec = fc_timestamp_usec / 1e6
        return fc_time_sec - self._state.offset

    def local_to_fc(self, local_time: float) -> int:
        """
        Convert local time to flight controller timestamp.

        Args:
            local_time: Local monotonic time in seconds

        Returns:
            FC timestamp in microseconds
        """
        fc_time_sec = local_time + self._state.offset
        return int(fc_time_sec * 1e6)

    def fc_boot_ms_to_local(self, time_boot_ms: int) -> float:
        """
        Convert time_boot_ms to local time.

        Note: time_boot_ms wraps at ~49 days, so this requires
        knowing the FC boot time or using time_usec instead.

        Args:
            time_boot_ms: FC boot time in milliseconds

        Returns:
            Local monotonic time in seconds
        """
        # Convert to seconds and apply offset
        fc_time_sec = time_boot_ms / 1000.0
        return fc_time_sec - self._state.offset

    @property
    def offset(self) -> float:
        """Current clock offset (FC_time - local_time)."""
        return self._state.offset

    @property
    def rtt(self) -> float:
        """Current round-trip time estimate."""
        return self._state.round_trip_time

    @property
    def is_converged(self) -> bool:
        """Whether time sync has converged."""
        return self._state.converged

    @property
    def offset_std(self) -> float:
        """Standard deviation of offset estimates."""
        return self._state.offset_std

    def get_status(self) -> dict:
        """Get synchronization status."""
        return {
            'offset_ms': self._state.offset * 1000,
            'rtt_ms': self._state.round_trip_time * 1000,
            'offset_std_ms': self._state.offset_std * 1000,
            'num_samples': self._state.num_samples,
            'converged': self._state.converged
        }

    def reset(self) -> None:
        """Reset synchronization state."""
        with self._lock:
            self._state = TimeSyncState()
            self._offset_filter.reset()
            self._rtt_filter.reset()


class SystemTimeTracker:
    """
    Track SYSTEM_TIME messages for additional time reference.

    SYSTEM_TIME provides:
    - time_unix_usec: UNIX epoch time (if GPS locked)
    - time_boot_ms: Time since FC boot
    """

    def __init__(self):
        self._boot_time_offset: Optional[float] = None  # boot_ms to local
        self._unix_time_offset: Optional[float] = None  # unix_usec to local
        self._last_update = 0.0

    def handle_system_time(self, msg, local_time: float) -> None:
        """
        Handle SYSTEM_TIME message.

        Args:
            msg: SYSTEM_TIME MAVLink message
            local_time: Local receive time
        """
        # Boot time offset
        boot_sec = msg.time_boot_ms / 1000.0
        self._boot_time_offset = boot_sec - local_time

        # UNIX time offset (if valid)
        if msg.time_unix_usec > 0:
            unix_sec = msg.time_unix_usec / 1e6
            self._unix_time_offset = unix_sec - local_time

        self._last_update = local_time

    def boot_ms_to_local(self, time_boot_ms: int) -> Optional[float]:
        """Convert time_boot_ms to local time."""
        if self._boot_time_offset is None:
            return None
        boot_sec = time_boot_ms / 1000.0
        return boot_sec - self._boot_time_offset

    def unix_usec_to_local(self, time_unix_usec: int) -> Optional[float]:
        """Convert UNIX timestamp to local time."""
        if self._unix_time_offset is None:
            return None
        unix_sec = time_unix_usec / 1e6
        return unix_sec - self._unix_time_offset
