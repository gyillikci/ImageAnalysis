"""
High-Rate MAVLink Streaming Module

Provides tools for requesting specific messages at high rates
using SET_MESSAGE_INTERVAL command (ArduPilot 4.0+).

This bypasses the stream group mechanism for finer control
over individual message rates.

References:
- https://ardupilot.org/dev/docs/mavlink-requesting-data.html
- https://github.com/ArduPilot/ardupilot/issues/3630
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable
from enum import IntEnum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import monotonic_time


class MAVLinkMessageID(IntEnum):
    """Common MAVLink message IDs for IMU data."""
    HEARTBEAT = 0
    SYS_STATUS = 1
    SYSTEM_TIME = 2
    GPS_RAW_INT = 24
    SCALED_IMU = 26
    RAW_IMU = 27
    SCALED_PRESSURE = 29
    ATTITUDE = 30
    ATTITUDE_QUATERNION = 31
    GLOBAL_POSITION_INT = 33
    RC_CHANNELS = 65
    VFR_HUD = 74
    COMMAND_LONG = 76
    COMMAND_ACK = 77
    HIGHRES_IMU = 105
    ATTITUDE_TARGET = 83
    SCALED_IMU2 = 116
    SCALED_IMU3 = 129
    TIMESYNC = 111
    VIBRATION = 241


class MAVCommand(IntEnum):
    """MAVLink command IDs."""
    SET_MESSAGE_INTERVAL = 511
    GET_MESSAGE_INTERVAL = 510
    REQUEST_MESSAGE = 512


@dataclass
class StreamConfig:
    """Configuration for a message stream."""
    message_id: int
    rate_hz: float
    enabled: bool = True


@dataclass
class StreamStatus:
    """Status of a message stream."""
    message_id: int
    requested_rate_hz: float
    actual_rate_hz: float
    message_count: int
    last_received: float


class HighRateStreamer:
    """
    Request and manage high-rate message streams.

    Uses SET_MESSAGE_INTERVAL command for precise rate control
    of individual messages.

    Usage:
        streamer = HighRateStreamer()
        streamer.set_connection(mavlink_connection)

        # Request HIGHRES_IMU at 200Hz
        streamer.request_message_rate(MAVLinkMessageID.HIGHRES_IMU, 200)

        # Or request RAW_IMU at 100Hz
        streamer.request_message_rate(MAVLinkMessageID.RAW_IMU, 100)
    """

    def __init__(self):
        self._connection = None
        self._streams: Dict[int, StreamConfig] = {}
        self._status: Dict[int, StreamStatus] = {}
        self._lock = threading.Lock()

        # Rate measurement
        self._message_times: Dict[int, List[float]] = {}
        self._rate_window = 1.0  # seconds for rate calculation

    def set_connection(self, connection) -> None:
        """Set MAVLink connection."""
        self._connection = connection

    def request_message_rate(self, message_id: int, rate_hz: float) -> bool:
        """
        Request a specific message at a given rate.

        Uses MAV_CMD_SET_MESSAGE_INTERVAL (511).

        Args:
            message_id: MAVLink message ID
            rate_hz: Desired rate in Hz (0 = disable)

        Returns:
            True if command was sent successfully
        """
        if not self._connection:
            print("No MAVLink connection")
            return False

        # Calculate interval in microseconds
        # -1 = default rate, 0 = disable, >0 = interval in us
        if rate_hz <= 0:
            interval_us = 0  # Disable
        else:
            interval_us = int(1e6 / rate_hz)

        try:
            # Send COMMAND_LONG with SET_MESSAGE_INTERVAL
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                MAVCommand.SET_MESSAGE_INTERVAL,  # command
                0,  # confirmation
                message_id,  # param1: message ID
                interval_us,  # param2: interval in microseconds
                0,  # param3: response target (0 = flight stack default)
                0, 0, 0, 0  # param4-7: unused
            )

            # Store configuration
            with self._lock:
                self._streams[message_id] = StreamConfig(
                    message_id=message_id,
                    rate_hz=rate_hz,
                    enabled=rate_hz > 0
                )
                self._status[message_id] = StreamStatus(
                    message_id=message_id,
                    requested_rate_hz=rate_hz,
                    actual_rate_hz=0.0,
                    message_count=0,
                    last_received=0.0
                )
                self._message_times[message_id] = []

            print(f"Requested message {message_id} at {rate_hz} Hz "
                  f"(interval: {interval_us} us)")
            return True

        except Exception as e:
            print(f"Failed to request message rate: {e}")
            return False

    def request_imu_streams(self, rate_hz: float = 200) -> None:
        """
        Request all IMU-related streams at specified rate.

        Args:
            rate_hz: Desired rate for IMU messages
        """
        imu_messages = [
            MAVLinkMessageID.RAW_IMU,
            MAVLinkMessageID.SCALED_IMU,
            MAVLinkMessageID.SCALED_IMU2,
            MAVLinkMessageID.HIGHRES_IMU,
        ]

        for msg_id in imu_messages:
            self.request_message_rate(msg_id, rate_hz)
            time.sleep(0.1)  # Small delay between requests

    def request_attitude_stream(self, rate_hz: float = 100) -> None:
        """Request attitude messages at specified rate."""
        self.request_message_rate(MAVLinkMessageID.ATTITUDE, rate_hz)
        time.sleep(0.1)
        self.request_message_rate(MAVLinkMessageID.ATTITUDE_QUATERNION, rate_hz)

    def disable_message(self, message_id: int) -> bool:
        """Disable a message stream."""
        return self.request_message_rate(message_id, 0)

    def on_message_received(self, message_id: int) -> None:
        """
        Call this when a message is received to track actual rates.

        Args:
            message_id: ID of received message
        """
        now = monotonic_time()

        with self._lock:
            if message_id not in self._status:
                return

            # Update status
            self._status[message_id].message_count += 1
            self._status[message_id].last_received = now

            # Track times for rate calculation
            times = self._message_times.get(message_id, [])
            times.append(now)

            # Remove old entries outside window
            cutoff = now - self._rate_window
            times = [t for t in times if t > cutoff]
            self._message_times[message_id] = times

            # Calculate actual rate
            if len(times) >= 2:
                duration = times[-1] - times[0]
                if duration > 0:
                    self._status[message_id].actual_rate_hz = (len(times) - 1) / duration

    def get_message_interval(self, message_id: int) -> Optional[int]:
        """
        Query current message interval from flight controller.

        Args:
            message_id: Message ID to query

        Returns:
            Interval in microseconds, or None if query failed
        """
        if not self._connection:
            return None

        try:
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                MAVCommand.GET_MESSAGE_INTERVAL,
                0,
                message_id,
                0, 0, 0, 0, 0, 0
            )
            # Note: Response comes via MESSAGE_INTERVAL message
            return None  # Would need async handling
        except Exception as e:
            print(f"Failed to query message interval: {e}")
            return None

    def request_single_message(self, message_id: int) -> bool:
        """
        Request a single instance of a message.

        Useful for getting current values without setting up a stream.

        Args:
            message_id: Message ID to request

        Returns:
            True if request was sent
        """
        if not self._connection:
            return False

        try:
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                MAVCommand.REQUEST_MESSAGE,
                0,
                message_id,
                0, 0, 0, 0, 0, 0
            )
            return True
        except Exception as e:
            print(f"Failed to request message: {e}")
            return False

    def get_stream_status(self, message_id: int) -> Optional[StreamStatus]:
        """Get status for a specific stream."""
        with self._lock:
            return self._status.get(message_id)

    def get_all_status(self) -> Dict[int, StreamStatus]:
        """Get status for all streams."""
        with self._lock:
            return dict(self._status)

    def print_status(self) -> None:
        """Print status of all streams."""
        print("\nStream Status:")
        print("-" * 60)
        with self._lock:
            for msg_id, status in sorted(self._status.items()):
                name = MAVLinkMessageID(msg_id).name if msg_id in MAVLinkMessageID._value2member_map_ else f"MSG_{msg_id}"
                print(f"  {name}: requested={status.requested_rate_hz:.0f}Hz, "
                      f"actual={status.actual_rate_hz:.1f}Hz, "
                      f"count={status.message_count}")
        print("-" * 60)


class StreamRateMonitor:
    """
    Monitor actual message rates and detect issues.

    Detects:
    - Rate drops below requested
    - Burst patterns (high jitter)
    - Lost messages
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._message_times: Dict[int, List[float]] = {}
        self._intervals: Dict[int, List[float]] = {}
        self._lock = threading.Lock()

    def on_message(self, message_id: int, timestamp: float) -> None:
        """Record message arrival time."""
        with self._lock:
            if message_id not in self._message_times:
                self._message_times[message_id] = []
                self._intervals[message_id] = []

            times = self._message_times[message_id]
            intervals = self._intervals[message_id]

            # Calculate interval from previous
            if times:
                interval = timestamp - times[-1]
                intervals.append(interval)
                if len(intervals) > self._window_size:
                    intervals.pop(0)

            times.append(timestamp)
            if len(times) > self._window_size:
                times.pop(0)

    def get_statistics(self, message_id: int) -> Optional[dict]:
        """
        Get timing statistics for a message.

        Returns:
            Dictionary with rate, jitter, min/max intervals
        """
        with self._lock:
            intervals = self._intervals.get(message_id)
            if not intervals or len(intervals) < 2:
                return None

            import numpy as np
            arr = np.array(intervals)

            return {
                'rate_hz': 1.0 / np.mean(arr) if np.mean(arr) > 0 else 0,
                'mean_interval_ms': np.mean(arr) * 1000,
                'std_interval_ms': np.std(arr) * 1000,
                'min_interval_ms': np.min(arr) * 1000,
                'max_interval_ms': np.max(arr) * 1000,
                'jitter_ms': (np.max(arr) - np.min(arr)) * 1000,
                'samples': len(arr)
            }

    def detect_bursts(self, message_id: int, threshold_factor: float = 3.0) -> List[int]:
        """
        Detect burst patterns in message timing.

        Args:
            message_id: Message to analyze
            threshold_factor: Intervals > mean * factor are considered bursts

        Returns:
            List of indices where bursts were detected
        """
        with self._lock:
            intervals = self._intervals.get(message_id)
            if not intervals or len(intervals) < 10:
                return []

            import numpy as np
            arr = np.array(intervals)
            mean_interval = np.mean(arr)
            threshold = mean_interval * threshold_factor

            burst_indices = np.where(arr > threshold)[0].tolist()
            return burst_indices

    def print_report(self) -> None:
        """Print timing report for all messages."""
        print("\nMessage Timing Report:")
        print("=" * 70)

        with self._lock:
            for msg_id in sorted(self._message_times.keys()):
                stats = self.get_statistics(msg_id)
                if not stats:
                    continue

                name = MAVLinkMessageID(msg_id).name if msg_id in MAVLinkMessageID._value2member_map_ else f"MSG_{msg_id}"
                bursts = self.detect_bursts(msg_id)

                print(f"\n{name} (ID {msg_id}):")
                print(f"  Rate: {stats['rate_hz']:.1f} Hz")
                print(f"  Interval: {stats['mean_interval_ms']:.2f} ms (std: {stats['std_interval_ms']:.2f} ms)")
                print(f"  Range: [{stats['min_interval_ms']:.2f}, {stats['max_interval_ms']:.2f}] ms")
                print(f"  Jitter: {stats['jitter_ms']:.2f} ms")
                if bursts:
                    print(f"  Bursts detected: {len(bursts)} events")

        print("\n" + "=" * 70)
