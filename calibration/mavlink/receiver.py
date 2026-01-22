"""
MAVLink receiver for real-time data from Orange Cube flight controller.

Supports UDP and serial connections.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

# Core imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import IMUData, AttitudeData, GPSData
from core.ring_buffer import TimestampedRingBuffer
from core.utils import monotonic_time, RateEstimator


class ConnectionType(Enum):
    UDP = "udp"
    SERIAL = "serial"
    TCP = "tcp"


@dataclass
class MAVLinkConfig:
    """MAVLink connection configuration."""
    connection_string: str = "udp:127.0.0.1:14550"
    source_system: int = 255
    source_component: int = 0
    request_data_streams: bool = True
    stream_rate_hz: int = 100
    buffer_size_imu: int = 10000
    buffer_size_attitude: int = 5000
    buffer_size_gps: int = 1000

    @classmethod
    def from_string(cls, conn_str: str) -> 'MAVLinkConfig':
        """Create config from connection string."""
        return cls(connection_string=conn_str)


class MAVLinkReceiver:
    """
    Real-time MAVLink data receiver.

    Runs in a background thread and populates ring buffers with
    IMU, attitude, and GPS data.

    Usage:
        receiver = MAVLinkReceiver(config)
        receiver.start()

        # Get latest data
        imu = receiver.imu_buffer.peek_newest()
        att = receiver.attitude_buffer.peek_newest()

        # Stop when done
        receiver.stop()
    """

    def __init__(self, config: MAVLinkConfig):
        """
        Initialize MAVLink receiver.

        Args:
            config: MAVLink configuration
        """
        self.config = config
        self._connection = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Data buffers
        self.imu_buffer = TimestampedRingBuffer[IMUData](config.buffer_size_imu)
        self.attitude_buffer = TimestampedRingBuffer[AttitudeData](config.buffer_size_attitude)
        self.gps_buffer = TimestampedRingBuffer[GPSData](config.buffer_size_gps)

        # Rate estimators
        self._imu_rate = RateEstimator(window_size=100)
        self._attitude_rate = RateEstimator(window_size=50)
        self._gps_rate = RateEstimator(window_size=20)

        # Callbacks
        self._callbacks: Dict[str, list] = {
            'imu': [],
            'attitude': [],
            'gps': [],
            'connected': [],
            'disconnected': []
        }

        # Status
        self._connected = False
        self._last_heartbeat = 0.0
        self._message_counts: Dict[str, int] = {}

    def add_callback(self, event: str, callback: Callable) -> None:
        """
        Add callback for specific event.

        Events: 'imu', 'attitude', 'gps', 'connected', 'disconnected'
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _fire_callbacks(self, event: str, data: Any = None) -> None:
        """Fire callbacks for event."""
        for callback in self._callbacks.get(event, []):
            try:
                if data is not None:
                    callback(data)
                else:
                    callback()
            except Exception as e:
                print(f"Callback error for {event}: {e}")

    def start(self) -> bool:
        """
        Start receiver thread.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            self._connect()
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Failed to start MAVLink receiver: {e}")
            return False

    def stop(self) -> None:
        """Stop receiver thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._disconnect()

    def _connect(self) -> None:
        """Establish MAVLink connection."""
        try:
            from pymavlink import mavutil
        except ImportError:
            raise ImportError("pymavlink is required. Install with: pip install pymavlink")

        conn_str = self.config.connection_string

        print(f"Connecting to MAVLink: {conn_str}")

        # Parse connection string
        if conn_str.startswith("udp:"):
            # Format: udp:host:port or udpin:host:port
            self._connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self.config.source_system,
                source_component=self.config.source_component
            )
        elif conn_str.startswith("serial:") or conn_str.startswith("/dev/"):
            # Format: serial:/dev/ttyUSB0:57600 or /dev/ttyUSB0
            if conn_str.startswith("serial:"):
                parts = conn_str[7:].split(":")
                device = parts[0]
                baud = int(parts[1]) if len(parts) > 1 else 57600
            else:
                device = conn_str.split(":")[0]
                baud = 57600
            self._connection = mavutil.mavlink_connection(
                device,
                baud=baud,
                source_system=self.config.source_system,
                source_component=self.config.source_component
            )
        elif conn_str.startswith("tcp:"):
            self._connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self.config.source_system,
                source_component=self.config.source_component
            )
        else:
            # Try direct connection string
            self._connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self.config.source_system,
                source_component=self.config.source_component
            )

        # Wait for heartbeat
        print("Waiting for heartbeat...")
        self._connection.wait_heartbeat(timeout=10)
        print(f"Heartbeat received from system {self._connection.target_system}")

        self._connected = True
        self._last_heartbeat = monotonic_time()
        self._fire_callbacks('connected')

        # Request data streams
        if self.config.request_data_streams:
            self._request_data_streams()

    def _disconnect(self) -> None:
        """Close MAVLink connection."""
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

        if self._connected:
            self._connected = False
            self._fire_callbacks('disconnected')

    def _request_data_streams(self) -> None:
        """Request data streams from flight controller."""
        if not self._connection:
            return

        # Request all data streams
        # MAV_DATA_STREAM_RAW_SENSORS = 1
        # MAV_DATA_STREAM_EXTENDED_STATUS = 2
        # MAV_DATA_STREAM_RC_CHANNELS = 3
        # MAV_DATA_STREAM_RAW_CONTROLLER = 4
        # MAV_DATA_STREAM_POSITION = 6
        # MAV_DATA_STREAM_EXTRA1 = 10 (attitude)
        # MAV_DATA_STREAM_EXTRA2 = 11
        # MAV_DATA_STREAM_EXTRA3 = 12

        rate = self.config.stream_rate_hz

        # Request raw sensors (IMU)
        self._connection.mav.request_data_stream_send(
            self._connection.target_system,
            self._connection.target_component,
            1,  # MAV_DATA_STREAM_RAW_SENSORS
            rate,
            1  # start
        )

        # Request attitude
        self._connection.mav.request_data_stream_send(
            self._connection.target_system,
            self._connection.target_component,
            10,  # MAV_DATA_STREAM_EXTRA1 (attitude)
            rate,
            1
        )

        # Request position (GPS)
        self._connection.mav.request_data_stream_send(
            self._connection.target_system,
            self._connection.target_component,
            6,  # MAV_DATA_STREAM_POSITION
            10,  # GPS at lower rate
            1
        )

        print(f"Requested data streams at {rate} Hz")

    def _receive_loop(self) -> None:
        """Main receive loop (runs in thread)."""
        while self._running:
            try:
                if not self._connection:
                    time.sleep(0.1)
                    continue

                # Receive message with timeout
                msg = self._connection.recv_match(blocking=True, timeout=0.1)

                if msg is None:
                    continue

                # Get local timestamp immediately
                local_ts = monotonic_time()
                msg_type = msg.get_type()

                # Track message counts
                self._message_counts[msg_type] = self._message_counts.get(msg_type, 0) + 1

                # Process message based on type
                if msg_type == 'HEARTBEAT':
                    self._last_heartbeat = local_ts

                elif msg_type == 'RAW_IMU':
                    self._handle_raw_imu(msg, local_ts)

                elif msg_type == 'SCALED_IMU' or msg_type == 'SCALED_IMU2':
                    self._handle_scaled_imu(msg, local_ts)

                elif msg_type == 'HIGHRES_IMU':
                    self._handle_highres_imu(msg, local_ts)

                elif msg_type == 'ATTITUDE':
                    self._handle_attitude(msg, local_ts)

                elif msg_type == 'ATTITUDE_QUATERNION':
                    self._handle_attitude_quaternion(msg, local_ts)

                elif msg_type == 'GPS_RAW_INT':
                    self._handle_gps(msg, local_ts)

                elif msg_type == 'GLOBAL_POSITION_INT':
                    pass  # Could add global position handling

            except Exception as e:
                if self._running:
                    print(f"MAVLink receive error: {e}")
                time.sleep(0.01)

    def _handle_raw_imu(self, msg, local_ts: float) -> None:
        """Handle RAW_IMU message."""
        imu = IMUData(
            timestamp=local_ts,
            fc_timestamp=msg.time_usec,
            p=msg.xgyro / 1000.0,  # mrad/s to rad/s
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0 * 9.81,  # mG to m/s²
            ay=msg.yacc / 1000.0 * 9.81,
            az=msg.zacc / 1000.0 * 9.81
        )

        self.imu_buffer.push(imu)
        self._imu_rate.update(local_ts)
        self._fire_callbacks('imu', imu)

    def _handle_scaled_imu(self, msg, local_ts: float) -> None:
        """Handle SCALED_IMU message."""
        imu = IMUData(
            timestamp=local_ts,
            fc_timestamp=msg.time_boot_ms * 1000,  # ms to us
            p=msg.xgyro / 1000.0,  # mrad/s to rad/s
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0,  # mG to G, then to m/s² (approx)
            ay=msg.yacc / 1000.0,
            az=msg.zacc / 1000.0
        )

        self.imu_buffer.push(imu)
        self._imu_rate.update(local_ts)
        self._fire_callbacks('imu', imu)

    def _handle_highres_imu(self, msg, local_ts: float) -> None:
        """Handle HIGHRES_IMU message (high-rate IMU)."""
        imu = IMUData(
            timestamp=local_ts,
            fc_timestamp=msg.time_usec,
            p=msg.xgyro,  # Already in rad/s
            q=msg.ygyro,
            r=msg.zgyro,
            ax=msg.xacc,  # Already in m/s²
            ay=msg.yacc,
            az=msg.zacc
        )

        self.imu_buffer.push(imu)
        self._imu_rate.update(local_ts)
        self._fire_callbacks('imu', imu)

    def _handle_attitude(self, msg, local_ts: float) -> None:
        """Handle ATTITUDE message."""
        att = AttitudeData(
            timestamp=local_ts,
            fc_timestamp=msg.time_boot_ms * 1000,  # ms to us
            roll=msg.roll,
            pitch=msg.pitch,
            yaw=msg.yaw,
            rollspeed=msg.rollspeed,
            pitchspeed=msg.pitchspeed,
            yawspeed=msg.yawspeed
        )

        self.attitude_buffer.push(att)
        self._attitude_rate.update(local_ts)
        self._fire_callbacks('attitude', att)

    def _handle_attitude_quaternion(self, msg, local_ts: float) -> None:
        """Handle ATTITUDE_QUATERNION message."""
        from core.utils import quaternion_to_euler

        q = [msg.q1, msg.q2, msg.q3, msg.q4]  # w, x, y, z
        roll, pitch, yaw = quaternion_to_euler(q)

        att = AttitudeData(
            timestamp=local_ts,
            fc_timestamp=msg.time_boot_ms * 1000,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            rollspeed=msg.rollspeed,
            pitchspeed=msg.pitchspeed,
            yawspeed=msg.yawspeed
        )

        self.attitude_buffer.push(att)
        self._attitude_rate.update(local_ts)
        self._fire_callbacks('attitude', att)

    def _handle_gps(self, msg, local_ts: float) -> None:
        """Handle GPS_RAW_INT message."""
        gps = GPSData(
            timestamp=local_ts,
            fc_timestamp=msg.time_usec,
            lat=msg.lat / 1e7,
            lon=msg.lon / 1e7,
            alt=msg.alt / 1000.0,  # mm to m
            vel=getattr(msg, 'vel', 0) / 100.0,  # cm/s to m/s
            hdg=getattr(msg, 'cog', 0) / 100.0,  # cdeg to deg
            satellites=getattr(msg, 'satellites_visible', 0),
            fix_type=msg.fix_type
        )

        self.gps_buffer.push(gps)
        self._gps_rate.update(local_ts)
        self._fire_callbacks('gps', gps)

    @property
    def is_connected(self) -> bool:
        """Check if connected and receiving heartbeats."""
        if not self._connected:
            return False
        # Check for recent heartbeat (within 3 seconds)
        return (monotonic_time() - self._last_heartbeat) < 3.0

    @property
    def imu_rate(self) -> float:
        """Current IMU message rate (Hz)."""
        return self._imu_rate.rate

    @property
    def attitude_rate(self) -> float:
        """Current attitude message rate (Hz)."""
        return self._attitude_rate.rate

    @property
    def gps_rate(self) -> float:
        """Current GPS message rate (Hz)."""
        return self._gps_rate.rate

    @property
    def message_counts(self) -> Dict[str, int]:
        """Get message counts by type."""
        return dict(self._message_counts)

    def get_status(self) -> dict:
        """Get current receiver status."""
        return {
            'connected': self.is_connected,
            'imu_rate': self.imu_rate,
            'attitude_rate': self.attitude_rate,
            'gps_rate': self.gps_rate,
            'imu_buffer_fill': self.imu_buffer.fill_ratio,
            'attitude_buffer_fill': self.attitude_buffer.fill_ratio,
            'gps_buffer_fill': self.gps_buffer.fill_ratio,
            'message_counts': self.message_counts
        }
