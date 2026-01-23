"""
Enhanced MAVLink Receiver with Time Synchronization and High-Rate Streaming

This module provides a production-ready MAVLink receiver that:
1. Synchronizes clocks using TIMESYNC protocol
2. Uses FC timestamps (not arrival times) for accurate timing
3. Supports high-rate IMU streaming via SET_MESSAGE_INTERVAL
4. Provides interpolated IMU data on regular time grid
5. Monitors message rates and detects burst patterns

Usage:
    receiver = EnhancedMAVLinkReceiver(config)
    receiver.start()

    # Get clock-corrected IMU data
    imu = receiver.get_latest_imu()
    local_time = receiver.fc_to_local(imu.fc_timestamp)

    # Get interpolated data for VQF
    times, gyro, accel = receiver.get_interpolated_imu(duration=1.0)
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import IMUData, AttitudeData, GPSData
from core.ring_buffer import TimestampedRingBuffer
from core.utils import monotonic_time, RateEstimator
from core.interpolator import IMUInterpolator, InterpolatorConfig, RealTimeInterpolator

from mavlink.timesync import TimeSynchronizer, TimeSyncConfig, SystemTimeTracker
from mavlink.streaming import HighRateStreamer, StreamRateMonitor, MAVLinkMessageID


@dataclass
class EnhancedMAVLinkConfig:
    """Enhanced MAVLink receiver configuration."""
    # Connection
    connection_string: str = "udp:127.0.0.1:14550"
    source_system: int = 255
    source_component: int = 0

    # High-rate streaming
    imu_rate_hz: float = 200.0  # Target IMU rate
    attitude_rate_hz: float = 100.0  # Target attitude rate
    use_highres_imu: bool = True  # Prefer HIGHRES_IMU if available

    # Time synchronization
    enable_timesync: bool = True
    timesync_interval: float = 1.0

    # Interpolation
    enable_interpolation: bool = True
    interpolation_rate_hz: float = 200.0
    interpolation_method: str = 'linear'

    # Buffers
    buffer_size_imu: int = 20000
    buffer_size_attitude: int = 5000
    buffer_size_gps: int = 1000

    # Monitoring
    enable_rate_monitoring: bool = True


class EnhancedMAVLinkReceiver:
    """
    Enhanced MAVLink receiver with time sync and high-rate streaming.
    """

    def __init__(self, config: Optional[EnhancedMAVLinkConfig] = None):
        self.config = config or EnhancedMAVLinkConfig()
        self._connection = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Data buffers (store with FC timestamps)
        self.imu_buffer = TimestampedRingBuffer[IMUData](self.config.buffer_size_imu)
        self.attitude_buffer = TimestampedRingBuffer[AttitudeData](self.config.buffer_size_attitude)
        self.gps_buffer = TimestampedRingBuffer[GPSData](self.config.buffer_size_gps)

        # Time synchronization
        self._timesync = TimeSynchronizer(TimeSyncConfig(
            sync_interval=self.config.timesync_interval
        ))
        self._system_time = SystemTimeTracker()

        # High-rate streaming
        self._streamer = HighRateStreamer()

        # Rate monitoring
        self._rate_monitor = StreamRateMonitor()
        self._imu_rate = RateEstimator(window_size=100)
        self._attitude_rate = RateEstimator(window_size=50)

        # Interpolation
        if self.config.enable_interpolation:
            self._interpolator = IMUInterpolator(InterpolatorConfig(
                target_rate_hz=self.config.interpolation_rate_hz,
                method=self.config.interpolation_method,
                use_fc_timestamps=True
            ))
            self._realtime_interpolator = RealTimeInterpolator(InterpolatorConfig(
                target_rate_hz=self.config.interpolation_rate_hz,
                use_fc_timestamps=True
            ))
        else:
            self._interpolator = None
            self._realtime_interpolator = None

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'imu': [],
            'imu_interpolated': [],
            'attitude': [],
            'gps': [],
            'connected': [],
            'disconnected': [],
            'timesync': []
        }

        # Status
        self._connected = False
        self._last_heartbeat = 0.0
        self._message_counts: Dict[str, int] = {}
        self._highres_imu_available = False

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add callback for event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def add_imu_callback(self, callback: Callable[[IMUData], None]) -> None:
        """Convenience method to add IMU callback."""
        self.add_callback('imu', callback)

    def _fire_callbacks(self, event: str, data: Any = None) -> None:
        """Fire callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                if data is not None:
                    cb(data)
                else:
                    cb()
            except Exception as e:
                print(f"Callback error ({event}): {e}")

    def start(self) -> bool:
        """Start receiver."""
        if self._running:
            return True

        try:
            self._connect()
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()

            # Start time synchronization
            if self.config.enable_timesync:
                self._timesync.set_connection(self._connection)
                self._timesync.start()

            return True
        except Exception as e:
            print(f"Failed to start enhanced receiver: {e}")
            return False

    def stop(self) -> None:
        """Stop receiver."""
        self._running = False

        if self.config.enable_timesync:
            self._timesync.stop()

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._disconnect()

    def _connect(self) -> None:
        """Establish connection."""
        try:
            from pymavlink import mavutil
        except ImportError:
            raise ImportError("pymavlink required")

        conn_str = self.config.connection_string
        print(f"Connecting to: {conn_str}")

        self._connection = mavutil.mavlink_connection(
            conn_str,
            source_system=self.config.source_system,
            source_component=self.config.source_component
        )

        print("Waiting for heartbeat...")
        self._connection.wait_heartbeat(timeout=10)
        print(f"Connected to system {self._connection.target_system}")

        self._connected = True
        self._last_heartbeat = monotonic_time()

        # Set up streaming
        self._streamer.set_connection(self._connection)
        self._request_streams()

        self._fire_callbacks('connected')

    def _disconnect(self) -> None:
        """Close connection."""
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

        if self._connected:
            self._connected = False
            self._fire_callbacks('disconnected')

    def _request_streams(self) -> None:
        """Request high-rate data streams."""
        # Try HIGHRES_IMU first
        if self.config.use_highres_imu:
            self._streamer.request_message_rate(
                MAVLinkMessageID.HIGHRES_IMU,
                self.config.imu_rate_hz
            )

        # Also request RAW_IMU as fallback
        self._streamer.request_message_rate(
            MAVLinkMessageID.RAW_IMU,
            self.config.imu_rate_hz
        )

        # Request attitude
        self._streamer.request_message_rate(
            MAVLinkMessageID.ATTITUDE,
            self.config.attitude_rate_hz
        )

        # Request TIMESYNC responses
        self._streamer.request_message_rate(
            MAVLinkMessageID.TIMESYNC,
            10  # 10 Hz for timesync
        )

        print(f"Requested IMU at {self.config.imu_rate_hz} Hz")

    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                if not self._connection:
                    time.sleep(0.1)
                    continue

                msg = self._connection.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue

                local_ts = monotonic_time()
                msg_type = msg.get_type()

                # Track counts
                self._message_counts[msg_type] = self._message_counts.get(msg_type, 0) + 1

                # Update rate monitor
                if self.config.enable_rate_monitoring:
                    msg_id = getattr(msg, '_header', {}).get('msgId', 0) if hasattr(msg, '_header') else 0
                    self._rate_monitor.on_message(msg_id, local_ts)
                    self._streamer.on_message_received(msg_id)

                # Process by type
                if msg_type == 'HEARTBEAT':
                    self._last_heartbeat = local_ts

                elif msg_type == 'TIMESYNC':
                    self._timesync.handle_timesync_response(msg)
                    self._fire_callbacks('timesync', self._timesync.get_status())

                elif msg_type == 'SYSTEM_TIME':
                    self._system_time.handle_system_time(msg, local_ts)

                elif msg_type == 'HIGHRES_IMU':
                    self._highres_imu_available = True
                    self._handle_highres_imu(msg, local_ts)

                elif msg_type == 'RAW_IMU':
                    # Skip if HIGHRES_IMU is available
                    if not self._highres_imu_available:
                        self._handle_raw_imu(msg, local_ts)

                elif msg_type == 'SCALED_IMU' or msg_type == 'SCALED_IMU2':
                    if not self._highres_imu_available:
                        self._handle_scaled_imu(msg, local_ts)

                elif msg_type == 'ATTITUDE':
                    self._handle_attitude(msg, local_ts)

                elif msg_type == 'GPS_RAW_INT':
                    self._handle_gps(msg, local_ts)

            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")
                time.sleep(0.01)

    def _handle_highres_imu(self, msg, local_ts: float) -> None:
        """Handle HIGHRES_IMU (best quality)."""
        # Use FC timestamp for accurate timing
        imu = IMUData(
            timestamp=self.fc_to_local(msg.time_usec),  # Convert to local time
            fc_timestamp=msg.time_usec,
            p=msg.xgyro,  # Already in rad/s
            q=msg.ygyro,
            r=msg.zgyro,
            ax=msg.xacc,  # Already in m/s²
            ay=msg.yacc,
            az=msg.zacc
        )

        self._process_imu(imu, local_ts)

    def _handle_raw_imu(self, msg, local_ts: float) -> None:
        """Handle RAW_IMU."""
        imu = IMUData(
            timestamp=self.fc_to_local(msg.time_usec),
            fc_timestamp=msg.time_usec,
            p=msg.xgyro / 1000.0,  # mrad/s to rad/s
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0 * 9.81,  # mG to m/s²
            ay=msg.yacc / 1000.0 * 9.81,
            az=msg.zacc / 1000.0 * 9.81
        )

        self._process_imu(imu, local_ts)

    def _handle_scaled_imu(self, msg, local_ts: float) -> None:
        """Handle SCALED_IMU."""
        fc_timestamp = msg.time_boot_ms * 1000  # ms to us

        imu = IMUData(
            timestamp=self.fc_to_local(fc_timestamp),
            fc_timestamp=fc_timestamp,
            p=msg.xgyro / 1000.0,
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0,
            ay=msg.yacc / 1000.0,
            az=msg.zacc / 1000.0
        )

        self._process_imu(imu, local_ts)

    def _process_imu(self, imu: IMUData, arrival_time: float) -> None:
        """Process IMU sample (common path)."""
        # Add to buffer
        self.imu_buffer.push(imu)
        self._imu_rate.update(arrival_time)

        # Add to interpolator
        if self._interpolator:
            self._interpolator.add_sample(imu)

        # Real-time interpolation
        if self._realtime_interpolator:
            offset = self._timesync.offset if self.config.enable_timesync else 0.0
            interp_samples = self._realtime_interpolator.on_sample(imu, offset)
            for s in interp_samples:
                self._fire_callbacks('imu_interpolated', s)

        # Fire callback
        self._fire_callbacks('imu', imu)

    def _handle_attitude(self, msg, local_ts: float) -> None:
        """Handle ATTITUDE."""
        fc_timestamp = msg.time_boot_ms * 1000

        att = AttitudeData(
            timestamp=self.fc_to_local(fc_timestamp),
            fc_timestamp=fc_timestamp,
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

    def _handle_gps(self, msg, local_ts: float) -> None:
        """Handle GPS."""
        gps = GPSData(
            timestamp=self.fc_to_local(msg.time_usec),
            fc_timestamp=msg.time_usec,
            lat=msg.lat / 1e7,
            lon=msg.lon / 1e7,
            alt=msg.alt / 1000.0,
            vel=getattr(msg, 'vel', 0) / 100.0,
            hdg=getattr(msg, 'cog', 0) / 100.0,
            satellites=getattr(msg, 'satellites_visible', 0),
            fix_type=msg.fix_type
        )

        self.gps_buffer.push(gps)
        self._fire_callbacks('gps', gps)

    # === Public API ===

    def fc_to_local(self, fc_timestamp_usec: int) -> float:
        """Convert FC timestamp to local time."""
        if self.config.enable_timesync and self._timesync.is_converged:
            return self._timesync.fc_to_local(fc_timestamp_usec)
        else:
            # Fallback: use arrival time approximation
            return fc_timestamp_usec / 1e6

    def local_to_fc(self, local_time: float) -> int:
        """Convert local time to FC timestamp."""
        if self.config.enable_timesync:
            return self._timesync.local_to_fc(local_time)
        else:
            return int(local_time * 1e6)

    def get_interpolated_imu(self, duration: float = 1.0
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get interpolated IMU data on regular time grid.

        Args:
            duration: Duration in seconds

        Returns:
            Tuple of (times[N], gyro[N,3], accel[N,3])
        """
        if not self._interpolator:
            return np.array([]), np.array([]), np.array([])

        offset = self._timesync.offset if self.config.enable_timesync else 0.0
        return self._interpolator.get_interpolated_arrays(duration, offset)

    def get_latest_imu(self) -> Optional[IMUData]:
        """Get most recent IMU sample."""
        return self.imu_buffer.peek_newest()

    def get_latest_attitude(self) -> Optional[AttitudeData]:
        """Get most recent attitude."""
        return self.attitude_buffer.peek_newest()

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        if not self._connected:
            return False
        return (monotonic_time() - self._last_heartbeat) < 3.0

    @property
    def is_time_synced(self) -> bool:
        """Check if time synchronization has converged."""
        return self._timesync.is_converged if self.config.enable_timesync else False

    @property
    def clock_offset(self) -> float:
        """Get current clock offset (FC_time - local_time)."""
        return self._timesync.offset if self.config.enable_timesync else 0.0

    @property
    def imu_rate(self) -> float:
        """Current IMU message rate."""
        return self._imu_rate.rate

    @property
    def attitude_rate(self) -> float:
        """Current attitude message rate."""
        return self._attitude_rate.rate

    def get_status(self) -> dict:
        """Get comprehensive status."""
        status = {
            'connected': self.is_connected,
            'time_synced': self.is_time_synced,
            'clock_offset_ms': self.clock_offset * 1000,
            'imu_rate': self.imu_rate,
            'attitude_rate': self.attitude_rate,
            'highres_imu_available': self._highres_imu_available,
            'imu_buffer_fill': self.imu_buffer.fill_ratio,
            'message_counts': dict(self._message_counts)
        }

        if self.config.enable_timesync:
            status['timesync'] = self._timesync.get_status()

        if self._interpolator:
            status['interpolator'] = {
                'sample_count': self._interpolator.sample_count,
                'actual_rate': self._interpolator.get_actual_rate()
            }

        return status

    def print_timing_report(self) -> None:
        """Print detailed timing report."""
        if self.config.enable_rate_monitoring:
            self._rate_monitor.print_report()
        self._streamer.print_status()
