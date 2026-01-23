"""
IMU Data Interpolation Module

Converts burst-delivered MAVLink IMU data to a regular time grid
using the flight controller timestamps as ground truth.

This is essential for:
- VQF/Madgwick filter input (requires regular sampling)
- Cross-correlation with visual data
- Accurate temporal calibration

The key insight: MAVLink delivers data in bursts, but the FC
timestamps (time_usec) represent true sensor sample times.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy import interpolate
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import IMUData
from core.ring_buffer import TimestampedRingBuffer


@dataclass
class InterpolatorConfig:
    """Configuration for IMU interpolation."""
    # Target output rate
    target_rate_hz: float = 200.0

    # Interpolation method: 'linear', 'cubic', 'nearest'
    method: str = 'linear'

    # Minimum samples required for interpolation
    min_samples: int = 10

    # Maximum gap to interpolate across (seconds)
    max_gap: float = 0.1

    # Use FC timestamps (True) or arrival timestamps (False)
    use_fc_timestamps: bool = True


class IMUInterpolator:
    """
    Interpolate IMU data to regular time grid.

    Takes burst-delivered IMU samples and produces evenly-spaced
    samples at the target rate.

    Usage:
        interp = IMUInterpolator(config)

        # Add samples as they arrive
        interp.add_sample(imu_data)

        # Get interpolated samples
        regular_samples = interp.get_interpolated(duration=1.0)
    """

    def __init__(self, config: Optional[InterpolatorConfig] = None):
        self.config = config or InterpolatorConfig()
        self._samples: List[IMUData] = []
        self._lock = threading.Lock()
        self._max_samples = int(self.config.target_rate_hz * 60)  # 1 minute buffer

    def add_sample(self, imu: IMUData) -> None:
        """Add an IMU sample to the buffer."""
        with self._lock:
            self._samples.append(imu)
            # Keep buffer bounded
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]

    def add_samples(self, samples: List[IMUData]) -> None:
        """Add multiple IMU samples."""
        with self._lock:
            self._samples.extend(samples)
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]

    def clear(self) -> None:
        """Clear sample buffer."""
        with self._lock:
            self._samples = []

    def get_interpolated(self,
                         duration: Optional[float] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         clock_offset: float = 0.0
                         ) -> List[IMUData]:
        """
        Get interpolated IMU samples on regular time grid.

        Args:
            duration: Duration to interpolate (from end of data)
            start_time: Start time (alternative to duration)
            end_time: End time (alternative to duration)
            clock_offset: Offset to apply to FC timestamps (FC_time - local_time)

        Returns:
            List of IMUData samples on regular time grid
        """
        with self._lock:
            samples = list(self._samples)

        if len(samples) < self.config.min_samples:
            return []

        # Extract timestamps
        if self.config.use_fc_timestamps:
            # Use FC timestamps (converted to seconds, with offset)
            times = np.array([(s.fc_timestamp / 1e6) - clock_offset for s in samples])
        else:
            # Use arrival timestamps
            times = np.array([s.timestamp for s in samples])

        # Determine time range
        if start_time is not None and end_time is not None:
            t_start, t_end = start_time, end_time
        elif duration is not None:
            t_end = times[-1]
            t_start = t_end - duration
        else:
            t_start, t_end = times[0], times[-1]

        # Filter samples within range (with some margin)
        margin = 0.1
        mask = (times >= t_start - margin) & (times <= t_end + margin)
        if np.sum(mask) < self.config.min_samples:
            return []

        times = times[mask]
        samples = [s for s, m in zip(samples, mask) if m]

        # Check for gaps
        intervals = np.diff(times)
        if np.any(intervals > self.config.max_gap):
            print(f"Warning: Gap of {np.max(intervals):.3f}s detected in IMU data")

        # Extract data channels
        p = np.array([s.p for s in samples])
        q = np.array([s.q for s in samples])
        r = np.array([s.r for s in samples])
        ax = np.array([s.ax for s in samples])
        ay = np.array([s.ay for s in samples])
        az = np.array([s.az for s in samples])

        # Create regular time grid
        dt = 1.0 / self.config.target_rate_hz
        grid_times = np.arange(max(t_start, times[0]), min(t_end, times[-1]), dt)

        if len(grid_times) < 2:
            return []

        # Interpolate each channel
        if self.config.method == 'linear':
            p_interp = np.interp(grid_times, times, p)
            q_interp = np.interp(grid_times, times, q)
            r_interp = np.interp(grid_times, times, r)
            ax_interp = np.interp(grid_times, times, ax)
            ay_interp = np.interp(grid_times, times, ay)
            az_interp = np.interp(grid_times, times, az)

        elif self.config.method == 'cubic':
            # Cubic spline interpolation
            p_func = interpolate.interp1d(times, p, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
            q_func = interpolate.interp1d(times, q, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
            r_func = interpolate.interp1d(times, r, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
            ax_func = interpolate.interp1d(times, ax, kind='cubic',
                                           bounds_error=False, fill_value='extrapolate')
            ay_func = interpolate.interp1d(times, ay, kind='cubic',
                                           bounds_error=False, fill_value='extrapolate')
            az_func = interpolate.interp1d(times, az, kind='cubic',
                                           bounds_error=False, fill_value='extrapolate')

            p_interp = p_func(grid_times)
            q_interp = q_func(grid_times)
            r_interp = r_func(grid_times)
            ax_interp = ax_func(grid_times)
            ay_interp = ay_func(grid_times)
            az_interp = az_func(grid_times)

        elif self.config.method == 'nearest':
            p_func = interpolate.interp1d(times, p, kind='nearest',
                                          bounds_error=False, fill_value='extrapolate')
            q_func = interpolate.interp1d(times, q, kind='nearest',
                                          bounds_error=False, fill_value='extrapolate')
            r_func = interpolate.interp1d(times, r, kind='nearest',
                                          bounds_error=False, fill_value='extrapolate')
            ax_func = interpolate.interp1d(times, ax, kind='nearest',
                                           bounds_error=False, fill_value='extrapolate')
            ay_func = interpolate.interp1d(times, ay, kind='nearest',
                                           bounds_error=False, fill_value='extrapolate')
            az_func = interpolate.interp1d(times, az, kind='nearest',
                                           bounds_error=False, fill_value='extrapolate')

            p_interp = p_func(grid_times)
            q_interp = q_func(grid_times)
            r_interp = r_func(grid_times)
            ax_interp = ax_func(grid_times)
            ay_interp = ay_func(grid_times)
            az_interp = az_func(grid_times)

        else:
            raise ValueError(f"Unknown interpolation method: {self.config.method}")

        # Create output samples
        result = []
        for i, t in enumerate(grid_times):
            imu = IMUData(
                timestamp=t,
                fc_timestamp=int((t + clock_offset) * 1e6),
                p=float(p_interp[i]),
                q=float(q_interp[i]),
                r=float(r_interp[i]),
                ax=float(ax_interp[i]),
                ay=float(ay_interp[i]),
                az=float(az_interp[i])
            )
            result.append(imu)

        return result

    def get_interpolated_arrays(self,
                                duration: float,
                                clock_offset: float = 0.0
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get interpolated data as numpy arrays (more efficient).

        Args:
            duration: Duration to interpolate
            clock_offset: FC clock offset

        Returns:
            Tuple of (times, gyro[N,3], accel[N,3])
        """
        samples = self.get_interpolated(duration=duration, clock_offset=clock_offset)

        if not samples:
            return np.array([]), np.array([]), np.array([])

        times = np.array([s.timestamp for s in samples])
        gyro = np.array([[s.p, s.q, s.r] for s in samples])
        accel = np.array([[s.ax, s.ay, s.az] for s in samples])

        return times, gyro, accel

    @property
    def sample_count(self) -> int:
        """Number of samples in buffer."""
        return len(self._samples)

    @property
    def time_range(self) -> Optional[Tuple[float, float]]:
        """Time range of buffered samples (using FC timestamps)."""
        with self._lock:
            if len(self._samples) < 2:
                return None

            if self.config.use_fc_timestamps:
                t_start = self._samples[0].fc_timestamp / 1e6
                t_end = self._samples[-1].fc_timestamp / 1e6
            else:
                t_start = self._samples[0].timestamp
                t_end = self._samples[-1].timestamp

            return (t_start, t_end)

    def get_actual_rate(self) -> float:
        """Calculate actual sample rate from FC timestamps."""
        with self._lock:
            if len(self._samples) < 10:
                return 0.0

            if self.config.use_fc_timestamps:
                times = [s.fc_timestamp / 1e6 for s in self._samples[-100:]]
            else:
                times = [s.timestamp for s in self._samples[-100:]]

            duration = times[-1] - times[0]
            if duration > 0:
                return (len(times) - 1) / duration
            return 0.0


class RealTimeInterpolator:
    """
    Real-time interpolation for streaming applications.

    Produces interpolated samples with minimal latency as new
    data arrives.
    """

    def __init__(self, config: Optional[InterpolatorConfig] = None):
        self.config = config or InterpolatorConfig()
        self._buffer_size = int(self.config.target_rate_hz * 0.5)  # 500ms buffer
        self._samples: List[IMUData] = []
        self._last_output_time: Optional[float] = None
        self._lock = threading.Lock()

        # Output callback
        self._callback = None

    def set_callback(self, callback) -> None:
        """Set callback for interpolated samples."""
        self._callback = callback

    def on_sample(self, imu: IMUData, clock_offset: float = 0.0) -> List[IMUData]:
        """
        Process incoming sample and emit interpolated samples.

        Args:
            imu: New IMU sample
            clock_offset: FC clock offset

        Returns:
            List of interpolated samples (may be empty)
        """
        with self._lock:
            self._samples.append(imu)

            # Keep buffer bounded
            if len(self._samples) > self._buffer_size:
                self._samples.pop(0)

            if len(self._samples) < 3:
                return []

            # Get FC time of new sample
            if self.config.use_fc_timestamps:
                new_time = (imu.fc_timestamp / 1e6) - clock_offset
            else:
                new_time = imu.timestamp

            # Initialize last output time
            if self._last_output_time is None:
                # Start one sample period back
                dt = 1.0 / self.config.target_rate_hz
                prev_time = (self._samples[-2].fc_timestamp / 1e6) - clock_offset
                self._last_output_time = prev_time - dt

            # Generate samples up to (but not past) the new sample
            dt = 1.0 / self.config.target_rate_hz
            output_samples = []

            next_output = self._last_output_time + dt

            while next_output < new_time - dt:  # Leave margin
                # Linear interpolation between surrounding samples
                sample = self._interpolate_at(next_output, clock_offset)
                if sample is not None:
                    output_samples.append(sample)
                    self._last_output_time = next_output
                next_output += dt

            # Fire callback
            if self._callback and output_samples:
                for s in output_samples:
                    self._callback(s)

            return output_samples

    def _interpolate_at(self, t: float, clock_offset: float) -> Optional[IMUData]:
        """Interpolate a single sample at time t."""
        if len(self._samples) < 2:
            return None

        # Find surrounding samples
        if self.config.use_fc_timestamps:
            times = [(s.fc_timestamp / 1e6) - clock_offset for s in self._samples]
        else:
            times = [s.timestamp for s in self._samples]

        # Find bracket
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                # Linear interpolation
                t0, t1 = times[i], times[i + 1]
                if t1 == t0:
                    alpha = 0.0
                else:
                    alpha = (t - t0) / (t1 - t0)

                s0, s1 = self._samples[i], self._samples[i + 1]

                return IMUData(
                    timestamp=t,
                    fc_timestamp=int((t + clock_offset) * 1e6),
                    p=s0.p + alpha * (s1.p - s0.p),
                    q=s0.q + alpha * (s1.q - s0.q),
                    r=s0.r + alpha * (s1.r - s0.r),
                    ax=s0.ax + alpha * (s1.ax - s0.ax),
                    ay=s0.ay + alpha * (s1.ay - s0.ay),
                    az=s0.az + alpha * (s1.az - s0.az)
                )

        return None


def interpolate_to_timestamps(samples: List[IMUData],
                               target_times: np.ndarray,
                               clock_offset: float = 0.0) -> List[IMUData]:
    """
    Interpolate IMU samples to specific target timestamps.

    Useful for aligning IMU data to camera frame timestamps.

    Args:
        samples: List of IMU samples
        target_times: Array of target timestamps (local time)
        clock_offset: FC clock offset

    Returns:
        List of interpolated IMUData at target times
    """
    if len(samples) < 2:
        return []

    # Extract source times and data
    src_times = np.array([(s.fc_timestamp / 1e6) - clock_offset for s in samples])
    p = np.array([s.p for s in samples])
    q = np.array([s.q for s in samples])
    r = np.array([s.r for s in samples])
    ax = np.array([s.ax for s in samples])
    ay = np.array([s.ay for s in samples])
    az = np.array([s.az for s in samples])

    # Filter target times to valid range
    valid_mask = (target_times >= src_times[0]) & (target_times <= src_times[-1])
    valid_times = target_times[valid_mask]

    if len(valid_times) == 0:
        return []

    # Interpolate
    result = []
    for t in valid_times:
        imu = IMUData(
            timestamp=t,
            fc_timestamp=int((t + clock_offset) * 1e6),
            p=float(np.interp(t, src_times, p)),
            q=float(np.interp(t, src_times, q)),
            r=float(np.interp(t, src_times, r)),
            ax=float(np.interp(t, src_times, ax)),
            ay=float(np.interp(t, src_times, ay)),
            az=float(np.interp(t, src_times, az))
        )
        result.append(imu)

    return result
