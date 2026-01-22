"""
Temporal synchronization module for camera-IMU calibration.

This module performs cross-correlation between IMU angular rates
and visual motion estimates to find the time offset between the
two data streams.
"""

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from scipy import signal, interpolate

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import IMUData, MotionEstimate, SyncResult
from core.ring_buffer import TimestampedRingBuffer
from core.utils import monotonic_time


@dataclass
class TemporalSyncConfig:
    """Configuration for temporal synchronization."""
    # Correlation parameters
    sample_rate: float = 60.0  # Hz for resampling
    min_window_duration: float = 5.0  # Minimum seconds of data needed
    max_window_duration: float = 30.0  # Maximum window for correlation

    # Filter parameters
    use_butterworth: bool = True
    butterworth_order: int = 2
    butterworth_cutoff: float = 10.0  # Hz

    # Convergence parameters
    min_correlation: float = 0.5  # Minimum correlation coefficient
    convergence_threshold: float = 0.01  # Seconds - offset must stabilize within this
    convergence_samples: int = 5  # Number of consistent estimates needed

    # Search range
    max_time_offset: float = 2.0  # Maximum expected offset in seconds

    # Update rate
    update_interval: float = 1.0  # Seconds between sync attempts

    # Axis selection for correlation
    use_axis: str = 'p'  # 'p', 'q', 'r', or 'all'


@dataclass
class TemporalSyncState:
    """Internal state for temporal synchronization."""
    time_offset: float = 0.0
    correlation: float = 0.0
    converged: bool = False
    num_estimates: int = 0
    recent_offsets: List[float] = field(default_factory=list)
    last_update: float = 0.0


class TemporalSynchronizer:
    """
    Real-time temporal synchronization between IMU and visual motion.

    Uses cross-correlation to find the time offset between IMU data
    and visual motion estimates from the camera.

    Usage:
        sync = TemporalSynchronizer(config)
        sync.start()

        # Feed data from callbacks
        imu_receiver.add_callback(sync.on_imu_data)
        motion_estimator.add_callback(sync.on_motion_data)

        # Check sync status
        if sync.is_converged:
            offset = sync.time_offset
    """

    def __init__(self, config: TemporalSyncConfig):
        """
        Initialize temporal synchronizer.

        Args:
            config: Synchronization configuration
        """
        self.config = config

        # Data buffers
        buffer_size = int(config.max_window_duration * config.sample_rate * 2)
        self._imu_buffer = TimestampedRingBuffer[IMUData](buffer_size)
        self._motion_buffer = TimestampedRingBuffer[MotionEstimate](buffer_size)

        # State
        self._state = TemporalSyncState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Callbacks
        self._callbacks: List[Callable[[SyncResult], None]] = []

        # Results buffer
        self._results_buffer = TimestampedRingBuffer[SyncResult](100)

    def add_callback(self, callback: Callable[[SyncResult], None]) -> None:
        """Add callback for sync results."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, result: SyncResult) -> None:
        """Fire callbacks with sync result."""
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                print(f"Sync callback error: {e}")

    def on_imu_data(self, imu: IMUData) -> None:
        """Callback for IMU data."""
        self._imu_buffer.push(imu)

    def on_motion_data(self, motion: MotionEstimate) -> None:
        """Callback for visual motion data."""
        if motion.valid:
            self._motion_buffer.push(motion)

    def start(self) -> None:
        """Start synchronization thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        print("Temporal synchronizer started")

    def stop(self) -> None:
        """Stop synchronization thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Temporal synchronizer stopped")

    def _sync_loop(self) -> None:
        """Main synchronization loop."""
        import time

        while self._running:
            now = monotonic_time()

            # Check if we should update
            if now - self._state.last_update >= self.config.update_interval:
                self._attempt_sync()
                self._state.last_update = now

            time.sleep(0.1)

    def _attempt_sync(self) -> None:
        """Attempt to compute time synchronization."""
        with self._lock:
            # Get data from buffers
            imu_data = self._imu_buffer.get_all()
            motion_data = self._motion_buffer.get_all()

            if not imu_data or not motion_data:
                return

            # Check if we have enough data
            imu_duration = imu_data[-1].timestamp - imu_data[0].timestamp
            motion_duration = motion_data[-1].timestamp - motion_data[0].timestamp

            min_duration = self.config.min_window_duration
            if imu_duration < min_duration or motion_duration < min_duration:
                return

            # Compute correlation
            try:
                offset, corr = self._compute_correlation(imu_data, motion_data)

                if corr < self.config.min_correlation:
                    return

                # Update state
                self._state.time_offset = offset
                self._state.correlation = corr
                self._state.num_estimates += 1

                # Track recent offsets for convergence
                self._state.recent_offsets.append(offset)
                if len(self._state.recent_offsets) > self.config.convergence_samples:
                    self._state.recent_offsets.pop(0)

                # Check convergence
                if len(self._state.recent_offsets) >= self.config.convergence_samples:
                    offsets = np.array(self._state.recent_offsets)
                    offset_std = np.std(offsets)
                    if offset_std < self.config.convergence_threshold:
                        self._state.converged = True

                # Create result
                result = SyncResult(
                    timestamp=monotonic_time(),
                    time_offset=offset,
                    correlation=corr,
                    converged=self._state.converged,
                    num_samples=len(imu_data)
                )

                self._results_buffer.push(result)
                self._fire_callbacks(result)

                if self._state.converged:
                    print(f"Temporal sync converged: offset={offset:.4f}s, corr={corr:.3f}")
                else:
                    print(f"Temporal sync: offset={offset:.4f}s, corr={corr:.3f}")

            except Exception as e:
                print(f"Correlation error: {e}")

    def _compute_correlation(self, imu_data: List[IMUData],
                             motion_data: List[MotionEstimate]) -> Tuple[float, float]:
        """
        Compute cross-correlation between IMU and visual motion.

        Args:
            imu_data: List of IMU samples
            motion_data: List of visual motion estimates

        Returns:
            Tuple of (time_offset, correlation_coefficient)
        """
        hz = self.config.sample_rate

        # Find overlap time range
        imu_t_min = imu_data[0].timestamp
        imu_t_max = imu_data[-1].timestamp
        motion_t_min = motion_data[0].timestamp
        motion_t_max = motion_data[-1].timestamp

        # Resample IMU data
        imu_times = np.array([d.timestamp for d in imu_data])

        if self.config.use_axis == 'p':
            imu_values = np.array([d.p for d in imu_data])
            motion_values = np.array([d.p for d in motion_data])
        elif self.config.use_axis == 'q':
            imu_values = np.array([d.q for d in imu_data])
            motion_values = np.array([d.q for d in motion_data])
        elif self.config.use_axis == 'r':
            imu_values = np.array([d.r for d in imu_data])
            motion_values = np.array([d.r for d in motion_data])
        else:  # 'all' - use magnitude
            imu_p = np.array([d.p for d in imu_data])
            imu_q = np.array([d.q for d in imu_data])
            imu_r = np.array([d.r for d in imu_data])
            imu_values = np.sqrt(imu_p**2 + imu_q**2 + imu_r**2)

            motion_p = np.array([d.p for d in motion_data])
            motion_q = np.array([d.q for d in motion_data])
            motion_r = np.array([d.r for d in motion_data])
            motion_values = np.sqrt(motion_p**2 + motion_q**2 + motion_r**2)

        motion_times = np.array([d.timestamp for d in motion_data])

        # Create interpolation functions
        imu_interp = interpolate.interp1d(
            imu_times, imu_values,
            bounds_error=False, fill_value=0.0
        )
        motion_interp = interpolate.interp1d(
            motion_times, motion_values,
            bounds_error=False, fill_value=0.0
        )

        # Resample to common time base
        imu_duration = imu_t_max - imu_t_min
        motion_duration = motion_t_max - motion_t_min

        imu_resampled = []
        for t in np.linspace(imu_t_min, imu_t_max, int(imu_duration * hz)):
            imu_resampled.append([t, imu_interp(t)])
        imu_resampled = np.array(imu_resampled)

        motion_resampled = []
        for t in np.linspace(motion_t_min, motion_t_max, int(motion_duration * hz)):
            motion_resampled.append([t, motion_interp(t)])
        motion_resampled = np.array(motion_resampled)

        # Apply Butterworth filter if enabled
        if self.config.use_butterworth:
            nyquist = hz / 2.0
            cutoff = min(self.config.butterworth_cutoff, nyquist * 0.9)
            b, a = signal.butter(
                self.config.butterworth_order,
                cutoff / nyquist
            )
            imu_filtered = signal.filtfilt(b, a, imu_resampled[:, 1])
            motion_filtered = signal.filtfilt(b, a, motion_resampled[:, 1])
        else:
            imu_filtered = imu_resampled[:, 1]
            motion_filtered = motion_resampled[:, 1]

        # Compute cross-correlation
        ycorr = np.correlate(imu_filtered, motion_filtered, mode='full')

        # Find peak
        max_index = np.argmax(ycorr)

        # Convert to time shift
        # Need to account for correlation array indexing
        shift_sec = max_index / hz - motion_duration
        start_diff = imu_resampled[0, 0] - motion_resampled[0, 0]
        time_offset = start_diff + shift_sec

        # Limit to max expected offset
        time_offset = np.clip(
            time_offset,
            -self.config.max_time_offset,
            self.config.max_time_offset
        )

        # Compute correlation coefficient
        if len(ycorr) > 0 and np.max(ycorr) > 0:
            # Normalized correlation
            corr = np.max(ycorr) / (len(motion_filtered) *
                                     np.std(imu_filtered) *
                                     np.std(motion_filtered))
            corr = min(corr, 1.0)  # Cap at 1.0
        else:
            corr = 0.0

        return time_offset, corr

    def compute_sync_now(self) -> Optional[SyncResult]:
        """
        Compute synchronization immediately with current data.

        Returns:
            SyncResult if successful, None otherwise
        """
        with self._lock:
            imu_data = self._imu_buffer.get_all()
            motion_data = self._motion_buffer.get_all()

            if not imu_data or not motion_data:
                return None

            try:
                offset, corr = self._compute_correlation(imu_data, motion_data)

                return SyncResult(
                    timestamp=monotonic_time(),
                    time_offset=offset,
                    correlation=corr,
                    converged=self._state.converged,
                    num_samples=len(imu_data)
                )
            except Exception as e:
                print(f"Sync computation error: {e}")
                return None

    @property
    def time_offset(self) -> float:
        """Current time offset estimate (camera_time - imu_time)."""
        return self._state.time_offset

    @property
    def correlation(self) -> float:
        """Current correlation coefficient."""
        return self._state.correlation

    @property
    def is_converged(self) -> bool:
        """Whether synchronization has converged."""
        return self._state.converged

    @property
    def num_estimates(self) -> int:
        """Number of offset estimates computed."""
        return self._state.num_estimates

    def reset(self) -> None:
        """Reset synchronization state."""
        with self._lock:
            self._imu_buffer.clear()
            self._motion_buffer.clear()
            self._state = TemporalSyncState()
            self._results_buffer.clear()

    def get_status(self) -> dict:
        """Get synchronization status."""
        return {
            'time_offset': self._state.time_offset,
            'correlation': self._state.correlation,
            'converged': self._state.converged,
            'num_estimates': self._state.num_estimates,
            'imu_samples': len(self._imu_buffer),
            'motion_samples': len(self._motion_buffer)
        }


def correlate_offline(imu_data: List[IMUData],
                      motion_data: List[MotionEstimate],
                      config: Optional[TemporalSyncConfig] = None) -> SyncResult:
    """
    Compute temporal synchronization offline (batch mode).

    Args:
        imu_data: List of IMU samples
        motion_data: List of visual motion estimates
        config: Optional configuration

    Returns:
        SyncResult with time offset and correlation
    """
    if config is None:
        config = TemporalSyncConfig()

    sync = TemporalSynchronizer(config)

    # Add all data
    for imu in imu_data:
        sync.on_imu_data(imu)
    for motion in motion_data:
        sync.on_motion_data(motion)

    # Compute sync
    result = sync.compute_sync_now()

    if result is None:
        return SyncResult(
            timestamp=monotonic_time(),
            time_offset=0.0,
            correlation=0.0,
            converged=False,
            num_samples=0
        )

    return result
