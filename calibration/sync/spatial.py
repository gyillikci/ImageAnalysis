"""
Spatial calibration module for camera-IMU calibration.

This module estimates the camera mount rotation (yaw, pitch, roll offsets)
relative to the IMU/body frame by comparing visual angular rates with
IMU angular rates.
"""

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from scipy.optimize import least_squares

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import IMUData, MotionEstimate, SpatialCalibration
from core.ring_buffer import TimestampedRingBuffer
from core.utils import (
    monotonic_time, euler_to_rotation_matrix, rotation_matrix_to_euler
)


@dataclass
class SpatialCalibConfig:
    """Configuration for spatial calibration."""
    # Data requirements
    min_samples: int = 100  # Minimum samples needed
    max_samples: int = 1000  # Maximum samples to use

    # Motion requirements
    min_angular_rate: float = 0.1  # rad/s - ignore low motion
    max_angular_rate: float = 3.0  # rad/s - ignore extreme motion

    # Optimizer parameters
    initial_yaw: float = 0.0  # Initial yaw offset guess (rad)
    initial_pitch: float = 0.0  # Initial pitch offset guess (rad)
    initial_roll: float = 0.0  # Initial roll offset guess (rad)

    # Convergence
    convergence_threshold: float = 0.001  # rad - change threshold
    max_iterations: int = 100

    # Update rate
    update_interval: float = 2.0  # Seconds between calibration attempts


@dataclass
class MountOffset:
    """Camera mount rotation offset."""
    yaw: float = 0.0  # rad
    pitch: float = 0.0  # rad
    roll: float = 0.0  # rad
    rmse: float = 0.0  # rad
    converged: bool = False


@dataclass
class SyncedSample:
    """Synchronized IMU and visual motion sample."""
    timestamp: float
    imu_p: float
    imu_q: float
    imu_r: float
    vis_p: float
    vis_q: float
    vis_r: float


class SpatialCalibrator:
    """
    Real-time spatial calibration between camera and IMU.

    Estimates the camera mount rotation by minimizing the difference
    between IMU angular rates and camera-derived angular rates after
    applying the mount rotation.

    Usage:
        calib = SpatialCalibrator(config)
        calib.start()

        # Feed synchronized data
        calib.add_sample(imu_data, motion_data, time_offset)

        # Check calibration status
        if calib.is_converged:
            offset = calib.mount_offset
    """

    def __init__(self, config: SpatialCalibConfig):
        """
        Initialize spatial calibrator.

        Args:
            config: Calibration configuration
        """
        self.config = config

        # Sample buffer
        self._samples: List[SyncedSample] = []
        self._max_samples = config.max_samples

        # State
        self._mount_offset = MountOffset(
            yaw=config.initial_yaw,
            pitch=config.initial_pitch,
            roll=config.initial_roll
        )
        self._rotation_matrix = np.eye(3)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Callbacks
        self._callbacks: List[Callable[[SpatialCalibration], None]] = []

        # Stats
        self._num_calibrations = 0
        self._last_update = 0.0

    def add_callback(self, callback: Callable[[SpatialCalibration], None]) -> None:
        """Add callback for calibration results."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, result: SpatialCalibration) -> None:
        """Fire callbacks with calibration result."""
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                print(f"Spatial calibration callback error: {e}")

    def add_sample(self, imu: IMUData, motion: MotionEstimate,
                   time_offset: float = 0.0) -> None:
        """
        Add a synchronized IMU and visual motion sample.

        Args:
            imu: IMU data sample
            motion: Visual motion estimate
            time_offset: Time offset to apply (from temporal calibration)
        """
        if not motion.valid:
            return

        # Check angular rate magnitude
        imu_mag = np.sqrt(imu.p**2 + imu.q**2 + imu.r**2)
        vis_mag = np.sqrt(motion.p**2 + motion.q**2 + motion.r**2)

        if imu_mag < self.config.min_angular_rate:
            return
        if imu_mag > self.config.max_angular_rate:
            return

        with self._lock:
            sample = SyncedSample(
                timestamp=motion.timestamp + time_offset,
                imu_p=imu.p,
                imu_q=imu.q,
                imu_r=imu.r,
                vis_p=motion.p,
                vis_q=motion.q,
                vis_r=motion.r
            )
            self._samples.append(sample)

            # Keep buffer bounded
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]

    def add_samples_from_buffers(self, imu_buffer: TimestampedRingBuffer,
                                  motion_buffer: TimestampedRingBuffer,
                                  time_offset: float = 0.0) -> int:
        """
        Add samples by matching IMU and motion data from buffers.

        Args:
            imu_buffer: Ring buffer of IMU data
            motion_buffer: Ring buffer of motion estimates
            time_offset: Time offset (camera_time + offset = imu_time)

        Returns:
            Number of samples added
        """
        motion_data = motion_buffer.get_all()
        count = 0

        for motion in motion_data:
            if not motion.valid:
                continue

            # Find matching IMU sample
            imu_time = motion.timestamp + time_offset
            imu = imu_buffer.get_nearest(imu_time)

            if imu is None:
                continue

            # Check time proximity (within 50ms)
            if abs(imu.timestamp - imu_time) > 0.05:
                continue

            self.add_sample(imu, motion, 0.0)  # offset already applied
            count += 1

        return count

    def start(self) -> None:
        """Start calibration thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._calibration_loop, daemon=True)
        self._thread.start()
        print("Spatial calibrator started")

    def stop(self) -> None:
        """Stop calibration thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Spatial calibrator stopped")

    def _calibration_loop(self) -> None:
        """Main calibration loop."""
        import time

        while self._running:
            now = monotonic_time()

            if now - self._last_update >= self.config.update_interval:
                self._attempt_calibration()
                self._last_update = now

            time.sleep(0.1)

    def _attempt_calibration(self) -> None:
        """Attempt to compute spatial calibration."""
        with self._lock:
            if len(self._samples) < self.config.min_samples:
                return

            samples = list(self._samples)

        try:
            result = self._optimize_mount(samples)

            if result is not None:
                self._mount_offset = MountOffset(
                    yaw=result.yaw,
                    pitch=result.pitch,
                    roll=result.roll,
                    rmse=result.rmse,
                    converged=True
                )
                self._rotation_matrix = result.rotation_matrix
                self._num_calibrations += 1

                self._fire_callbacks(result)

                print(f"Spatial calibration: yaw={np.degrees(result.yaw):.2f}deg, "
                      f"pitch={np.degrees(result.pitch):.2f}deg, "
                      f"roll={np.degrees(result.roll):.2f}deg, "
                      f"rmse={np.degrees(result.rmse):.3f}deg")

        except Exception as e:
            print(f"Spatial calibration error: {e}")

    def _optimize_mount(self, samples: List[SyncedSample]) -> Optional[SpatialCalibration]:
        """
        Optimize camera mount rotation.

        Args:
            samples: List of synchronized samples

        Returns:
            SpatialCalibration result or None
        """
        # Initial guess
        x0 = np.array([
            self._mount_offset.yaw,
            self._mount_offset.pitch,
            self._mount_offset.roll
        ])

        # Extract data arrays
        imu_rates = np.array([[s.imu_p, s.imu_q, s.imu_r] for s in samples])
        vis_rates = np.array([[s.vis_p, s.vis_q, s.vis_r] for s in samples])

        def error_func(x):
            """Compute residuals for given mount rotation."""
            yaw, pitch, roll = x

            # Build rotation matrix from camera to body frame
            R = euler_to_rotation_matrix(roll, pitch, yaw)

            # Transform visual rates to body frame
            transformed = np.array([R @ v for v in vis_rates])

            # Compute residuals
            residuals = (imu_rates - transformed).flatten()

            return residuals

        # Run optimization
        result = least_squares(
            error_func,
            x0,
            method='lm',
            max_nfev=self.config.max_iterations
        )

        if not result.success:
            return None

        yaw, pitch, roll = result.x
        R = euler_to_rotation_matrix(roll, pitch, yaw)

        # Compute RMSE
        residuals = error_func(result.x)
        rmse = np.sqrt(np.mean(residuals**2))

        return SpatialCalibration(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_matrix=R,
            rmse=rmse,
            samples_used=len(samples)
        )

    def calibrate_now(self) -> Optional[SpatialCalibration]:
        """
        Run calibration immediately with current data.

        Returns:
            SpatialCalibration result or None
        """
        with self._lock:
            if len(self._samples) < self.config.min_samples:
                return None
            samples = list(self._samples)

        return self._optimize_mount(samples)

    @property
    def mount_offset(self) -> MountOffset:
        """Current mount offset estimate."""
        return self._mount_offset

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Current rotation matrix from camera to body frame."""
        return self._rotation_matrix.copy()

    @property
    def is_converged(self) -> bool:
        """Whether calibration has converged."""
        return self._mount_offset.converged

    @property
    def num_samples(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    @property
    def num_calibrations(self) -> int:
        """Number of calibration runs completed."""
        return self._num_calibrations

    def reset(self) -> None:
        """Reset calibration state."""
        with self._lock:
            self._samples = []
            self._mount_offset = MountOffset(
                yaw=self.config.initial_yaw,
                pitch=self.config.initial_pitch,
                roll=self.config.initial_roll
            )
            self._rotation_matrix = np.eye(3)
            self._num_calibrations = 0

    def get_status(self) -> dict:
        """Get calibration status."""
        return {
            'num_samples': len(self._samples),
            'num_calibrations': self._num_calibrations,
            'converged': self._mount_offset.converged,
            'yaw_deg': np.degrees(self._mount_offset.yaw),
            'pitch_deg': np.degrees(self._mount_offset.pitch),
            'roll_deg': np.degrees(self._mount_offset.roll),
            'rmse_deg': np.degrees(self._mount_offset.rmse)
        }

    def transform_rates(self, vis_p: float, vis_q: float, vis_r: float
                        ) -> Tuple[float, float, float]:
        """
        Transform visual angular rates to body frame.

        Args:
            vis_p: Visual roll rate
            vis_q: Visual pitch rate
            vis_r: Visual yaw rate

        Returns:
            Tuple of (body_p, body_q, body_r)
        """
        vis = np.array([vis_p, vis_q, vis_r])
        body = self._rotation_matrix @ vis
        return tuple(body)


def calibrate_offline(imu_data: List[IMUData],
                      motion_data: List[MotionEstimate],
                      time_offset: float = 0.0,
                      config: Optional[SpatialCalibConfig] = None
                      ) -> Optional[SpatialCalibration]:
    """
    Compute spatial calibration offline (batch mode).

    Args:
        imu_data: List of IMU samples
        motion_data: List of visual motion estimates
        time_offset: Time offset (camera_time + offset = imu_time)
        config: Optional configuration

    Returns:
        SpatialCalibration result or None
    """
    if config is None:
        config = SpatialCalibConfig()

    calib = SpatialCalibrator(config)

    # Match samples by timestamp
    for motion in motion_data:
        if not motion.valid:
            continue

        # Find nearest IMU sample
        imu_time = motion.timestamp + time_offset
        best_imu = None
        best_diff = float('inf')

        for imu in imu_data:
            diff = abs(imu.timestamp - imu_time)
            if diff < best_diff:
                best_diff = diff
                best_imu = imu

        if best_imu is not None and best_diff < 0.05:  # 50ms tolerance
            calib.add_sample(best_imu, motion, 0.0)

    # Run calibration
    return calib.calibrate_now()


class AxisAlignmentHelper:
    """
    Helper class for determining camera axis alignment.

    Useful for determining if camera is forward, down, rear facing
    by analyzing the correlation between visual and IMU angular rates.
    """

    @staticmethod
    def detect_orientation(imu_rates: np.ndarray,
                           vis_rates: np.ndarray) -> str:
        """
        Detect camera orientation based on rate correlations.

        Args:
            imu_rates: Array of IMU angular rates [N, 3] (p, q, r)
            vis_rates: Array of visual angular rates [N, 3] (p, q, r)

        Returns:
            Estimated orientation: 'forward', 'down', 'rear', or 'unknown'
        """
        # Compute correlations between each axis
        corr_pp = np.corrcoef(imu_rates[:, 0], vis_rates[:, 0])[0, 1]
        corr_qq = np.corrcoef(imu_rates[:, 1], vis_rates[:, 1])[0, 1]
        corr_rr = np.corrcoef(imu_rates[:, 2], vis_rates[:, 2])[0, 1]

        # Check for cross-correlations (axis swaps)
        corr_pr = np.corrcoef(imu_rates[:, 0], vis_rates[:, 2])[0, 1]
        corr_rp = np.corrcoef(imu_rates[:, 2], vis_rates[:, 0])[0, 1]

        print(f"Axis correlations: p-p={corr_pp:.2f}, q-q={corr_qq:.2f}, "
              f"r-r={corr_rr:.2f}")
        print(f"Cross correlations: p-r={corr_pr:.2f}, r-p={corr_rp:.2f}")

        # Forward facing: visual p ~ imu p, visual q ~ imu q
        if corr_pp > 0.7 and corr_qq > 0.7:
            return 'forward'

        # Rear facing: visual p ~ -imu p
        if corr_pp < -0.7:
            return 'rear'

        # Down facing: visual p ~ imu r, visual r ~ -imu p
        if abs(corr_pr) > 0.7 or abs(corr_rp) > 0.7:
            return 'down'

        return 'unknown'
