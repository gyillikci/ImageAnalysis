"""
Visual motion estimator - computes angular rates from frame-to-frame motion.
"""

import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import FrameData, MotionEstimate
from core.ring_buffer import TimestampedRingBuffer
from core.utils import (
    monotonic_time, RateEstimator,
    rodrigues_to_rotation_matrix, rotation_matrix_to_euler,
    angular_rate_from_rotation
)
from .feature_tracker import FeatureTracker


@dataclass
class MotionEstimatorConfig:
    """Motion estimator configuration."""
    max_features: int = 500
    min_features: int = 10  # Lowered for more motion estimates
    quality_level: float = 0.01
    min_distance: int = 10
    ransac_threshold: float = 2.0  # Increased for more RANSAC inliers
    confidence: float = 0.99
    buffer_size: int = 1000

    # Camera intrinsics (required for essential matrix)
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 360.0


class MotionEstimator:
    """
    Visual motion estimator using essential matrix decomposition.

    Estimates camera rotation (angular rates) from frame-to-frame feature
    correspondence using the essential matrix.

    Usage:
        estimator = MotionEstimator(config)
        estimator.start()

        # Process frames (or use with camera callback)
        for frame in camera.frames:
            motion = estimator.process_frame(frame)
            if motion.valid:
                print(f"Angular rates: {motion.p}, {motion.q}, {motion.r}")

        estimator.stop()
    """

    def __init__(self, config: MotionEstimatorConfig):
        """
        Initialize motion estimator.

        Args:
            config: Estimator configuration
        """
        self.config = config

        # Feature tracker
        self._tracker = FeatureTracker(
            max_features=config.max_features,
            quality_level=config.quality_level,
            min_distance=config.min_distance
        )

        # Camera matrix
        self._K = np.array([
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Motion buffer
        self.motion_buffer = TimestampedRingBuffer[MotionEstimate](config.buffer_size)

        # Rate estimator
        self._rate_estimator = RateEstimator(window_size=60)

        # State
        self._prev_R = np.eye(3)
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks = []

    def add_callback(self, callback: Callable[[MotionEstimate], None]) -> None:
        """Add motion estimate callback."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, motion: MotionEstimate) -> None:
        """Fire callbacks."""
        for cb in self._callbacks:
            try:
                cb(motion)
            except Exception as e:
                print(f"Motion callback error: {e}")

    def process_frame(self, frame_data: FrameData) -> MotionEstimate:
        """
        Process a single frame and estimate motion using optical flow.

        Args:
            frame_data: Input frame data

        Returns:
            MotionEstimate with angular rates
        """
        import cv2

        timestamp = frame_data.timestamp
        frame = frame_data.image

        with self._lock:
            # Track features
            prev_feats, curr_feats = self._tracker.process(frame, timestamp)

            # Check if we have enough features
            if prev_feats is None or curr_feats is None:
                return self._invalid_motion(timestamp, 0.0)

            if len(prev_feats.points) < self.config.min_features:
                return self._invalid_motion(timestamp, 0.0)

            dt = curr_feats.timestamp - prev_feats.timestamp
            if dt <= 0:
                return self._invalid_motion(timestamp, 0.0)

            # Get matched points
            prev_pts = prev_feats.points
            curr_pts = curr_feats.points
            
            # Compute optical flow
            flow = curr_pts - prev_pts
            
            # Mean flow gives us pitch and yaw
            mean_flow = np.mean(flow, axis=0)  # [dx, dy] in pixels per frame
            
            # omega_y (yaw) ≈ flow_x / (fx * dt)
            # omega_x (pitch) ≈ -flow_y / (fy * dt)
            omega_y = mean_flow[0] / (self.config.fx * dt)  # yaw rate
            omega_x = -mean_flow[1] / (self.config.fy * dt)  # pitch rate
            
            # For roll (rotation about optical axis)
            # Roll causes rotational flow pattern - pixels move in circles around center
            # First, remove the translational component (mean flow) from each point
            flow_centered = flow - mean_flow  # Remove translation to isolate rotation
            
            # Position vectors from image center
            pos = prev_pts - np.array([self.config.cx, self.config.cy])
            r_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
            r_dist = np.maximum(r_dist, 10.0)  # Avoid division by zero
            
            # For pure roll, flow should be perpendicular to position vector
            # Tangent direction at each point is (-y, x) / r (counter-clockwise)
            tangent_x = -pos[:, 1] / r_dist
            tangent_y = pos[:, 0] / r_dist
            tangential = flow_centered[:, 0] * tangent_x + flow_centered[:, 1] * tangent_y
            
            # Angular velocity = tangential velocity / radius
            omega_per_point = tangential / r_dist
            omega_z = -np.median(omega_per_point) / dt  # roll rate (negated for IMU convention)

            # Angular rates in camera frame
            p = omega_z   # Roll rate
            q = omega_x   # Pitch rate
            r = omega_y   # Yaw rate
            
            # Sanity check: cap maximum angular rate at 10 rad/s (~573 deg/s)
            if max(abs(p), abs(q), abs(r)) > 10.0:
                return self._invalid_motion(timestamp, dt)

            # Create motion estimate
            motion = MotionEstimate(
                timestamp=timestamp,
                dt=dt,
                rotation=np.array([p * dt, q * dt, r * dt]),  # Approximate rotation vector
                translation=None,
                p=p,
                q=q,
                r=r,
                num_features=len(prev_pts),
                inlier_ratio=1.0,
                valid=True
            )

            # Store in buffer
            self.motion_buffer.push(motion)

            # Update rate
            self._rate_estimator.update(timestamp)

            # Fire callbacks
            self._fire_callbacks(motion)

            return motion

    def _invalid_motion(self, timestamp: float, dt: float) -> MotionEstimate:
        """Create invalid motion estimate."""
        motion = MotionEstimate(
            timestamp=timestamp,
            dt=dt,
            valid=False
        )
        self.motion_buffer.push(motion)
        return motion

    def set_camera_matrix(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Update camera intrinsics."""
        with self._lock:
            self._K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            self.config.fx = fx
            self.config.fy = fy
            self.config.cx = cx
            self.config.cy = cy

    def reset(self) -> None:
        """Reset estimator state."""
        with self._lock:
            self._tracker.reset()
            self._prev_R = np.eye(3)
            self.motion_buffer.clear()

    @property
    def rate(self) -> float:
        """Current motion estimation rate (Hz)."""
        return self._rate_estimator.rate

    @property
    def num_features(self) -> int:
        """Current number of tracked features."""
        return self._tracker.num_features

    def get_status(self) -> dict:
        """Get estimator status."""
        return {
            'rate': self.rate,
            'num_features': self.num_features,
            'buffer_fill': self.motion_buffer.fill_ratio
        }


class IntegratingMotionEstimator(MotionEstimator):
    """
    Motion estimator that also integrates rotation to get attitude.

    Useful for comparing visual attitude with IMU attitude.
    """

    def __init__(self, config: MotionEstimatorConfig):
        super().__init__(config)
        self._integrated_R = np.eye(3)
        self._integrated_roll = 0.0
        self._integrated_pitch = 0.0
        self._integrated_yaw = 0.0

    def process_frame(self, frame_data: FrameData) -> MotionEstimate:
        """Process frame and update integrated attitude."""
        motion = super().process_frame(frame_data)

        if motion.valid and motion.rotation is not None:
            # Update integrated rotation
            R = rodrigues_to_rotation_matrix(motion.rotation)
            self._integrated_R = R @ self._integrated_R

            # Extract Euler angles
            self._integrated_roll, self._integrated_pitch, self._integrated_yaw = \
                rotation_matrix_to_euler(self._integrated_R)

        return motion

    @property
    def integrated_attitude(self) -> Tuple[float, float, float]:
        """Get integrated attitude (roll, pitch, yaw) in radians."""
        return (self._integrated_roll, self._integrated_pitch, self._integrated_yaw)

    def reset_integration(self) -> None:
        """Reset integrated attitude to zero."""
        with self._lock:
            self._integrated_R = np.eye(3)
            self._integrated_roll = 0.0
            self._integrated_pitch = 0.0
            self._integrated_yaw = 0.0
