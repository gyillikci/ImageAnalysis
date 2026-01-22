"""
Utility functions for calibration system.
"""

import time
import math
import numpy as np
from typing import Tuple, Optional


def monotonic_time() -> float:
    """
    Return monotonic timestamp in seconds.

    This is the primary time source for all local timestamps.
    """
    return time.monotonic()


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix (ZYX convention).

    Args:
        roll: Roll angle (rad)
        pitch: Pitch angle (rad)
        yaw: Yaw angle (rad)

    Returns:
        3x3 rotation matrix
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (ZYX convention).

    Args:
        R: 3x3 rotation matrix

    Returns:
        (roll, pitch, yaw) in radians
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return (roll, pitch, yaw)


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (ZYX convention).

    Args:
        q: Quaternion [w, x, y, z] or [x, y, z, w]

    Returns:
        (roll, pitch, yaw) in radians
    """
    # Assume [w, x, y, z] format
    if len(q) != 4:
        raise ValueError("Quaternion must have 4 elements")

    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R


def angular_rate_from_rotation(R1: np.ndarray, R2: np.ndarray,
                                dt: float) -> np.ndarray:
    """
    Compute angular rate from two rotation matrices.

    Args:
        R1: First rotation matrix
        R2: Second rotation matrix
        dt: Time delta (seconds)

    Returns:
        Angular velocity vector [wx, wy, wz] (rad/s)
    """
    if dt <= 0:
        return np.zeros(3)

    # R2 = R_delta @ R1
    # R_delta = R2 @ R1.T
    R_delta = R2 @ R1.T

    # Extract rotation angle and axis using Rodrigues formula
    # trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R_delta)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-6:
        return np.zeros(3)

    # Rotation axis (unnormalized)
    axis = np.array([
        R_delta[2, 1] - R_delta[1, 2],
        R_delta[0, 2] - R_delta[2, 0],
        R_delta[1, 0] - R_delta[0, 1]
    ])

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np.zeros(3)

    axis = axis / axis_norm

    # Angular velocity
    omega = axis * theta / dt

    return omega


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues rotation vector to rotation matrix.

    Args:
        rvec: Rodrigues vector (angle * axis)

    Returns:
        3x3 rotation matrix
    """
    theta = np.linalg.norm(rvec)

    if theta < 1e-6:
        return np.eye(3)

    k = rvec / theta  # axis
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    return R


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Rodrigues vector.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Rodrigues vector (angle * axis)
    """
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-6:
        return np.zeros(3)

    # Rotation axis
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np.zeros(3)

    axis = axis / axis_norm

    return axis * theta


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(a1: float, a2: float) -> float:
    """Compute signed angle difference (a1 - a2), normalized to [-pi, pi]."""
    return normalize_angle(a1 - a2)


def interpolate_imu(imu1, imu2, alpha: float):
    """
    Linear interpolation between two IMU samples.

    Args:
        imu1: First IMU sample
        imu2: Second IMU sample
        alpha: Interpolation factor [0, 1]

    Returns:
        Interpolated IMU sample
    """
    from .data_types import IMUData

    return IMUData(
        timestamp=imu1.timestamp + alpha * (imu2.timestamp - imu1.timestamp),
        fc_timestamp=int(imu1.fc_timestamp + alpha * (imu2.fc_timestamp - imu1.fc_timestamp)),
        p=imu1.p + alpha * (imu2.p - imu1.p),
        q=imu1.q + alpha * (imu2.q - imu1.q),
        r=imu1.r + alpha * (imu2.r - imu1.r),
        ax=imu1.ax + alpha * (imu2.ax - imu1.ax),
        ay=imu1.ay + alpha * (imu2.ay - imu1.ay),
        az=imu1.az + alpha * (imu2.az - imu1.az)
    )


def interpolate_attitude(att1, att2, alpha: float):
    """
    Linear interpolation between two attitude samples.

    Note: This is a simple linear interpolation, not proper
    quaternion SLERP. For small attitude changes this is acceptable.

    Args:
        att1: First attitude sample
        att2: Second attitude sample
        alpha: Interpolation factor [0, 1]

    Returns:
        Interpolated attitude sample
    """
    from .data_types import AttitudeData

    # Interpolate angles carefully (handle wraparound)
    roll = att1.roll + alpha * angle_difference(att2.roll, att1.roll)
    pitch = att1.pitch + alpha * angle_difference(att2.pitch, att1.pitch)
    yaw = att1.yaw + alpha * angle_difference(att2.yaw, att1.yaw)

    return AttitudeData(
        timestamp=att1.timestamp + alpha * (att2.timestamp - att1.timestamp),
        fc_timestamp=int(att1.fc_timestamp + alpha * (att2.fc_timestamp - att1.fc_timestamp)),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        rollspeed=att1.rollspeed + alpha * (att2.rollspeed - att1.rollspeed),
        pitchspeed=att1.pitchspeed + alpha * (att2.pitchspeed - att1.pitchspeed),
        yawspeed=att1.yawspeed + alpha * (att2.yawspeed - att1.yawspeed)
    )


class RateEstimator:
    """
    Estimate rate (Hz) of incoming data.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize rate estimator.

        Args:
            window_size: Number of samples to average
        """
        self._timestamps = []
        self._window_size = window_size
        self._rate = 0.0

    def update(self, timestamp: float) -> float:
        """
        Update with new timestamp and return current rate estimate.

        Args:
            timestamp: New sample timestamp

        Returns:
            Current rate estimate (Hz)
        """
        self._timestamps.append(timestamp)

        # Keep only recent timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]

        # Compute rate
        if len(self._timestamps) >= 2:
            duration = self._timestamps[-1] - self._timestamps[0]
            if duration > 0:
                self._rate = (len(self._timestamps) - 1) / duration

        return self._rate

    @property
    def rate(self) -> float:
        return self._rate

    def reset(self) -> None:
        self._timestamps = []
        self._rate = 0.0
