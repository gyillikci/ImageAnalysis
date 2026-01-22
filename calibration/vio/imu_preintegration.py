"""
IMU Preintegration for VIO.

Preintegrates IMU measurements between keyframes to create
relative motion constraints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.spatial.transform import Rotation

from .state import (
    quaternion_multiply, quaternion_from_rotation_vector,
    skew_symmetric
)


@dataclass
class IMUMeasurement:
    """Single IMU measurement."""
    timestamp: float
    gyro: np.ndarray  # [wx, wy, wz] in rad/s
    accel: np.ndarray  # [ax, ay, az] in m/s^2


@dataclass 
class PreintegratedIMU:
    """
    Preintegrated IMU measurements between two keyframes.
    
    Contains the relative rotation, velocity, and position change
    as well as the Jacobians for bias correction.
    """
    # Preintegrated measurements
    delta_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # Rotation
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Velocity
    delta_p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Position
    
    # Integration time
    dt: float = 0.0
    
    # Jacobians for bias correction
    d_R_d_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    d_v_d_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    d_v_d_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    d_p_d_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    d_p_d_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Covariance
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))
    
    # Biases used during preintegration
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Number of measurements
    num_measurements: int = 0


class IMUPreintegrator:
    """
    IMU Preintegration class.
    
    Accumulates IMU measurements and computes preintegrated
    relative motion between keyframes.
    """
    
    # Gravity vector (NED frame)
    GRAVITY = np.array([0, 0, 9.81])
    
    def __init__(self, 
                 gyro_noise: float = 0.01,
                 accel_noise: float = 0.1,
                 gyro_bias_noise: float = 0.001,
                 accel_bias_noise: float = 0.01):
        """
        Initialize preintegrator.
        
        Args:
            gyro_noise: Gyroscope noise (rad/s/sqrt(Hz))
            accel_noise: Accelerometer noise (m/s^2/sqrt(Hz))
            gyro_bias_noise: Gyroscope bias random walk
            accel_bias_noise: Accelerometer bias random walk
        """
        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.accel_bias_noise = accel_bias_noise
        
        # Noise covariance
        self._Q = np.diag([
            gyro_noise**2, gyro_noise**2, gyro_noise**2,
            accel_noise**2, accel_noise**2, accel_noise**2
        ])
        
        self.reset()
    
    def reset(self, gyro_bias: np.ndarray = None, accel_bias: np.ndarray = None):
        """Reset preintegration."""
        self._preint = PreintegratedIMU()
        
        if gyro_bias is not None:
            self._preint.gyro_bias = gyro_bias.copy()
        if accel_bias is not None:
            self._preint.accel_bias = accel_bias.copy()
        
        self._measurements: List[IMUMeasurement] = []
        self._last_timestamp = None
    
    def add_measurement(self, timestamp: float, gyro: np.ndarray, accel: np.ndarray):
        """
        Add an IMU measurement and integrate.
        
        Args:
            timestamp: Measurement timestamp
            gyro: Gyroscope reading [wx, wy, wz] in rad/s
            accel: Accelerometer reading [ax, ay, az] in m/s^2
        """
        meas = IMUMeasurement(timestamp=timestamp, gyro=gyro.copy(), accel=accel.copy())
        self._measurements.append(meas)
        
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return
        
        dt = timestamp - self._last_timestamp
        if dt <= 0:
            return
        
        self._integrate(meas, dt)
        self._last_timestamp = timestamp
    
    def _integrate(self, meas: IMUMeasurement, dt: float):
        """Integrate a single measurement."""
        # Bias-corrected measurements
        omega = meas.gyro - self._preint.gyro_bias
        acc = meas.accel - self._preint.accel_bias
        
        # Current rotation
        R = self._preint.delta_R
        
        # Rotation increment (first-order)
        theta = omega * dt
        dR = Rotation.from_rotvec(theta).as_matrix()
        
        # Update preintegrated measurements
        # Position: p += v*dt + 0.5*R*a*dt^2
        self._preint.delta_p += self._preint.delta_v * dt + 0.5 * R @ acc * dt**2
        
        # Velocity: v += R*a*dt
        self._preint.delta_v += R @ acc * dt
        
        # Rotation: R = R * dR
        self._preint.delta_R = R @ dR
        
        # Update Jacobians for bias correction
        # These allow correcting preintegration when biases change
        
        # d_R_d_bg update
        Jr = self._right_jacobian(theta)
        self._preint.d_R_d_bg = dR.T @ self._preint.d_R_d_bg - Jr * dt
        
        # d_v_d_bg and d_v_d_ba update
        self._preint.d_v_d_bg -= R @ skew_symmetric(acc) @ self._preint.d_R_d_bg * dt
        self._preint.d_v_d_ba -= R * dt
        
        # d_p_d_bg and d_p_d_ba update
        self._preint.d_p_d_bg += self._preint.d_v_d_bg * dt - 0.5 * R @ skew_symmetric(acc) @ self._preint.d_R_d_bg * dt**2
        self._preint.d_p_d_ba += self._preint.d_v_d_ba * dt - 0.5 * R * dt**2
        
        # Update covariance
        A = np.eye(9)
        A[0:3, 0:3] = dR.T
        A[3:6, 0:3] = -R @ skew_symmetric(acc) * dt
        A[6:9, 0:3] = -0.5 * R @ skew_symmetric(acc) * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = R * dt
        B[6:9, 3:6] = 0.5 * R * dt**2
        
        self._preint.covariance = A @ self._preint.covariance @ A.T + B @ self._Q @ B.T
        
        # Update integration time
        self._preint.dt += dt
        self._preint.num_measurements += 1
    
    def _right_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Compute right Jacobian of SO(3)."""
        angle = np.linalg.norm(theta)
        if angle < 1e-6:
            return np.eye(3)
        
        axis = theta / angle
        K = skew_symmetric(axis)
        
        Jr = np.eye(3) - (1 - np.cos(angle)) / angle * K + (1 - np.sin(angle) / angle) * K @ K
        return Jr
    
    def get_preintegrated(self) -> PreintegratedIMU:
        """Get current preintegrated measurements."""
        return self._preint
    
    def get_delta_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get relative pose change (rotation quaternion and translation).
        
        Returns:
            Tuple of (quaternion [w,x,y,z], translation [x,y,z])
        """
        r = Rotation.from_matrix(self._preint.delta_R)
        q = r.as_quat()  # [x,y,z,w]
        quat = np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]
        
        return quat, self._preint.delta_p
    
    @property
    def integration_time(self) -> float:
        """Total integration time."""
        return self._preint.dt
    
    @property
    def num_measurements(self) -> int:
        """Number of measurements integrated."""
        return self._preint.num_measurements
