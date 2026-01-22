"""
Extended Kalman Filter for Visual-Inertial Odometry.

Fuses IMU predictions with visual measurements.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation

from .state import VIOState, Pose, quaternion_multiply, quaternion_from_rotation_vector, skew_symmetric
from .imu_preintegration import IMUPreintegrator, PreintegratedIMU


@dataclass
class VIOConfig:
    """VIO configuration parameters."""
    # IMU noise parameters
    gyro_noise: float = 0.01  # rad/s/sqrt(Hz)
    accel_noise: float = 0.1  # m/s^2/sqrt(Hz)
    gyro_bias_noise: float = 0.0001  # rad/s^2/sqrt(Hz)
    accel_bias_noise: float = 0.001  # m/s^3/sqrt(Hz)
    
    # Visual measurement noise
    pixel_noise: float = 1.0  # pixels
    
    # Camera-IMU calibration
    time_offset: float = 0.0  # seconds (camera_time = imu_time + offset)
    R_cam_imu: np.ndarray = None  # Rotation from IMU to camera frame
    p_cam_imu: np.ndarray = None  # Translation from IMU to camera in IMU frame
    
    # Camera intrinsics
    fx: float = 1175.0
    fy: float = 1175.0
    cx: float = 640.0
    cy: float = 360.0
    
    # Gravity
    gravity: np.ndarray = None
    
    def __post_init__(self):
        if self.R_cam_imu is None:
            self.R_cam_imu = np.eye(3)
        if self.p_cam_imu is None:
            self.p_cam_imu = np.zeros(3)
        if self.gravity is None:
            self.gravity = np.array([0, 0, 9.81])


class VIOEKF:
    """
    Extended Kalman Filter for VIO.
    
    State: [position(3), velocity(3), orientation(4), gyro_bias(3), accel_bias(3)]
    Error state: [delta_p(3), delta_v(3), delta_theta(3), delta_bg(3), delta_ba(3)] = 15 dims
    
    The filter uses error-state formulation where the quaternion is updated
    using a 3D rotation error.
    """
    
    def __init__(self, config: VIOConfig = None):
        """Initialize EKF."""
        self.config = config or VIOConfig()
        
        # Current state
        self._state = VIOState()
        
        # IMU preintegrator
        self._preintegrator = IMUPreintegrator(
            gyro_noise=self.config.gyro_noise,
            accel_noise=self.config.accel_noise,
            gyro_bias_noise=self.config.gyro_bias_noise,
            accel_bias_noise=self.config.accel_bias_noise
        )
        
        # Process noise
        self._Q = np.diag([
            0.01, 0.01, 0.01,  # position
            0.1, 0.1, 0.1,    # velocity
            0.01, 0.01, 0.01,  # orientation
            self.config.gyro_bias_noise**2,
            self.config.gyro_bias_noise**2,
            self.config.gyro_bias_noise**2,
            self.config.accel_bias_noise**2,
            self.config.accel_bias_noise**2,
            self.config.accel_bias_noise**2
        ])
        
        # Initialized flag
        self._initialized = False
        
        # Last IMU timestamp
        self._last_imu_time = None
    
    def initialize(self, position: np.ndarray = None, 
                   orientation: np.ndarray = None,
                   timestamp: float = 0.0):
        """
        Initialize the filter.
        
        Args:
            position: Initial position [x, y, z]
            orientation: Initial orientation quaternion [w, x, y, z]
            timestamp: Initial timestamp
        """
        self._state = VIOState()
        self._state.timestamp = timestamp
        
        if position is not None:
            self._state.position = position.copy()
        if orientation is not None:
            self._state.orientation = orientation.copy() / np.linalg.norm(orientation)
        
        # Initial covariance
        self._state.covariance = np.diag([
            0.01, 0.01, 0.01,  # position (1cm)
            0.1, 0.1, 0.1,     # velocity (0.1 m/s)
            0.01, 0.01, 0.01,  # orientation (0.6 deg)
            0.001, 0.001, 0.001,  # gyro bias
            0.01, 0.01, 0.01   # accel bias
        ])
        
        self._preintegrator.reset(self._state.gyro_bias, self._state.accel_bias)
        self._initialized = True
        self._last_imu_time = timestamp
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def state(self) -> VIOState:
        return self._state
    
    @property
    def pose(self) -> Pose:
        return self._state.to_pose()
    
    def predict_imu(self, timestamp: float, gyro: np.ndarray, accel: np.ndarray):
        """
        IMU prediction step.
        
        Args:
            timestamp: IMU timestamp
            gyro: Gyroscope reading [wx, wy, wz] in rad/s
            accel: Accelerometer reading [ax, ay, az] in m/s^2
        """
        if not self._initialized:
            return
        
        if self._last_imu_time is None:
            self._last_imu_time = timestamp
            return
        
        dt = timestamp - self._last_imu_time
        if dt <= 0:
            return
        
        # Bias-corrected measurements
        omega = gyro - self._state.gyro_bias
        acc = accel - self._state.accel_bias
        
        # Current rotation matrix
        R = self._state.rotation_matrix()
        
        # Predict orientation (first-order integration)
        theta = omega * dt
        dq = quaternion_from_rotation_vector(theta)
        self._state.orientation = quaternion_multiply(self._state.orientation, dq)
        self._state.orientation /= np.linalg.norm(self._state.orientation)
        
        # Predict velocity (account for gravity)
        a_world = R @ acc - self.config.gravity
        self._state.velocity += a_world * dt
        
        # Predict position
        self._state.position += self._state.velocity * dt + 0.5 * a_world * dt**2
        
        # Update covariance
        F = self._compute_F(omega, acc, R, dt)
        G = self._compute_G(R, dt)
        
        self._state.covariance = F @ self._state.covariance @ F.T + G @ self._Q @ G.T
        
        # Update timestamp
        self._state.timestamp = timestamp
        self._last_imu_time = timestamp
        
        # Also add to preintegrator for keyframe-based updates
        self._preintegrator.add_measurement(timestamp, gyro, accel)
    
    def _compute_F(self, omega: np.ndarray, accel: np.ndarray, R: np.ndarray, dt: float) -> np.ndarray:
        """Compute state transition Jacobian."""
        F = np.eye(15)
        
        # Position wrt velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Position wrt orientation
        F[0:3, 6:9] = -0.5 * R @ skew_symmetric(accel) * dt**2
        
        # Position wrt accel bias
        F[0:3, 12:15] = -0.5 * R * dt**2
        
        # Velocity wrt orientation
        F[3:6, 6:9] = -R @ skew_symmetric(accel) * dt
        
        # Velocity wrt accel bias
        F[3:6, 12:15] = -R * dt
        
        # Orientation wrt gyro bias
        F[6:9, 9:12] = -np.eye(3) * dt
        
        return F
    
    def _compute_G(self, R: np.ndarray, dt: float) -> np.ndarray:
        """Compute noise Jacobian."""
        G = np.zeros((15, 15))
        
        # Position noise
        G[0:3, 0:3] = np.eye(3)
        
        # Velocity noise
        G[3:6, 3:6] = np.eye(3)
        
        # Orientation noise
        G[6:9, 6:9] = np.eye(3)
        
        # Bias random walk
        G[9:12, 9:12] = np.eye(3)
        G[12:15, 12:15] = np.eye(3)
        
        return G
    
    def update_visual(self, features_2d: np.ndarray, features_3d: np.ndarray,
                      timestamp: float = None):
        """
        Visual measurement update.
        
        Uses 2D-3D correspondences to update the state.
        
        Args:
            features_2d: Nx2 array of 2D feature observations [u, v]
            features_3d: Nx3 array of 3D feature positions in world frame [x, y, z]
            timestamp: Observation timestamp (applies time offset)
        """
        if not self._initialized:
            return
        
        if len(features_2d) < 4:
            return
        
        # Apply camera-IMU calibration
        R_c_i = self.config.R_cam_imu
        p_c_i = self.config.p_cam_imu
        
        # Current body pose
        R_w_b = self._state.rotation_matrix()
        p_w_b = self._state.position
        
        # Camera pose in world frame
        R_w_c = R_w_b @ R_c_i.T
        p_w_c = p_w_b + R_w_b @ p_c_i
        
        # Transform 3D points to camera frame
        points_cam = (R_w_c.T @ (features_3d.T - p_w_c.reshape(3, 1))).T
        
        # Project to image and compute residuals
        z_pred = np.zeros((len(features_2d), 2))
        H = np.zeros((len(features_2d) * 2, 15))
        
        valid_indices = []
        
        for i, (p3d_cam, obs) in enumerate(zip(points_cam, features_2d)):
            x, y, z = p3d_cam
            
            if z <= 0.1:  # Behind camera
                continue
            
            valid_indices.append(i)
            
            # Predicted projection
            u_pred = self.config.fx * x / z + self.config.cx
            v_pred = self.config.fy * y / z + self.config.cy
            z_pred[i] = [u_pred, v_pred]
            
            # Jacobian of projection wrt camera-frame point
            J_proj = np.array([
                [self.config.fx / z, 0, -self.config.fx * x / z**2],
                [0, self.config.fy / z, -self.config.fy * y / z**2]
            ])
            
            # Jacobian of camera-frame point wrt body pose
            # p_cam = R_c_i @ R_w_b.T @ (p_world - p_w_b)
            p3d_world = features_3d[i]
            p_body = R_w_b.T @ (p3d_world - p_w_b)
            
            # wrt position
            J_p = -R_c_i @ R_w_b.T
            
            # wrt orientation (using right perturbation)
            J_theta = R_c_i @ skew_symmetric(p_body)
            
            # Fill Jacobian rows
            H[2*i:2*i+2, 0:3] = J_proj @ J_p
            H[2*i:2*i+2, 6:9] = J_proj @ J_theta
        
        if len(valid_indices) < 4:
            return
        
        # Residual
        residual = features_2d - z_pred
        residual = residual.flatten()
        
        # Measurement noise
        R_meas = np.eye(len(features_2d) * 2) * self.config.pixel_noise**2
        
        # Kalman gain
        P = self._state.covariance
        S = H @ P @ H.T + R_meas
        
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        # State update (error-state)
        dx = K @ residual
        
        # Apply correction
        self._state.position += dx[0:3]
        self._state.velocity += dx[3:6]
        
        # Orientation update (quaternion)
        dtheta = dx[6:9]
        dq = quaternion_from_rotation_vector(dtheta)
        self._state.orientation = quaternion_multiply(self._state.orientation, dq)
        self._state.orientation /= np.linalg.norm(self._state.orientation)
        
        # Bias update
        self._state.gyro_bias += dx[9:12]
        self._state.accel_bias += dx[12:15]
        
        # Covariance update
        I_KH = np.eye(15) - K @ H
        self._state.covariance = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
    
    def update_relative_pose(self, delta_R: np.ndarray, delta_p: np.ndarray,
                             covariance: np.ndarray = None):
        """
        Update with relative pose measurement (e.g., from visual odometry).
        
        Args:
            delta_R: 3x3 rotation matrix of relative rotation
            delta_p: 3D relative translation
            covariance: 6x6 measurement covariance (optional)
        """
        if not self._initialized:
            return
        
        if covariance is None:
            covariance = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
        
        # This is a simplified update - in practice you'd compare
        # against the IMU preintegration
        
        # Current rotation
        R = self._state.rotation_matrix()
        
        # Predicted vs measured rotation error
        R_pred = self._preintegrator.get_preintegrated().delta_R
        R_error = R_pred.T @ delta_R
        
        # Convert to axis-angle
        r = Rotation.from_matrix(R_error)
        theta_error = r.as_rotvec()
        
        # Translation error
        p_pred = self._preintegrator.get_preintegrated().delta_p
        p_error = p_pred - delta_p
        
        # Combined residual
        residual = np.concatenate([theta_error, p_error])
        
        # Measurement Jacobian (simplified)
        H = np.zeros((6, 15))
        H[0:3, 6:9] = np.eye(3)  # Rotation error wrt orientation
        H[3:6, 0:3] = np.eye(3)  # Position error wrt position
        
        # Kalman update
        P = self._state.covariance
        S = H @ P @ H.T + covariance
        
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        dx = K @ residual
        
        # Apply correction
        self._state.position += dx[0:3]
        self._state.velocity += dx[3:6]
        
        dtheta = dx[6:9]
        dq = quaternion_from_rotation_vector(dtheta)
        self._state.orientation = quaternion_multiply(self._state.orientation, dq)
        self._state.orientation /= np.linalg.norm(self._state.orientation)
        
        self._state.gyro_bias += dx[9:12]
        self._state.accel_bias += dx[12:15]
        
        I_KH = np.eye(15) - K @ H
        self._state.covariance = I_KH @ P
        
        # Reset preintegrator
        self._preintegrator.reset(self._state.gyro_bias, self._state.accel_bias)
    
    def reset_preintegration(self):
        """Reset the IMU preintegrator (call after keyframe)."""
        self._preintegrator.reset(self._state.gyro_bias, self._state.accel_bias)
