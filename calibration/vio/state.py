"""
VIO State and Pose representations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    """6-DOF pose (position + orientation)."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] in meters
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # quaternion [w, x, y, z]
    timestamp: float = 0.0
    
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from quaternion."""
        r = Rotation.from_quat([self.orientation[1], self.orientation[2], 
                                 self.orientation[3], self.orientation[0]])  # scipy uses [x,y,z,w]
        return r.as_matrix()
    
    def euler_degrees(self) -> np.ndarray:
        """Get Euler angles [roll, pitch, yaw] in degrees."""
        r = Rotation.from_quat([self.orientation[1], self.orientation[2], 
                                 self.orientation[3], self.orientation[0]])
        return r.as_euler('xyz', degrees=True)
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from body frame to world frame."""
        return self.rotation_matrix() @ point + self.position
    
    def inverse(self) -> 'Pose':
        """Get inverse pose."""
        R = self.rotation_matrix()
        R_inv = R.T
        p_inv = -R_inv @ self.position
        r = Rotation.from_matrix(R_inv)
        q = r.as_quat()  # [x,y,z,w]
        return Pose(
            position=p_inv,
            orientation=np.array([q[3], q[0], q[1], q[2]]),  # [w,x,y,z]
            timestamp=self.timestamp
        )
    
    def compose(self, other: 'Pose') -> 'Pose':
        """Compose two poses: self * other."""
        R1 = self.rotation_matrix()
        R2 = other.rotation_matrix()
        R = R1 @ R2
        p = R1 @ other.position + self.position
        r = Rotation.from_matrix(R)
        q = r.as_quat()  # [x,y,z,w]
        return Pose(
            position=p,
            orientation=np.array([q[3], q[0], q[1], q[2]]),  # [w,x,y,z]
            timestamp=other.timestamp
        )


@dataclass
class VIOState:
    """
    Full VIO state vector.
    
    State: [position(3), velocity(3), orientation(4), gyro_bias(3), accel_bias(3)]
    Total: 16 dimensions (orientation is quaternion)
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] in meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz] in m/s
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # quaternion [w, x, y, z]
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [bwx, bwy, bwz] in rad/s
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [bax, bay, baz] in m/s^2
    timestamp: float = 0.0
    
    # Covariance matrix (15x15 for error state)
    covariance: np.ndarray = field(default_factory=lambda: np.eye(15) * 0.01)
    
    def to_pose(self) -> Pose:
        """Extract pose from state."""
        return Pose(
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            timestamp=self.timestamp
        )
    
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from orientation quaternion."""
        r = Rotation.from_quat([self.orientation[1], self.orientation[2], 
                                 self.orientation[3], self.orientation[0]])
        return r.as_matrix()
    
    def to_vector(self) -> np.ndarray:
        """Convert state to vector [pos(3), vel(3), quat(4), bg(3), ba(3)]."""
        return np.concatenate([
            self.position,
            self.velocity, 
            self.orientation,
            self.gyro_bias,
            self.accel_bias
        ])
    
    @staticmethod
    def from_vector(vec: np.ndarray, timestamp: float = 0.0) -> 'VIOState':
        """Create state from vector."""
        return VIOState(
            position=vec[0:3],
            velocity=vec[3:6],
            orientation=vec[6:10] / np.linalg.norm(vec[6:10]),  # normalize quaternion
            gyro_bias=vec[10:13],
            accel_bias=vec[13:16],
            timestamp=timestamp
        )
    
    def copy(self) -> 'VIOState':
        """Deep copy of state."""
        return VIOState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            timestamp=self.timestamp,
            covariance=self.covariance.copy()
        )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.
    Returns [w, x, y, z].
    """
    if np.abs(angle) < 1e-10:
        return np.array([1, 0, 0, 0])
    
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_from_rotation_vector(rv: np.ndarray) -> np.ndarray:
    """
    Create quaternion from rotation vector (axis * angle).
    Returns [w, x, y, z].
    """
    angle = np.linalg.norm(rv)
    if angle < 1e-10:
        return np.array([1, 0, 0, 0])
    axis = rv / angle
    return quaternion_from_axis_angle(axis, angle)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
