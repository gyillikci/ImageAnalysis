"""
Visual-Inertial Odometry (VIO) Module

Fuses camera and IMU data for 6-DOF pose estimation.
"""

from .state import VIOState, Pose
from .imu_preintegration import IMUPreintegrator
from .ekf import VIOEKF
from .vio import VisualInertialOdometry

__all__ = [
    'VIOState',
    'Pose', 
    'IMUPreintegrator',
    'VIOEKF',
    'VisualInertialOdometry'
]
