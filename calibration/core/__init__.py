"""
Core data structures and utilities for calibration system.
"""

from .data_types import (
    IMUData,
    AttitudeData,
    GPSData,
    FrameData,
    MotionEstimate,
    SyncResult,
    CalibrationResult
)
from .ring_buffer import RingBuffer, TimestampedRingBuffer
from .utils import (
    monotonic_time,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    quaternion_to_euler,
    angular_rate_from_rotation
)
from .interpolator import (
    IMUInterpolator,
    InterpolatorConfig,
    RealTimeInterpolator,
    interpolate_to_timestamps
)

__all__ = [
    # Data types
    'IMUData',
    'AttitudeData',
    'GPSData',
    'FrameData',
    'MotionEstimate',
    'SyncResult',
    'CalibrationResult',
    # Buffers
    'RingBuffer',
    'TimestampedRingBuffer',
    # Utilities
    'monotonic_time',
    'euler_to_rotation_matrix',
    'rotation_matrix_to_euler',
    'quaternion_to_euler',
    'angular_rate_from_rotation',
    # Interpolation
    'IMUInterpolator',
    'InterpolatorConfig',
    'RealTimeInterpolator',
    'interpolate_to_timestamps'
]
