"""
Real-time Camera-IMU Calibration System

This package provides tools for temporal and spatial calibration
between a camera (such as Harrier) and a flight controller
(Orange Cube, Pixhawk, etc.) using MAVLink.

Modules:
    core: Data types, ring buffers, and utility functions
    mavlink: MAVLink receiver for flight controller data
    camera: Camera capture and motion estimation
    sync: Temporal and spatial calibration algorithms

Usage:
    from calibration.mavlink.receiver import MAVLinkReceiver
    from calibration.camera.harrier import HarrierCamera
    from calibration.sync.temporal import TemporalSynchronizer
    from calibration.sync.spatial import SpatialCalibrator
"""

__version__ = "1.0.0"
__author__ = "ImageAnalysis Project"

# Convenience imports
from .core.data_types import (
    IMUData,
    AttitudeData,
    GPSData,
    FrameData,
    MotionEstimate,
    CalibrationResult,
    TemporalCalibration,
    SpatialCalibration
)

__all__ = [
    'IMUData',
    'AttitudeData',
    'GPSData',
    'FrameData',
    'MotionEstimate',
    'CalibrationResult',
    'TemporalCalibration',
    'SpatialCalibration'
]
