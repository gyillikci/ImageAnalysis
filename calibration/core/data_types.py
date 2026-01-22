"""
Data type definitions for the calibration system.

All timestamps are in seconds from a monotonic clock source.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from enum import Enum


class SyncState(Enum):
    """Synchronization state machine states."""
    INITIALIZING = "initializing"
    ACQUIRING = "acquiring"
    CORRELATING = "correlating"
    SYNCHRONIZED = "synchronized"
    LOST = "lost"


@dataclass
class IMUData:
    """
    IMU measurement data from flight controller.

    Attributes:
        timestamp: Local monotonic timestamp (seconds)
        fc_timestamp: Flight controller timestamp (microseconds)
        p: Roll rate (rad/s)
        q: Pitch rate (rad/s)
        r: Yaw rate (rad/s)
        ax: X acceleration (m/s²)
        ay: Y acceleration (m/s²)
        az: Z acceleration (m/s²)
    """
    timestamp: float
    fc_timestamp: int
    p: float
    q: float
    r: float
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    def gyro_vector(self) -> np.ndarray:
        """Return gyro rates as numpy array [p, q, r]."""
        return np.array([self.p, self.q, self.r])

    def accel_vector(self) -> np.ndarray:
        """Return accelerations as numpy array [ax, ay, az]."""
        return np.array([self.ax, self.ay, self.az])


@dataclass
class AttitudeData:
    """
    Attitude estimate from flight controller EKF.

    Attributes:
        timestamp: Local monotonic timestamp (seconds)
        fc_timestamp: Flight controller timestamp (microseconds)
        roll: Roll angle (rad)
        pitch: Pitch angle (rad)
        yaw: Yaw angle (rad)
        rollspeed: Roll rate (rad/s)
        pitchspeed: Pitch rate (rad/s)
        yawspeed: Yaw rate (rad/s)
    """
    timestamp: float
    fc_timestamp: int
    roll: float
    pitch: float
    yaw: float
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    def euler_vector(self) -> np.ndarray:
        """Return Euler angles as numpy array [roll, pitch, yaw]."""
        return np.array([self.roll, self.pitch, self.yaw])

    def rate_vector(self) -> np.ndarray:
        """Return angular rates as numpy array."""
        return np.array([self.rollspeed, self.pitchspeed, self.yawspeed])


@dataclass
class GPSData:
    """
    GPS position data.

    Attributes:
        timestamp: Local monotonic timestamp (seconds)
        fc_timestamp: Flight controller timestamp (microseconds)
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        alt: Altitude MSL (meters)
        vel: Ground speed (m/s)
        hdg: Heading (degrees)
        satellites: Number of satellites
        fix_type: GPS fix type (0=no fix, 2=2D, 3=3D)
    """
    timestamp: float
    fc_timestamp: int
    lat: float
    lon: float
    alt: float
    vel: float = 0.0
    hdg: float = 0.0
    satellites: int = 0
    fix_type: int = 0


@dataclass
class FrameData:
    """
    Camera frame data.

    Attributes:
        timestamp: Local monotonic timestamp (seconds)
        frame_number: Sequential frame counter
        image: Numpy array of image data (H, W, C)
        width: Frame width
        height: Frame height
    """
    timestamp: float
    frame_number: int
    image: np.ndarray
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]


@dataclass
class MotionEstimate:
    """
    Visual motion estimate from frame-to-frame tracking.

    Attributes:
        timestamp: Timestamp of the second frame
        dt: Time delta between frames (seconds)
        rotation: Rotation vector (Rodrigues) or None if estimation failed
        translation: Translation vector (normalized) or None
        p: Estimated roll rate (rad/s)
        q: Estimated pitch rate (rad/s)
        r: Estimated yaw rate (rad/s)
        num_features: Number of features used
        inlier_ratio: Ratio of inliers in estimation
        valid: Whether estimation is valid
    """
    timestamp: float
    dt: float
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    num_features: int = 0
    inlier_ratio: float = 0.0
    valid: bool = False

    def gyro_vector(self) -> np.ndarray:
        """Return estimated gyro rates as numpy array [p, q, r]."""
        return np.array([self.p, self.q, self.r])


@dataclass
class TemporalCalibration:
    """
    Temporal calibration result.

    Attributes:
        time_offset: Camera time + offset = IMU time (seconds)
        confidence: Confidence score [0, 1]
        correlation_peak: Peak correlation value
        samples_used: Number of samples used in estimation
    """
    time_offset: float
    confidence: float
    correlation_peak: float
    samples_used: int

    def apply(self, camera_timestamp: float) -> float:
        """Convert camera timestamp to IMU timestamp."""
        return camera_timestamp + self.time_offset


@dataclass
class SpatialCalibration:
    """
    Spatial calibration result (camera mount offset).

    Attributes:
        yaw: Yaw offset (rad)
        pitch: Pitch offset (rad)
        roll: Roll offset (rad)
        rotation_matrix: 3x3 rotation matrix from camera to body frame
        rmse: Root mean square error (rad)
        samples_used: Number of samples used
    """
    yaw: float
    pitch: float
    roll: float
    rotation_matrix: np.ndarray
    rmse: float
    samples_used: int

    def yaw_deg(self) -> float:
        return np.degrees(self.yaw)

    def pitch_deg(self) -> float:
        return np.degrees(self.pitch)

    def roll_deg(self) -> float:
        return np.degrees(self.roll)

    def transform_attitude(self, camera_attitude: np.ndarray) -> np.ndarray:
        """Transform camera attitude to body frame."""
        return self.rotation_matrix @ camera_attitude


@dataclass
class SyncResult:
    """
    Result from temporal synchronization attempt.

    Attributes:
        timestamp: When this result was computed
        time_offset: Estimated time offset (camera_time - imu_time)
        correlation: Correlation coefficient
        converged: Whether sync has converged
        num_samples: Number of samples used
    """
    timestamp: float
    time_offset: float
    correlation: float
    converged: bool
    num_samples: int


@dataclass
class CalibrationResult:
    """
    Complete calibration result.

    Attributes:
        temporal: Temporal calibration result
        spatial: Spatial calibration result
        timestamp: When calibration was performed
        duration: How long calibration took (seconds)
        state: Final calibration state
    """
    temporal: Optional[TemporalCalibration] = None
    spatial: Optional[SpatialCalibration] = None
    timestamp: float = 0.0
    duration: float = 0.0
    state: SyncState = SyncState.INITIALIZING

    def is_valid(self) -> bool:
        """Check if calibration is complete and valid."""
        return (self.temporal is not None and
                self.spatial is not None and
                self.state == SyncState.SYNCHRONIZED)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "state": self.state.value,
            "valid": self.is_valid()
        }

        if self.temporal:
            result["temporal_calibration"] = {
                "time_offset_seconds": self.temporal.time_offset,
                "confidence": self.temporal.confidence,
                "correlation_peak": self.temporal.correlation_peak,
                "samples_used": self.temporal.samples_used
            }

        if self.spatial:
            result["spatial_calibration"] = {
                "mount_rotation": {
                    "yaw_deg": self.spatial.yaw_deg(),
                    "pitch_deg": self.spatial.pitch_deg(),
                    "roll_deg": self.spatial.roll_deg()
                },
                "rmse_deg": np.degrees(self.spatial.rmse),
                "samples_used": self.spatial.samples_used
            }

        return result


@dataclass
class SystemStatus:
    """
    Real-time system status.

    Attributes:
        mavlink_connected: MAVLink connection status
        camera_connected: Camera connection status
        imu_rate: Current IMU message rate (Hz)
        frame_rate: Current camera frame rate (Hz)
        sync_state: Current synchronization state
        time_offset: Current time offset estimate
        buffer_fill: Ring buffer fill levels
    """
    mavlink_connected: bool = False
    camera_connected: bool = False
    imu_rate: float = 0.0
    frame_rate: float = 0.0
    attitude_rate: float = 0.0
    sync_state: SyncState = SyncState.INITIALIZING
    time_offset: float = 0.0
    time_offset_confidence: float = 0.0
    imu_buffer_fill: float = 0.0
    frame_buffer_fill: float = 0.0
