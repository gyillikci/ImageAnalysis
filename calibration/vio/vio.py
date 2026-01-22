"""
Visual-Inertial Odometry main class.

Combines camera tracking, IMU integration, and EKF fusion.
"""

import numpy as np
import cv2
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple
from collections import deque
from scipy.spatial.transform import Rotation

from .state import VIOState, Pose
from .ekf import VIOEKF, VIOConfig


@dataclass
class VIOCalibration:
    """Camera-IMU calibration parameters."""
    # Temporal
    time_offset: float = -0.083  # seconds (camera leads IMU)
    
    # Spatial (camera mount offset relative to IMU)
    yaw: float = 4.56    # degrees
    pitch: float = -11.25  # degrees
    roll: float = -1.09   # degrees
    
    # Camera intrinsics
    fx: float = 1175.0
    fy: float = 1175.0
    cx: float = 640.0
    cy: float = 360.0
    
    def get_R_cam_imu(self) -> np.ndarray:
        """Get rotation matrix from IMU to camera frame."""
        r = Rotation.from_euler('zyx', [self.yaw, self.pitch, self.roll], degrees=True)
        return r.as_matrix()


@dataclass
class TrackedFeature:
    """A tracked feature across frames."""
    id: int
    positions: List[np.ndarray]  # 2D positions in each frame
    timestamps: List[float]
    position_3d: Optional[np.ndarray] = None  # Triangulated 3D position
    
    @property
    def last_position(self) -> np.ndarray:
        return self.positions[-1] if self.positions else None
    
    @property
    def track_length(self) -> int:
        return len(self.positions)


class FeatureTracker:
    """Feature detection and tracking for VIO."""
    
    def __init__(self, max_features: int = 200, min_distance: int = 20):
        self.max_features = max_features
        self.min_distance = min_distance
        
        # Feature detection params
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=7
        )
        
        # LK optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.prev_pts = None
        self.feature_id = 0
        self.tracks: List[TrackedFeature] = []
    
    def process(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Process a frame and track features.
        
        Returns:
            Tuple of (prev_points, curr_points, track_ids)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.prev_gray is None:
            # First frame - detect features
            pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if pts is not None:
                self.prev_pts = pts.reshape(-1, 2)
                for pt in self.prev_pts:
                    self.tracks.append(TrackedFeature(
                        id=self.feature_id,
                        positions=[pt.copy()],
                        timestamps=[timestamp]
                    ))
                    self.feature_id += 1
            else:
                self.prev_pts = np.array([]).reshape(-1, 2)
            
            self.prev_gray = gray
            return None, None, None
        
        if len(self.prev_pts) < 10:
            # Need to detect new features
            pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if pts is not None:
                self.prev_pts = pts.reshape(-1, 2)
                self.tracks = []
                for pt in self.prev_pts:
                    self.tracks.append(TrackedFeature(
                        id=self.feature_id,
                        positions=[pt.copy()],
                        timestamps=[timestamp]
                    ))
                    self.feature_id += 1
            
            self.prev_gray = gray
            return None, None, None
        
        # Track features
        prev_pts_cv = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts_cv, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray
            return None, None, None
        
        # Filter good matches
        status = status.flatten()
        good_prev = self.prev_pts[status == 1]
        good_curr = curr_pts[status == 1].reshape(-1, 2)
        good_tracks = [t for t, s in zip(self.tracks, status) if s == 1]
        
        # Update tracks
        for track, pt in zip(good_tracks, good_curr):
            track.positions.append(pt.copy())
            track.timestamps.append(timestamp)
        
        # Get track IDs
        track_ids = [t.id for t in good_tracks]
        
        # Add new features if needed
        if len(good_curr) < self.max_features // 2:
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            for pt in good_curr:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), self.min_distance, 0, -1)
            
            new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if new_pts is not None:
                new_pts = new_pts.reshape(-1, 2)
                for pt in new_pts:
                    good_tracks.append(TrackedFeature(
                        id=self.feature_id,
                        positions=[pt.copy()],
                        timestamps=[timestamp]
                    ))
                    self.feature_id += 1
                good_curr = np.vstack([good_curr, new_pts])
        
        self.prev_gray = gray
        self.prev_pts = good_curr
        self.tracks = good_tracks
        
        return good_prev, good_curr, track_ids
    
    def get_long_tracks(self, min_length: int = 5) -> List[TrackedFeature]:
        """Get features tracked for at least min_length frames."""
        return [t for t in self.tracks if t.track_length >= min_length]


class VisualInertialOdometry:
    """
    Main VIO class.
    
    Integrates feature tracking, IMU measurements, and EKF fusion
    to estimate 6-DOF pose in real-time.
    """
    
    def __init__(self, calibration: VIOCalibration = None):
        """Initialize VIO."""
        self.calibration = calibration or VIOCalibration()
        
        # VIO config
        config = VIOConfig(
            time_offset=self.calibration.time_offset,
            R_cam_imu=self.calibration.get_R_cam_imu(),
            fx=self.calibration.fx,
            fy=self.calibration.fy,
            cx=self.calibration.cx,
            cy=self.calibration.cy
        )
        
        # EKF
        self.ekf = VIOEKF(config)
        
        # Feature tracker
        self.tracker = FeatureTracker()
        
        # IMU buffer (for time offset correction)
        self.imu_buffer: deque = deque(maxlen=1000)
        
        # Keyframe management
        self.keyframes: List[Tuple[float, Pose, np.ndarray]] = []  # (timestamp, pose, features)
        self.last_keyframe_time = None
        self.keyframe_interval = 0.1  # seconds
        
        # Callbacks
        self._pose_callbacks: List[Callable[[Pose], None]] = []
        
        # State
        self._lock = threading.Lock()
        self._initialized = False
        
        # Map (simple feature triangulation)
        self.map_points: dict = {}  # feature_id -> 3D position
    
    def add_pose_callback(self, callback: Callable[[Pose], None]):
        """Add callback for pose updates."""
        self._pose_callbacks.append(callback)
    
    def _fire_pose_callbacks(self, pose: Pose):
        """Fire pose callbacks."""
        for cb in self._pose_callbacks:
            try:
                cb(pose)
            except Exception as e:
                print(f"Pose callback error: {e}")
    
    def initialize(self, timestamp: float = 0.0):
        """Initialize VIO at origin."""
        with self._lock:
            self.ekf.initialize(timestamp=timestamp)
            self._initialized = True
            print(f"VIO initialized at t={timestamp:.3f}")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def pose(self) -> Pose:
        """Current pose estimate."""
        return self.ekf.pose
    
    @property
    def state(self) -> VIOState:
        """Current state estimate."""
        return self.ekf.state
    
    def on_imu(self, timestamp: float, gyro: np.ndarray, accel: np.ndarray):
        """
        Process IMU measurement.
        
        Args:
            timestamp: IMU timestamp
            gyro: Gyroscope [wx, wy, wz] in rad/s
            accel: Accelerometer [ax, ay, az] in m/s^2
        """
        # Buffer IMU for time offset correction
        self.imu_buffer.append((timestamp, gyro.copy(), accel.copy()))
        
        if not self._initialized:
            return
        
        with self._lock:
            self.ekf.predict_imu(timestamp, gyro, accel)
    
    def on_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Pose]:
        """
        Process camera frame.
        
        Args:
            frame: Camera image
            timestamp: Camera timestamp
            
        Returns:
            Updated pose estimate
        """
        # Apply time offset
        adjusted_time = timestamp + self.calibration.time_offset
        
        # Track features
        prev_pts, curr_pts, track_ids = self.tracker.process(frame, adjusted_time)
        
        if prev_pts is None or len(prev_pts) < 10:
            return self.pose if self._initialized else None
        
        if not self._initialized:
            self.initialize(adjusted_time)
        
        with self._lock:
            # Compute visual odometry (simple essential matrix approach)
            # For a full VIO, you'd use the triangulated map points
            
            # Check if we should create a keyframe
            if self._should_create_keyframe(adjusted_time, prev_pts, curr_pts):
                self._create_keyframe(adjusted_time, frame, curr_pts, track_ids)
            
            # Visual update using tracked features with known 3D positions
            features_2d = []
            features_3d = []
            
            for track_id, pt in zip(track_ids, curr_pts):
                if track_id in self.map_points:
                    features_2d.append(pt)
                    features_3d.append(self.map_points[track_id])
            
            if len(features_2d) >= 4:
                features_2d = np.array(features_2d)
                features_3d = np.array(features_3d)
                self.ekf.update_visual(features_2d, features_3d, adjusted_time)
            
            pose = self.ekf.pose
        
        self._fire_pose_callbacks(pose)
        return pose
    
    def _should_create_keyframe(self, timestamp: float, prev_pts: np.ndarray, 
                                 curr_pts: np.ndarray) -> bool:
        """Determine if we should create a new keyframe."""
        if self.last_keyframe_time is None:
            return True
        
        # Time-based keyframe
        if timestamp - self.last_keyframe_time > self.keyframe_interval:
            return True
        
        # Motion-based keyframe
        if prev_pts is not None and curr_pts is not None:
            # Use minimum length to handle mismatched arrays
            min_len = min(len(prev_pts), len(curr_pts))
            if min_len > 0:
                flow = np.mean(np.linalg.norm(curr_pts[:min_len] - prev_pts[:min_len], axis=1))
                if flow > 30:  # pixels
                    return True
        
        return False
    
    def _create_keyframe(self, timestamp: float, frame: np.ndarray,
                         features: np.ndarray, track_ids: List[int]):
        """Create a new keyframe and triangulate features."""
        pose = self.ekf.pose
        self.keyframes.append((timestamp, pose, features.copy()))
        self.last_keyframe_time = timestamp
        
        # Triangulate new features using last two keyframes
        if len(self.keyframes) >= 2:
            self._triangulate_features(track_ids)
        
        # Reset preintegration
        self.ekf.reset_preintegration()
    
    def _triangulate_features(self, current_track_ids: List[int]):
        """Triangulate features from keyframe pairs."""
        if len(self.keyframes) < 2:
            return
        
        # Get last two keyframes
        t1, pose1, pts1 = self.keyframes[-2]
        t2, pose2, pts2 = self.keyframes[-1]
        
        # Find common tracks
        tracks = self.tracker.get_long_tracks(min_length=2)
        
        for track in tracks:
            if track.id in self.map_points:
                continue
            
            if len(track.positions) < 2:
                continue
            
            # Get observations at both keyframes
            pt1 = track.positions[-2]
            pt2 = track.positions[-1]
            
            # Triangulate
            p3d = self._triangulate_point(pose1, pose2, pt1, pt2)
            
            if p3d is not None:
                self.map_points[track.id] = p3d
    
    def _triangulate_point(self, pose1: Pose, pose2: Pose, 
                           pt1: np.ndarray, pt2: np.ndarray) -> Optional[np.ndarray]:
        """Triangulate a 3D point from two views."""
        # Camera matrices
        K = np.array([
            [self.calibration.fx, 0, self.calibration.cx],
            [0, self.calibration.fy, self.calibration.cy],
            [0, 0, 1]
        ])
        
        # Camera-IMU transform
        R_c_i = self.calibration.get_R_cam_imu()
        
        # Camera poses in world frame
        R1 = pose1.rotation_matrix() @ R_c_i.T
        t1 = pose1.position
        
        R2 = pose2.rotation_matrix() @ R_c_i.T
        t2 = pose2.position
        
        # Projection matrices
        P1 = K @ np.hstack([R1.T, -R1.T @ t1.reshape(3, 1)])
        P2 = K @ np.hstack([R2.T, -R2.T @ t2.reshape(3, 1)])
        
        # Triangulate
        pts_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
        
        if pts_4d[3] == 0:
            return None
        
        pt_3d = pts_4d[:3] / pts_4d[3]
        pt_3d = pt_3d.flatten()
        
        # Check if point is in front of both cameras
        p1_cam = R1.T @ (pt_3d - t1)
        p2_cam = R2.T @ (pt_3d - t2)
        
        if p1_cam[2] < 0.1 or p2_cam[2] < 0.1:
            return None
        
        # Check reprojection error
        pt1_reproj = P1 @ np.append(pt_3d, 1)
        pt1_reproj = pt1_reproj[:2] / pt1_reproj[2]
        
        error = np.linalg.norm(pt1_reproj - pt1)
        if error > 5.0:  # pixels
            return None
        
        return pt_3d
    
    def get_trajectory(self) -> List[Pose]:
        """Get list of keyframe poses."""
        return [pose for _, pose, _ in self.keyframes]
    
    def get_map_points(self) -> np.ndarray:
        """Get all 3D map points."""
        if not self.map_points:
            return np.array([]).reshape(0, 3)
        return np.array(list(self.map_points.values()))
