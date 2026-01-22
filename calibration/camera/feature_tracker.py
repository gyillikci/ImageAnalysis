"""
Feature tracker for frame-to-frame motion estimation.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrackedFeatures:
    """Container for tracked feature points."""
    points: np.ndarray          # Nx2 array of (x, y) points
    ids: np.ndarray             # N array of feature IDs
    ages: np.ndarray            # N array of track ages (frames)
    timestamp: float            # Frame timestamp


class FeatureTracker:
    """
    Feature point tracker using optical flow.

    Tracks features across frames using Lucas-Kanade optical flow
    and maintains feature IDs for long-term tracking.
    """

    def __init__(self,
                 max_features: int = 500,
                 quality_level: float = 0.01,
                 min_distance: int = 10,
                 block_size: int = 7,
                 use_harris: bool = False,
                 lk_win_size: Tuple[int, int] = (21, 21),
                 lk_max_level: int = 3):
        """
        Initialize feature tracker.

        Args:
            max_features: Maximum number of features to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between features
            block_size: Block size for corner detection
            use_harris: Use Harris corner detector (vs Shi-Tomasi)
            lk_win_size: Window size for Lucas-Kanade
            lk_max_level: Max pyramid level for Lucas-Kanade
        """
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.use_harris = use_harris
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level

        # LK optical flow parameters
        self._lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(3, 10, 0.03)  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
        )

        # State
        self._prev_gray = None
        self._prev_points = None
        self._feature_ids = None
        self._feature_ages = None
        self._next_id = 0
        self._prev_timestamp = 0.0

    def reset(self) -> None:
        """Reset tracker state."""
        self._prev_gray = None
        self._prev_points = None
        self._feature_ids = None
        self._feature_ages = None
        self._next_id = 0
        self._prev_timestamp = 0.0

    def process(self, frame: np.ndarray,
                timestamp: float) -> Tuple[Optional[TrackedFeatures],
                                            Optional[TrackedFeatures]]:
        """
        Process new frame and track features.

        Args:
            frame: Input frame (BGR or grayscale)
            timestamp: Frame timestamp

        Returns:
            (prev_features, curr_features) or (None, None) if tracking failed
        """
        import cv2

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # First frame - detect features
        if self._prev_gray is None:
            self._detect_features(gray, timestamp)
            return (None, None)

        # Track features using optical flow
        if self._prev_points is None or len(self._prev_points) == 0:
            self._detect_features(gray, timestamp)
            return (None, None)

        # Run optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_points,
            None,
            **self._lk_params
        )

        # Filter good points
        if curr_points is None:
            self._detect_features(gray, timestamp)
            return (None, None)

        status = status.reshape(-1)
        good_mask = status == 1

        # Create feature containers
        prev_features = TrackedFeatures(
            points=self._prev_points[good_mask],
            ids=self._feature_ids[good_mask],
            ages=self._feature_ages[good_mask],
            timestamp=self._prev_timestamp
        )

        curr_features = TrackedFeatures(
            points=curr_points[good_mask],
            ids=self._feature_ids[good_mask],
            ages=self._feature_ages[good_mask] + 1,
            timestamp=timestamp
        )

        # Update state
        self._prev_gray = gray
        self._prev_points = curr_points[good_mask]
        self._feature_ids = self._feature_ids[good_mask]
        self._feature_ages = self._feature_ages[good_mask] + 1
        self._prev_timestamp = timestamp

        # Detect new features if needed
        if len(self._prev_points) < self.max_features * 0.5:
            self._add_new_features(gray)

        return (prev_features, curr_features)

    def _detect_features(self, gray: np.ndarray, timestamp: float) -> None:
        """Detect new features in frame."""
        import cv2

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris
        )

        if corners is not None:
            self._prev_points = corners.reshape(-1, 2).astype(np.float32)
            n = len(self._prev_points)
            self._feature_ids = np.arange(self._next_id, self._next_id + n)
            self._feature_ages = np.zeros(n, dtype=np.int32)
            self._next_id += n
        else:
            self._prev_points = np.array([], dtype=np.float32).reshape(0, 2)
            self._feature_ids = np.array([], dtype=np.int64)
            self._feature_ages = np.array([], dtype=np.int32)

        self._prev_gray = gray
        self._prev_timestamp = timestamp

    def _add_new_features(self, gray: np.ndarray) -> None:
        """Add new features to existing tracks."""
        import cv2

        # Create mask to avoid existing features
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for pt in self._prev_points:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), self.min_distance, 0, -1)

        # Detect new corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features - len(self._prev_points),
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris,
            mask=mask
        )

        if corners is not None and len(corners) > 0:
            new_points = corners.reshape(-1, 2).astype(np.float32)
            n = len(new_points)
            new_ids = np.arange(self._next_id, self._next_id + n)
            new_ages = np.zeros(n, dtype=np.int32)
            self._next_id += n

            # Append to existing
            self._prev_points = np.vstack([self._prev_points, new_points])
            self._feature_ids = np.concatenate([self._feature_ids, new_ids])
            self._feature_ages = np.concatenate([self._feature_ages, new_ages])

    @property
    def num_features(self) -> int:
        """Current number of tracked features."""
        if self._prev_points is None:
            return 0
        return len(self._prev_points)


class StereoFeatureTracker:
    """
    Feature tracker that maintains correspondence between frames.

    Useful for computing essential matrix and recovering rotation.
    """

    def __init__(self, **kwargs):
        """Initialize with same parameters as FeatureTracker."""
        self._tracker = FeatureTracker(**kwargs)
        self._history: List[TrackedFeatures] = []
        self._max_history = 10

    def reset(self) -> None:
        """Reset tracker."""
        self._tracker.reset()
        self._history.clear()

    def process(self, frame: np.ndarray,
                timestamp: float) -> Tuple[Optional[np.ndarray],
                                            Optional[np.ndarray],
                                            float]:
        """
        Process frame and return matched point pairs.

        Args:
            frame: Input frame
            timestamp: Frame timestamp

        Returns:
            (prev_points, curr_points, dt) - matched point arrays and time delta
        """
        prev_feats, curr_feats = self._tracker.process(frame, timestamp)

        if prev_feats is None or curr_feats is None:
            return (None, None, 0.0)

        if len(prev_feats.points) < 8:
            return (None, None, 0.0)

        dt = curr_feats.timestamp - prev_feats.timestamp

        return (prev_feats.points, curr_feats.points, dt)

    @property
    def num_features(self) -> int:
        return self._tracker.num_features
