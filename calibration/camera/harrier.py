"""
Harrier camera interface using OpenCV (UVC compatible).

The Harrier camera uses USB Video Class (UVC), so it works with
standard OpenCV VideoCapture.
"""

import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import FrameData
from core.ring_buffer import TimestampedRingBuffer
from core.utils import monotonic_time, RateEstimator


@dataclass
class CameraConfig:
    """Camera configuration."""
    device: int = 0                    # Device index or path
    width: int = 1920                  # Frame width
    height: int = 1080                 # Frame height
    fps: int = 60                      # Target frame rate
    buffer_size: int = 300             # Frame buffer size
    auto_exposure: bool = True         # Enable auto exposure
    auto_focus: bool = True            # Enable auto focus
    exposure: int = -1                 # Manual exposure (-1 = auto)
    gain: int = -1                     # Manual gain (-1 = auto)

    # Camera intrinsics (optional, for undistortion)
    fx: float = 0.0                    # Focal length x
    fy: float = 0.0                    # Focal length y
    cx: float = 0.0                    # Principal point x
    cy: float = 0.0                    # Principal point y
    dist_coeffs: list = field(default_factory=lambda: [0, 0, 0, 0, 0])

    @property
    def has_intrinsics(self) -> bool:
        """Check if camera intrinsics are set."""
        return self.fx > 0 and self.fy > 0

    def get_camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_dist_coeffs(self) -> np.ndarray:
        """Get distortion coefficients."""
        return np.array(self.dist_coeffs, dtype=np.float64)


class HarrierCamera:
    """
    Harrier camera capture interface.

    Captures frames in a background thread and stores them in a ring buffer.
    Compatible with any UVC camera (including Harrier USB/HDMI).

    Usage:
        camera = HarrierCamera(config)
        camera.start()

        # Get latest frame
        frame = camera.get_latest_frame()

        # Or use callback
        camera.add_callback(my_frame_handler)

        camera.stop()
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize camera.

        Args:
            config: Camera configuration
        """
        self.config = config
        self._capture = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame buffer
        self.frame_buffer = TimestampedRingBuffer[FrameData](config.buffer_size)

        # Rate estimator
        self._frame_rate = RateEstimator(window_size=60)

        # Frame counter
        self._frame_count = 0

        # Callbacks
        self._callbacks = []

        # Status
        self._connected = False
        self._last_frame_time = 0.0

        # Undistortion maps (computed once if intrinsics available)
        self._undistort_map1 = None
        self._undistort_map2 = None

    def add_callback(self, callback: Callable[[FrameData], None]) -> None:
        """Add callback for new frames."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove frame callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire_callbacks(self, frame: FrameData) -> None:
        """Fire callbacks for new frame."""
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                print(f"Frame callback error: {e}")

    def start(self) -> bool:
        """
        Start camera capture.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            self._connect()
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False

    def stop(self) -> None:
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._disconnect()

    def _connect(self) -> None:
        """Open camera connection."""
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")

        device = self.config.device

        print(f"Opening camera: {device}")

        # Open capture
        if isinstance(device, str):
            self._capture = cv2.VideoCapture(device)
        else:
            self._capture = cv2.VideoCapture(device, cv2.CAP_V4L2)  # Use V4L2 on Linux

        if not self._capture.isOpened():
            # Try without V4L2 backend
            self._capture = cv2.VideoCapture(device)

        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera: {device}")

        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        # Set frame rate
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Set buffer size (reduce latency)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Configure exposure
        if not self.config.auto_exposure and self.config.exposure > 0:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            self._capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)

        # Read actual settings
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")

        # Update config with actual values
        self.config.width = actual_width
        self.config.height = actual_height

        # Compute undistortion maps if intrinsics available
        if self.config.has_intrinsics:
            self._compute_undistort_maps()

        self._connected = True

    def _disconnect(self) -> None:
        """Close camera connection."""
        if self._capture:
            self._capture.release()
            self._capture = None
        self._connected = False

    def _compute_undistort_maps(self) -> None:
        """Pre-compute undistortion maps for efficiency."""
        import cv2

        K = self.config.get_camera_matrix()
        dist = self.config.get_dist_coeffs()
        size = (self.config.width, self.config.height)

        # Compute optimal camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)

        # Compute undistortion maps
        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, size, cv2.CV_16SC2
        )

        print("Undistortion maps computed")

    def _capture_loop(self) -> None:
        """Main capture loop (runs in thread)."""
        import cv2

        while self._running:
            try:
                if not self._capture or not self._capture.isOpened():
                    time.sleep(0.01)
                    continue

                # Grab frame
                ret, frame = self._capture.read()

                # Get timestamp immediately after capture
                local_ts = monotonic_time()

                if not ret or frame is None:
                    time.sleep(0.001)
                    continue

                # Apply undistortion if available
                if self._undistort_map1 is not None:
                    frame = cv2.remap(frame, self._undistort_map1,
                                     self._undistort_map2, cv2.INTER_LINEAR)

                # Create frame data
                self._frame_count += 1
                frame_data = FrameData(
                    timestamp=local_ts,
                    frame_number=self._frame_count,
                    image=frame
                )

                # Store in buffer
                self.frame_buffer.push(frame_data)

                # Update rate
                self._frame_rate.update(local_ts)
                self._last_frame_time = local_ts

                # Fire callbacks
                self._fire_callbacks(frame_data)

            except Exception as e:
                if self._running:
                    print(f"Capture error: {e}")
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[FrameData]:
        """Get most recent frame."""
        return self.frame_buffer.peek_newest()

    def get_frame_at_time(self, timestamp: float) -> Optional[FrameData]:
        """Get frame nearest to specified timestamp."""
        return self.frame_buffer.get_nearest(timestamp)

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected and capturing."""
        if not self._connected:
            return False
        # Check for recent frame
        return (monotonic_time() - self._last_frame_time) < 1.0

    @property
    def frame_rate(self) -> float:
        """Current frame rate (Hz)."""
        return self._frame_rate.rate

    @property
    def frame_count(self) -> int:
        """Total frames captured."""
        return self._frame_count

    def get_status(self) -> dict:
        """Get current camera status."""
        return {
            'connected': self.is_connected,
            'frame_rate': self.frame_rate,
            'frame_count': self.frame_count,
            'buffer_fill': self.frame_buffer.fill_ratio,
            'resolution': (self.config.width, self.config.height)
        }

    def capture_single(self) -> Optional[FrameData]:
        """
        Capture a single frame (blocking).

        Useful for testing without starting the capture thread.
        """
        import cv2

        if self._capture is None:
            self._connect()

        if not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        local_ts = monotonic_time()

        if not ret or frame is None:
            return None

        if self._undistort_map1 is not None:
            frame = cv2.remap(frame, self._undistort_map1,
                             self._undistort_map2, cv2.INTER_LINEAR)

        self._frame_count += 1
        return FrameData(
            timestamp=local_ts,
            frame_number=self._frame_count,
            image=frame
        )

    def set_exposure(self, value: int) -> None:
        """Set manual exposure value."""
        import cv2
        if self._capture:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, value)

    def set_auto_exposure(self, enabled: bool = True) -> None:
        """Enable/disable auto exposure."""
        import cv2
        if self._capture:
            if enabled:
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
