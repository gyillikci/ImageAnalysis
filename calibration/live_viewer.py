#!/usr/bin/env python3
"""
Live Calibration Viewer

Real-time visualization of camera-IMU calibration process.
Shows camera feed, tracked features, and calibration progress.

Usage:
    python live_viewer.py --mavlink udp:127.0.0.1:14550 --camera 0

Author: ImageAnalysis Project
"""

import argparse
import time
import sys
import numpy as np
import cv2

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import FrameData
from core.utils import monotonic_time
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig
from camera.feature_tracker import FeatureTracker
from camera.motion_estimator import MotionEstimator, MotionEstimatorConfig
from sync.temporal import TemporalSynchronizer, TemporalSyncConfig
from sync.spatial import SpatialCalibrator, SpatialCalibConfig


class LiveViewer:
    """
    Live visualization of calibration process.
    """

    def __init__(self, args):
        """Initialize live viewer."""
        self.args = args

        # Configure MAVLink
        self.mavlink_config = MAVLinkConfig(
            connection_string=args.mavlink,
            baudrate=args.baudrate
        )

        # Configure camera
        camera_id = args.camera
        if camera_id.isdigit():
            camera_id = int(camera_id)

        self.camera_config = CameraConfig(
            device_id=camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps
        )

        # Motion config
        self.motion_config = MotionEstimatorConfig(
            max_features=args.max_features,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx if args.cx else args.width / 2,
            cy=args.cy if args.cy else args.height / 2
        )

        # Components
        self.mavlink = None
        self.camera = None
        self.feature_tracker = None
        self.motion_estimator = None
        self.temporal_sync = None
        self.spatial_calib = None

        # State
        self.running = False
        self.show_features = True
        self.show_flow = True
        self.show_info = True

    def setup(self) -> bool:
        """Initialize all components."""
        print("Initializing live viewer...")

        # MAVLink
        self.mavlink = MAVLinkReceiver(self.mavlink_config)
        if not self.mavlink.start():
            print("Failed to start MAVLink receiver")
            return False

        # Camera
        self.camera = HarrierCamera(self.camera_config)
        if not self.camera.start():
            print("Failed to start camera")
            return False

        # Feature tracker
        self.feature_tracker = FeatureTracker(
            max_features=self.args.max_features,
            quality_level=0.01,
            min_distance=10
        )

        # Motion estimator
        self.motion_estimator = MotionEstimator(self.motion_config)

        # Temporal sync
        self.temporal_sync = TemporalSynchronizer(TemporalSyncConfig(
            sample_rate=self.args.fps
        ))
        self.temporal_sync.start()

        # Spatial calibration
        self.spatial_calib = SpatialCalibrator(SpatialCalibConfig())
        self.spatial_calib.start()

        # Connect callbacks
        self.mavlink.add_imu_callback(self.temporal_sync.on_imu_data)

        print("Setup complete!")
        return True

    def run(self) -> None:
        """Run the live viewer."""
        self.running = True

        print("\nLive Viewer Controls:")
        print("  'q' - Quit")
        print("  'f' - Toggle features")
        print("  'o' - Toggle optical flow")
        print("  'i' - Toggle info overlay")
        print("  'r' - Reset calibration")
        print()

        cv2.namedWindow('Live Calibration', cv2.WINDOW_NORMAL)

        frame_count = 0
        fps_counter = 0
        fps_time = monotonic_time()
        display_fps = 0.0

        while self.running:
            # Get frame
            frame_data = self.camera.get_latest_frame()
            if frame_data is None:
                time.sleep(0.001)
                continue

            frame_count += 1

            # Track features
            gray = cv2.cvtColor(frame_data.image, cv2.COLOR_BGR2GRAY)
            prev_feats, curr_feats = self.feature_tracker.process(
                gray, frame_data.timestamp
            )

            # Estimate motion
            motion = self.motion_estimator.process_frame(frame_data)

            # Feed to synchronizers
            self.temporal_sync.on_motion_data(motion)

            if self.temporal_sync.is_converged:
                time_offset = self.temporal_sync.time_offset
                imu = self.mavlink.imu_buffer.get_nearest(
                    motion.timestamp + time_offset
                )
                if imu is not None and motion.valid:
                    self.spatial_calib.add_sample(imu, motion, 0.0)

            # Create display frame
            display = frame_data.image.copy()

            # Draw features
            if self.show_features and curr_feats is not None:
                for pt in curr_feats.points:
                    cv2.circle(display, (int(pt[0]), int(pt[1])),
                               3, (0, 255, 0), -1)

            # Draw optical flow
            if self.show_flow and prev_feats is not None and curr_feats is not None:
                for i in range(min(len(prev_feats.points), len(curr_feats.points))):
                    p1 = prev_feats.points[i]
                    p2 = curr_feats.points[i]
                    cv2.arrowedLine(display,
                                   (int(p1[0]), int(p1[1])),
                                   (int(p2[0]), int(p2[1])),
                                   (255, 0, 0), 1, tipLength=0.3)

            # Draw info overlay
            if self.show_info:
                self._draw_info(display, motion)

            # Update FPS counter
            fps_counter += 1
            now = monotonic_time()
            if now - fps_time >= 1.0:
                display_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # Draw FPS
            cv2.putText(display, f"FPS: {display_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Live Calibration', display)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('f'):
                self.show_features = not self.show_features
            elif key == ord('o'):
                self.show_flow = not self.show_flow
            elif key == ord('i'):
                self.show_info = not self.show_info
            elif key == ord('r'):
                self._reset_calibration()

        cv2.destroyAllWindows()

    def _draw_info(self, frame, motion) -> None:
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (255, 255, 255)
        x = w - 290
        y = 25
        dy = 25

        # MAVLink status
        mav = self.mavlink.get_status()
        cv2.putText(frame, f"MAVLink: {'OK' if mav['connected'] else 'NO'}",
                    (x, y), font, scale, color, 1)
        y += dy

        cv2.putText(frame, f"IMU Rate: {mav['imu_rate']:.0f} Hz",
                    (x, y), font, scale, color, 1)
        y += dy

        # Motion
        if motion.valid:
            cv2.putText(frame, f"Features: {motion.num_features}",
                        (x, y), font, scale, color, 1)
            y += dy

            cv2.putText(frame, f"p:{motion.p*57.3:.1f} q:{motion.q*57.3:.1f} r:{motion.r*57.3:.1f}",
                        (x, y), font, scale, color, 1)
            y += dy

        # Temporal sync
        ts = self.temporal_sync.get_status()
        status_color = (0, 255, 0) if ts['converged'] else (0, 165, 255)
        cv2.putText(frame, f"Time Offset: {ts['time_offset']*1000:.1f} ms",
                    (x, y), font, scale, status_color, 1)
        y += dy

        cv2.putText(frame, f"Correlation: {ts['correlation']:.3f}",
                    (x, y), font, scale, status_color, 1)
        y += dy

        # Spatial calibration
        sc = self.spatial_calib.get_status()
        status_color = (0, 255, 0) if sc['converged'] else (0, 165, 255)
        cv2.putText(frame, f"Mount Y:{sc['yaw_deg']:.1f} P:{sc['pitch_deg']:.1f} R:{sc['roll_deg']:.1f}",
                    (x, y), font, scale, status_color, 1)
        y += dy

        # Convergence status
        y = h - 30
        if self.temporal_sync.is_converged and self.spatial_calib.is_converged:
            cv2.putText(frame, "CALIBRATION COMPLETE",
                        (w//2 - 120, y), font, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Calibrating...",
                        (w//2 - 60, y), font, 0.8, (0, 165, 255), 2)

    def _reset_calibration(self) -> None:
        """Reset calibration state."""
        print("Resetting calibration...")
        self.temporal_sync.reset()
        self.spatial_calib.reset()
        self.feature_tracker.reset()

    def cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up...")

        if self.temporal_sync:
            self.temporal_sync.stop()

        if self.spatial_calib:
            self.spatial_calib.stop()

        if self.camera:
            self.camera.stop()

        if self.mavlink:
            self.mavlink.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Live Calibration Viewer'
    )

    # MAVLink
    parser.add_argument('--mavlink', required=True,
                        help='MAVLink connection string')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate')

    # Camera
    parser.add_argument('--camera', default='0',
                        help='Camera device ID')
    parser.add_argument('--width', type=int, default=1280,
                        help='Frame width')
    parser.add_argument('--height', type=int, default=720,
                        help='Frame height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame rate')

    # Camera intrinsics
    parser.add_argument('--fx', type=float, default=800.0,
                        help='Focal length X')
    parser.add_argument('--fy', type=float, default=800.0,
                        help='Focal length Y')
    parser.add_argument('--cx', type=float, default=None,
                        help='Principal point X')
    parser.add_argument('--cy', type=float, default=None,
                        help='Principal point Y')

    # Features
    parser.add_argument('--max-features', type=int, default=500,
                        help='Maximum features to track')

    args = parser.parse_args()

    viewer = LiveViewer(args)

    try:
        if not viewer.setup():
            print("Setup failed!")
            sys.exit(1)

        viewer.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        viewer.cleanup()


if __name__ == '__main__':
    main()
