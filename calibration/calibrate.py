#!/usr/bin/env python3
"""
Real-time Camera-IMU Calibration Application

This application performs temporal and spatial calibration between a
Harrier camera and an Orange Cube flight controller using MAVLink.

Usage:
    python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0
    python calibrate.py --mavlink /dev/ttyACM0 --camera /dev/video0

Author: ImageAnalysis Project
"""

import argparse
import time
import json
import signal
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import (
    IMUData, MotionEstimate, CalibrationResult,
    TemporalCalibration, SpatialCalibration, SyncState
)
from core.utils import monotonic_time
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig
from camera.motion_estimator import MotionEstimator, MotionEstimatorConfig
from sync.temporal import TemporalSynchronizer, TemporalSyncConfig
from sync.spatial import SpatialCalibrator, SpatialCalibConfig


class CalibrationApp:
    """
    Main calibration application.

    Coordinates all modules to perform real-time camera-IMU calibration.
    """

    def __init__(self, args):
        """
        Initialize calibration application.

        Args:
            args: Command line arguments
        """
        self.args = args

        # MAVLink configuration
        self.mavlink_config = MAVLinkConfig(
            connection_string=args.mavlink,
            baudrate=args.baudrate
        )

        # Camera configuration
        camera_id = args.camera
        if camera_id.isdigit():
            camera_id = int(camera_id)

        self.camera_config = CameraConfig(
            device_id=camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps
        )

        # Motion estimator configuration
        self.motion_config = MotionEstimatorConfig(
            max_features=args.max_features,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx if args.cx else args.width / 2,
            cy=args.cy if args.cy else args.height / 2
        )

        # Temporal sync configuration
        self.temporal_config = TemporalSyncConfig(
            sample_rate=args.fps,
            min_window_duration=args.min_duration,
            use_axis=args.sync_axis
        )

        # Spatial calibration configuration
        self.spatial_config = SpatialCalibConfig(
            min_samples=args.min_samples,
            initial_yaw=np.radians(args.init_yaw),
            initial_pitch=np.radians(args.init_pitch),
            initial_roll=np.radians(args.init_roll)
        )

        # Components
        self.mavlink = None
        self.camera = None
        self.motion_estimator = None
        self.temporal_sync = None
        self.spatial_calib = None

        # State
        self.running = False
        self.calibration_result = CalibrationResult()
        self.start_time = 0.0

    def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful
        """
        print("=" * 60)
        print("Real-time Camera-IMU Calibration")
        print("=" * 60)

        # Create MAVLink receiver
        print("\nInitializing MAVLink receiver...")
        self.mavlink = MAVLinkReceiver(self.mavlink_config)
        if not self.mavlink.start():
            print("Failed to start MAVLink receiver")
            return False
        print(f"  Connected to: {self.mavlink_config.connection_string}")

        # Create camera
        print("\nInitializing camera...")
        self.camera = HarrierCamera(self.camera_config)
        if not self.camera.start():
            print("Failed to start camera")
            return False
        print(f"  Resolution: {self.camera_config.width}x{self.camera_config.height}")
        print(f"  FPS: {self.camera_config.fps}")

        # Create motion estimator
        print("\nInitializing motion estimator...")
        self.motion_estimator = MotionEstimator(self.motion_config)

        # Create temporal synchronizer
        print("\nInitializing temporal synchronizer...")
        self.temporal_sync = TemporalSynchronizer(self.temporal_config)
        self.temporal_sync.start()

        # Create spatial calibrator
        print("\nInitializing spatial calibrator...")
        self.spatial_calib = SpatialCalibrator(self.spatial_config)
        self.spatial_calib.start()

        # Connect callbacks
        self.mavlink.add_imu_callback(self.temporal_sync.on_imu_data)

        print("\nSetup complete!")
        print("=" * 60)

        return True

    def run(self) -> CalibrationResult:
        """
        Run the calibration process.

        Returns:
            CalibrationResult with final calibration
        """
        self.running = True
        self.start_time = monotonic_time()
        self.calibration_result.state = SyncState.ACQUIRING

        print("\nStarting calibration...")
        print("Move the aircraft to generate rotation in all axes.")
        print("Press Ctrl+C to stop.\n")

        frame_count = 0
        last_status_time = monotonic_time()
        status_interval = 2.0

        try:
            while self.running:
                # Get camera frame
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                frame_count += 1

                # Estimate visual motion
                motion = self.motion_estimator.process_frame(frame)

                # Feed motion to temporal sync
                self.temporal_sync.on_motion_data(motion)

                # Once temporal sync converges, feed to spatial calibrator
                if self.temporal_sync.is_converged:
                    self.calibration_result.state = SyncState.CORRELATING

                    # Get synchronized IMU data
                    time_offset = self.temporal_sync.time_offset
                    imu = self.mavlink.imu_buffer.get_nearest(
                        motion.timestamp + time_offset
                    )

                    if imu is not None and motion.valid:
                        self.spatial_calib.add_sample(imu, motion, 0.0)

                # Check if spatial calibration converged
                if self.spatial_calib.is_converged:
                    self.calibration_result.state = SyncState.SYNCHRONIZED

                # Print status periodically
                now = monotonic_time()
                if now - last_status_time >= status_interval:
                    self._print_status()
                    last_status_time = now

                    # Check if calibration is complete
                    if self._check_complete():
                        print("\nCalibration complete!")
                        break

        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user.")

        self.running = False
        return self._finalize_result()

    def _print_status(self) -> None:
        """Print current calibration status."""
        elapsed = monotonic_time() - self.start_time

        print(f"\n--- Status ({elapsed:.1f}s elapsed) ---")

        # MAVLink status
        mav_status = self.mavlink.get_status()
        print(f"MAVLink: connected={mav_status['connected']}, "
              f"imu_rate={mav_status['imu_rate']:.1f}Hz")

        # Camera status
        cam_status = self.camera.get_status()
        print(f"Camera: connected={cam_status['connected']}, "
              f"fps={cam_status['actual_fps']:.1f}")

        # Motion estimator status
        me_status = self.motion_estimator.get_status()
        print(f"Motion: rate={me_status['rate']:.1f}Hz, "
              f"features={me_status['num_features']}")

        # Temporal sync status
        ts_status = self.temporal_sync.get_status()
        print(f"Temporal: offset={ts_status['time_offset']*1000:.1f}ms, "
              f"corr={ts_status['correlation']:.3f}, "
              f"converged={ts_status['converged']}")

        # Spatial calibration status
        sc_status = self.spatial_calib.get_status()
        print(f"Spatial: samples={sc_status['num_samples']}, "
              f"converged={sc_status['converged']}")
        if sc_status['converged']:
            print(f"  Mount offset: yaw={sc_status['yaw_deg']:.2f}deg, "
                  f"pitch={sc_status['pitch_deg']:.2f}deg, "
                  f"roll={sc_status['roll_deg']:.2f}deg")

    def _check_complete(self) -> bool:
        """Check if calibration is complete."""
        if not self.temporal_sync.is_converged:
            return False

        if not self.spatial_calib.is_converged:
            return False

        # Require minimum elapsed time
        elapsed = monotonic_time() - self.start_time
        if elapsed < self.args.min_duration:
            return False

        return True

    def _finalize_result(self) -> CalibrationResult:
        """Finalize calibration result."""
        elapsed = monotonic_time() - self.start_time

        # Create temporal calibration
        if self.temporal_sync.is_converged:
            self.calibration_result.temporal = TemporalCalibration(
                time_offset=self.temporal_sync.time_offset,
                confidence=self.temporal_sync.correlation,
                correlation_peak=self.temporal_sync.correlation,
                samples_used=self.temporal_sync.num_estimates
            )

        # Create spatial calibration
        if self.spatial_calib.is_converged:
            offset = self.spatial_calib.mount_offset
            self.calibration_result.spatial = SpatialCalibration(
                yaw=offset.yaw,
                pitch=offset.pitch,
                roll=offset.roll,
                rotation_matrix=self.spatial_calib.rotation_matrix,
                rmse=offset.rmse,
                samples_used=self.spatial_calib.num_samples
            )

        self.calibration_result.timestamp = monotonic_time()
        self.calibration_result.duration = elapsed

        return self.calibration_result

    def cleanup(self) -> None:
        """Clean up all components."""
        print("\nCleaning up...")

        if self.temporal_sync:
            self.temporal_sync.stop()

        if self.spatial_calib:
            self.spatial_calib.stop()

        if self.camera:
            self.camera.stop()

        if self.mavlink:
            self.mavlink.stop()

        print("Cleanup complete.")

    def save_result(self, filename: str) -> None:
        """
        Save calibration result to file.

        Args:
            filename: Output filename (JSON)
        """
        result_dict = self.calibration_result.to_dict()

        # Add metadata
        result_dict['metadata'] = {
            'date': datetime.now().isoformat(),
            'mavlink': self.mavlink_config.connection_string,
            'camera': str(self.camera_config.device_id),
            'camera_resolution': f"{self.camera_config.width}x{self.camera_config.height}"
        }

        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Result saved to: {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time Camera-IMU Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # UDP connection (SITL or network)
  python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0

  # Serial connection (USB)
  python calibrate.py --mavlink /dev/ttyACM0 --baudrate 115200 --camera 0

  # Specify camera intrinsics
  python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0 \\
      --fx 800 --fy 800 --cx 640 --cy 360

  # Save result to file
  python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0 \\
      --output calibration_result.json
"""
    )

    # MAVLink options
    mavlink_group = parser.add_argument_group('MAVLink')
    mavlink_group.add_argument(
        '--mavlink', required=True,
        help='MAVLink connection string (e.g., udp:127.0.0.1:14550 or /dev/ttyACM0)'
    )
    mavlink_group.add_argument(
        '--baudrate', type=int, default=115200,
        help='Serial baudrate (default: 115200)'
    )

    # Camera options
    camera_group = parser.add_argument_group('Camera')
    camera_group.add_argument(
        '--camera', default='0',
        help='Camera device ID or path (default: 0)'
    )
    camera_group.add_argument(
        '--width', type=int, default=1280,
        help='Frame width (default: 1280)'
    )
    camera_group.add_argument(
        '--height', type=int, default=720,
        help='Frame height (default: 720)'
    )
    camera_group.add_argument(
        '--fps', type=int, default=30,
        help='Frame rate (default: 30)'
    )

    # Camera intrinsics
    intrinsics_group = parser.add_argument_group('Camera Intrinsics')
    intrinsics_group.add_argument(
        '--fx', type=float, default=800.0,
        help='Focal length X (default: 800)'
    )
    intrinsics_group.add_argument(
        '--fy', type=float, default=800.0,
        help='Focal length Y (default: 800)'
    )
    intrinsics_group.add_argument(
        '--cx', type=float, default=None,
        help='Principal point X (default: width/2)'
    )
    intrinsics_group.add_argument(
        '--cy', type=float, default=None,
        help='Principal point Y (default: height/2)'
    )

    # Calibration options
    calib_group = parser.add_argument_group('Calibration')
    calib_group.add_argument(
        '--min-duration', type=float, default=10.0,
        help='Minimum calibration duration (seconds, default: 10)'
    )
    calib_group.add_argument(
        '--min-samples', type=int, default=100,
        help='Minimum samples for spatial calibration (default: 100)'
    )
    calib_group.add_argument(
        '--max-features', type=int, default=500,
        help='Maximum features to track (default: 500)'
    )
    calib_group.add_argument(
        '--sync-axis', choices=['p', 'q', 'r', 'all'], default='p',
        help='Axis for temporal synchronization (default: p)'
    )

    # Initial mount offset guess
    mount_group = parser.add_argument_group('Initial Mount Offset')
    mount_group.add_argument(
        '--init-yaw', type=float, default=0.0,
        help='Initial yaw offset guess (degrees, default: 0)'
    )
    mount_group.add_argument(
        '--init-pitch', type=float, default=0.0,
        help='Initial pitch offset guess (degrees, default: 0)'
    )
    mount_group.add_argument(
        '--init-roll', type=float, default=0.0,
        help='Initial roll offset guess (degrees, default: 0)'
    )

    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output', '-o',
        help='Save calibration result to JSON file'
    )

    args = parser.parse_args()

    # Create and run application
    app = CalibrationApp(args)

    # Handle signals
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal...")
        app.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Setup
        if not app.setup():
            print("Setup failed!")
            sys.exit(1)

        # Run calibration
        result = app.run()

        # Print final result
        print("\n" + "=" * 60)
        print("CALIBRATION RESULT")
        print("=" * 60)

        if result.is_valid():
            print(f"\nStatus: SUCCESS")
            print(f"Duration: {result.duration:.1f} seconds")

            print(f"\nTemporal Calibration:")
            print(f"  Time offset: {result.temporal.time_offset*1000:.2f} ms")
            print(f"  Correlation: {result.temporal.confidence:.3f}")

            print(f"\nSpatial Calibration (Camera Mount Offset):")
            print(f"  Yaw:   {result.spatial.yaw_deg():.2f} deg")
            print(f"  Pitch: {result.spatial.pitch_deg():.2f} deg")
            print(f"  Roll:  {result.spatial.roll_deg():.2f} deg")
            print(f"  RMSE:  {np.degrees(result.spatial.rmse):.3f} deg")
        else:
            print(f"\nStatus: INCOMPLETE")
            print(f"State: {result.state.value}")
            print("Calibration did not converge. Try:")
            print("  - Moving the aircraft more vigorously")
            print("  - Ensuring good lighting for feature tracking")
            print("  - Running for a longer duration")

        # Save result
        if args.output:
            app.save_result(args.output)

    finally:
        app.cleanup()


if __name__ == '__main__':
    main()
