#!/usr/bin/env python3
"""
Gyroflow-Compatible Video and IMU Recorder

Records synchronized video (MP4) and IMU data (GCSV format) for use with Gyroflow.
The GCSV format is Gyroflow's standard text-based IMU log format.

Usage:
    python gyroflow_recorder.py --mavlink COM6 --baudrate 115200 --camera 0 --output my_recording

This will create:
    - my_recording.mp4 (video file)
    - my_recording.gcsv (IMU data in Gyroflow format)
    - my_recording.json (lens profile)

IMU Orientation (--orientation):
    The 3-letter orientation string maps IMU axes to camera axes.
    Format: [X-axis][Y-axis][Z-axis] where each letter is one of X,Y,Z,x,y,z
    
    Capital letter = positive axis direction
    Lowercase letter = negative axis direction
    
    Examples:
    - "XYZ" = IMU axes align with camera axes (default)
    - "yXZ" = Camera X = -IMU Y, Camera Y = +IMU X, Camera Z = +IMU Z
    - "ZYx" = Camera X = +IMU Z, Camera Y = +IMU Y, Camera Z = -IMU X
    
    The camera coordinate system is typically:
    - X = right
    - Y = down  
    - Z = forward (into the scene)
    
    If stabilization looks wrong, try different orientations in Gyroflow.
"""

import argparse
import time
import numpy as np
import cv2
from collections import deque
from datetime import datetime
import threading
import queue
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig


class GCSVWriter:
    """Writes IMU data in Gyroflow GCSV format."""
    
    def __init__(self, filepath: str, orientation: str = "zxy",
                 video_filename: str = None, include_accel: bool = True,
                 lens_profile: str = None):
        """
        Initialize GCSV writer.
        
        Args:
            filepath: Path to output .gcsv file
            orientation: IMU orientation string (default: zxy for typical setup)
            video_filename: Name of the corresponding video file
            include_accel: Whether to include accelerometer data
            lens_profile: Optional lens profile identifier for Gyroflow
        """
        self.filepath = filepath
        self.orientation = orientation
        self.video_filename = video_filename or os.path.basename(filepath).replace('.gcsv', '.mp4')
        self.include_accel = include_accel
        self.lens_profile = lens_profile
        
        self.file = None
        self.start_time = None
        self.sample_count = 0
        
        # Scale factors
        # We'll write raw integers and use scale factors
        # Gyro: store in 0.001 rad/s units -> gscale = 0.001
        # Accel: store in 0.001 g units -> ascale = 0.001
        self.gscale = 0.001  # 1 unit = 0.001 rad/s
        self.ascale = 0.001  # 1 unit = 0.001 g
        self.tscale = 0.000001  # 1 unit = 1 microsecond
        
    def open(self):
        """Open the GCSV file and write header."""
        self.file = open(self.filepath, 'w', newline='')
        self.start_time = None  # Will be set from first IMU sample
        
        # Write GCSV header
        lines = [
            "GYROFLOW IMU LOG",
            "version,1.3",
            "id,orange_cube_mavlink",
            f"orientation,{self.orientation}",
            "note,Recorded with ImageAnalysis calibration tools",
            "fwversion,ArduPilot",
            f"timestamp,{int(time.time())}",
            "vendor,CubePilot",
            f"videofilename,{self.video_filename}",
        ]
        
        # Add lens profile if specified
        if self.lens_profile:
            lines.append(f"lensprofile,{self.lens_profile}")
        
        lines.extend([
            f"tscale,{self.tscale}",
            f"gscale,{self.gscale}",
        ])
        
        if self.include_accel:
            lines.append(f"ascale,{self.ascale}")
            lines.append("t,gx,gy,gz,ax,ay,az")
        else:
            lines.append("t,gx,gy,gz")
        
        for line in lines:
            self.file.write(line + '\n')
        
        self.file.flush()
        print(f"[GCSV] Opened {self.filepath}")
        
    def write(self, imu: IMUData):
        """Write a single IMU sample."""
        if self.file is None:
            return
        
        # Calculate timestamp in microseconds from start
        # Use monotonic time from the IMU data
        if self.start_time is None:
            self.start_time = imu.timestamp
        
        # Skip samples that are before start (from queue before recording began)
        if imu.timestamp < self.start_time:
            return
        
        t_us = int((imu.timestamp - self.start_time) * 1_000_000)
        
        # Convert gyro from rad/s to scaled integers (divide by gscale)
        gx = int(imu.p / self.gscale)
        gy = int(imu.q / self.gscale)
        gz = int(imu.r / self.gscale)
        
        if self.include_accel and imu.ax is not None:
            # Convert accel from m/s^2 to g, then to scaled integers
            ax = int((imu.ax / 9.81) / self.ascale)
            ay = int((imu.ay / 9.81) / self.ascale)
            az = int((imu.az / 9.81) / self.ascale)
            self.file.write(f"{t_us},{gx},{gy},{gz},{ax},{ay},{az}\n")
        else:
            self.file.write(f"{t_us},{gx},{gy},{gz}\n")
        
        self.sample_count += 1
        self.last_t_us = t_us
        
        # Flush periodically
        if self.sample_count % 100 == 0:
            self.file.flush()
    
    def close(self):
        """Close the GCSV file."""
        if self.file:
            self.file.flush()
            self.file.close()
            duration = self.last_t_us / 1_000_000 if hasattr(self, 'last_t_us') else 0
            print(f"[GCSV] Closed {self.filepath} ({self.sample_count} samples, {duration:.2f}s)")
            self.file = None


def create_lens_profile(filepath: str, width: int, height: int, 
                        fx: float, fy: float, cx: float = None, cy: float = None,
                        distortion_coeffs: list = None, fps: float = 30.0):
    """
    Create a Gyroflow-compatible lens profile JSON file.
    
    Args:
        filepath: Path to save the .json lens profile
        width: Video width in pixels
        height: Video height in pixels  
        fx: Focal length in pixels (x-axis)
        fy: Focal length in pixels (y-axis)
        cx: Principal point x (default: width/2)
        cy: Principal point y (default: height/2)
        distortion_coeffs: OpenCV fisheye distortion [k1, k2, k3, k4] (default: zeros)
        fps: Video frame rate
    """
    import json
    from datetime import datetime
    
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    if distortion_coeffs is None:
        distortion_coeffs = [0.0, 0.0, 0.0, 0.0]
    
    profile = {
        "name": f"Custom Camera {width}x{height}",
        "note": "Generated by ImageAnalysis gyroflow_recorder",
        "calibrated_by": "ImageAnalysis",
        "camera_brand": "Generic",
        "camera_model": "USB Camera",
        "lens_model": f"fx={fx:.0f}",
        "camera_setting": "",
        "calib_dimension": {"w": width, "h": height},
        "orig_dimension": {"w": width, "h": height},
        "output_dimension": {"w": width, "h": height},
        "frame_readout_time": None,
        "input_horizontal_stretch": 1.0,
        "input_vertical_stretch": 1.0,
        "num_images": 0,
        "fps": fps,
        "official": False,
        "asymmetrical": False,
        "fisheye_params": {
            "RMS_error": 0.0,
            "camera_matrix": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ],
            "distortion_coeffs": distortion_coeffs
        },
        "distortion_model": "opencv_fisheye",
        "identifier": f"generic-usb_camera-{width}x{height}",
        "calibrator_version": "1.0.0",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    with open(filepath, 'w') as f:
        json.dump(profile, f, indent=4)
    
    print(f"[Lens] Created lens profile: {filepath}")
    return profile


class VideoWriter:
    """Writes video frames to MP4 file."""
    
    def __init__(self, filepath: str, width: int, height: int, fps: float = 30.0):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        
        self.writer = None
        self.frame_count = 0
        self.start_time = None
        
    def open(self):
        """Open the video file for writing."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filepath, fourcc, self.fps, (self.width, self.height)
        )
        self.start_time = time.time()
        print(f"[Video] Opened {self.filepath} ({self.width}x{self.height} @ {self.fps} fps)")
        
    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if self.writer is None:
            return
        
        self.writer.write(frame)
        self.frame_count += 1
        
    def close(self):
        """Close the video file."""
        if self.writer:
            self.writer.release()
            duration = time.time() - self.start_time if self.start_time else 0
            actual_fps = self.frame_count / duration if duration > 0 else 0
            print(f"[Video] Closed {self.filepath} ({self.frame_count} frames, {actual_fps:.1f} actual fps)")
            self.writer = None


class GyroflowRecorder:
    """Records synchronized video and IMU data for Gyroflow."""
    
    def __init__(self, mavlink_config: MAVLinkConfig, camera_config: CameraConfig,
                 output_base: str, orientation: str = "zxy",
                 fx: float = None, fy: float = None, fps: float = 30.0):
        self.mavlink_config = mavlink_config
        self.camera_config = camera_config
        self.output_base = output_base
        self.orientation = orientation
        self.fx = fx
        self.fy = fy
        self.fps = fps
        
        # Output files
        self.video_path = output_base + '.mp4'
        self.gcsv_path = output_base + '.gcsv'
        self.lens_profile_path = output_base + '.json'
        
        # Components
        self.mavlink = None
        self.camera = None
        self.video_writer = None
        self.gcsv_writer = None
        
        # State
        self.recording = False
        self.running = False
        
        # IMU queue for thread-safe access (kept for stats display)
        self.imu_queue = queue.Queue(maxsize=1000)
        
        # Lock for thread-safe GCSV writing
        self.gcsv_lock = threading.Lock()
        
        # Frame rate control
        self.last_frame_number = -1
        self.frame_interval = 1.0 / fps  # Target interval between frames
        self.last_frame_time = 0
        
        # Stats
        self.imu_count = 0
        self.frame_count = 0
        
    def _on_imu(self, imu: IMUData):
        """Callback for IMU data - writes directly for full rate."""
        self.imu_count += 1
        
        # Write directly to GCSV (callback runs in MAVLink thread)
        if self.recording and self.gcsv_writer:
            with self.gcsv_lock:
                self.gcsv_writer.write(imu)
        
    def start(self):
        """Start the recorder (preview mode, not yet recording)."""
        print("Initializing recorder...")
        
        # Start MAVLink
        self.mavlink = MAVLinkReceiver(self.mavlink_config)
        self.mavlink.add_callback('imu', self._on_imu)
        self.mavlink.start()
        print(f"[MAVLink] Connected to {self.mavlink_config.connection_string}")
        
        # Start camera
        self.camera = HarrierCamera(self.camera_config)
        self.camera.start()
        time.sleep(0.5)
        
        frame_data = self.camera.get_latest_frame()
        if frame_data is not None:
            print(f"[Camera] Started ({frame_data.width}x{frame_data.height})")
        else:
            print("[Camera] Started (waiting for frames...)")
        
        self.running = True
        print("\nRecorder ready. Press 'R' to start/stop recording, 'Q' to quit.\n")
        
    def start_recording(self):
        """Start recording video and IMU data."""
        if self.recording:
            return
        
        # Get frame dimensions
        frame_data = self.camera.get_latest_frame()
        if frame_data is None:
            print("[Error] No camera frame available!")
            return
        
        w, h = frame_data.width, frame_data.height
        
        # Create lens profile if fx/fy provided
        if self.fx and self.fy:
            create_lens_profile(
                self.lens_profile_path,
                width=w, height=h,
                fx=self.fx, fy=self.fy,
                fps=self.fps
            )
        
        # Create writers
        video_filename = os.path.basename(self.video_path)
        self.gcsv_writer = GCSVWriter(
            self.gcsv_path,
            orientation=self.orientation,
            video_filename=video_filename
        )
        self.video_writer = VideoWriter(self.video_path, w, h, fps=self.fps)
        
        # Open files
        self.gcsv_writer.open()
        self.video_writer.open()
        
        # Clear IMU queue
        while not self.imu_queue.empty():
            try:
                self.imu_queue.get_nowait()
            except queue.Empty:
                break
        
        self.recording = True
        self.frame_count = 0
        self.imu_count = 0
        self.last_frame_number = -1
        self.last_frame_time = time.time()
        print("\n*** RECORDING STARTED ***\n")
        
    def stop_recording(self):
        """Stop recording."""
        if not self.recording:
            return
        
        self.recording = False
        
        # Close writers
        if self.video_writer:
            self.video_writer.close()
            self.video_writer = None
        
        if self.gcsv_writer:
            self.gcsv_writer.close()
            self.gcsv_writer = None
        
        print("\n*** RECORDING STOPPED ***")
        print(f"Output files:")
        print(f"  Video:        {self.video_path}")
        print(f"  IMU:          {self.gcsv_path}")
        if self.fx and self.fy:
            print(f"  Lens profile: {self.lens_profile_path}")
        print()
        print("To use with Gyroflow:")
        print("  1. Open Gyroflow")
        print("  2. Load the .mp4 video")
        print("  3. Load the .gcsv IMU file (or it may auto-detect)")
        print("  4. Load the .json lens profile")
        print("  5. Sync and stabilize!\n")
        
    def run(self):
        """Main loop with preview."""
        try:
            while self.running:
                # IMU data is written directly in callback for full rate
                
                # Get camera frame
                frame_data = self.camera.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue
                
                frame = frame_data.image
                
                # Write frame if recording - only unique frames at target fps
                if self.recording and self.video_writer:
                    # Check if this is a new frame (different frame number)
                    if frame_data.frame_number != self.last_frame_number:
                        # Also check if enough time has passed (frame rate limiting)
                        current_time = time.time()
                        if current_time - self.last_frame_time >= self.frame_interval * 0.9:
                            self.video_writer.write(frame)
                            self.frame_count += 1
                            self.last_frame_number = frame_data.frame_number
                            self.last_frame_time = current_time
                
                # Create preview with overlay
                preview = frame.copy()
                
                # Status overlay
                h, w = preview.shape[:2]
                
                if self.recording:
                    # Red recording indicator
                    cv2.circle(preview, (30, 30), 15, (0, 0, 255), -1)
                    status_text = f"REC | Frames: {self.frame_count} | IMU: {self.gcsv_writer.sample_count if self.gcsv_writer else 0}"
                    cv2.putText(preview, status_text, (55, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Preview indicator
                    cv2.putText(preview, "PREVIEW (Press R to record)", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # IMU rate indicator
                cv2.putText(preview, f"IMU samples: {self.imu_count}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show preview
                cv2.imshow('Gyroflow Recorder', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or Esc
                    break
                elif key == ord('r'):  # R to toggle recording
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the recorder."""
        self.running = False
        
        # Stop recording if active
        if self.recording:
            self.stop_recording()
        
        # Cleanup
        if self.camera:
            self.camera.stop()
        if self.mavlink:
            self.mavlink.stop()
        
        cv2.destroyAllWindows()
        print("Recorder stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Record synchronized video and IMU data for Gyroflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gyroflow_recorder.py --mavlink COM6 --camera 0 --output my_clip
    python gyroflow_recorder.py --mavlink COM6 --camera 0 --output test --orientation YxZ

Output:
    Creates files with the same base name:
    - <output>.mp4  - Video file
    - <output>.gcsv - IMU data in Gyroflow format
    - <output>.json - Lens profile

Controls:
    R     - Start/stop recording
    Q/Esc - Quit

IMU Orientation (3 letters mapping IMU to Camera axes):
    Capital = positive, lowercase = negative
    Example: "YxZ" means Camera_X=+IMU_Y, Camera_Y=-IMU_X, Camera_Z=+IMU_Z
    
    If stabilization looks wrong in Gyroflow, try adjusting the orientation.
        """
    )
    
    parser.add_argument('--mavlink', required=True,
                        help='MAVLink connection string (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='MAVLink baudrate (default: 115200)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height (default: 720)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output base filename (default: recording_YYYYMMDD_HHMMSS)')
    parser.add_argument('--orientation', default='XYZ',
                        help='IMU orientation: 3 letters (XYZ/xyz). Capital=+, lower=- (default: XYZ)')
    parser.add_argument('--fx', type=float, default=1175,
                        help='Focal length in pixels (x-axis, default: 1175 for 64deg FOV)')
    parser.add_argument('--fy', type=float, default=1175,
                        help='Focal length in pixels (y-axis, default: 1175 for 64deg FOV)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Video frame rate (default: 30.0)')
    parser.add_argument('--imu-rate', type=int, default=200,
                        help='IMU sample rate in Hz (default: 200, max ~400)')
    
    args = parser.parse_args()
    
    # Generate default output name if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'recording_{timestamp}'
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure MAVLink
    mavlink_config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate,
        stream_rate_hz=args.imu_rate
    )
    
    # Configure camera
    camera_config = CameraConfig(
        device=args.camera,
        width=args.width,
        height=args.height
    )
    
    # Create and run recorder
    recorder = GyroflowRecorder(
        mavlink_config=mavlink_config,
        camera_config=camera_config,
        output_base=args.output,
        orientation=args.orientation,
        fx=args.fx,
        fy=args.fy,
        fps=args.fps
    )
    
    recorder.start()
    recorder.run()


if __name__ == '__main__':
    main()
