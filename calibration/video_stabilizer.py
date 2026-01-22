#!/usr/bin/env python3
"""
Real-time Video Stabilization using IMU

Uses IMU angular rates to stabilize camera video by:
1. Integrating IMU gyroscope to get orientation
2. Computing the rotation needed to stabilize
3. Applying homography transform to warp frames
"""

import argparse
import time
import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig


class OrientationEstimator:
    """Estimates orientation from IMU gyroscope data."""
    
    def __init__(self):
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
        self.last_timestamp = None
        self._lock = threading.Lock()
        
        # Smoothed orientation for stabilization target
        self.target_orientation = self.orientation.copy()
        self.smooth_factor = 0.0005  # Very low = very smooth
        
        # Gyro lowpass filter state
        self.gyro_filtered = np.zeros(3)
        self.gyro_alpha = 0.05  # Strong lowpass (lower = more filtering)
        
        # Deadband to ignore small rotations (noise)
        self.deadband = 0.08  # rad/s - ignore rotations below this (~4.5 deg/s)
    
    def update(self, timestamp: float, gyro: np.ndarray):
        """Update orientation with gyroscope measurement."""
        with self._lock:
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                self.gyro_filtered = gyro.copy()
                return
            
            dt = timestamp - self.last_timestamp
            if dt <= 0 or dt > 0.1:  # Skip if dt is invalid
                self.last_timestamp = timestamp
                return
            
            # Lowpass filter the gyro to reduce noise
            self.gyro_filtered = (self.gyro_alpha * gyro + 
                                  (1 - self.gyro_alpha) * self.gyro_filtered)
            
            # Apply deadband to ignore small noisy rotations
            omega = self.gyro_filtered.copy()
            for i in range(3):
                if abs(omega[i]) < self.deadband:
                    omega[i] = 0.0
            
            # Integrate angular velocity to get rotation increment
            angle = np.linalg.norm(omega) * dt
            
            if angle > 1e-8:
                axis = omega / np.linalg.norm(omega)
                dq = self._quaternion_from_axis_angle(axis, angle)
                self.orientation = self._quaternion_multiply(self.orientation, dq)
                self.orientation /= np.linalg.norm(self.orientation)
            
            # Update smoothed target with very slow lowpass
            self.target_orientation = (
                (1 - self.smooth_factor) * self.target_orientation +
                self.smooth_factor * self.orientation
            )
            self.target_orientation /= np.linalg.norm(self.target_orientation)
            
            self.last_timestamp = timestamp
    
    def get_stabilization_rotation(self) -> np.ndarray:
        """Get rotation matrix to stabilize from current to target orientation."""
        with self._lock:
            # Current orientation
            R_current = Rotation.from_quat([
                self.orientation[1], self.orientation[2], 
                self.orientation[3], self.orientation[0]
            ]).as_matrix()
            
            # Target (smoothed) orientation
            R_target = Rotation.from_quat([
                self.target_orientation[1], self.target_orientation[2],
                self.target_orientation[3], self.target_orientation[0]
            ]).as_matrix()
            
            # Rotation from current to target: R_stab = R_target * R_current^-1
            R_stab = R_target @ R_current.T
            
            return R_stab
    
    def get_euler_degrees(self) -> np.ndarray:
        """Get current orientation as Euler angles [roll, pitch, yaw] in degrees."""
        with self._lock:
            r = Rotation.from_quat([
                self.orientation[1], self.orientation[2],
                self.orientation[3], self.orientation[0]
            ])
            return r.as_euler('xyz', degrees=True)
    
    def reset(self):
        """Reset orientation to identity."""
        with self._lock:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
            self.target_orientation = self.orientation.copy()
            self.last_timestamp = None
    
    @staticmethod
    def _quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion [w, x, y, z] from axis-angle."""
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class VideoStabilizer:
    """Real-time video stabilization using IMU."""
    
    def __init__(self, width: int, height: int, fx: float, fy: float,
                 mount_yaw: float = 0, mount_pitch: float = 0, mount_roll: float = 0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = width / 2
        self.cy = height / 2
        
        # Camera matrix
        self.K = np.array([
            [fx, 0, self.cx],
            [0, fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        
        # Camera-IMU mount rotation
        self.R_cam_imu = Rotation.from_euler(
            'zyx', [mount_yaw, mount_pitch, mount_roll], degrees=True
        ).as_matrix()
        
        # Orientation estimator
        self.orientation = OrientationEstimator()
        
        # Stabilization mode
        self.enabled = True
        self.mode = 'full'  # 'full', 'horizon', 'smooth'
        
        # Crop factor (to avoid showing black borders)
        self.crop_factor = 0.85
        
        # Maximum correction angle (radians) to prevent extreme warps
        self.max_correction_angle = np.radians(15)
    
    def on_imu(self, timestamp: float, gyro: np.ndarray):
        """Process IMU measurement."""
        # Transform gyro from IMU frame to camera frame
        gyro_cam = self.R_cam_imu @ gyro
        self.orientation.update(timestamp, gyro_cam)
    
    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize a frame using current IMU orientation."""
        if not self.enabled:
            return frame
        
        # Get stabilization rotation
        R_stab = self.orientation.get_stabilization_rotation()
        
        if self.mode == 'horizon':
            # Only correct roll (keep horizon level)
            euler = Rotation.from_matrix(R_stab).as_euler('xyz')
            R_stab = Rotation.from_euler('xyz', [euler[0], 0, 0]).as_matrix()
        
        # Limit the correction magnitude to avoid extreme warping
        rotvec = Rotation.from_matrix(R_stab).as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle > self.max_correction_angle:
            rotvec = rotvec * (self.max_correction_angle / angle)
            R_stab = Rotation.from_rotvec(rotvec).as_matrix()
        
        # Compute homography: H = K * R * K^-1
        H = self.K @ R_stab @ self.K_inv
        
        # Apply crop to avoid black borders
        if self.crop_factor < 1.0:
            # Scale homography to zoom in slightly
            scale = 1.0 / self.crop_factor
            S = np.array([
                [scale, 0, self.cx * (1 - scale)],
                [0, scale, self.cy * (1 - scale)],
                [0, 0, 1]
            ])
            H = S @ H
        
        # Warp frame
        stabilized = cv2.warpPerspective(
            frame, H, (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return stabilized
    
    def reset(self):
        """Reset stabilization."""
        self.orientation.reset()
    
    def set_smoothness(self, value: float):
        """Set smoothness (0-1, lower = smoother)."""
        self.orientation.smooth_factor = np.clip(value, 0.001, 0.5)


def main():
    parser = argparse.ArgumentParser(description='IMU-based Video Stabilization')
    
    # Connections
    parser.add_argument('--mavlink', required=True, help='MAVLink port (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', type=int, default=0)
    
    # Camera
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fx', type=float, default=1175.0)
    parser.add_argument('--fy', type=float, default=1175.0)
    
    # Calibration (mount angles)
    parser.add_argument('--yaw', type=float, default=4.56)
    parser.add_argument('--pitch', type=float, default=-11.25)
    parser.add_argument('--roll', type=float, default=-1.09)
    
    # Stabilization
    parser.add_argument('--smoothness', type=float, default=0.02,
                       help='Smoothness factor (0.001-0.5, lower=smoother)')
    parser.add_argument('--crop', type=float, default=0.85,
                       help='Crop factor to avoid black borders (0.7-1.0)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMU Video Stabilization")
    print("=" * 60)
    print("Controls:")
    print("  Q/ESC  - Quit")
    print("  SPACE  - Toggle stabilization on/off")
    print("  R      - Reset orientation")
    print("  M      - Cycle mode (full/horizon/off)")
    print("  +/-    - Adjust smoothness")
    print("  S      - Start/stop recording")
    print("=" * 60)
    
    # Setup MAVLink
    print(f"\nConnecting to MAVLink: {args.mavlink}")
    mavlink_config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )
    mavlink = MAVLinkReceiver(mavlink_config)
    
    if not mavlink.start():
        print("Failed to connect to MAVLink!")
        return
    
    # Setup camera
    print(f"Opening camera: {args.camera}")
    camera_config = CameraConfig(
        device=args.camera,
        width=args.width,
        height=args.height,
        fps=60
    )
    camera = HarrierCamera(camera_config)
    
    if not camera.start():
        print("Failed to start camera!")
        mavlink.stop()
        return
    
    print(f"Camera: {args.width}x{args.height}")
    
    # Create stabilizer
    stabilizer = VideoStabilizer(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        mount_yaw=args.yaw,
        mount_pitch=args.pitch,
        mount_roll=args.roll
    )
    stabilizer.set_smoothness(args.smoothness)
    stabilizer.crop_factor = args.crop
    
    # IMU callback
    def imu_callback(imu: IMUData):
        gyro = np.array([imu.p, imu.q, imu.r])
        stabilizer.on_imu(imu.timestamp, gyro)
    
    mavlink.add_callback('imu', imu_callback)
    
    # Recording
    video_writer = None
    recording = False
    
    # Stats
    frame_times = deque(maxlen=30)
    modes = ['full', 'horizon', 'off']
    mode_idx = 0
    
    print("\nRunning... Press Q to quit")
    
    cv2.namedWindow('Stabilized', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame_data = camera.get_latest_frame()
            if frame_data is None:
                time.sleep(0.001)
                continue
            
            t_start = time.time()
            
            frame = frame_data.image
            
            # Stabilize
            if stabilizer.mode != 'off':
                stabilized = stabilizer.stabilize(frame)
            else:
                stabilized = frame.copy()
            
            frame_times.append(time.time() - t_start)
            
            # Draw info on frames
            euler = stabilizer.orientation.get_euler_degrees()
            
            # Original frame info
            frame_vis = frame.copy()
            cv2.putText(frame_vis, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_vis, f"R:{euler[0]:.1f} P:{euler[1]:.1f} Y:{euler[2]:.1f}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stabilized frame info
            status = "ON" if stabilizer.enabled and stabilizer.mode != 'off' else "OFF"
            mode_str = stabilizer.mode.upper() if stabilizer.mode != 'off' else 'OFF'
            
            cv2.putText(stabilized, f"STABILIZED ({mode_str})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if len(frame_times) > 0:
                fps = 1.0 / max(np.mean(frame_times), 0.001)
                cv2.putText(stabilized, f"FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(stabilized, f"Smooth: {stabilizer.orientation.smooth_factor:.3f}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if recording:
                cv2.circle(stabilized, (args.width - 30, 30), 15, (0, 0, 255), -1)
                cv2.putText(stabilized, "REC", (args.width - 80, 37),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                video_writer.write(stabilized)
            
            # Draw horizon line on both
            h_y = args.height // 2
            # On original - tilted horizon
            roll_rad = np.radians(euler[0])
            dx = int(args.width * 0.4 * np.cos(roll_rad))
            dy = int(args.width * 0.4 * np.sin(roll_rad))
            cv2.line(frame_vis, (args.width//2 - dx, h_y + dy),
                    (args.width//2 + dx, h_y - dy), (0, 255, 255), 2)
            
            # On stabilized - should be level
            cv2.line(stabilized, (args.width//4, h_y),
                    (3*args.width//4, h_y), (0, 255, 0), 2)
            
            cv2.imshow('Original', frame_vis)
            cv2.imshow('Stabilized', stabilized)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                stabilizer.enabled = not stabilizer.enabled
                print(f"Stabilization: {'ON' if stabilizer.enabled else 'OFF'}")
            elif key == ord('r'):
                stabilizer.reset()
                print("Orientation reset")
            elif key == ord('m'):
                mode_idx = (mode_idx + 1) % len(modes)
                stabilizer.mode = modes[mode_idx]
                print(f"Mode: {stabilizer.mode}")
            elif key == ord('+') or key == ord('='):
                new_smooth = min(stabilizer.orientation.smooth_factor * 1.5, 0.5)
                stabilizer.set_smoothness(new_smooth)
                print(f"Smoothness: {new_smooth:.3f} (less smooth)")
            elif key == ord('-'):
                new_smooth = max(stabilizer.orientation.smooth_factor / 1.5, 0.001)
                stabilizer.set_smoothness(new_smooth)
                print(f"Smoothness: {new_smooth:.3f} (smoother)")
            elif key == ord('s'):
                if not recording:
                    filename = f"stabilized_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(filename, fourcc, 30,
                                                   (args.width, args.height))
                    recording = True
                    print(f"Recording started: {filename}")
                else:
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print("Recording stopped")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        camera.stop()
        mavlink.stop()
        print("Cleanup complete")


if __name__ == '__main__':
    main()
