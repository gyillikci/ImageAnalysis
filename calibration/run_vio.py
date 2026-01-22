#!/usr/bin/env python3
"""
Real-time Visual-Inertial Odometry Demo.

Runs VIO using camera and IMU, displays live trajectory and map.
"""

import argparse
import time
import numpy as np
import cv2
from collections import deque
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig
from vio.vio import VisualInertialOdometry, VIOCalibration


class VIODemo:
    """Real-time VIO demonstration."""
    
    def __init__(self, args):
        self.args = args
        
        # Calibration from our calibration results
        self.calibration = VIOCalibration(
            time_offset=args.time_offset / 1000.0,  # ms to seconds
            yaw=args.yaw,
            pitch=args.pitch,
            roll=args.roll,
            fx=args.fx,
            fy=args.fy,
            cx=args.width / 2,
            cy=args.height / 2
        )
        
        # VIO
        self.vio = VisualInertialOdometry(self.calibration)
        
        # Trajectory history for plotting
        self.trajectory = deque(maxlen=1000)
        
        # Running flag
        self.running = False
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        
    def run(self):
        """Run the VIO demo."""
        print("=" * 60)
        print("Visual-Inertial Odometry Demo")
        print("=" * 60)
        print("Controls:")
        print("  Q/ESC - Quit")
        print("  R     - Reset VIO")
        print("  S     - Save trajectory")
        print("=" * 60)
        
        # Setup MAVLink
        print(f"\nConnecting to MAVLink: {self.args.mavlink}")
        mavlink_config = MAVLinkConfig(
            connection_string=self.args.mavlink,
            baudrate=self.args.baudrate
        )
        mavlink = MAVLinkReceiver(mavlink_config)
        
        if not mavlink.start():
            print("Failed to connect to MAVLink!")
            return
        
        # Setup camera
        print(f"Opening camera: {self.args.camera}")
        camera_config = CameraConfig(
            device=self.args.camera,
            width=self.args.width,
            height=self.args.height,
            fps=60
        )
        camera = HarrierCamera(camera_config)
        
        if not camera.start():
            print("Failed to start camera!")
            mavlink.disconnect()
            return
        
        print(f"Camera: {self.args.width}x{self.args.height}")
        
        # Register IMU callback
        def imu_callback(imu: IMUData):
            # IMU data: p, q, r are gyro rates; ax, ay, az are accel
            gyro = np.array([imu.p, imu.q, imu.r])
            accel = np.array([imu.ax, imu.ay, imu.az])
            self.vio.on_imu(imu.timestamp, gyro, accel)
        
        mavlink.add_callback('imu', imu_callback)
        
        self.running = True
        print("\nRunning VIO... Press Q to quit")
        
        # Create windows
        cv2.namedWindow('VIO - Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('VIO - Trajectory', cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Get frame
                frame_data = camera.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue
                
                t_start = time.time()
                
                # Process frame
                pose = self.vio.on_frame(frame_data.image, frame_data.timestamp)
                
                # Track FPS
                self.frame_times.append(time.time() - t_start)
                
                if pose is not None:
                    self.trajectory.append(pose.position.copy())
                
                # Visualize
                vis_camera = self._draw_camera_view(frame_data.image)
                vis_traj = self._draw_trajectory()
                
                cv2.imshow('VIO - Camera', vis_camera)
                cv2.imshow('VIO - Trajectory', vis_traj)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self._reset_vio()
                elif key == ord('s'):
                    self._save_trajectory()
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            self.running = False
            cv2.destroyAllWindows()
            camera.stop()
            mavlink.stop()
            print("Cleanup complete")
    
    def _draw_camera_view(self, frame: np.ndarray) -> np.ndarray:
        """Draw camera view with feature tracks."""
        vis = frame.copy()
        
        # Draw tracked features
        if self.vio.tracker.prev_pts is not None:
            for pt in self.vio.tracker.prev_pts:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        
        # Draw feature tracks
        for track in self.vio.tracker.tracks:
            if len(track.positions) >= 2:
                pts = np.array(track.positions[-10:], dtype=np.int32)  # Last 10 positions
                for i in range(len(pts) - 1):
                    color = (0, 255 - i * 20, i * 20)  # Fade from green to red
                    cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), color, 1)
        
        # Draw info
        state = self.vio.state
        if state is not None:
            euler = np.array([0, 0, 0])
            try:
                from scipy.spatial.transform import Rotation
                r = Rotation.from_quat([state.orientation[1], state.orientation[2],
                                        state.orientation[3], state.orientation[0]])
                euler = r.as_euler('xyz', degrees=True)
            except:
                pass
            
            info = [
                f"Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}] m",
                f"Velocity: [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}] m/s",
                f"Attitude: R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f} deg",
                f"Features: {len(self.vio.tracker.tracks)} tracked, {len(self.vio.map_points)} mapped"
            ]
            
            if len(self.frame_times) > 0:
                fps = 1.0 / np.mean(self.frame_times)
                info.append(f"VIO: {fps:.1f} Hz")
            
            for i, text in enumerate(info):
                cv2.putText(vis, text, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis
    
    def _draw_trajectory(self) -> np.ndarray:
        """Draw top-down trajectory view."""
        size = 600
        vis = np.zeros((size, size, 3), dtype=np.uint8)
        vis[:] = (40, 40, 40)
        
        # Draw grid
        center = size // 2
        scale = 50  # pixels per meter
        
        for i in range(-10, 11):
            alpha = 0.3 if i != 0 else 0.7
            color = (int(80 * alpha), int(80 * alpha), int(80 * alpha))
            cv2.line(vis, (0, center + i * scale), (size, center + i * scale), color, 1)
            cv2.line(vis, (center + i * scale, 0), (center + i * scale, size), color, 1)
        
        # Draw axis labels
        cv2.putText(vis, "X", (size - 20, center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "Y", (center + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(list(self.trajectory))
            
            for i in range(len(traj) - 1):
                # Use X-Y plane (top-down view)
                x1 = int(center + traj[i, 0] * scale)
                y1 = int(center - traj[i, 1] * scale)  # Y is flipped
                x2 = int(center + traj[i+1, 0] * scale)
                y2 = int(center - traj[i+1, 1] * scale)
                
                # Fade color based on age
                age = (len(traj) - i) / len(traj)
                color = (int(255 * age), int(100 * age), int(50 * age))
                
                cv2.line(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw current position
        if len(self.trajectory) > 0:
            pos = self.trajectory[-1]
            x = int(center + pos[0] * scale)
            y = int(center - pos[1] * scale)
            
            # Draw direction arrow
            if self.vio.is_initialized:
                euler = self.vio.pose.euler_degrees()
                yaw_rad = np.radians(euler[2])
                arrow_len = 20
                dx = int(arrow_len * np.cos(yaw_rad))
                dy = int(-arrow_len * np.sin(yaw_rad))
                cv2.arrowedLine(vis, (x, y), (x + dx, y + dy), (0, 255, 255), 2, tipLength=0.3)
            
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        
        # Draw map points (projected to XY plane)
        map_pts = self.vio.get_map_points()
        if len(map_pts) > 0:
            for pt in map_pts[:500]:  # Limit number drawn
                x = int(center + pt[0] * scale)
                y = int(center - pt[1] * scale)
                if 0 <= x < size and 0 <= y < size:
                    cv2.circle(vis, (x, y), 2, (100, 100, 255), -1)
        
        # Draw info
        info = [
            "Top-Down View (X-Y plane)",
            f"Scale: {1/scale:.2f} m/pixel",
            f"Trajectory: {len(self.trajectory)} poses",
            f"Map points: {len(map_pts)}"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(vis, text, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis
    
    def _reset_vio(self):
        """Reset VIO state."""
        print("Resetting VIO...")
        self.vio = VisualInertialOdometry(self.calibration)
        self.trajectory.clear()
    
    def _save_trajectory(self):
        """Save trajectory to file."""
        if len(self.trajectory) == 0:
            print("No trajectory to save")
            return
        
        filename = f"trajectory_{int(time.time())}.npy"
        traj = np.array(list(self.trajectory))
        np.save(filename, traj)
        print(f"Saved trajectory to {filename}")


def main():
    parser = argparse.ArgumentParser(description='VIO Demo')
    
    # Connection
    parser.add_argument('--mavlink', required=True, help='MAVLink port (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    
    # Camera
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fx', type=float, default=1175.0)
    parser.add_argument('--fy', type=float, default=1175.0)
    
    # Calibration (from our calibration results)
    parser.add_argument('--time-offset', type=float, default=-83.22,
                       help='Time offset in ms (camera_time = imu_time + offset)')
    parser.add_argument('--yaw', type=float, default=4.56, help='Camera yaw offset (deg)')
    parser.add_argument('--pitch', type=float, default=-11.25, help='Camera pitch offset (deg)')
    parser.add_argument('--roll', type=float, default=-1.09, help='Camera roll offset (deg)')
    
    args = parser.parse_args()
    
    demo = VIODemo(args)
    demo.run()


if __name__ == '__main__':
    main()
