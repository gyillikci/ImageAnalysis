#!/usr/bin/env python3
"""
Plot IMU gyro and camera motion signals for visual comparison.
Uses simple optical flow for camera motion estimation (more robust).
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData, FrameData
from core.utils import monotonic_time
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig


class SimpleMotionEstimator:
    """Simple optical flow based motion estimator."""
    
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        
        # LK optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Feature detection params
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )
    
    def process(self, frame, timestamp):
        """
        Process frame and return (p, q, r) angular rates in rad/s.
        Returns None if estimation fails.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # First frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_time = timestamp
            return None
        
        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_time = timestamp
            return None
        
        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_time = timestamp
            return None
        
        # Filter good points
        status = status.flatten()
        good_prev = self.prev_pts[status == 1].reshape(-1, 2)
        good_curr = curr_pts[status == 1].reshape(-1, 2)
        
        if len(good_prev) < 10:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_time = timestamp
            return None
        
        dt = timestamp - self.prev_time
        if dt <= 0:
            return None
        
        # Compute flow
        flow = good_curr - good_prev
        
        # Convert pixel flow to angular velocity
        # For a pinhole camera:
        # flow_x = fx * omega_y - (x - cx) * omega_z  (approximately, ignoring translation)
        # flow_y = -fy * omega_x - (y - cy) * omega_z
        
        # Simplified: use mean flow and assume rotation about optical axis (yaw) dominates
        mean_flow = np.mean(flow, axis=0)  # [dx, dy] in pixels per frame
        
        # Convert to angular rate
        # omega_y (yaw around vertical) ≈ flow_x / (fx * dt)
        # omega_x (pitch) ≈ -flow_y / (fy * dt)
        
        omega_y = mean_flow[0] / (self.fx * dt)  # yaw rate
        omega_x = -mean_flow[1] / (self.fy * dt)  # pitch rate
        
        # For roll (rotation about optical axis)
        # Roll causes rotational flow pattern - pixels move in circles around center
        # First, remove the translational component (mean flow) from each point
        flow_centered = flow - mean_flow  # Remove translation to isolate rotation
        
        # Position vectors from image center
        pos = good_prev - np.array([self.cx, self.cy])
        r_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        r_dist = np.maximum(r_dist, 10.0)  # Avoid division by zero
        
        # For pure roll, flow should be perpendicular to position vector
        # Tangent direction at each point is (-y, x) / r (counter-clockwise)
        tangent_x = -pos[:, 1] / r_dist
        tangent_y = pos[:, 0] / r_dist
        tangential = flow_centered[:, 0] * tangent_x + flow_centered[:, 1] * tangent_y
        
        # Angular velocity = tangential velocity / radius
        omega_per_point = tangential / r_dist
        omega_z = -np.median(omega_per_point) / dt  # roll rate (negated for IMU convention)
        
        # Update state
        self.prev_gray = gray
        self.prev_pts = good_curr.reshape(-1, 1, 2).astype(np.float32)
        self.prev_time = timestamp
        
        # Refresh features if too few
        if len(self.prev_pts) < 50:
            new_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if new_pts is not None:
                self.prev_pts = new_pts
        
        # Return p, q, r (roll, pitch, yaw rates)
        # Map to body frame (depends on camera orientation)
        p = omega_z   # roll
        q = omega_x   # pitch
        r = omega_y   # yaw
        
        return p, q, r


def main():
    parser = argparse.ArgumentParser(description='Plot IMU and camera signals')
    parser.add_argument('--mavlink', required=True, help='MAVLink connection (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', default='0')
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration (seconds)')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fx', type=float, default=1762, help='Focal length X')
    parser.add_argument('--fy', type=float, default=1762, help='Focal length Y')
    args = parser.parse_args()

    print("=" * 60)
    print("Signal Plotter - IMU vs Camera Motion")
    print("=" * 60)
    
    # Storage for data
    imu_times = []
    imu_p = []  # roll rate
    imu_q = []  # pitch rate
    imu_r = []  # yaw rate
    
    cam_times = []
    cam_p = []
    cam_q = []
    cam_r = []
    
    start_time = [None]  # Use list to allow modification in callback
    
    # --- MAVLink ---
    mavlink_config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )
    mavlink = MAVLinkReceiver(mavlink_config)
    
    def on_imu(imu: IMUData):
        if start_time[0] is None:
            start_time[0] = imu.timestamp
        t = imu.timestamp - start_time[0]
        imu_times.append(t)
        imu_p.append(imu.p)
        imu_q.append(imu.q)
        imu_r.append(imu.r)
    
    mavlink.add_callback('imu', on_imu)
    
    if not mavlink.start():
        print("Failed to start MAVLink")
        return
    
    # --- Camera ---
    camera_id = int(args.camera) if args.camera.isdigit() else args.camera
    camera_config = CameraConfig(
        device=camera_id,
        width=args.width,
        height=args.height
    )
    camera = HarrierCamera(camera_config)
    if not camera.start():
        print("Failed to start camera")
        mavlink.stop()
        return
    print(f"Camera: {camera_config.width}x{camera_config.height}")
    
    # --- Motion Estimator (simple optical flow) ---
    motion_estimator = SimpleMotionEstimator(
        fx=args.fx,
        fy=args.fy,
        cx=args.width / 2,
        cy=args.height / 2
    )
    
    print(f"\nRecording for {args.duration} seconds...")
    print(">>> ROTATE THE DEVICE BACK AND FORTH! <<<\n")
    
    record_start = time.time()
    frame_count = 0
    valid_count = 0
    
    try:
        while time.time() - record_start < args.duration:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            result = motion_estimator.process(frame.image, frame.timestamp)
            
            if result is not None:
                p, q, r = result
                valid_count += 1
                if start_time[0] is None:
                    start_time[0] = frame.timestamp
                t = frame.timestamp - start_time[0]
                cam_times.append(t)
                cam_p.append(p)
                cam_q.append(q)
                cam_r.append(r)
            
            # Progress indicator
            elapsed = time.time() - record_start
            if frame_count % 30 == 0:
                print(f"  t={elapsed:.1f}s: IMU={len(imu_times)}, Cam={valid_count}/{frame_count}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    # Cleanup
    mavlink.stop()
    camera.stop()
    
    print(f"\n{'='*60}")
    print(f"Recorded: IMU={len(imu_times)}, Camera={len(cam_times)}")
    print(f"{'='*60}")
    
    if len(imu_times) < 10 or len(cam_times) < 10:
        print("Not enough data to plot!")
        return
    
    # Convert to numpy
    imu_times = np.array(imu_times)
    imu_p = np.array(imu_p)
    imu_q = np.array(imu_q)
    imu_r = np.array(imu_r)
    
    cam_times = np.array(cam_times)
    cam_p = np.array(cam_p)
    cam_q = np.array(cam_q)
    cam_r = np.array(cam_r)
    
    # Print stats
    print(f"\nIMU stats:")
    print(f"  p (roll):  mean={np.mean(imu_p):.3f}, std={np.std(imu_p):.3f} rad/s")
    print(f"  q (pitch): mean={np.mean(imu_q):.3f}, std={np.std(imu_q):.3f} rad/s")
    print(f"  r (yaw):   mean={np.mean(imu_r):.3f}, std={np.std(imu_r):.3f} rad/s")
    
    print(f"\nCamera stats:")
    print(f"  p (roll):  mean={np.mean(cam_p):.3f}, std={np.std(cam_p):.3f} rad/s")
    print(f"  q (pitch): mean={np.mean(cam_q):.3f}, std={np.std(cam_q):.3f} rad/s")
    print(f"  r (yaw):   mean={np.mean(cam_r):.3f}, std={np.std(cam_r):.3f} rad/s")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('IMU vs Camera Angular Rates', fontsize=14)
    
    # Plot roll (p)
    ax = axes[0]
    ax.plot(imu_times, imu_p, 'b-', alpha=0.7, label='IMU p (roll)', linewidth=0.8)
    ax.plot(cam_times, cam_p, 'r-', alpha=0.7, label='Cam p (roll)', linewidth=0.8)
    ax.set_ylabel('Roll Rate (rad/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Roll Rate (p) - rotation around optical axis')
    
    # Plot pitch (q)
    ax = axes[1]
    ax.plot(imu_times, imu_q, 'b-', alpha=0.7, label='IMU q (pitch)', linewidth=0.8)
    ax.plot(cam_times, cam_q, 'r-', alpha=0.7, label='Cam q (pitch)', linewidth=0.8)
    ax.set_ylabel('Pitch Rate (rad/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Pitch Rate (q) - tilting up/down')
    
    # Plot yaw (r)
    ax = axes[2]
    ax.plot(imu_times, imu_r, 'b-', alpha=0.7, label='IMU r (yaw)', linewidth=0.8)
    ax.plot(cam_times, cam_r, 'r-', alpha=0.7, label='Cam r (yaw)', linewidth=0.8)
    ax.set_ylabel('Yaw Rate (rad/s)')
    ax.set_xlabel('Time (seconds)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Yaw Rate (r) - panning left/right')
    
    plt.tight_layout()
    
    # Save to file
    output_file = 'signals_plot.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    
    # Also save data for further analysis
    np.savez('signals_data.npz',
             imu_times=imu_times, imu_p=imu_p, imu_q=imu_q, imu_r=imu_r,
             cam_times=cam_times, cam_p=cam_p, cam_q=cam_q, cam_r=cam_r)
    print("Data saved to: signals_data.npz")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    main()
