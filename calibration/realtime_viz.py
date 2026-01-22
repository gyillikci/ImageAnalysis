#!/usr/bin/env python3
"""
Real-time visualization of IMU and camera motion signals.
Shows camera feed with optical flow and live signal plots.
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


class SimpleMotionEstimator:
    """Simple optical flow based motion estimator with visualization."""
    
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.last_flow = None
        self.last_good_prev = None
        self.last_good_curr = None
        
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
        Also stores flow vectors for visualization.
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
        
        # Store for visualization
        self.last_good_prev = good_prev
        self.last_good_curr = good_curr
        
        # Compute flow
        flow = good_curr - good_prev
        self.last_flow = flow
        
        # Convert pixel flow to angular velocity
        mean_flow = np.mean(flow, axis=0)  # [dx, dy] in pixels per frame
        
        # omega_y (yaw) ≈ flow_x / (fx * dt)
        # omega_x (pitch) ≈ -flow_y / (fy * dt)
        omega_y = mean_flow[0] / (self.fx * dt)  # yaw rate
        omega_x = -mean_flow[1] / (self.fy * dt)  # pitch rate
        
        # For roll (rotation about optical axis)
        # Roll causes rotational flow pattern - pixels move in circles around center
        # First, remove the translational component (mean flow) from each point
        flow_centered = flow - mean_flow  # Remove translation to isolate rotation
        
        # Position vectors from image center
        pos = good_prev - np.array([self.cx, self.cy])
        r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        r = np.maximum(r, 10.0)  # Avoid division by zero, also ignore center points
        
        # For pure roll, flow should be perpendicular to position vector
        # Tangent direction at each point is (-y, x) / r (counter-clockwise)
        # Tangential flow component = flow_centered dot tangent
        tangent_x = -pos[:, 1] / r  # Unit tangent x
        tangent_y = pos[:, 0] / r   # Unit tangent y
        tangential = flow_centered[:, 0] * tangent_x + flow_centered[:, 1] * tangent_y
        
        # Angular velocity = tangential velocity / radius
        # Use median to be robust to outliers
        omega_per_point = tangential / r
        omega_z = -np.median(omega_per_point) / dt  # roll rate (negated to match IMU convention)
        
        # Update state
        self.prev_gray = gray
        self.prev_pts = good_curr.reshape(-1, 1, 2).astype(np.float32)
        self.prev_time = timestamp
        
        # Refresh features if too few
        if len(self.prev_pts) < 50:
            new_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if new_pts is not None:
                self.prev_pts = new_pts
        
        p = omega_z   # roll
        q = omega_x   # pitch  
        r = omega_y   # yaw
        
        return p, q, r
    
    def draw_flow(self, frame, scale=5.0):
        """Draw optical flow vectors on frame."""
        vis = frame.copy()
        
        if self.last_good_prev is not None and self.last_good_curr is not None:
            for (x1, y1), (x2, y2) in zip(self.last_good_prev, self.last_good_curr):
                # Draw flow vector (scaled)
                dx = (x2 - x1) * scale
                dy = (y2 - y1) * scale
                
                pt1 = (int(x1), int(y1))
                pt2 = (int(x1 + dx), int(y1 + dy))
                
                # Color by direction: red=right, blue=left, green=up, magenta=down
                angle = np.arctan2(dy, dx)
                hue = int((angle + np.pi) / (2 * np.pi) * 180)
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                
                cv2.arrowedLine(vis, pt1, pt2, color, 1, tipLength=0.3)
                cv2.circle(vis, pt1, 2, (0, 255, 0), -1)
        
        # Draw center crosshair
        cx, cy = int(self.cx), int(self.cy)
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
        
        return vis


def create_plot_image(imu_history, cam_history, width=600, height=400):
    """Create a plot image from signal histories."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background
    
    # Plot areas
    plot_h = height // 3 - 10
    y_offsets = [10, height // 3 + 5, 2 * height // 3]
    labels = ['Roll (p)', 'Pitch (q)', 'Yaw (r)']
    imu_keys = ['p', 'q', 'r']
    cam_keys = ['p', 'q', 'r']
    
    for i, (label, imu_key, cam_key) in enumerate(zip(labels, imu_keys, cam_keys)):
        y_off = y_offsets[i]
        
        # Draw plot background
        cv2.rectangle(img, (50, y_off), (width - 10, y_off + plot_h), (60, 60, 60), -1)
        
        # Draw zero line
        zero_y = y_off + plot_h // 2
        cv2.line(img, (50, zero_y), (width - 10, zero_y), (100, 100, 100), 1)
        
        # Get data
        imu_data = list(imu_history[imu_key])
        cam_data = list(cam_history[cam_key])
        
        if len(imu_data) < 2:
            continue
        
        # Scale: ±3 rad/s = full height
        scale = plot_h / 6.0
        
        # Draw IMU signal (blue)
        pts_imu = []
        for j, val in enumerate(imu_data):
            x = 50 + int(j * (width - 60) / len(imu_data))
            y = zero_y - int(val * scale)
            y = max(y_off, min(y_off + plot_h, y))
            pts_imu.append((x, y))
        
        if len(pts_imu) > 1:
            cv2.polylines(img, [np.array(pts_imu)], False, (255, 100, 100), 1)
        
        # Draw camera signal (red/orange)
        pts_cam = []
        n_cam = len(cam_data)
        n_imu = len(imu_data)
        for j, val in enumerate(cam_data):
            # Align camera samples to same time scale as IMU
            x = 50 + int(j * (width - 60) / max(n_cam, 1))
            y = zero_y - int(val * scale)
            y = max(y_off, min(y_off + plot_h, y))
            pts_cam.append((x, y))
        
        if len(pts_cam) > 1:
            cv2.polylines(img, [np.array(pts_cam)], False, (100, 100, 255), 1)
        
        # Label
        cv2.putText(img, label, (5, y_off + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Scale markers
        cv2.putText(img, '+3', (5, y_off + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(img, '-3', (5, y_off + plot_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    # Legend
    cv2.putText(img, 'IMU', (width - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    cv2.putText(img, 'Cam', (width - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Real-time signal visualization')
    parser.add_argument('--mavlink', required=True, help='MAVLink connection (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', default='0')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fx', type=float, default=1175, help='Focal length X')
    parser.add_argument('--fy', type=float, default=1175, help='Focal length Y')
    parser.add_argument('--history', type=int, default=200, help='History length for plots')
    args = parser.parse_args()

    print("=" * 60)
    print("Real-time Signal Visualizer")
    print("=" * 60)
    print("Controls:")
    print("  Q/ESC - Quit")
    print("  S     - Save screenshot")
    print("=" * 60)
    
    # Signal history buffers
    history_len = args.history
    imu_history = {
        'p': deque(maxlen=history_len),
        'q': deque(maxlen=history_len),
        'r': deque(maxlen=history_len)
    }
    cam_history = {
        'p': deque(maxlen=history_len),
        'q': deque(maxlen=history_len),
        'r': deque(maxlen=history_len)
    }
    
    last_imu = [None]
    
    # --- MAVLink ---
    mavlink_config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )
    mavlink = MAVLinkReceiver(mavlink_config)
    
    def on_imu(imu: IMUData):
        last_imu[0] = imu
        imu_history['p'].append(imu.p)
        imu_history['q'].append(imu.q)
        imu_history['r'].append(imu.r)
    
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
    
    actual_width = camera_config.width
    actual_height = camera_config.height
    print(f"Camera: {actual_width}x{actual_height}")
    
    # --- Motion Estimator ---
    motion_estimator = SimpleMotionEstimator(
        fx=args.fx,
        fy=args.fy,
        cx=actual_width / 2,
        cy=actual_height / 2
    )
    
    # Create windows
    cv2.namedWindow('Camera + Flow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Signals', cv2.WINDOW_NORMAL)
    
    print("\nRunning... Press Q to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frame
            frame_data = camera.get_latest_frame()
            if frame_data is None:
                time.sleep(0.001)
                continue
            
            frame = frame_data.image
            timestamp = frame_data.timestamp
            frame_count += 1
            
            # Estimate motion
            result = motion_estimator.process(frame, timestamp)
            
            if result is not None:
                p, q, r = result
                cam_history['p'].append(p)
                cam_history['q'].append(q)
                cam_history['r'].append(r)
            
            # Draw flow visualization
            vis_frame = motion_estimator.draw_flow(frame, scale=8.0)
            
            # Add text overlay
            elapsed = time.time() - start_time
            fps = frame_count / max(elapsed, 0.001)
            
            # Current values
            if last_imu[0] is not None:
                imu = last_imu[0]
                cv2.putText(vis_frame, f"IMU:  p={imu.p:+.2f}  q={imu.q:+.2f}  r={imu.r:+.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            
            if result is not None:
                cv2.putText(vis_frame, f"CAM:  p={p:+.2f}  q={q:+.2f}  r={r:+.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, actual_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add roll indicator (shows rotation about optical axis)
            if result is not None:
                # Draw rotation indicator at center
                cx, cy = actual_width // 2, actual_height // 2
                radius = 50
                angle = p * 10  # Scale for visibility
                end_x = int(cx + radius * np.sin(angle))
                end_y = int(cy - radius * np.cos(angle))
                cv2.circle(vis_frame, (cx, cy), radius, (100, 100, 100), 1)
                cv2.line(vis_frame, (cx, cy), (end_x, end_y), (0, 255, 255), 2)
                cv2.putText(vis_frame, "ROLL", (cx - 20, cy + radius + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Create signal plot
            plot_img = create_plot_image(imu_history, cam_history, 600, 400)
            
            # Show windows
            cv2.imshow('Camera + Flow', vis_frame)
            cv2.imshow('Signals', plot_img)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Save screenshot
                ts = int(time.time())
                cv2.imwrite(f'screenshot_camera_{ts}.png', vis_frame)
                cv2.imwrite(f'screenshot_signals_{ts}.png', plot_img)
                print(f"Screenshots saved with timestamp {ts}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    # Cleanup
    cv2.destroyAllWindows()
    mavlink.stop()
    camera.stop()
    print("Done")


if __name__ == '__main__':
    main()
