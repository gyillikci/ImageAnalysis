#!/usr/bin/env python3
"""
Calibration Validation Tool

Validates the camera-IMU calibration by:
1. Applying the time offset correction
2. Applying the spatial (mount angle) correction
3. Computing correlation before and after correction
4. Plotting aligned signals
"""

import argparse
import time
import numpy as np
import cv2
from collections import deque
from scipy import signal, interpolate
from scipy.spatial.transform import Rotation

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData
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
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )
    
    def process(self, frame, timestamp):
        """Process frame and return (p, q, r) angular rates in rad/s."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
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
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_time = timestamp
            return None
        
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
        
        flow = good_curr - good_prev
        mean_flow = np.mean(flow, axis=0)
        
        omega_y = mean_flow[0] / (self.fx * dt)
        omega_x = -mean_flow[1] / (self.fy * dt)
        
        flow_centered = flow - mean_flow
        pos = good_prev - np.array([self.cx, self.cy])
        r_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        r_dist = np.maximum(r_dist, 10.0)
        
        tangent_x = -pos[:, 1] / r_dist
        tangent_y = pos[:, 0] / r_dist
        tangential = flow_centered[:, 0] * tangent_x + flow_centered[:, 1] * tangent_y
        
        omega_per_point = tangential / r_dist
        omega_z = -np.median(omega_per_point) / dt
        
        self.prev_gray = gray
        self.prev_pts = good_curr.reshape(-1, 1, 2).astype(np.float32)
        self.prev_time = timestamp
        
        if len(self.prev_pts) < 50:
            new_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if new_pts is not None:
                self.prev_pts = new_pts
        
        p = omega_z
        q = omega_x
        r = omega_y
        
        return p, q, r


def compute_correlation(imu_times, imu_data, cam_times, cam_data, time_offset=0.0):
    """Compute correlation with optional time offset."""
    # Apply time offset to camera data
    cam_times_shifted = cam_times + time_offset
    
    # Find overlapping time range
    t_min = max(imu_times[0], cam_times_shifted[0])
    t_max = min(imu_times[-1], cam_times_shifted[-1])
    
    if t_max <= t_min:
        return 0.0
    
    # Resample to common time base
    hz = 60.0
    t_common = np.arange(t_min, t_max, 1.0/hz)
    
    if len(t_common) < 10:
        return 0.0
    
    # Interpolate
    imu_interp = interpolate.interp1d(imu_times, imu_data, bounds_error=False, fill_value=0.0)
    cam_interp = interpolate.interp1d(cam_times_shifted, cam_data, bounds_error=False, fill_value=0.0)
    
    imu_resampled = imu_interp(t_common)
    cam_resampled = cam_interp(t_common)
    
    # Filter
    if len(t_common) > 20:
        nyquist = hz / 2.0
        b, a = signal.butter(2, 10.0 / nyquist)
        imu_filtered = signal.filtfilt(b, a, imu_resampled)
        cam_filtered = signal.filtfilt(b, a, cam_resampled)
    else:
        imu_filtered = imu_resampled
        cam_filtered = cam_resampled
    
    # Correlation coefficient
    if np.std(imu_filtered) > 0 and np.std(cam_filtered) > 0:
        corr = np.corrcoef(imu_filtered, cam_filtered)[0, 1]
        return corr
    return 0.0


def apply_spatial_correction(cam_p, cam_q, cam_r, yaw_deg, pitch_deg, roll_deg):
    """Apply spatial (mount angle) correction to camera angular rates."""
    # Create rotation matrix from mount angles
    R = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    
    # Transform camera angular rates to IMU frame
    corrected = []
    for p, q, r in zip(cam_p, cam_q, cam_r):
        omega_cam = np.array([p, q, r])
        omega_imu = R @ omega_cam
        corrected.append(omega_imu)
    
    corrected = np.array(corrected)
    return corrected[:, 0], corrected[:, 1], corrected[:, 2]


def main():
    parser = argparse.ArgumentParser(description='Validate Camera-IMU Calibration')
    parser.add_argument('--mavlink', required=True, help='MAVLink connection (e.g., COM6)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fx', type=float, default=1175.0)
    parser.add_argument('--fy', type=float, default=1175.0)
    parser.add_argument('--duration', type=float, default=20.0)
    
    # Calibration parameters (from our calibration)
    parser.add_argument('--time-offset', type=float, default=-83.22,
                       help='Time offset in ms')
    parser.add_argument('--yaw', type=float, default=4.56)
    parser.add_argument('--pitch', type=float, default=-11.25)
    parser.add_argument('--roll', type=float, default=-1.09)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Calibration Validation")
    print("=" * 60)
    print(f"\nCalibration to validate:")
    print(f"  Time offset: {args.time_offset:.2f} ms")
    print(f"  Spatial: yaw={args.yaw:.2f}°, pitch={args.pitch:.2f}°, roll={args.roll:.2f}°")
    print()
    
    # Setup MAVLink
    print(f"Connecting to MAVLink: {args.mavlink}")
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
    
    print(f"Camera: {args.width}x{args.height}\n")
    
    # Motion estimator
    estimator = SimpleMotionEstimator(
        fx=args.fx, fy=args.fy,
        cx=args.width/2, cy=args.height/2
    )
    
    # Data collection
    imu_data = []
    cam_data = []
    
    def imu_callback(imu: IMUData):
        imu_data.append({
            'timestamp': imu.timestamp,
            'p': imu.p, 'q': imu.q, 'r': imu.r
        })
    
    mavlink.add_callback('imu', imu_callback)
    
    print(f"Recording for {args.duration:.1f} seconds...")
    print(">>> ROTATE THE DEVICE IN ALL AXES! <<<\n")
    
    start_time = time.time()
    last_print = start_time
    
    try:
        while time.time() - start_time < args.duration:
            frame_data = camera.get_latest_frame()
            if frame_data is not None:
                result = estimator.process(frame_data.image, frame_data.timestamp)
                if result is not None:
                    p, q, r = result
                    cam_data.append({
                        'timestamp': frame_data.timestamp,
                        'p': p, 'q': q, 'r': r
                    })
            
            # Progress
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                print(f"  t={elapsed:.1f}s: IMU={len(imu_data)}, Cam={len(cam_data)}")
                last_print = now
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        camera.stop()
        mavlink.stop()
    
    print(f"\nRecorded: IMU={len(imu_data)}, Camera={len(cam_data)}")
    
    if len(imu_data) < 100 or len(cam_data) < 50:
        print("Not enough data!")
        return
    
    # Extract arrays
    imu_times = np.array([d['timestamp'] for d in imu_data])
    imu_p = np.array([d['p'] for d in imu_data])
    imu_q = np.array([d['q'] for d in imu_data])
    imu_r = np.array([d['r'] for d in imu_data])
    
    cam_times = np.array([d['timestamp'] for d in cam_data])
    cam_p = np.array([d['p'] for d in cam_data])
    cam_q = np.array([d['q'] for d in cam_data])
    cam_r = np.array([d['r'] for d in cam_data])
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    # Compute correlations WITHOUT correction
    print("\n--- WITHOUT Calibration ---")
    corr_p_raw = compute_correlation(imu_times, imu_p, cam_times, cam_p, 0.0)
    corr_q_raw = compute_correlation(imu_times, imu_q, cam_times, cam_q, 0.0)
    corr_r_raw = compute_correlation(imu_times, imu_r, cam_times, cam_r, 0.0)
    avg_raw = (abs(corr_p_raw) + abs(corr_q_raw) + abs(corr_r_raw)) / 3
    
    print(f"  Roll (p):  correlation = {corr_p_raw:.3f}")
    print(f"  Pitch (q): correlation = {corr_q_raw:.3f}")
    print(f"  Yaw (r):   correlation = {corr_r_raw:.3f}")
    print(f"  Average:   {avg_raw:.3f}")
    
    # Compute correlations WITH time offset only
    print("\n--- WITH Time Offset Only ---")
    time_offset_sec = args.time_offset / 1000.0
    corr_p_time = compute_correlation(imu_times, imu_p, cam_times, cam_p, time_offset_sec)
    corr_q_time = compute_correlation(imu_times, imu_q, cam_times, cam_q, time_offset_sec)
    corr_r_time = compute_correlation(imu_times, imu_r, cam_times, cam_r, time_offset_sec)
    avg_time = (abs(corr_p_time) + abs(corr_q_time) + abs(corr_r_time)) / 3
    
    print(f"  Roll (p):  correlation = {corr_p_time:.3f}")
    print(f"  Pitch (q): correlation = {corr_q_time:.3f}")
    print(f"  Yaw (r):   correlation = {corr_r_time:.3f}")
    print(f"  Average:   {avg_time:.3f}")
    
    # Apply spatial correction
    cam_p_corr, cam_q_corr, cam_r_corr = apply_spatial_correction(
        cam_p, cam_q, cam_r, args.yaw, args.pitch, args.roll
    )
    
    # Compute correlations WITH full calibration
    print("\n--- WITH Full Calibration ---")
    corr_p_full = compute_correlation(imu_times, imu_p, cam_times, cam_p_corr, time_offset_sec)
    corr_q_full = compute_correlation(imu_times, imu_q, cam_times, cam_q_corr, time_offset_sec)
    corr_r_full = compute_correlation(imu_times, imu_r, cam_times, cam_r_corr, time_offset_sec)
    avg_full = (abs(corr_p_full) + abs(corr_q_full) + abs(corr_r_full)) / 3
    
    print(f"  Roll (p):  correlation = {corr_p_full:.3f}")
    print(f"  Pitch (q): correlation = {corr_q_full:.3f}")
    print(f"  Yaw (r):   correlation = {corr_r_full:.3f}")
    print(f"  Average:   {avg_full:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCorrelation improvement:")
    print(f"  Raw:              {avg_raw:.3f}")
    print(f"  + Time offset:    {avg_time:.3f} ({(avg_time/avg_raw - 1)*100:+.1f}%)")
    print(f"  + Spatial:        {avg_full:.3f} ({(avg_full/avg_raw - 1)*100:+.1f}%)")
    
    if avg_full > 0.7:
        print(f"\n✓ CALIBRATION VALIDATED - Strong correlation ({avg_full:.3f})")
    elif avg_full > 0.5:
        print(f"\n~ CALIBRATION ACCEPTABLE - Moderate correlation ({avg_full:.3f})")
    else:
        print(f"\n✗ CALIBRATION POOR - Weak correlation ({avg_full:.3f})")
    
    # Create plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Calibration Validation', fontsize=14)
        
        # Resample for plotting
        t_min = max(imu_times[0], cam_times[0] + time_offset_sec)
        t_max = min(imu_times[-1], cam_times[-1] + time_offset_sec)
        t_plot = np.linspace(t_min, t_max, 500)
        t_plot_rel = t_plot - t_min
        
        for i, (name, imu_vals, cam_vals_raw, cam_vals_corr) in enumerate([
            ('Roll (p)', imu_p, cam_p, cam_p_corr),
            ('Pitch (q)', imu_q, cam_q, cam_q_corr),
            ('Yaw (r)', imu_r, cam_r, cam_r_corr)
        ]):
            # Interpolate
            imu_interp = interpolate.interp1d(imu_times, imu_vals, bounds_error=False, fill_value=0.0)
            cam_raw_interp = interpolate.interp1d(cam_times, cam_vals_raw, bounds_error=False, fill_value=0.0)
            cam_corr_interp = interpolate.interp1d(cam_times + time_offset_sec, cam_vals_corr, bounds_error=False, fill_value=0.0)
            
            imu_plot = imu_interp(t_plot)
            cam_raw_plot = cam_raw_interp(t_plot)
            cam_corr_plot = cam_corr_interp(t_plot)
            
            # Raw (left column)
            axes[i, 0].plot(t_plot_rel, imu_plot, 'b-', label='IMU', linewidth=1)
            axes[i, 0].plot(t_plot_rel, cam_raw_plot, 'r-', label='Camera (raw)', linewidth=1, alpha=0.7)
            axes[i, 0].set_ylabel(f'{name} (rad/s)')
            axes[i, 0].legend(loc='upper right')
            axes[i, 0].set_title(f'{name} - Raw')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Corrected (right column)
            axes[i, 1].plot(t_plot_rel, imu_plot, 'b-', label='IMU', linewidth=1)
            axes[i, 1].plot(t_plot_rel, cam_corr_plot, 'g-', label='Camera (corrected)', linewidth=1, alpha=0.7)
            axes[i, 1].set_ylabel(f'{name} (rad/s)')
            axes[i, 1].legend(loc='upper right')
            axes[i, 1].set_title(f'{name} - After Calibration')
            axes[i, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('calibration_validation.png', dpi=150)
        print(f"\nPlot saved to: calibration_validation.png")
        
    except ImportError:
        print("\nNote: matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
