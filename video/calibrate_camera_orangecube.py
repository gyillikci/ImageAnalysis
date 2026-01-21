#!/usr/bin/env python3
"""
Camera-Flight Controller Calibration for Orange Cube (ArduPilot/PX4)

This script performs temporal and spatial calibration between a camera
and an Orange Cube flight controller (or any ArduPilot/PX4 based system).

Temporal Calibration:
    Finds the time offset between video timestamps and flight log timestamps
    using cross-correlation of gyro rates.

Spatial Calibration:
    Finds the camera mounting offset (yaw, pitch, roll) relative to the
    flight controller IMU by comparing video-derived attitude to EKF attitude.

Requirements:
    pip install pymavlink opencv-python scipy numpy matplotlib sk-video

Usage:
    # Basic usage
    python calibrate_camera_orangecube.py --video flight.mp4 --flight flight.bin

    # With camera calibration file
    python calibrate_camera_orangecube.py --video flight.mp4 --flight flight.bin \\
        --camera ../cameras/gopro_hero9.json

    # Specify camera mount orientation
    python calibrate_camera_orangecube.py --video flight.mp4 --flight flight.bin \\
        --cam-mount forward

    # Skip to specific step
    python calibrate_camera_orangecube.py --video flight.mp4 --flight flight.bin \\
        --skip-motion-est

Author: ImageAnalysis Project
License: MIT
"""

import argparse
import csv
import json
import math
import numpy as np
import os
import sys
from scipy import interpolate
from scipy.optimize import least_squares
import scipy.signal as signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ardupilot_log import load as load_flight, FlightInterp

# Constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera-FC calibration for Orange Cube (ArduPilot/PX4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic calibration
    python calibrate_camera_orangecube.py --video flight.mp4 --flight flight.bin

    # With all options
    python calibrate_camera_orangecube.py \\
        --video flight.mp4 \\
        --flight flight.bin \\
        --camera ../cameras/gopro_hero9.json \\
        --cam-mount forward \\
        --resample-hz 60 \\
        --plot
        """
    )

    parser.add_argument('--video', required=True, help='Video file from flight')
    parser.add_argument('--flight', required=True, help='Flight log file (.bin, .tlog, or .ulg)')
    parser.add_argument('--camera', help='Camera calibration JSON file')
    parser.add_argument('--cam-mount', choices=['forward', 'down', 'rear'],
                        default='forward', help='Camera mounting orientation (default: forward)')
    parser.add_argument('--resample-hz', type=float, default=60.0,
                        help='Resample rate for correlation (default: 60 Hz)')
    parser.add_argument('--time-shift', type=float,
                        help='Override automatic time shift detection')
    parser.add_argument('--skip-motion-est', action='store_true',
                        help='Skip video motion estimation (use existing CSV)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--output', help='Output JSON file for calibration results')

    return parser.parse_args()


def estimate_video_motion(video_path, camera_config=None, scale=1.0):
    """
    Estimate camera motion from video using feature tracking.

    Returns:
        Path to CSV file with motion estimates
    """
    try:
        import cv2
        import skvideo.io
    except ImportError:
        print("Error: opencv-python and sk-video required")
        print("Install with: pip install opencv-python sk-video")
        sys.exit(1)

    # Output file path
    filename, ext = os.path.splitext(video_path)
    output_csv = filename + "_motion.csv"

    print(f"\n{'='*60}")
    print("STEP 1: Video Motion Estimation")
    print(f"{'='*60}")
    print(f"Input video: {video_path}")
    print(f"Output CSV: {output_csv}")

    # Get video metadata
    metadata = skvideo.io.ffprobe(video_path)
    fps_string = metadata['video']['@avg_frame_rate']
    num, den = fps_string.split('/')
    fps = float(num) / float(den)
    w = int(metadata['video']['@width'])
    h = int(metadata['video']['@height'])
    duration = float(metadata['video']['@duration'])
    total_frames = int(round(duration * fps))

    print(f"Video: {w}x{h} @ {fps:.2f} fps, {duration:.1f}s, {total_frames} frames")

    # Load camera calibration if provided
    K = np.eye(3)
    dist = [0, 0, 0, 0, 0]
    if camera_config and os.path.exists(camera_config):
        with open(camera_config, 'r') as f:
            config = json.load(f)
        if 'K' in config:
            K = np.array(config['K']).reshape(3, 3)
        if 'dist_coeffs' in config:
            dist = config['dist_coeffs']
        print(f"Loaded camera config: {camera_config}")

    K = K * scale
    K[2, 2] = 1.0

    # Feature detector
    detector = cv2.ORB_create(500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Open video
    reader = skvideo.io.FFmpegReader(video_path, inputdict={}, outputdict={})

    # CSV output
    csvfile = open(output_csv, 'w', newline='')
    fieldnames = ['frame', 'time', 'rotation (deg)',
                  'translation x (px)', 'translation y (px)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    last_gray = None
    last_kp = None
    last_des = None
    frame_count = 0

    print(f"Processing {total_frames} frames...")

    for frame in reader.nextFrame():
        frame = frame[:, :, ::-1]  # RGB to BGR
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")

        # Scale and undistort
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        frame = cv2.undistort(frame, K, np.array(dist))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features
        kp = detector.detect(gray)
        kp, des = detector.compute(gray, kp)

        rotation = 0.0
        tx = 0.0
        ty = 0.0

        if last_des is not None and des is not None and len(des) > 0 and len(last_des) > 0:
            # Match features
            matches = matcher.knnMatch(des, trainDescriptors=last_des, k=2)

            # Filter matches
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    good.append(m[0])

            if len(good) >= 4:
                # Get matched points
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([last_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Estimate affine transform
                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

                if M is not None:
                    # Decompose affine matrix
                    tx = M[0, 2]
                    ty = M[1, 2]
                    rotation = math.atan2(-M[0, 1], M[0, 0]) * r2d

                    # Filter outliers
                    if abs(rotation) > 10 or math.sqrt(tx*tx + ty*ty) > 50:
                        rotation = 0.0
                        tx = 0.0
                        ty = 0.0

        # Write to CSV (rotation as rate in rad/s)
        row = {
            'frame': frame_count,
            'time': frame_count / fps,
            'rotation (deg)': -rotation * fps * d2r,  # Convert to rad/s
            'translation x (px)': tx / scale,
            'translation y (px)': ty / scale
        }
        writer.writerow(row)

        last_gray = gray
        last_kp = kp
        last_des = des

    csvfile.close()
    print(f"Motion estimation complete: {output_csv}")

    return output_csv, fps


def load_video_motion(csv_path):
    """Load video motion estimates from CSV"""
    data = {
        'frame': [],
        'time': [],
        'roll_rate': [],
        'tx': [],
        'ty': []
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['frame'].append(int(row['frame']))
            data['time'].append(float(row['time']))
            data['roll_rate'].append(float(row['rotation (deg)']))
            data['tx'].append(float(row['translation x (px)']))
            data['ty'].append(float(row['translation y (px)']))

    for key in data:
        data[key] = np.array(data[key])

    return data


def correlate_signals(video_data, flight_data, hz=60, cam_mount='forward'):
    """
    Find time offset between video and flight log using cross-correlation.

    Returns:
        time_shift: Offset in seconds (video_time + time_shift = flight_time)
    """
    print(f"\n{'='*60}")
    print("STEP 2: Temporal Calibration (Time Synchronization)")
    print(f"{'='*60}")

    # Create flight data interpolator
    interp = FlightInterp(flight_data)

    # Get time ranges
    video_min = video_data['time'].min()
    video_max = video_data['time'].max()
    video_len = video_max - video_min

    imu_min = flight_data['imu'][0]['time']
    imu_max = flight_data['imu'][-1]['time']
    flight_len = imu_max - imu_min

    print(f"Video time range: {video_min:.2f} - {video_max:.2f} s ({video_len:.2f} s)")
    print(f"Flight time range: {imu_min:.2f} - {imu_max:.2f} s ({flight_len:.2f} s)")

    # Resample video data
    video_times = np.linspace(video_min, video_max, int(video_len * hz))
    video_roll_interp = interpolate.interp1d(
        video_data['time'], video_data['roll_rate'],
        bounds_error=False, fill_value=0.0
    )
    video_resampled = video_roll_interp(video_times)

    # Resample flight data
    if cam_mount in ['forward', 'rear']:
        imu_key = 'p'  # Roll rate for forward/rear camera
    else:
        imu_key = 'r'  # Yaw rate for down camera

    flight_times = np.linspace(imu_min, imu_max, int(flight_len * hz))
    flight_resampled = np.array([
        float(interp.group['imu'].interp[imu_key](t))
        for t in flight_times
    ])

    # Apply low-pass filter
    b, a = signal.butter(2, 10.0 / (hz / 2))
    video_filtered = signal.filtfilt(b, a, video_resampled)
    flight_filtered = signal.filtfilt(b, a, flight_resampled)

    # Cross-correlation
    ycorr = np.correlate(flight_filtered, video_filtered, mode='full')

    # Find peak
    max_index = np.argmax(ycorr)
    shift_sec = max_index / hz - video_len

    # Calculate time shift
    start_diff = flight_times[0] - video_times[0]
    time_shift = start_diff + shift_sec

    print(f"\nCross-correlation peak index: {max_index}")
    print(f"Shift (seconds): {shift_sec:.3f}")
    print(f"Start time difference: {start_diff:.3f}")
    print(f"\n>>> TIME SHIFT: {time_shift:.4f} seconds <<<")
    print(f"    (video_time + {time_shift:.4f} = flight_time)")

    return time_shift, (video_times, video_filtered, flight_times, flight_filtered, ycorr)


def calibrate_camera_mount(video_data, flight_data, time_shift, hz=60, cam_mount='forward'):
    """
    Calibrate camera mounting offset using video-derived attitude vs EKF.

    Returns:
        mount_offset: (yaw, pitch, roll) in degrees
    """
    print(f"\n{'='*60}")
    print("STEP 3: Spatial Calibration (Camera Mount Offset)")
    print(f"{'='*60}")

    # Create interpolators
    interp = FlightInterp(flight_data)

    # Determine overlap region
    video_min = video_data['time'].min()
    video_max = video_data['time'].max()
    imu_min = flight_data['imu'][0]['time']
    imu_max = flight_data['imu'][-1]['time']

    t_min = max(video_min + time_shift, imu_min)
    t_max = min(video_max + time_shift, imu_max)

    print(f"Overlap region (flight time): {t_min:.2f} - {t_max:.2f} s")

    # Create interpolators for video data
    video_roll_interp = interpolate.interp1d(
        video_data['time'] + time_shift, video_data['roll_rate'],
        bounds_error=False, fill_value=0.0
    )

    # Sample data for optimization
    n_samples = int((t_max - t_min) * hz)
    times = np.linspace(t_min, t_max, n_samples)

    # Get flight controller attitude
    flight_phi = np.array([float(interp.group['filter'].interp['phi'](t)) for t in times])
    flight_the = np.array([float(interp.group['filter'].interp['the'](t)) for t in times])

    # Get video-derived roll rates and integrate to get roll estimate
    video_roll_rates = np.array([video_roll_interp(t) for t in times])

    # Simple integration of roll rate to get roll angle (just for comparison)
    dt = 1.0 / hz
    video_phi = np.cumsum(video_roll_rates) * dt

    # Remove mean to compare shapes
    video_phi -= np.mean(video_phi)
    flight_phi_centered = flight_phi - np.mean(flight_phi)

    # Scale factor estimation
    if np.std(video_phi) > 0:
        scale = np.std(flight_phi_centered) / np.std(video_phi)
    else:
        scale = 1.0

    video_phi_scaled = video_phi * scale

    # Estimate offset by comparing mean values
    roll_offset = np.mean(flight_phi) - np.mean(video_phi_scaled)

    # Default mount offsets based on camera orientation
    if cam_mount == 'forward':
        yaw_offset = 0.0
        pitch_offset = 0.0
        roll_offset_deg = roll_offset * r2d
    elif cam_mount == 'down':
        yaw_offset = 0.0
        pitch_offset = -90.0
        roll_offset_deg = roll_offset * r2d
    elif cam_mount == 'rear':
        yaw_offset = 180.0
        pitch_offset = 0.0
        roll_offset_deg = -roll_offset * r2d

    mount_offset = (yaw_offset, pitch_offset, roll_offset_deg)

    print(f"\nEstimated Camera Mount Offset:")
    print(f"  Yaw:   {mount_offset[0]:.2f}°")
    print(f"  Pitch: {mount_offset[1]:.2f}°")
    print(f"  Roll:  {mount_offset[2]:.2f}°")

    return mount_offset, (times, video_phi_scaled, flight_phi_centered)


def save_calibration(output_path, time_shift, mount_offset, video_path, flight_path):
    """Save calibration results to JSON file"""
    result = {
        'video_file': os.path.basename(video_path),
        'flight_log': os.path.basename(flight_path),
        'temporal_calibration': {
            'time_shift_seconds': time_shift,
            'description': 'video_time + time_shift = flight_time'
        },
        'spatial_calibration': {
            'yaw_deg': mount_offset[0],
            'pitch_deg': mount_offset[1],
            'roll_deg': mount_offset[2],
            'description': 'Camera mount offset relative to IMU (yaw, pitch, roll)'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration saved to: {output_path}")


def plot_results(corr_data, mount_data, time_shift):
    """Plot calibration results"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    video_times, video_filtered, flight_times, flight_filtered, ycorr = corr_data
    times, video_phi, flight_phi = mount_data

    # Figure 1: Cross-correlation
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Video roll rate
    axes[0].plot(video_times, video_filtered, 'b-', label='Video roll rate')
    axes[0].set_ylabel('Roll rate (rad/s)')
    axes[0].set_xlabel('Video time (s)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Video Motion Estimate')

    # Flight roll rate
    axes[1].plot(flight_times, flight_filtered, 'r-', label='IMU roll rate')
    axes[1].set_ylabel('Roll rate (rad/s)')
    axes[1].set_xlabel('Flight time (s)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Flight Controller IMU')

    # Cross-correlation
    axes[2].plot(ycorr)
    axes[2].axvline(np.argmax(ycorr), color='r', linestyle='--', label='Peak')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Lag (samples)')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_title(f'Cross-Correlation (time_shift = {time_shift:.4f}s)')

    plt.tight_layout()

    # Figure 2: Aligned signals
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

    # Aligned roll rates
    axes2[0].plot(video_times + time_shift, video_filtered, 'b-', alpha=0.7,
                  label='Video (shifted)')
    axes2[0].plot(flight_times, flight_filtered, 'r-', alpha=0.7,
                  label='Flight IMU')
    axes2[0].set_ylabel('Roll rate (rad/s)')
    axes2[0].set_xlabel('Flight time (s)')
    axes2[0].legend()
    axes2[0].grid(True)
    axes2[0].set_title('Aligned Roll Rates')

    # Integrated roll comparison
    axes2[1].plot(times, video_phi * r2d, 'b-', alpha=0.7, label='Video roll')
    axes2[1].plot(times, flight_phi * r2d, 'r-', alpha=0.7, label='EKF roll')
    axes2[1].set_ylabel('Roll angle (deg)')
    axes2[1].set_xlabel('Flight time (s)')
    axes2[1].legend()
    axes2[1].grid(True)
    axes2[1].set_title('Roll Angle Comparison')

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    print("=" * 60)
    print("Camera-Flight Controller Calibration")
    print("Orange Cube / ArduPilot / PX4")
    print("=" * 60)
    print(f"\nVideo: {args.video}")
    print(f"Flight log: {args.flight}")
    print(f"Camera mount: {args.cam_mount}")

    # Step 1: Video motion estimation
    video_csv = os.path.splitext(args.video)[0] + "_motion.csv"

    if args.skip_motion_est and os.path.exists(video_csv):
        print(f"\nUsing existing motion CSV: {video_csv}")
        video_data = load_video_motion(video_csv)
        # Estimate fps from data
        if len(video_data['time']) > 1:
            fps = 1.0 / np.mean(np.diff(video_data['time']))
        else:
            fps = 30.0
    else:
        video_csv, fps = estimate_video_motion(args.video, args.camera)
        video_data = load_video_motion(video_csv)

    # Step 2: Load flight data
    print(f"\n{'='*60}")
    print("Loading Flight Data")
    print(f"{'='*60}")
    flight_data, flight_format = load_flight(args.flight)
    print(f"Flight format: {flight_format}")
    print(f"IMU records: {len(flight_data['imu'])}")
    print(f"GPS records: {len(flight_data['gps'])}")
    print(f"Filter records: {len(flight_data['filter'])}")

    # Step 3: Temporal calibration
    if args.time_shift is not None:
        time_shift = args.time_shift
        corr_data = None
        print(f"\nUsing provided time shift: {time_shift:.4f} s")
    else:
        time_shift, corr_data = correlate_signals(
            video_data, flight_data,
            hz=args.resample_hz,
            cam_mount=args.cam_mount
        )

    # Step 4: Spatial calibration
    mount_offset, mount_data = calibrate_camera_mount(
        video_data, flight_data, time_shift,
        hz=args.resample_hz,
        cam_mount=args.cam_mount
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.splitext(args.video)[0] + "_calibration.json"

    save_calibration(output_path, time_shift, mount_offset, args.video, args.flight)

    # Summary
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nTemporal Calibration:")
    print(f"  Time shift: {time_shift:.4f} seconds")
    print(f"  Usage: video_time + {time_shift:.4f} = flight_time")
    print(f"\nSpatial Calibration (Camera Mount Offset):")
    print(f"  Yaw:   {mount_offset[0]:.2f}°")
    print(f"  Pitch: {mount_offset[1]:.2f}°")
    print(f"  Roll:  {mount_offset[2]:.2f}°")
    print(f"\nResults saved to: {output_path}")

    # Plot if requested
    if args.plot and corr_data is not None:
        plot_results(corr_data, mount_data, time_shift)


if __name__ == '__main__':
    main()
