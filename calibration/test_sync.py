#!/usr/bin/env python3
"""
Simple sync test - validates that IMU and camera motion data can be correlated.
"""
import sys
import time
import threading
sys.path.insert(0, '.')

import cv2
import numpy as np
from collections import deque

from mavlink.receiver import MAVLinkConfig, MAVLinkReceiver

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Camera-IMU Correlation')
    parser.add_argument('--mavlink', default='COM6', help='MAVLink connection string')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate')
    parser.add_argument('--camera', default='0', help='Camera device')
    parser.add_argument('--duration', type=float, default=10.0, help='Test duration in seconds')
    parser.add_argument('--show', action='store_true', help='Show video preview')
    args = parser.parse_args()

    print("=" * 60)
    print("Camera-IMU Correlation Test")
    print("=" * 60)
    print("\nThis test checks if camera motion correlates with IMU gyro.")
    print("ROTATE the device back and forth during the test!\n")

    # IMU setup
    config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )
    receiver = MAVLinkReceiver(config)

    # Buffers for data
    imu_gyro_z = deque(maxlen=1000)  # Store gyro Z (yaw rate)
    cam_omega_z = deque(maxlen=1000)  # Store camera yaw rate
    imu_times = deque(maxlen=1000)
    cam_times = deque(maxlen=1000)

    def on_imu(data):
        imu_times.append(time.time())
        imu_gyro_z.append(data.r)  # r = yaw rate (rad/s)

    receiver.add_callback('imu', on_imu)

    # Camera setup
    camera_id = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("✗ Failed to open camera!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera opened: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print(f"\nConnecting to {args.mavlink}...")
    receiver.start()

    # Optical flow setup
    prev_gray = None
    prev_pts = None
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    fov_rad = np.pi / 2
    px_per_rad = width / fov_rad
    dt = 1.0 / 50.0

    print(f"\nCollecting data for {args.duration} seconds...")
    print(">>> ROTATE THE DEVICE BACK AND FORTH! <<<\n")

    start_time = time.time()
    while time.time() - start_time < args.duration:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and prev_pts is not None and len(prev_pts) > 10:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

            if curr_pts is not None:
                good_old = prev_pts[status.flatten() == 1]
                good_new = curr_pts[status.flatten() == 1]

                if len(good_old) > 10:
                    flow = good_new - good_old
                    mean_flow = np.mean(flow, axis=0).flatten()
                    if len(mean_flow) >= 1:
                        omega_z = float(-mean_flow[0] / px_per_rad / dt)  # yaw from horizontal motion

                        cam_times.append(time.time())
                        cam_omega_z.append(omega_z)

        prev_pts = cv2.goodFeaturesToTrack(gray, **feature_params)
        prev_gray = gray

        # Print status every 0.5s
        elapsed = time.time() - start_time
        if int(elapsed * 2) % 1 == 0 and len(imu_gyro_z) > 0 and len(cam_omega_z) > 0:
            print(f"t={elapsed:5.1f}s | IMU gyro_z: {float(imu_gyro_z[-1]):7.3f} | Cam omega_z: {float(cam_omega_z[-1]):7.3f}")

        if args.show:
            cv2.imshow('Sync Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    receiver.stop()
    if args.show:
        cv2.destroyAllWindows()

    # Compute correlation
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    imu_data = np.array(list(imu_gyro_z))
    cam_data = np.array(list(cam_omega_z))

    print(f"\nIMU samples:    {len(imu_data)}")
    print(f"Camera samples: {len(cam_data)}")

    if len(imu_data) > 100 and len(cam_data) > 100:
        # Resample to same length for correlation
        min_len = min(len(imu_data), len(cam_data))
        imu_resamp = np.interp(np.linspace(0, 1, min_len),
                               np.linspace(0, 1, len(imu_data)), imu_data)
        cam_resamp = np.interp(np.linspace(0, 1, min_len),
                               np.linspace(0, 1, len(cam_data)), cam_data)

        # Compute correlation
        imu_std = np.std(imu_resamp)
        cam_std = np.std(cam_resamp)

        print(f"\nIMU gyro_z std:  {imu_std:.4f} rad/s")
        print(f"Cam omega_z std: {cam_std:.4f} rad/s")

        if imu_std > 0.01 and cam_std > 0.01:
            # Normalized cross-correlation
            corr = np.correlate(
                (imu_resamp - np.mean(imu_resamp)) / imu_std,
                (cam_resamp - np.mean(cam_resamp)) / cam_std,
                mode='full'
            ) / min_len

            max_corr = np.max(np.abs(corr))
            lag = np.argmax(np.abs(corr)) - min_len + 1

            print(f"\nMax correlation: {max_corr:.3f}")
            print(f"Lag (samples):   {lag}")

            if max_corr > 0.3:
                print("\n✓ Good correlation detected!")
                print("  Camera and IMU motion are correlated.")
                print("  The calibration should be able to find the time offset.")
            elif max_corr > 0.1:
                print("\n~ Weak correlation detected")
                print("  Try rotating more vigorously around the yaw axis.")
            else:
                print("\n✗ No correlation detected")
                print("  Check that camera and IMU are rigidly attached.")
                print("  Try rotating the device more.")
        else:
            print("\n✗ Not enough motion detected!")
            print("  IMU or camera variance is too low.")
            print("  You need to ROTATE the device during the test.")
    else:
        print("\n✗ Not enough samples collected")

    print("=" * 60)

if __name__ == '__main__':
    main()
