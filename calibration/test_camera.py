#!/usr/bin/env python3
"""
Simple Camera test - validates camera capture and motion estimation.
"""
import sys
import time
sys.path.insert(0, '.')

import cv2
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Camera and Motion Estimation')
    parser.add_argument('--camera', default='0', help='Camera device')
    parser.add_argument('--duration', type=float, default=10.0, help='Test duration in seconds')
    parser.add_argument('--no-show', action='store_true', help='Disable video preview')
    args = parser.parse_args()

    print("=" * 60)
    print("Camera Motion Test (with Visualization)")
    print("=" * 60)
    print("Press 'q' to quit early")

    camera_id = int(args.camera) if args.camera.isdigit() else args.camera

    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("✗ Failed to open camera!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} @ {fps} fps")

    # Simple optical flow motion estimation
    prev_gray = None
    prev_pts = None
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    frame_count = 0
    motion_samples = []
    current_omega = (0, 0)

    print(f"\nCapturing for {args.duration} seconds...")
    print("Rotate the camera to see angular velocity estimates.\n")

    start_time = time.time()
    while time.time() - start_time < args.duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        good_old = None
        good_new = None

        if prev_gray is not None and prev_pts is not None and len(prev_pts) > 10:
            # Track features
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

            if curr_pts is not None:
                good_old = prev_pts[status.flatten() == 1]
                good_new = curr_pts[status.flatten() == 1]

                if len(good_old) > 10:
                    # Compute average motion
                    flow = good_new - good_old
                    mean_flow = np.mean(flow, axis=0).flatten()

                    # Simple angular velocity estimate (pixels/frame -> approximate rad/s)
                    fov_rad = np.pi / 2
                    px_per_rad = width / fov_rad
                    dt = 1.0 / 50.0
                    
                    if len(mean_flow) >= 2:
                        omega_y = -mean_flow[0] / px_per_rad / dt
                        omega_x = mean_flow[1] / px_per_rad / dt
                        current_omega = (omega_x, omega_y)
                        motion_samples.append((omega_x, omega_y))

                        if frame_count % 25 == 0:
                            print(f"Frame {frame_count:4d} | Features: {len(good_old):3d} | "
                                  f"ω: [{omega_x:7.3f}, {omega_y:7.3f}] rad/s")

                    # Draw flow vectors
                    for old, new in zip(good_old, good_new):
                        ox, oy = old.ravel().astype(int)
                        nx, ny = new.ravel().astype(int)
                        cv2.line(display_frame, (ox, oy), (nx, ny), (0, 255, 0), 2)
                        cv2.circle(display_frame, (nx, ny), 3, (255, 0, 0), -1)

        # Find new features
        new_pts = cv2.goodFeaturesToTrack(gray, **feature_params)
        if new_pts is not None:
            for pt in new_pts:
                x, y = pt.ravel().astype(int)
                cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
        
        prev_pts = new_pts
        prev_gray = gray

        # Draw info overlay
        elapsed = time.time() - start_time
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
        
        info = [f"Time: {elapsed:.1f}s", f"Frame: {frame_count}",
                f"Features: {len(new_pts) if new_pts is not None else 0}",
                f"omega_x: {current_omega[0]:+.3f} rad/s",
                f"omega_y: {current_omega[1]:+.3f} rad/s"]
        for i, text in enumerate(info):
            cv2.putText(display_frame, text, (20, 35 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw motion arrow at center
        cx, cy = width // 2, height // 2
        ax = int(cx + current_omega[1] * 100)
        ay = int(cy + current_omega[0] * 100)
        cv2.arrowedLine(display_frame, (cx, cy), (ax, ay), (0, 255, 255), 3)
        cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)

        if not args.no_show:
            cv2.imshow('Camera Motion Test - Press Q to quit', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    actual_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"RESULT: Captured {frame_count} frames in {actual_duration:.1f}s")
    print(f"        Rate: {frame_count / actual_duration:.1f} fps")
    print(f"        Motion samples: {len(motion_samples)}")

    if len(motion_samples) > 10:
        omega = np.array(motion_samples)
        print(f"        ω range: [{omega.min():.3f}, {omega.max():.3f}] rad/s")
        print("        ✓ Camera motion estimation working!")
    else:
        print("        ✗ Not enough motion samples - check camera")
    print("=" * 60)

if __name__ == '__main__':
    main()
