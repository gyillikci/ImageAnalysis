#!/usr/bin/env python3
"""
Debug calibration data flow - checks what data reaches the temporal synchronizer.
"""

import argparse
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import IMUData, MotionEstimate, FrameData
from core.utils import monotonic_time
from mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from camera.harrier import HarrierCamera, CameraConfig
from camera.motion_estimator import MotionEstimator, MotionEstimatorConfig
from sync.temporal import TemporalSynchronizer, TemporalSyncConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mavlink', required=True)
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--camera', default='0')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Calibration Data Flow Debug")
    print("=" * 60)
    
    # Counters for callbacks
    imu_to_sync_count = [0]
    motion_to_sync_count = [0]
    
    # --- MAVLink ---
    mavlink_config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )
    mavlink = MAVLinkReceiver(mavlink_config)
    if not mavlink.start():
        print("Failed to start MAVLink")
        return
    
    # --- Camera ---
    camera_id = int(args.camera) if args.camera.isdigit() else args.camera
    camera_config = CameraConfig(device=camera_id)
    camera = HarrierCamera(camera_config)
    if not camera.start():
        print("Failed to start camera")
        return
    print(f"Camera opened: {camera_config.width}x{camera_config.height}")
    
    # --- Motion Estimator ---
    motion_config = MotionEstimatorConfig(
        fx=800.0, fy=800.0,
        cx=camera_config.width / 2,
        cy=camera_config.height / 2
    )
    motion_estimator = MotionEstimator(motion_config)
    
    # --- Temporal Synchronizer ---
    temporal_config = TemporalSyncConfig(
        sample_rate=50.0,
        min_window_duration=3.0,  # Lower for debug
        min_correlation=0.3,  # Lower threshold
        use_axis='r'  # Use yaw since test_sync showed good yaw correlation
    )
    temporal_sync = TemporalSynchronizer(temporal_config)
    
    # --- Hook IMU callback ---
    def imu_to_sync_callback(imu: IMUData):
        imu_to_sync_count[0] += 1
        temporal_sync.on_imu_data(imu)
    
    mavlink.add_callback('imu', imu_to_sync_callback)
    
    # Start temporal sync
    temporal_sync.start()
    
    print("\nCollecting data - ROTATE THE DEVICE!")
    print("=" * 60)
    
    start_time = time.time()
    last_print = start_time
    motion_count = 0
    valid_motion_count = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Get frame
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            # Estimate motion
            motion = motion_estimator.process_frame(frame)
            motion_count += 1
            
            if motion.valid:
                valid_motion_count += 1
                # Feed to temporal sync
                temporal_sync.on_motion_data(motion)
                motion_to_sync_count[0] += 1
            
            # Print status every second
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                
                # Check temporal sync buffers
                imu_in_buffer = len(temporal_sync._imu_buffer.get_all())
                motion_in_buffer = len(temporal_sync._motion_buffer.get_all())
                
                ts_status = temporal_sync.get_status()
                
                print(f"\nt={elapsed:.1f}s")
                print(f"  IMU callbacks fired: {imu_to_sync_count[0]}")
                print(f"  Motion valid/total:  {valid_motion_count}/{motion_count}")
                print(f"  Motion to sync:      {motion_to_sync_count[0]}")
                print(f"  Sync IMU buffer:     {imu_in_buffer}")
                print(f"  Sync Motion buffer:  {motion_in_buffer}")
                print(f"  Sync offset:         {ts_status['time_offset']*1000:.1f}ms")
                print(f"  Sync correlation:    {ts_status['correlation']:.3f}")
                print(f"  Sync converged:      {ts_status['converged']}")
                
                # Sample last motion if available
                if motion.valid:
                    print(f"  Last motion: p={motion.p:.3f}, q={motion.q:.3f}, r={motion.r:.3f}")
                
                last_print = now
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    # Final analysis
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)
    
    imu_data = temporal_sync._imu_buffer.get_all()
    motion_data = temporal_sync._motion_buffer.get_all()
    
    print(f"\nIMU samples in sync buffer:    {len(imu_data)}")
    print(f"Motion samples in sync buffer: {len(motion_data)}")
    
    if imu_data:
        imu_duration = imu_data[-1].timestamp - imu_data[0].timestamp
        imu_r_values = [d.r for d in imu_data]
        print(f"IMU duration: {imu_duration:.1f}s")
        print(f"IMU r std:    {np.std(imu_r_values):.4f} rad/s")
    
    if motion_data:
        motion_duration = motion_data[-1].timestamp - motion_data[0].timestamp
        motion_r_values = [d.r for d in motion_data]
        print(f"Motion duration: {motion_duration:.1f}s")
        print(f"Motion r std:    {np.std(motion_r_values):.4f} rad/s")
    
    # Try manual correlation
    if len(imu_data) > 100 and len(motion_data) > 30:
        print("\n--- Manual Correlation Check ---")
        
        # Downsample both to common rate
        from scipy import interpolate
        
        imu_times = np.array([d.timestamp for d in imu_data])
        imu_r = np.array([d.r for d in imu_data])
        motion_times = np.array([d.timestamp for d in motion_data])
        motion_r = np.array([d.r for d in motion_data])
        
        # Common time range
        t_start = max(imu_times[0], motion_times[0])
        t_end = min(imu_times[-1], motion_times[-1])
        overlap = t_end - t_start
        print(f"Time overlap: {overlap:.2f}s")
        
        if overlap > 2.0:
            # Resample
            common_times = np.linspace(t_start, t_end, int(overlap * 50))
            imu_interp = interpolate.interp1d(imu_times, imu_r, bounds_error=False, fill_value=0)
            motion_interp = interpolate.interp1d(motion_times, motion_r, bounds_error=False, fill_value=0)
            
            imu_resampled = imu_interp(common_times)
            motion_resampled = motion_interp(common_times)
            
            # Cross-correlation
            imu_norm = imu_resampled - np.mean(imu_resampled)
            motion_norm = motion_resampled - np.mean(motion_resampled)
            
            if np.std(imu_norm) > 0.01 and np.std(motion_norm) > 0.01:
                corr = np.correlate(imu_norm, motion_norm, mode='full')
                corr = corr / (np.std(imu_norm) * np.std(motion_norm) * len(imu_norm))
                
                max_corr = np.max(np.abs(corr))
                max_lag = np.argmax(np.abs(corr)) - len(imu_norm) + 1
                lag_seconds = max_lag / 50.0
                
                print(f"Max correlation: {max_corr:.3f}")
                print(f"Lag: {max_lag} samples ({lag_seconds*1000:.0f} ms)")
                
                if max_corr > 0.3:
                    print("✓ Good correlation detected - sync should work!")
                else:
                    print("✗ Low correlation - need more rotation motion")
            else:
                print("✗ Not enough variance in data")
    
    # Cleanup
    temporal_sync.stop()
    mavlink.stop()
    camera.stop()


if __name__ == '__main__':
    main()
