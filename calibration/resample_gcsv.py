#!/usr/bin/env python3
"""
Resample GCSV IMU data to uniform timestamps.

ArduPilot's MAVLink scheduler sends IMU data in bursts, causing non-uniform
timing. This tool resamples the data to uniform intervals for use with Gyroflow.

Usage:
    python resample_gcsv.py input.gcsv output.gcsv --rate 200
"""

import argparse
import numpy as np
from scipy import interpolate
import os


def parse_gcsv(filepath: str):
    """Parse GCSV file and return header lines and data."""
    header_lines = []
    timestamps = []
    gyro = []
    accel = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a data line (starts with a digit)
            if line[0].isdigit():
                parts = line.split(',')
                if len(parts) >= 7:
                    timestamps.append(int(parts[0]))
                    gyro.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    accel.append([float(parts[4]), float(parts[5]), float(parts[6])])
            else:
                header_lines.append(line)
    
    return header_lines, np.array(timestamps), np.array(gyro), np.array(accel)


def resample_data(timestamps, gyro, accel, target_rate_hz: float):
    """Resample IMU data to uniform timestamps at target rate."""
    
    # Convert timestamps to seconds for interpolation
    t_sec = timestamps / 1e6
    
    # Create uniform time grid
    t_start = t_sec[0]
    t_end = t_sec[-1]
    dt = 1.0 / target_rate_hz
    t_uniform = np.arange(t_start, t_end, dt)
    
    # Interpolate each channel
    gyro_resampled = np.zeros((len(t_uniform), 3))
    accel_resampled = np.zeros((len(t_uniform), 3))
    
    for i in range(3):
        # Use cubic interpolation for smooth results
        f_gyro = interpolate.interp1d(t_sec, gyro[:, i], kind='cubic', 
                                       fill_value='extrapolate')
        f_accel = interpolate.interp1d(t_sec, accel[:, i], kind='cubic',
                                        fill_value='extrapolate')
        
        gyro_resampled[:, i] = f_gyro(t_uniform)
        accel_resampled[:, i] = f_accel(t_uniform)
    
    # Convert back to microseconds
    timestamps_uniform = (t_uniform * 1e6).astype(np.int64)
    
    return timestamps_uniform, gyro_resampled, accel_resampled


def write_gcsv(filepath: str, header_lines: list, timestamps, gyro, accel):
    """Write resampled data to GCSV file."""
    
    with open(filepath, 'w') as f:
        # Write header
        for line in header_lines:
            f.write(line + '\n')
        
        # Write data
        for i in range(len(timestamps)):
            t = timestamps[i]
            gx, gy, gz = gyro[i]
            ax, ay, az = accel[i]
            f.write(f"{t},{gx:.0f},{gy:.0f},{gz:.0f},{ax:.0f},{ay:.0f},{az:.0f}\n")


def analyze_timing(timestamps, label=""):
    """Analyze and print timing statistics."""
    intervals = np.diff(timestamps)
    
    print(f"\n{label} Timing Analysis:")
    print(f"  Samples: {len(timestamps)}")
    print(f"  Duration: {timestamps[-1]/1e6:.2f} s")
    print(f"  Rate: {len(timestamps) / (timestamps[-1]/1e6):.1f} Hz")
    print(f"  Interval min: {intervals.min()} µs")
    print(f"  Interval max: {intervals.max()} µs")
    print(f"  Interval mean: {intervals.mean():.1f} µs")
    print(f"  Interval std: {intervals.std():.1f} µs")
    print(f"  Gaps > 10ms: {np.sum(intervals > 10000)}")
    print(f"  Gaps > 50ms: {np.sum(intervals > 50000)}")


def main():
    parser = argparse.ArgumentParser(
        description='Resample GCSV IMU data to uniform timestamps.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python resample_gcsv.py recording.gcsv recording_uniform.gcsv --rate 200
    python resample_gcsv.py recording.gcsv recording_uniform.gcsv --rate 400
        """
    )
    
    parser.add_argument('input', help='Input GCSV file')
    parser.add_argument('output', help='Output GCSV file')
    parser.add_argument('--rate', type=float, default=200,
                        help='Target sample rate in Hz (default: 200)')
    parser.add_argument('--analyze', action='store_true',
                        help='Show timing analysis before and after')
    
    args = parser.parse_args()
    
    print(f"Reading: {args.input}")
    header_lines, timestamps, gyro, accel = parse_gcsv(args.input)
    
    if args.analyze:
        analyze_timing(timestamps, "Original")
    
    print(f"\nResampling to {args.rate} Hz...")
    timestamps_uniform, gyro_uniform, accel_uniform = resample_data(
        timestamps, gyro, accel, args.rate
    )
    
    if args.analyze:
        analyze_timing(timestamps_uniform, "Resampled")
    
    print(f"\nWriting: {args.output}")
    write_gcsv(args.output, header_lines, timestamps_uniform, gyro_uniform, accel_uniform)
    
    print(f"\nDone! Resampled {len(timestamps)} -> {len(timestamps_uniform)} samples")


if __name__ == '__main__':
    main()
