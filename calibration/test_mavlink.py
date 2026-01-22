#!/usr/bin/env python3
"""
Simple MAVLink IMU test - validates IMU data reception.
"""
import sys
import time
sys.path.insert(0, '.')

from mavlink.receiver import MAVLinkConfig, MAVLinkReceiver

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test MAVLink IMU reception')
    parser.add_argument('--mavlink', default='COM6', help='MAVLink connection string')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate')
    parser.add_argument('--duration', type=float, default=5.0, help='Test duration in seconds')
    args = parser.parse_args()

    print("=" * 60)
    print("MAVLink IMU Test")
    print("=" * 60)

    config = MAVLinkConfig(
        connection_string=args.mavlink,
        baudrate=args.baudrate
    )

    receiver = MAVLinkReceiver(config)

    # Track received data
    imu_count = 0
    last_imu = None

    def on_imu(data):
        nonlocal imu_count, last_imu
        imu_count += 1
        last_imu = data

    receiver.add_callback('imu', on_imu)

    print(f"\nConnecting to {args.mavlink}...")
    receiver.start()

    print(f"Collecting IMU data for {args.duration} seconds...")
    print("Move the IMU to see gyro values change.\n")

    start_time = time.time()
    while time.time() - start_time < args.duration:
        time.sleep(0.5)
        if last_imu:
            print(f"IMU count: {imu_count:5d} | "
                  f"gyro: [{last_imu.p:7.3f}, {last_imu.q:7.3f}, {last_imu.r:7.3f}] rad/s | "
                  f"accel: [{last_imu.ax:7.2f}, {last_imu.ay:7.2f}, {last_imu.az:7.2f}] m/s²")
        else:
            print("Waiting for IMU data...")

    receiver.stop()

    print("\n" + "=" * 60)
    print(f"RESULT: Received {imu_count} IMU samples in {args.duration}s")
    print(f"        Rate: {imu_count / args.duration:.1f} Hz")
    if imu_count > 0:
        print("        ✓ MAVLink IMU working!")
    else:
        print("        ✗ No IMU data received - check connection")
    print("=" * 60)

if __name__ == '__main__':
    main()
