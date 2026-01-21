#!/usr/bin/env python3
"""
Unified Flight Log Loader for ImageAnalysis

This module provides a unified interface for loading flight logs from various
autopilot systems. It serves as a drop-in replacement for aurauas_flightdata
when working with ArduPilot, PX4, or DJI flight controllers.

Supported formats:
    - ArduPilot DataFlash binary (.bin)
    - ArduPilot telemetry log (.tlog)
    - PX4 ULog (.ulg)
    - DJI verbose CSV (from phantomhelp.com)
    - DJI SRT subtitle files

Usage:
    # Replace this:
    from aurauas_flightdata import flight_loader, flight_interp

    # With this:
    from flight_loader_local import flight_loader, flight_interp

Author: ImageAnalysis Project
License: MIT
"""

import os


# Import the ArduPilot adapter
from ardupilot_log import (
    load as load_ardupilot,
    FlightInterp,
    DataGroup
)

# Try to import DJI log support
try:
    from djilog import djicsv, djisrt
    HAS_DJI = True
except ImportError:
    HAS_DJI = False


def load(filepath, recal_file=None):
    """
    Auto-detect and load flight log file.

    Args:
        filepath: Path to flight log file
        recal_file: Optional recalibration file (for compatibility)

    Returns:
        tuple: (flight_data dict, format string)

    Supported formats:
        .bin  - ArduPilot DataFlash binary
        .tlog - ArduPilot telemetry log
        .ulg  - PX4 ULog
        .csv  - DJI verbose CSV (from phantomhelp.com)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Flight log not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    # ArduPilot/PX4 formats
    if ext in ['.bin', '.tlog', '.ulg']:
        return load_ardupilot(filepath, recal_file)

    # DJI CSV format
    elif ext == '.csv' and HAS_DJI:
        return load_dji_csv(filepath)

    # Try ArduPilot loader as fallback
    else:
        try:
            return load_ardupilot(filepath, recal_file)
        except Exception as e:
            raise ValueError(
                f"Unknown flight log format: {ext}\n"
                f"Supported: .bin, .tlog, .ulg, .csv\n"
                f"Error: {e}"
            )


def load_dji_csv(filepath):
    """
    Load DJI verbose CSV log file.

    Args:
        filepath: Path to DJI CSV file (exported from phantomhelp.com)

    Returns:
        tuple: (flight_data dict, 'dji' format string)
    """
    import math

    dji = djicsv()
    dji.load(filepath)

    flight_data = {
        'imu': [],
        'gps': [],
        'filter': [],
        'air': []
    }

    # Convert DJI log to standard format
    last_time = None
    for record in dji.log:
        t = record['unix_sec']

        # Skip duplicates
        if last_time is not None and t == last_time:
            continue
        last_time = t

        # GPS data
        flight_data['gps'].append({
            'time': t,
            'lat': record['lat'],
            'lon': record['lon'],
            'alt': record['baro_alt'] * 0.3048  # ft to m
        })

        # Gimbal attitude as filter (note: this is gimbal, not aircraft)
        roll_rad = math.radians(record['roll'])
        pitch_rad = math.radians(record['pitch'])
        yaw_rad = math.radians(record['yaw'])

        flight_data['filter'].append({
            'time': t,
            'phi': roll_rad,
            'the': pitch_rad,
            'psi': yaw_rad,
            'psix': math.cos(yaw_rad),
            'psiy': math.sin(yaw_rad),
            'alt': record['baro_alt'] * 0.3048
        })

    print(f"Loaded DJI CSV: gps={len(flight_data['gps'])}, "
          f"filter={len(flight_data['filter'])}")

    return flight_data, 'dji'


class InterpolationGroup:
    """
    Wrapper class providing aurauas_flightdata-compatible interface.

    Usage:
        interp = InterpolationGroup(flight_data)
        p_rate = interp.group['imu'].interp['p'](timestamp)
    """

    def __init__(self, flight_data):
        self._flight_interp = FlightInterp(flight_data)
        self.group = self._flight_interp.group

    def query(self, timestamp, group_name):
        """Query all channels in a group at a specific timestamp"""
        return self._flight_interp.query(timestamp, group_name)


class IterateGroup:
    """
    Iterator for flight data records.

    Usage:
        iter_group = IterateGroup(flight_data)
        for record in iter_group.iterate('imu'):
            print(record['time'], record['p'])
    """

    def __init__(self, flight_data):
        self.flight_data = flight_data

    def iterate(self, group_name):
        """Iterate over records in a group"""
        if group_name in self.flight_data:
            for record in self.flight_data[group_name]:
                yield record


# Create module-level aliases for drop-in compatibility
class _FlightLoader:
    """Module-like object for flight_loader.load() syntax"""
    @staticmethod
    def load(filepath, recal_file=None):
        return load(filepath, recal_file)


class _FlightInterp:
    """Module-like object providing InterpolationGroup and IterateGroup"""
    InterpolationGroup = InterpolationGroup
    IterateGroup = IterateGroup


# Export as module-like objects
flight_loader = _FlightLoader()
flight_interp = _FlightInterp()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Unified Flight Log Loader")
        print("=" * 40)
        print("\nUsage: python flight_loader_local.py <flight_log_file>")
        print("\nSupported formats:")
        print("  .bin  - ArduPilot DataFlash binary log")
        print("  .tlog - ArduPilot telemetry log")
        print("  .ulg  - PX4 ULog")
        print("  .csv  - DJI verbose CSV (from phantomhelp.com)")
        print("\nExample:")
        print("  python flight_loader_local.py flight.bin")
        sys.exit(0)

    filepath = sys.argv[1]

    try:
        # Test the unified loader
        data, fmt = flight_loader.load(filepath)

        print(f"\nLoaded: {filepath}")
        print(f"Format: {fmt}")
        print(f"\nRecords:")
        print(f"  IMU: {len(data['imu'])}")
        print(f"  GPS: {len(data['gps'])}")
        print(f"  Filter: {len(data['filter'])}")

        # Test interpolation
        interp = flight_interp.InterpolationGroup(data)

        if data['imu']:
            t_start = data['imu'][0]['time']
            t_end = data['imu'][-1]['time']
            t_mid = (t_start + t_end) / 2

            print(f"\nIMU time range: {t_start:.2f} - {t_end:.2f} s")
            print(f"\nInterpolated values at t={t_mid:.2f}:")

            if 'imu' in interp.group:
                p = interp.group['imu'].interp['p'](t_mid)
                q = interp.group['imu'].interp['q'](t_mid)
                r = interp.group['imu'].interp['r'](t_mid)
                print(f"  IMU: p={p:.4f}, q={q:.4f}, r={r:.4f} rad/s")

            if 'filter' in interp.group:
                import math
                phi = interp.group['filter'].interp['phi'](t_mid)
                the = interp.group['filter'].interp['the'](t_mid)
                print(f"  Filter: phi={math.degrees(phi):.2f}°, the={math.degrees(the):.2f}°")

        print("\nTest successful!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
