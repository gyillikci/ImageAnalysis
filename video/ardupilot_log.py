#!/usr/bin/env python3
"""
ArduPilot/PX4 Flight Log Adapter for ImageAnalysis

This module provides compatibility between ArduPilot/PX4 flight logs
(from Orange Cube, Pixhawk, etc.) and the ImageAnalysis calibration tools.

Supports:
- ArduPilot DataFlash binary logs (.bin)
- ArduPilot telemetry logs (.tlog)
- PX4 ULog files (.ulg)

Requirements:
    pip install pymavlink pyulog

Usage:
    from ardupilot_log import load, FlightInterp

    # Load flight data
    flight_data, flight_format = load('/path/to/flight.bin')

    # Create interpolation group
    interp = FlightInterp(flight_data)

    # Query interpolated IMU data
    p_rate = interp.group['imu'].interp['p'](timestamp)

Author: ImageAnalysis Project
License: MIT
"""

import math
import numpy as np
from scipy import interpolate
import os


class DataGroup:
    """Container for interpolated data channels within a group (imu, gps, filter)"""
    def __init__(self):
        self.interp = {}


class FlightInterp:
    """
    Interpolation group matching aurauas_flightdata interface.

    Provides:
        interp.group['imu'].interp['p']  - roll rate interpolator
        interp.group['imu'].interp['q']  - pitch rate interpolator
        interp.group['imu'].interp['r']  - yaw rate interpolator
        interp.group['filter'].interp['phi'] - roll angle
        interp.group['filter'].interp['the'] - pitch angle
        interp.group['filter'].interp['psi'] - yaw angle
        interp.group['filter'].interp['alt'] - altitude
        etc.
    """

    def __init__(self, flight_data):
        self.group = {}
        self._build_interpolators(flight_data)

    def _build_interpolators(self, flight_data):
        """Build scipy interpolators for all data channels"""

        # IMU interpolators
        if 'imu' in flight_data and len(flight_data['imu']) > 1:
            self.group['imu'] = DataGroup()
            imu_data = flight_data['imu']
            times = [r['time'] for r in imu_data]

            for key in imu_data[0].keys():
                if key != 'time':
                    values = [r.get(key, 0.0) for r in imu_data]
                    self.group['imu'].interp[key] = interpolate.interp1d(
                        times, values,
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

        # GPS interpolators
        if 'gps' in flight_data and len(flight_data['gps']) > 1:
            self.group['gps'] = DataGroup()
            gps_data = flight_data['gps']
            times = [r['time'] for r in gps_data]

            for key in gps_data[0].keys():
                if key != 'time':
                    values = [r.get(key, 0.0) for r in gps_data]
                    self.group['gps'].interp[key] = interpolate.interp1d(
                        times, values,
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

        # Filter/EKF interpolators
        if 'filter' in flight_data and len(flight_data['filter']) > 1:
            self.group['filter'] = DataGroup()
            filter_data = flight_data['filter']
            times = [r['time'] for r in filter_data]

            for key in filter_data[0].keys():
                if key != 'time':
                    values = [r.get(key, 0.0) for r in filter_data]
                    self.group['filter'].interp[key] = interpolate.interp1d(
                        times, values,
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

    def query(self, timestamp, group_name):
        """Query all channels in a group at a specific timestamp"""
        result = {'time': timestamp}
        if group_name in self.group:
            for key, interp_func in self.group[group_name].interp.items():
                result[key] = float(interp_func(timestamp))
        return result


def load_ardupilot_bin(filepath):
    """
    Load ArduPilot DataFlash binary log (.bin) file.

    Args:
        filepath: Path to .bin file

    Returns:
        tuple: (flight_data dict, 'ardupilot' format string)
    """
    try:
        from pymavlink import mavutil
    except ImportError:
        raise ImportError(
            "pymavlink is required for ArduPilot logs.\n"
            "Install with: pip install pymavlink"
        )

    flight_data = {
        'imu': [],
        'gps': [],
        'filter': [],
        'air': []
    }

    print(f"Loading ArduPilot log: {filepath}")
    mlog = mavutil.mavlink_connection(filepath)

    msg_count = {'IMU': 0, 'GPS': 0, 'ATT': 0, 'AHR2': 0, 'XKF1': 0}

    while True:
        try:
            msg = mlog.recv_match(blocking=False)
        except Exception as e:
            print(f"Warning: Error reading message: {e}")
            continue

        if msg is None:
            break

        msg_type = msg.get_type()

        # IMU data (gyro rates and accelerations)
        if msg_type == 'IMU':
            try:
                record = {
                    'time': msg.TimeUS / 1e6,
                    'p': msg.GyrX,      # roll rate (rad/s)
                    'q': msg.GyrY,      # pitch rate (rad/s)
                    'r': msg.GyrZ,      # yaw rate (rad/s)
                    'ax': msg.AccX,     # acceleration X (m/s^2)
                    'ay': msg.AccY,     # acceleration Y (m/s^2)
                    'az': msg.AccZ      # acceleration Z (m/s^2)
                }
                flight_data['imu'].append(record)
                msg_count['IMU'] += 1
            except AttributeError:
                pass

        # Alternative IMU message format (some firmware versions)
        elif msg_type == 'IMU2' or msg_type == 'IMU3':
            # Skip additional IMU instances for now, use primary
            pass

        # GPS data
        elif msg_type == 'GPS':
            try:
                record = {
                    'time': msg.TimeUS / 1e6,
                    'lat': msg.Lat / 1e7 if hasattr(msg, 'Lat') else msg.Lat,
                    'lon': msg.Lng / 1e7 if hasattr(msg, 'Lng') else msg.Lon,
                    'alt': msg.Alt,
                    'speed': getattr(msg, 'Spd', 0.0),
                    'course': getattr(msg, 'GCrs', 0.0),
                    'num_sats': getattr(msg, 'NSats', 0),
                    'hdop': getattr(msg, 'HDop', 0.0)
                }
                flight_data['gps'].append(record)
                msg_count['GPS'] += 1
            except AttributeError:
                pass

        # Attitude from EKF (primary attitude source)
        elif msg_type == 'ATT':
            try:
                roll_rad = math.radians(msg.Roll)
                pitch_rad = math.radians(msg.Pitch)
                yaw_rad = math.radians(msg.Yaw)

                record = {
                    'time': msg.TimeUS / 1e6,
                    'phi': roll_rad,                    # roll (rad)
                    'the': pitch_rad,                   # pitch (rad)
                    'psi': yaw_rad,                     # yaw (rad)
                    'psix': math.cos(yaw_rad),          # yaw x component
                    'psiy': math.sin(yaw_rad),          # yaw y component
                    'roll_deg': msg.Roll,
                    'pitch_deg': msg.Pitch,
                    'yaw_deg': msg.Yaw
                }
                flight_data['filter'].append(record)
                msg_count['ATT'] += 1
            except AttributeError:
                pass

        # Alternative attitude from AHRS2
        elif msg_type == 'AHR2':
            # Use as backup if ATT not available
            if len(flight_data['filter']) == 0 or msg_count['ATT'] == 0:
                try:
                    roll_rad = msg.Roll
                    pitch_rad = msg.Pitch
                    yaw_rad = msg.Yaw

                    record = {
                        'time': msg.TimeUS / 1e6,
                        'phi': roll_rad,
                        'the': pitch_rad,
                        'psi': yaw_rad,
                        'psix': math.cos(yaw_rad),
                        'psiy': math.sin(yaw_rad),
                        'alt': getattr(msg, 'Alt', 0.0)
                    }
                    flight_data['filter'].append(record)
                    msg_count['AHR2'] += 1
                except AttributeError:
                    pass

        # EKF state (XKF1) - contains altitude and velocities
        elif msg_type == 'XKF1':
            try:
                # Try to add altitude to existing filter records
                time = msg.TimeUS / 1e6
                alt = getattr(msg, 'PD', 0.0)  # Position Down (negative of altitude)

                # Find closest filter record and add altitude
                for rec in reversed(flight_data['filter']):
                    if abs(rec['time'] - time) < 0.1:
                        rec['alt'] = -alt  # Convert down to altitude
                        break
                msg_count['XKF1'] += 1
            except AttributeError:
                pass

        # Barometer data for altitude
        elif msg_type == 'BARO':
            try:
                time = msg.TimeUS / 1e6
                alt = getattr(msg, 'Alt', 0.0)

                # Add to airdata
                record = {
                    'time': time,
                    'alt': alt,
                    'press': getattr(msg, 'Press', 0.0),
                    'temp': getattr(msg, 'Temp', 0.0)
                }
                flight_data['air'].append(record)

                # Also add altitude to filter records if missing
                for rec in reversed(flight_data['filter']):
                    if abs(rec['time'] - time) < 0.1 and 'alt' not in rec:
                        rec['alt'] = alt
                        break
            except AttributeError:
                pass

    # If filter records don't have altitude, try to interpolate from baro
    if flight_data['air'] and flight_data['filter']:
        air_times = [r['time'] for r in flight_data['air']]
        air_alts = [r['alt'] for r in flight_data['air']]
        if len(air_times) > 1:
            alt_interp = interpolate.interp1d(
                air_times, air_alts,
                bounds_error=False,
                fill_value='extrapolate'
            )
            for rec in flight_data['filter']:
                if 'alt' not in rec:
                    rec['alt'] = float(alt_interp(rec['time']))

    # Ensure all filter records have altitude
    for rec in flight_data['filter']:
        if 'alt' not in rec:
            rec['alt'] = 0.0

    print(f"Loaded: IMU={msg_count['IMU']}, GPS={msg_count['GPS']}, "
          f"ATT={msg_count['ATT']}, AHR2={msg_count['AHR2']}, XKF1={msg_count['XKF1']}")
    print(f"Records: imu={len(flight_data['imu'])}, gps={len(flight_data['gps'])}, "
          f"filter={len(flight_data['filter'])}, air={len(flight_data['air'])}")

    return flight_data, 'ardupilot'


def load_ardupilot_tlog(filepath):
    """
    Load ArduPilot telemetry log (.tlog) file.

    Args:
        filepath: Path to .tlog file

    Returns:
        tuple: (flight_data dict, 'ardupilot_tlog' format string)
    """
    try:
        from pymavlink import mavutil
    except ImportError:
        raise ImportError(
            "pymavlink is required for ArduPilot logs.\n"
            "Install with: pip install pymavlink"
        )

    flight_data = {
        'imu': [],
        'gps': [],
        'filter': [],
        'air': []
    }

    print(f"Loading ArduPilot telemetry log: {filepath}")
    mlog = mavutil.mavlink_connection(filepath)

    msg_count = {'RAW_IMU': 0, 'SCALED_IMU': 0, 'GPS': 0, 'ATTITUDE': 0, 'GLOBAL_POSITION': 0}

    while True:
        try:
            msg = mlog.recv_match(blocking=False)
        except Exception as e:
            print(f"Warning: Error reading message: {e}")
            continue

        if msg is None:
            break

        msg_type = msg.get_type()
        timestamp = getattr(msg, '_timestamp', 0)

        # Raw IMU data
        if msg_type == 'RAW_IMU':
            try:
                record = {
                    'time': msg.time_usec / 1e6 if hasattr(msg, 'time_usec') else timestamp,
                    'p': msg.xgyro / 1000.0,    # mrad/s to rad/s
                    'q': msg.ygyro / 1000.0,
                    'r': msg.zgyro / 1000.0,
                    'ax': msg.xacc / 1000.0 * 9.81,  # mG to m/s^2
                    'ay': msg.yacc / 1000.0 * 9.81,
                    'az': msg.zacc / 1000.0 * 9.81
                }
                flight_data['imu'].append(record)
                msg_count['RAW_IMU'] += 1
            except AttributeError:
                pass

        # Scaled IMU (already in SI units)
        elif msg_type == 'SCALED_IMU' or msg_type == 'SCALED_IMU2':
            try:
                record = {
                    'time': msg.time_boot_ms / 1000.0 if hasattr(msg, 'time_boot_ms') else timestamp,
                    'p': msg.xgyro / 1000.0,    # mrad/s to rad/s
                    'q': msg.ygyro / 1000.0,
                    'r': msg.zgyro / 1000.0,
                    'ax': msg.xacc / 1000.0,    # mg to g, then to m/s^2
                    'ay': msg.yacc / 1000.0,
                    'az': msg.zacc / 1000.0
                }
                flight_data['imu'].append(record)
                msg_count['SCALED_IMU'] += 1
            except AttributeError:
                pass

        # GPS raw
        elif msg_type == 'GPS_RAW_INT':
            try:
                record = {
                    'time': msg.time_usec / 1e6 if hasattr(msg, 'time_usec') else timestamp,
                    'lat': msg.lat / 1e7,
                    'lon': msg.lon / 1e7,
                    'alt': msg.alt / 1000.0,    # mm to m
                    'speed': getattr(msg, 'vel', 0) / 100.0,  # cm/s to m/s
                    'num_sats': getattr(msg, 'satellites_visible', 0)
                }
                flight_data['gps'].append(record)
                msg_count['GPS'] += 1
            except AttributeError:
                pass

        # Attitude
        elif msg_type == 'ATTITUDE':
            try:
                record = {
                    'time': msg.time_boot_ms / 1000.0 if hasattr(msg, 'time_boot_ms') else timestamp,
                    'phi': msg.roll,
                    'the': msg.pitch,
                    'psi': msg.yaw,
                    'psix': math.cos(msg.yaw),
                    'psiy': math.sin(msg.yaw),
                    'p': msg.rollspeed,
                    'q': msg.pitchspeed,
                    'r': msg.yawspeed
                }
                flight_data['filter'].append(record)
                msg_count['ATTITUDE'] += 1
            except AttributeError:
                pass

        # Global position (has altitude)
        elif msg_type == 'GLOBAL_POSITION_INT':
            try:
                time = msg.time_boot_ms / 1000.0 if hasattr(msg, 'time_boot_ms') else timestamp
                alt = msg.relative_alt / 1000.0  # mm to m

                # Add altitude to closest filter record
                for rec in reversed(flight_data['filter']):
                    if abs(rec['time'] - time) < 0.5:
                        rec['alt'] = alt
                        break
                msg_count['GLOBAL_POSITION'] += 1
            except AttributeError:
                pass

    # Fill in missing altitudes
    for rec in flight_data['filter']:
        if 'alt' not in rec:
            rec['alt'] = 0.0

    print(f"Loaded: RAW_IMU={msg_count['RAW_IMU']}, SCALED_IMU={msg_count['SCALED_IMU']}, "
          f"GPS={msg_count['GPS']}, ATTITUDE={msg_count['ATTITUDE']}")

    return flight_data, 'ardupilot_tlog'


def load_px4_ulog(filepath):
    """
    Load PX4 ULog file (.ulg).

    Args:
        filepath: Path to .ulg file

    Returns:
        tuple: (flight_data dict, 'px4' format string)
    """
    try:
        from pyulog import ULog
    except ImportError:
        raise ImportError(
            "pyulog is required for PX4 logs.\n"
            "Install with: pip install pyulog"
        )

    flight_data = {
        'imu': [],
        'gps': [],
        'filter': [],
        'air': []
    }

    print(f"Loading PX4 ULog: {filepath}")
    ulog = ULog(filepath)

    # Find sensor_combined (IMU data)
    for d in ulog.data_list:
        if d.name == 'sensor_combined':
            times = d.data['timestamp'] / 1e6  # us to s
            for i in range(len(times)):
                record = {
                    'time': times[i],
                    'p': d.data['gyro_rad[0]'][i],
                    'q': d.data['gyro_rad[1]'][i],
                    'r': d.data['gyro_rad[2]'][i],
                    'ax': d.data['accelerometer_m_s2[0]'][i],
                    'ay': d.data['accelerometer_m_s2[1]'][i],
                    'az': d.data['accelerometer_m_s2[2]'][i]
                }
                flight_data['imu'].append(record)

        # Vehicle attitude
        elif d.name == 'vehicle_attitude':
            times = d.data['timestamp'] / 1e6
            for i in range(len(times)):
                # Quaternion to Euler
                q0 = d.data['q[0]'][i]
                q1 = d.data['q[1]'][i]
                q2 = d.data['q[2]'][i]
                q3 = d.data['q[3]'][i]

                # Convert quaternion to Euler (ZYX convention)
                phi = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
                the = math.asin(max(-1, min(1, 2*(q0*q2 - q3*q1))))
                psi = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

                record = {
                    'time': times[i],
                    'phi': phi,
                    'the': the,
                    'psi': psi,
                    'psix': math.cos(psi),
                    'psiy': math.sin(psi)
                }
                flight_data['filter'].append(record)

        # GPS position
        elif d.name == 'vehicle_gps_position':
            times = d.data['timestamp'] / 1e6
            for i in range(len(times)):
                record = {
                    'time': times[i],
                    'lat': d.data['lat'][i] / 1e7,
                    'lon': d.data['lon'][i] / 1e7,
                    'alt': d.data['alt'][i] / 1000.0
                }
                flight_data['gps'].append(record)

        # Local position (for altitude)
        elif d.name == 'vehicle_local_position':
            times = d.data['timestamp'] / 1e6
            for i in range(len(times)):
                time = times[i]
                alt = -d.data['z'][i]  # NED z is down

                # Add altitude to filter records
                for rec in flight_data['filter']:
                    if abs(rec['time'] - time) < 0.1 and 'alt' not in rec:
                        rec['alt'] = alt
                        break

    # Ensure altitude in filter records
    for rec in flight_data['filter']:
        if 'alt' not in rec:
            rec['alt'] = 0.0

    print(f"Loaded: imu={len(flight_data['imu'])}, gps={len(flight_data['gps'])}, "
          f"filter={len(flight_data['filter'])}")

    return flight_data, 'px4'


def load(filepath, recal_file=None):
    """
    Auto-detect and load flight log file.

    Supports:
        - ArduPilot .bin (DataFlash binary)
        - ArduPilot .tlog (telemetry log)
        - PX4 .ulg (ULog)

    Args:
        filepath: Path to flight log file
        recal_file: Optional recalibration file (unused, for compatibility)

    Returns:
        tuple: (flight_data dict, format string)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Flight log not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.bin':
        return load_ardupilot_bin(filepath)
    elif ext == '.tlog':
        return load_ardupilot_tlog(filepath)
    elif ext == '.ulg':
        return load_px4_ulog(filepath)
    else:
        # Try to auto-detect by reading magic bytes
        with open(filepath, 'rb') as f:
            header = f.read(16)

        if b'\xa3\x95' in header:  # ArduPilot bin magic
            return load_ardupilot_bin(filepath)
        elif b'ULog' in header:  # PX4 ULog magic
            return load_px4_ulog(filepath)
        else:
            # Default to trying bin format
            print(f"Warning: Unknown format for {filepath}, trying ArduPilot bin...")
            return load_ardupilot_bin(filepath)


# Compatibility alias
flight_loader = type('flight_loader', (), {'load': staticmethod(load)})()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ardupilot_log.py <flight_log_file>")
        print("\nSupported formats:")
        print("  .bin  - ArduPilot DataFlash binary log")
        print("  .tlog - ArduPilot telemetry log")
        print("  .ulg  - PX4 ULog")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        flight_data, flight_format = load(filepath)

        print(f"\nFlight format: {flight_format}")
        print(f"\nData summary:")
        print(f"  IMU records: {len(flight_data['imu'])}")
        print(f"  GPS records: {len(flight_data['gps'])}")
        print(f"  Filter records: {len(flight_data['filter'])}")

        if flight_data['imu']:
            imu_start = flight_data['imu'][0]['time']
            imu_end = flight_data['imu'][-1]['time']
            print(f"\nIMU time range: {imu_start:.2f} - {imu_end:.2f} s ({imu_end-imu_start:.2f} s)")

        if flight_data['filter']:
            filt_start = flight_data['filter'][0]['time']
            filt_end = flight_data['filter'][-1]['time']
            print(f"Filter time range: {filt_start:.2f} - {filt_end:.2f} s ({filt_end-filt_start:.2f} s)")

        # Test interpolation
        print("\nTesting interpolation...")
        interp = FlightInterp(flight_data)

        if 'imu' in interp.group:
            mid_time = (imu_start + imu_end) / 2
            p = interp.group['imu'].interp['p'](mid_time)
            q = interp.group['imu'].interp['q'](mid_time)
            r = interp.group['imu'].interp['r'](mid_time)
            print(f"  IMU at t={mid_time:.2f}: p={p:.4f}, q={q:.4f}, r={r:.4f} rad/s")

        if 'filter' in interp.group:
            mid_time = (filt_start + filt_end) / 2
            phi = interp.group['filter'].interp['phi'](mid_time)
            the = interp.group['filter'].interp['the'](mid_time)
            alt = interp.group['filter'].interp['alt'](mid_time)
            print(f"  Filter at t={mid_time:.2f}: phi={math.degrees(phi):.2f}°, "
                  f"the={math.degrees(the):.2f}°, alt={alt:.2f}m")

        print("\nAdapter test successful!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
