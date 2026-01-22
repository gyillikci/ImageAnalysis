"""
MAVLink message parsing helpers.
"""

import math
from typing import Tuple, Optional

from core.data_types import IMUData, AttitudeData, GPSData
from core.utils import monotonic_time, quaternion_to_euler


def parse_attitude(msg, local_timestamp: Optional[float] = None) -> AttitudeData:
    """
    Parse ATTITUDE message to AttitudeData.

    Args:
        msg: MAVLink ATTITUDE message
        local_timestamp: Local monotonic timestamp (uses current time if None)

    Returns:
        AttitudeData object
    """
    if local_timestamp is None:
        local_timestamp = monotonic_time()

    return AttitudeData(
        timestamp=local_timestamp,
        fc_timestamp=msg.time_boot_ms * 1000,  # ms to us
        roll=msg.roll,
        pitch=msg.pitch,
        yaw=msg.yaw,
        rollspeed=msg.rollspeed,
        pitchspeed=msg.pitchspeed,
        yawspeed=msg.yawspeed
    )


def parse_attitude_quaternion(msg, local_timestamp: Optional[float] = None) -> AttitudeData:
    """
    Parse ATTITUDE_QUATERNION message to AttitudeData.

    Args:
        msg: MAVLink ATTITUDE_QUATERNION message
        local_timestamp: Local monotonic timestamp

    Returns:
        AttitudeData object
    """
    if local_timestamp is None:
        local_timestamp = monotonic_time()

    # Convert quaternion to Euler angles
    q = [msg.q1, msg.q2, msg.q3, msg.q4]
    roll, pitch, yaw = quaternion_to_euler(q)

    return AttitudeData(
        timestamp=local_timestamp,
        fc_timestamp=msg.time_boot_ms * 1000,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        rollspeed=msg.rollspeed,
        pitchspeed=msg.pitchspeed,
        yawspeed=msg.yawspeed
    )


def parse_imu(msg, local_timestamp: Optional[float] = None,
              msg_type: str = 'RAW_IMU') -> IMUData:
    """
    Parse IMU message to IMUData.

    Args:
        msg: MAVLink IMU message (RAW_IMU, SCALED_IMU, or HIGHRES_IMU)
        local_timestamp: Local monotonic timestamp
        msg_type: Type of IMU message

    Returns:
        IMUData object
    """
    if local_timestamp is None:
        local_timestamp = monotonic_time()

    if msg_type == 'RAW_IMU':
        return IMUData(
            timestamp=local_timestamp,
            fc_timestamp=msg.time_usec,
            p=msg.xgyro / 1000.0,  # mrad/s to rad/s
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0 * 9.81,  # mG to m/s²
            ay=msg.yacc / 1000.0 * 9.81,
            az=msg.zacc / 1000.0 * 9.81
        )

    elif msg_type in ['SCALED_IMU', 'SCALED_IMU2', 'SCALED_IMU3']:
        return IMUData(
            timestamp=local_timestamp,
            fc_timestamp=msg.time_boot_ms * 1000,
            p=msg.xgyro / 1000.0,
            q=msg.ygyro / 1000.0,
            r=msg.zgyro / 1000.0,
            ax=msg.xacc / 1000.0,  # Already in mg, convert to m/s²
            ay=msg.yacc / 1000.0,
            az=msg.zacc / 1000.0
        )

    elif msg_type == 'HIGHRES_IMU':
        return IMUData(
            timestamp=local_timestamp,
            fc_timestamp=msg.time_usec,
            p=msg.xgyro,  # Already rad/s
            q=msg.ygyro,
            r=msg.zgyro,
            ax=msg.xacc,  # Already m/s²
            ay=msg.yacc,
            az=msg.zacc
        )

    else:
        raise ValueError(f"Unknown IMU message type: {msg_type}")


def parse_gps(msg, local_timestamp: Optional[float] = None) -> GPSData:
    """
    Parse GPS_RAW_INT message to GPSData.

    Args:
        msg: MAVLink GPS_RAW_INT message
        local_timestamp: Local monotonic timestamp

    Returns:
        GPSData object
    """
    if local_timestamp is None:
        local_timestamp = monotonic_time()

    return GPSData(
        timestamp=local_timestamp,
        fc_timestamp=msg.time_usec,
        lat=msg.lat / 1e7,
        lon=msg.lon / 1e7,
        alt=msg.alt / 1000.0,  # mm to m
        vel=getattr(msg, 'vel', 0) / 100.0,  # cm/s to m/s
        hdg=getattr(msg, 'cog', 0) / 100.0,  # cdeg to deg
        satellites=getattr(msg, 'satellites_visible', 0),
        fix_type=msg.fix_type
    )


def parse_global_position(msg, local_timestamp: Optional[float] = None) -> GPSData:
    """
    Parse GLOBAL_POSITION_INT message to GPSData.

    This provides fused GPS position from the EKF.

    Args:
        msg: MAVLink GLOBAL_POSITION_INT message
        local_timestamp: Local monotonic timestamp

    Returns:
        GPSData object
    """
    if local_timestamp is None:
        local_timestamp = monotonic_time()

    return GPSData(
        timestamp=local_timestamp,
        fc_timestamp=msg.time_boot_ms * 1000,
        lat=msg.lat / 1e7,
        lon=msg.lon / 1e7,
        alt=msg.relative_alt / 1000.0,  # mm to m (relative altitude)
        vel=math.sqrt(msg.vx**2 + msg.vy**2) / 100.0,  # cm/s to m/s
        hdg=msg.hdg / 100.0 if msg.hdg != 65535 else 0.0,  # cdeg to deg
        satellites=0,  # Not available in this message
        fix_type=3  # Assume 3D fix if we're getting this message
    )


# Message type to parser mapping
MESSAGE_PARSERS = {
    'ATTITUDE': parse_attitude,
    'ATTITUDE_QUATERNION': parse_attitude_quaternion,
    'RAW_IMU': lambda msg, ts: parse_imu(msg, ts, 'RAW_IMU'),
    'SCALED_IMU': lambda msg, ts: parse_imu(msg, ts, 'SCALED_IMU'),
    'SCALED_IMU2': lambda msg, ts: parse_imu(msg, ts, 'SCALED_IMU2'),
    'SCALED_IMU3': lambda msg, ts: parse_imu(msg, ts, 'SCALED_IMU3'),
    'HIGHRES_IMU': lambda msg, ts: parse_imu(msg, ts, 'HIGHRES_IMU'),
    'GPS_RAW_INT': parse_gps,
    'GLOBAL_POSITION_INT': parse_global_position
}


def get_message_type(msg) -> str:
    """Get message type string from MAVLink message."""
    return msg.get_type()


def can_parse(msg) -> bool:
    """Check if we have a parser for this message type."""
    return get_message_type(msg) in MESSAGE_PARSERS


def parse_message(msg, local_timestamp: Optional[float] = None):
    """
    Parse any supported MAVLink message.

    Args:
        msg: MAVLink message
        local_timestamp: Local monotonic timestamp

    Returns:
        Parsed data object or None if not supported
    """
    msg_type = get_message_type(msg)
    parser = MESSAGE_PARSERS.get(msg_type)

    if parser:
        return parser(msg, local_timestamp)

    return None
