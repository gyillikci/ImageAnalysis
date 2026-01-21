# Camera-Flight Controller Calibration for Orange Cube

This guide explains how to perform temporal and spatial calibration between
a camera and an Orange Cube flight controller (or any ArduPilot/PX4 system).

## Overview

**Temporal Calibration**: Finds the time offset between video and flight log timestamps.

**Spatial Calibration**: Finds the camera mounting offset (yaw, pitch, roll) relative to the IMU.

## Requirements

```bash
# Required packages
pip install pymavlink    # For ArduPilot .bin/.tlog files
pip install pyulog       # For PX4 .ulg files (optional)
pip install opencv-python scipy numpy matplotlib
pip install sk-video     # For video processing
```

## Supported Flight Log Formats

| Format | Extension | Autopilot |
|--------|-----------|-----------|
| DataFlash Binary | `.bin` | ArduPilot |
| Telemetry Log | `.tlog` | ArduPilot |
| ULog | `.ulg` | PX4 |

## Quick Start

### 1. Basic Calibration

```bash
cd video/
python calibrate_camera_orangecube.py \
    --video flight_video.mp4 \
    --flight flight.bin \
    --plot
```

### 2. With Camera Calibration

```bash
python calibrate_camera_orangecube.py \
    --video flight_video.mp4 \
    --flight flight.bin \
    --camera ../cameras/gopro_hero9.json \
    --cam-mount forward \
    --plot
```

### 3. Output

The script produces:
- `flight_video_motion.csv` - Video motion estimates
- `flight_video_calibration.json` - Calibration results

Example output:
```json
{
  "video_file": "flight_video.mp4",
  "flight_log": "flight.bin",
  "temporal_calibration": {
    "time_shift_seconds": 2.3456,
    "description": "video_time + time_shift = flight_time"
  },
  "spatial_calibration": {
    "yaw_deg": 0.0,
    "pitch_deg": -90.0,
    "roll_deg": 1.23,
    "description": "Camera mount offset relative to IMU"
  }
}
```

## Flight Data Collection Tips

### ArduPilot Logging Setup

Enable high-rate IMU logging in Mission Planner:

```
LOG_BITMASK: Include IMU, ATT, GPS, BARO
LOG_FILE_RATEMAX: 0 (no limit)
INS_LOG_BAT_OPT: 0 (log all IMU data)
```

### Recommended Flight Maneuvers

For best calibration results, perform during flight:

1. **Roll maneuvers**: ±30° left and right
2. **Pitch maneuvers**: ±20° nose up and down
3. **Yaw rotations**: Slow 360° turns
4. **Combined**: Figure-8 patterns

### Data Requirements

- Video and flight log from the **same flight**
- Camera **rigidly mounted** to airframe
- **Clear horizon** visible (for horizon-based calibration)
- At least **60 seconds** of maneuvering flight

## Module Usage

### Using the ArduPilot Log Adapter Directly

```python
from ardupilot_log import load, FlightInterp

# Load flight data
flight_data, flight_format = load('flight.bin')

# Create interpolator
interp = FlightInterp(flight_data)

# Query IMU data at any timestamp
p_rate = interp.group['imu'].interp['p'](timestamp)
q_rate = interp.group['imu'].interp['q'](timestamp)
r_rate = interp.group['imu'].interp['r'](timestamp)

# Query EKF attitude
phi = interp.group['filter'].interp['phi'](timestamp)  # Roll (rad)
the = interp.group['filter'].interp['the'](timestamp)  # Pitch (rad)
psi = interp.group['filter'].interp['psi'](timestamp)  # Yaw (rad)
alt = interp.group['filter'].interp['alt'](timestamp)  # Altitude (m)
```

### Drop-in Replacement for aurauas_flightdata

```python
# Replace this:
# from aurauas_flightdata import flight_loader, flight_interp

# With this:
from flight_loader_local import flight_loader, flight_interp

# Same interface
data, fmt = flight_loader.load('flight.bin')
interp = flight_interp.InterpolationGroup(data)
```

## Troubleshooting

### "No IMU data found"

- Check that LOG_BITMASK includes IMU logging
- Verify the .bin file is not corrupted
- Try using MAVExplorer to inspect the log

### "Cross-correlation failed"

- Ensure video and flight log are from the same flight
- Check that there's significant motion in the video
- Try adjusting `--resample-hz` (default 60)

### "Poor calibration accuracy"

- Perform more aggressive maneuvers during flight
- Ensure camera is rigidly mounted (no vibration dampening)
- Check for correct camera orientation setting (`--cam-mount`)

### PX4 ULog Support

```bash
pip install pyulog
python calibrate_camera_orangecube.py --video video.mp4 --flight log.ulg
```

## Files

| File | Description |
|------|-------------|
| `ardupilot_log.py` | ArduPilot/PX4 log parser and interpolator |
| `flight_loader_local.py` | Unified loader (drop-in for aurauas_flightdata) |
| `calibrate_camera_orangecube.py` | Complete calibration script |

## Theory

### Temporal Calibration

Uses cross-correlation between:
- Video-derived roll rate (from feature tracking)
- IMU roll rate (from flight log)

The peak of the correlation function indicates the time offset.

### Spatial Calibration

Compares video-derived attitude to EKF attitude after time alignment.
The difference reveals the camera mounting offset.

```
video_attitude = EKF_attitude × mount_rotation
mount_rotation = EKF_attitude⁻¹ × video_attitude
```

## See Also

- `5a-horizon-tracker.py` - Horizon-based attitude estimation
- `5b-cam-mount-from-horiz.py` - Full horizon-based calibration
- `5b-cam-mount-from-gyro.py` - Gyro-based mount calibration
