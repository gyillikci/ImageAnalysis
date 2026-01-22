# Real-Time Camera-IMU Calibration System

## Orange Cube + Harrier Camera Calibration

This system performs real-time temporal and spatial calibration between
an Orange Cube flight controller (via MAVLink) and a Harrier camera.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│   │ Orange Cube │         │   Harrier   │         │    Host     │       │
│   │   (FC)      │         │   Camera    │         │  Computer   │       │
│   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘       │
│          │                       │                       │              │
│          │ MAVLink               │ USB/UVC               │              │
│          │ (UDP/Serial)          │ Video                 │              │
│          │                       │                       │              │
│          ▼                       ▼                       │              │
│   ┌──────────────────────────────────────────────────────┴───┐          │
│   │                    DATA ACQUISITION LAYER                 │          │
│   ├───────────────────────┬──────────────────────────────────┤          │
│   │                       │                                   │          │
│   │  ┌─────────────────┐  │  ┌─────────────────┐             │          │
│   │  │ MAVLink Receiver│  │  │ Camera Capture  │             │          │
│   │  │                 │  │  │                 │             │          │
│   │  │ • Attitude      │  │  │ • Frame grab    │             │          │
│   │  │ • IMU rates     │  │  │ • Timestamps    │             │          │
│   │  │ • GPS           │  │  │ • Metadata      │             │          │
│   │  │ • Timestamps    │  │  │                 │             │          │
│   │  └────────┬────────┘  │  └────────┬────────┘             │          │
│   │           │           │           │                       │          │
│   └───────────┼───────────┴───────────┼───────────────────────┘          │
│               │                       │                                  │
│               ▼                       ▼                                  │
│   ┌───────────────────────────────────────────────────────────┐          │
│   │                  SYNCHRONIZATION LAYER                     │          │
│   ├───────────────────────────────────────────────────────────┤          │
│   │                                                            │          │
│   │  ┌─────────────────┐      ┌─────────────────┐             │          │
│   │  │ Ring Buffers    │      │ Time Sync       │             │          │
│   │  │                 │      │                 │             │          │
│   │  │ • IMU buffer    │      │ • Clock offset  │             │          │
│   │  │ • Frame buffer  │      │ • Drift comp    │             │          │
│   │  │ • Attitude buf  │      │ • Correlation   │             │          │
│   │  └────────┬────────┘      └────────┬────────┘             │          │
│   │           │                        │                       │          │
│   └───────────┼────────────────────────┼───────────────────────┘          │
│               │                        │                                  │
│               ▼                        ▼                                  │
│   ┌───────────────────────────────────────────────────────────┐          │
│   │                   CALIBRATION LAYER                        │          │
│   ├───────────────────────────────────────────────────────────┤          │
│   │                                                            │          │
│   │  ┌─────────────────┐      ┌─────────────────┐             │          │
│   │  │ Temporal Calib  │      │ Spatial Calib   │             │          │
│   │  │                 │      │                 │             │          │
│   │  │ • Online xcorr  │      │ • Mount offset  │             │          │
│   │  │ • Delay est.    │      │ • Optimization  │             │          │
│   │  │ • Latency comp  │      │ • Validation    │             │          │
│   │  └────────┬────────┘      └────────┬────────┘             │          │
│   │           │                        │                       │          │
│   └───────────┼────────────────────────┼───────────────────────┘          │
│               │                        │                                  │
│               ▼                        ▼                                  │
│   ┌───────────────────────────────────────────────────────────┐          │
│   │                    OUTPUT / VISUALIZATION                  │          │
│   ├───────────────────────────────────────────────────────────┤          │
│   │  • Real-time display    • Calibration results             │          │
│   │  • Status indicators    • Export to JSON                  │          │
│   │  • Debug plots          • Integration params              │          │
│   └───────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
calibration/
├── __init__.py
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   ├── __init__.py
│   ├── default.yaml            # Default configuration
│   └── harrier_camera.yaml     # Harrier camera settings
├── core/
│   ├── __init__.py
│   ├── data_types.py           # Data structures (IMUData, FrameData, etc.)
│   ├── ring_buffer.py          # Thread-safe ring buffers
│   └── utils.py                # Utility functions
├── mavlink/
│   ├── __init__.py
│   ├── receiver.py             # MAVLink message receiver
│   └── messages.py             # Message parsing helpers
├── camera/
│   ├── __init__.py
│   ├── harrier.py              # Harrier camera interface
│   ├── feature_tracker.py      # Real-time feature tracking
│   └── motion_estimator.py     # Frame-to-frame motion estimation
├── sync/
│   ├── __init__.py
│   ├── temporal.py             # Temporal synchronization
│   ├── spatial.py              # Spatial calibration (mount offset)
│   └── online_correlation.py   # Online cross-correlation
├── viz/
│   ├── __init__.py
│   ├── display.py              # Real-time visualization
│   └── plots.py                # Calibration result plots
├── tests/
│   ├── __init__.py
│   ├── test_mavlink.py
│   ├── test_camera.py
│   └── test_sync.py
├── calibrate.py                # Main calibration script
└── live_viewer.py              # Live camera + IMU viewer
```

## Data Flow

### 1. Data Acquisition (Parallel Threads)

```
Thread 1: MAVLink Receiver
    └── Receives: ATTITUDE, RAW_IMU, SCALED_IMU, GPS_RAW_INT
    └── Timestamps with local monotonic clock
    └── Pushes to IMU ring buffer

Thread 2: Camera Capture
    └── Captures frames via OpenCV (UVC)
    └── Timestamps with local monotonic clock
    └── Pushes to Frame ring buffer

Thread 3: Feature Tracker
    └── Processes frames from buffer
    └── Extracts features, estimates motion
    └── Computes visual gyro rates
```

### 2. Temporal Synchronization

```
Online Cross-Correlation:
    └── Sliding window of IMU rates and visual rates
    └── Compute correlation at different lags
    └── Estimate time offset: t_camera + offset = t_imu
    └── Update Kalman filter for drift compensation
```

### 3. Spatial Calibration

```
Mount Offset Estimation:
    └── After temporal sync, align attitude streams
    └── Compare visual attitude to EKF attitude
    └── Optimize rotation matrix R_cam_to_body
    └── Output: yaw, pitch, roll offsets
```

## Key Algorithms

### Temporal Calibration

Uses **online cross-correlation** between:
- IMU angular rates (p, q, r) from MAVLink
- Visual angular rates from frame-to-frame feature tracking

```python
# Sliding window correlation
for lag in range(-max_lag, max_lag):
    corr[lag] = correlate(imu_rates[t-window:t],
                          visual_rates[t-window-lag:t-lag])
time_offset = lag_at_max_correlation
```

### Spatial Calibration

After temporal alignment, estimate mount rotation:

```python
# Minimize attitude difference
def cost(R_mount):
    visual_attitude = R_mount @ camera_attitude
    error = visual_attitude - imu_attitude
    return sum(error**2)

R_optimal = optimize(cost, initial_guess)
yaw, pitch, roll = rotation_to_euler(R_optimal)
```

## Requirements

- Python 3.8+
- OpenCV 4.x
- pymavlink
- numpy, scipy
- PyYAML
- (optional) matplotlib for visualization

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run calibration
python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0

# With configuration file
python calibrate.py --config config/default.yaml
```

### Live Viewer (Debug)

```bash
# View live camera + IMU data
python live_viewer.py --mavlink udp:127.0.0.1:14550 --camera 0
```

## Configuration

```yaml
# config/default.yaml
mavlink:
  connection: "udp:127.0.0.1:14550"
  # or "serial:/dev/ttyUSB0:57600"

camera:
  device: 0                    # /dev/video0
  width: 1920
  height: 1080
  fps: 60

calibration:
  temporal:
    window_size: 5.0           # seconds
    max_lag: 1.0               # seconds
    update_rate: 10.0          # Hz
  spatial:
    min_samples: 500
    convergence_threshold: 0.001

output:
  save_path: "calibration_result.json"
```

## Output Format

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "temporal_calibration": {
    "time_offset_seconds": 0.0234,
    "confidence": 0.95,
    "method": "cross_correlation"
  },
  "spatial_calibration": {
    "mount_rotation": {
      "yaw_deg": 0.5,
      "pitch_deg": -89.2,
      "roll_deg": 1.1
    },
    "covariance": [...],
    "rmse_deg": 0.3
  },
  "system_info": {
    "camera": "Harrier 40x",
    "flight_controller": "Orange Cube",
    "mavlink_version": "2.0"
  }
}
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.x with Python bindings
- A working camera (Harrier or any UVC-compatible camera)
- MAVLink-compatible flight controller (Orange Cube, Pixhawk, etc.)

### Install Dependencies

```bash
cd calibration
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy opencv-python pymavlink PyYAML
```

## Detailed Usage

### 1. Basic Calibration

Connect your camera and flight controller, then run:

```bash
python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0
```

### 2. Serial Connection (USB)

For direct USB connection to flight controller:

```bash
python calibrate.py --mavlink /dev/ttyACM0 --baudrate 115200 --camera 0
```

### 3. Specify Camera Intrinsics

If you know your camera's intrinsic parameters:

```bash
python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0 \
    --fx 800 --fy 800 --cx 640 --cy 360
```

### 4. Save Calibration Result

```bash
python calibrate.py --mavlink udp:127.0.0.1:14550 --camera 0 \
    --output calibration_result.json
```

### 5. Live Viewer (for debugging)

To visualize the calibration process in real-time:

```bash
python live_viewer.py --mavlink udp:127.0.0.1:14550 --camera 0
```

Controls:
- `q` - Quit
- `f` - Toggle feature display
- `o` - Toggle optical flow display
- `i` - Toggle info overlay
- `r` - Reset calibration

## Calibration Procedure

1. **Setup**: Connect camera and flight controller, ensure MAVLink stream is active
2. **Start Calibration**: Run the calibrate.py script
3. **Generate Motion**: Move the aircraft/gimbal to generate rotation in all axes
   - Roll the aircraft left and right
   - Pitch up and down
   - Yaw left and right
4. **Wait for Convergence**: The system will automatically detect convergence
5. **Review Results**: Check the temporal offset and mount rotation values

### Tips for Best Results

- **Good lighting**: Ensure adequate lighting for feature tracking
- **Textured environment**: Point camera at textured surfaces for better features
- **Varied motion**: Include rotation in all three axes
- **Moderate speed**: Avoid extremely fast motions
- **Minimum duration**: Run for at least 10-15 seconds

## Troubleshooting

### No MAVLink Connection

1. Check connection string format:
   - UDP: `udp:IP:PORT` (e.g., `udp:127.0.0.1:14550`)
   - Serial: `/dev/ttyACM0` or `COM3`

2. Verify MAVLink stream is active:
   ```bash
   python -c "from pymavlink import mavutil; m = mavutil.mavlink_connection('udp:127.0.0.1:14550'); print(m.recv_match())"
   ```

### Camera Not Found

1. List available cameras:
   ```bash
   ls /dev/video*
   ```

2. Test camera with OpenCV:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print(f"Camera working: {ret}")
   ```

### Poor Calibration Results

- **Low correlation**: Move the aircraft more vigorously
- **High RMSE**: Check for sensor noise or timing issues
- **No convergence**: Ensure both camera and IMU are providing data

## API Usage

The calibration modules can also be used programmatically:

```python
from calibration.mavlink.receiver import MAVLinkReceiver, MAVLinkConfig
from calibration.camera.harrier import HarrierCamera, CameraConfig
from calibration.sync.temporal import TemporalSynchronizer, TemporalSyncConfig
from calibration.sync.spatial import SpatialCalibrator, SpatialCalibConfig

# Create components
mavlink = MAVLinkReceiver(MAVLinkConfig(connection_string="udp:127.0.0.1:14550"))
camera = HarrierCamera(CameraConfig(device_id=0))

# Start data acquisition
mavlink.start()
camera.start()

# Create synchronizers
temporal_sync = TemporalSynchronizer(TemporalSyncConfig())
spatial_calib = SpatialCalibrator(SpatialCalibConfig())

# Connect callbacks
mavlink.add_imu_callback(temporal_sync.on_imu_data)

# Process frames and estimate motion...
# Feed data to calibrators...

# Get results
time_offset = temporal_sync.time_offset
mount_offset = spatial_calib.mount_offset
```

## References

- [Active Silicon Harrier Camera](https://www.activesilicon.com/products/harrier-40x-af-zoom-camera-usb-hdmi/)
- [MAVLink Protocol](https://mavlink.io/en/)
- [ArduPilot MAVLink Messages](https://ardupilot.org/dev/docs/mavlink-commands.html)
- [Orange Cube (CubePilot)](https://docs.cubepilot.org/)
