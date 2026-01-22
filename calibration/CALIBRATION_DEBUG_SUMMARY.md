# Camera-IMU Calibration Debug Summary

## Overview

This document summarizes the debugging process for the real-time camera-IMU calibration system. The calibration estimates the temporal offset (time delay) and spatial offset (mount angles) between a camera and an IMU (Orange Cube flight controller).

**Final Result:** ✅ SUCCESS
- Temporal offset: -83.22 ms (correlation: 0.971)
- Spatial offset: Yaw=4.56°, Pitch=-11.25°, Roll=-1.09°

---

## Issues Encountered and Solutions

### 1. MAVLinkConfig Missing `baudrate` Field

**Problem:** The `MAVLinkConfig` dataclass didn't have a `baudrate` parameter.

**Solution:** Added `baudrate: int = 115200` to the `MAVLinkConfig` dataclass in `mavlink/receiver.py`.

---

### 2. CameraConfig Wrong Parameter Name

**Problem:** Camera initialization failed because the parameter was named `device_id` but should be `device`.

**Solution:** Changed `device_id` to `device` in the `CameraConfig` dataclass.

---

### 3. IMU Callback Method Name Mismatch

**Problem:** Code called `add_imu_callback()` but the actual method was `add_callback('imu', ...)`.

**Solution:** Updated callback registration to use the correct method signature.

---

### 4. Camera Status Key Mismatch

**Problem:** Code checked for `actual_fps` in camera status but the key was `frame_rate`.

**Solution:** Updated status key references.

---

### 5. Temporal Sync Stuck at corr=0.000

**Problem:** The temporal synchronizer never found any correlation between IMU and camera signals.

**Root Cause Analysis:**
1. First suspected: Thresholds too strict → Lowered `min_correlation` from 0.5 to 0.3
2. Then found: Camera motion estimates were ~100x too large (67 rad/s vs 0.8 rad/s for IMU)
3. Added sanity check to filter values > 10 rad/s → Now estimates were 10x too small
4. Root cause: **Essential matrix decomposition was unreliable**

**Solution:** Replaced essential matrix approach with simple optical flow method:
```python
# Mean flow gives pitch and yaw
mean_flow = np.mean(flow, axis=0)
omega_y = mean_flow[0] / (fx * dt)   # yaw rate
omega_x = -mean_flow[1] / (fy * dt)  # pitch rate
```

---

### 6. Camera FOV Not Considered

**Problem:** Default focal length values (fx=800, fy=800) were incorrect for the camera.

**Camera Specs:**
- Diagonal FOV: 64°
- Focal length: 5.1mm

**Solution:** Calculated correct focal length:
- For 1280x720: `fx = fy = 1175`
- For 1920x1080: `fx = fy = 1762`

Formula: `fx = (width/2) / tan(hfov/2)` where `hfov = 2 * atan(tan(dfov/2) * width / diagonal)`

---

### 7. Roll Axis Responding to All Movements (3x Error)

**Problem:** The roll (p) estimate from camera was 3x larger than IMU and responded to pitch/yaw movements.

**Root Cause:** The roll calculation wasn't removing the translational component (mean flow) before computing the rotational component.

**Original (Wrong):**
```python
centered = good_curr - np.array([cx, cy])
tangential = (flow[:, 0] * (-centered[:, 1]) + flow[:, 1] * centered[:, 0]) / r
omega_z = np.median(tangential) / (np.mean(r) * dt)
```

**Fixed:**
```python
# Remove translation first
flow_centered = flow - mean_flow

# Position from image center (using prev points)
pos = good_prev - np.array([cx, cy])
r_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
r_dist = np.maximum(r_dist, 10.0)

# Proper tangent calculation
tangent_x = -pos[:, 1] / r_dist
tangent_y = pos[:, 0] / r_dist
tangential = flow_centered[:, 0] * tangent_x + flow_centered[:, 1] * tangent_y

# Angular velocity per point
omega_per_point = tangential / r_dist
omega_z = -np.median(omega_per_point) / dt  # Negated for IMU convention
```

---

### 8. Roll Sign Inverted

**Problem:** Roll signals from camera and IMU had opposite signs.

**Solution:** Added negation: `omega_z = -np.median(omega_per_point) / dt`

---

### 9. IMU Buffer Too Small

**Problem:** IMU buffer only held 9 seconds of data but sync needed 10+ seconds.

**Root Cause:** Buffer size was calculated as `max_window_duration * sample_rate * 2` = 30 * 60 * 2 = 3600 samples, but IMU runs at ~200Hz, so 3600 samples = only 18 seconds max, and the ring buffer was filling up.

**Solution:** Increased IMU buffer size:
```python
imu_buffer_size = int(config.max_window_duration * 250)  # Assume max 250Hz IMU
```

---

### 10. Negative Correlation Not Handled

**Problem:** Cross-correlation only looked for positive peak, but signals could be inverted.

**Solution:** Check both positive and negative peaks:
```python
max_pos_index = np.argmax(ycorr)
max_neg_index = np.argmin(ycorr)

if abs(ycorr[max_pos_index]) >= abs(ycorr[max_neg_index]):
    max_index = max_pos_index
else:
    max_index = max_neg_index
```

---

## Files Modified

| File | Changes |
|------|---------|
| `mavlink/receiver.py` | Added `baudrate` field to `MAVLinkConfig` |
| `camera/motion_estimator.py` | Replaced essential matrix with optical flow, fixed roll calculation |
| `sync/temporal.py` | Fixed buffer sizes, lowered thresholds, handle negative correlation |
| `calibrate.py` | Fixed callback method, status keys, added `--fx`/`--fy` args |

---

## Debugging Tools Created

### 1. `test_mavlink.py`
Tests MAVLink connection and IMU data reception.

### 2. `test_camera.py`
Tests camera capture and motion estimation.

### 3. `test_sync.py`
Tests temporal synchronization with both data streams.

### 4. `plot_signals.py`
Records IMU and camera signals, plots comparison, saves to PNG.

### 5. `realtime_viz.py`
Real-time visualization with:
- Live camera feed with optical flow vectors
- Roll indicator
- Live signal plots (IMU blue, Camera red)

---

## Key Learnings

1. **Essential matrix decomposition is unreliable for angular rate estimation** - It requires good depth variation and can produce wildly wrong results. Simple mean optical flow works much better for pure rotation.

2. **Roll requires special handling** - Must subtract mean flow (translation) before computing rotational component, otherwise pitch/yaw contaminate the roll estimate.

3. **Buffer sizes must account for actual data rates** - IMU at 200Hz needs larger buffers than camera at 50Hz.

4. **Visual debugging is essential** - The real-time visualization tool was crucial for understanding why roll was responding to all axes.

5. **Camera intrinsics matter** - Correct focal length values are critical for converting pixel motion to angular rates.

---

## Usage

```bash
# Run calibration
python calibrate.py --mavlink COM6 --baudrate 115200 --camera 0 --fx 1175 --fy 1175

# Run real-time visualization
python realtime_viz.py --mavlink COM6 --baudrate 115200 --camera 0

# Run signal comparison plot
python plot_signals.py --mavlink COM6 --baudrate 115200 --camera 0 --duration 15
```
