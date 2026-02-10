"""
Reinforcement Learning modules for calibration system.

Applications:
1. Gimbal Stabilization - Use calibrated IMU for smooth gimbal control
2. System Identification - Learn IMU noise, gimbal dynamics, time delays
3. Adaptive Time Sync - Learn optimal filtering for clock synchronization
4. Calibration Motion Planning - Optimal motions for faster convergence
"""

from .gimbal_rl import (
    GimbalState,
    GimbalAction,
    GimbalSimulator,
    GimbalRLController,
    train_gimbal_agent,
    evaluate_with_calibration
)

from .system_id import (
    IMUNoiseModel,
    GimbalDynamicsModel,
    TimeDelayModel,
    IMUBiasEstimator,
    GimbalSystemID,
    TimeDelayEstimator,
    OnlineSystemID,
    train_active_sysid
)

try:
    from .gimbal_rl import GimbalEnv
    from .system_id import ActiveSystemIDEnv
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

__all__ = [
    # Gimbal RL
    'GimbalState',
    'GimbalAction',
    'GimbalSimulator',
    'GimbalRLController',
    'train_gimbal_agent',
    'evaluate_with_calibration',
    # System ID
    'IMUNoiseModel',
    'GimbalDynamicsModel',
    'TimeDelayModel',
    'IMUBiasEstimator',
    'GimbalSystemID',
    'TimeDelayEstimator',
    'OnlineSystemID',
    'train_active_sysid',
    # Flag
    'HAS_GYM'
]

if HAS_GYM:
    __all__.extend(['GimbalEnv', 'ActiveSystemIDEnv'])
