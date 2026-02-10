"""
Reinforcement Learning modules for calibration system.

Applications:
1. Gimbal Stabilization - Use calibrated IMU for smooth gimbal control
2. Adaptive Time Sync - Learn optimal filtering for clock synchronization
3. Calibration Motion Planning - Optimal motions for faster convergence
4. Visual-Inertial Odometry - Adaptive feature selection
"""

from .gimbal_rl import (
    GimbalState,
    GimbalAction,
    GimbalSimulator,
    GimbalRLController,
    train_gimbal_agent,
    evaluate_with_calibration
)

try:
    from .gimbal_rl import GimbalEnv
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

__all__ = [
    'GimbalState',
    'GimbalAction',
    'GimbalSimulator',
    'GimbalRLController',
    'train_gimbal_agent',
    'evaluate_with_calibration',
    'HAS_GYM'
]

if HAS_GYM:
    __all__.append('GimbalEnv')
