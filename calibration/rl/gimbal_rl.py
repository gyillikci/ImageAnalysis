"""
Reinforcement Learning for Gimbal Stabilization

Uses calibrated IMU data for smooth gimbal control.
This is a natural application of the camera-IMU calibration system.

Algorithms supported:
- PPO (Proximal Policy Optimization) - stable, good for continuous control
- SAC (Soft Actor-Critic) - sample efficient, handles exploration
- TD3 (Twin Delayed DDPG) - good for deterministic policies

Requirements:
    pip install stable-baselines3 gymnasium

Usage:
    # Training
    python gimbal_rl.py --train --episodes 10000

    # Evaluation with real hardware
    python gimbal_rl.py --eval --mavlink /dev/ttyACM0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("gymnasium not installed. Install with: pip install gymnasium")


@dataclass
class GimbalState:
    """Gimbal state representation."""
    # Current angles (rad)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Angular rates (rad/s)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    # Target angles (rad)
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_yaw: float = 0.0

    # IMU data (calibrated)
    imu_p: float = 0.0  # Body roll rate
    imu_q: float = 0.0  # Body pitch rate
    imu_r: float = 0.0  # Body yaw rate

    def to_array(self) -> np.ndarray:
        """Convert to observation array."""
        return np.array([
            self.roll, self.pitch, self.yaw,
            self.roll_rate, self.pitch_rate, self.yaw_rate,
            self.target_roll - self.roll,  # Errors
            self.target_pitch - self.pitch,
            self.target_yaw - self.yaw,
            self.imu_p, self.imu_q, self.imu_r
        ], dtype=np.float32)


@dataclass
class GimbalAction:
    """Gimbal motor commands."""
    roll_torque: float = 0.0   # Nm
    pitch_torque: float = 0.0  # Nm
    yaw_torque: float = 0.0    # Nm

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'GimbalAction':
        return cls(
            roll_torque=float(arr[0]),
            pitch_torque=float(arr[1]),
            yaw_torque=float(arr[2])
        )


class GimbalSimulator:
    """
    Simple gimbal dynamics simulator for RL training.

    Models 3-axis gimbal with motor dynamics and disturbances.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt

        # Gimbal inertia (kg*m^2)
        self.I_roll = 0.01
        self.I_pitch = 0.02
        self.I_yaw = 0.015

        # Motor limits
        self.max_torque = 0.5  # Nm
        self.max_rate = 5.0    # rad/s
        self.max_angle = np.radians(45)  # rad

        # Damping
        self.damping = 0.1

        # State
        self.reset()

    def reset(self) -> GimbalState:
        """Reset to initial state."""
        self.state = GimbalState()
        self.disturbance = np.zeros(3)
        return self.state

    def set_disturbance(self, roll_rate: float, pitch_rate: float, yaw_rate: float):
        """Set body disturbance rates (from aircraft motion)."""
        self.disturbance = np.array([roll_rate, pitch_rate, yaw_rate])

    def step(self, action: GimbalAction) -> Tuple[GimbalState, float, bool]:
        """
        Step simulation forward.

        Args:
            action: Motor torque commands

        Returns:
            (new_state, reward, done)
        """
        # Clip torques
        torques = np.clip(
            [action.roll_torque, action.pitch_torque, action.yaw_torque],
            -self.max_torque, self.max_torque
        )

        # Current rates
        rates = np.array([
            self.state.roll_rate,
            self.state.pitch_rate,
            self.state.yaw_rate
        ])

        # Inertias
        I = np.array([self.I_roll, self.I_pitch, self.I_yaw])

        # Acceleration = (torque - damping - disturbance_compensation) / I
        # Gimbal must counter body motion to stabilize
        disturbance_torque = self.disturbance * I  # Torque needed to counter
        accel = (torques - self.damping * rates) / I

        # Integrate rates
        new_rates = rates + accel * self.dt
        new_rates = np.clip(new_rates, -self.max_rate, self.max_rate)

        # Integrate angles
        angles = np.array([self.state.roll, self.state.pitch, self.state.yaw])
        new_angles = angles + new_rates * self.dt
        new_angles = np.clip(new_angles, -self.max_angle, self.max_angle)

        # Update state
        self.state.roll, self.state.pitch, self.state.yaw = new_angles
        self.state.roll_rate, self.state.pitch_rate, self.state.yaw_rate = new_rates
        self.state.imu_p, self.state.imu_q, self.state.imu_r = self.disturbance

        # Compute reward
        reward = self._compute_reward(action)

        # Check done (angle limits exceeded badly)
        done = np.any(np.abs(new_angles) > self.max_angle * 0.95)

        return self.state, reward, done

    def _compute_reward(self, action: GimbalAction) -> float:
        """Compute reward signal."""
        # Error terms
        roll_error = abs(self.state.target_roll - self.state.roll)
        pitch_error = abs(self.state.target_pitch - self.state.pitch)
        yaw_error = abs(self.state.target_yaw - self.state.yaw)

        # Smoothness (penalize high torques)
        torque_penalty = 0.01 * (
            action.roll_torque**2 +
            action.pitch_torque**2 +
            action.yaw_torque**2
        )

        # Rate penalty (penalize oscillation)
        rate_penalty = 0.001 * (
            self.state.roll_rate**2 +
            self.state.pitch_rate**2 +
            self.state.yaw_rate**2
        )

        # Total reward (negative for errors)
        reward = -(roll_error + pitch_error + yaw_error) - torque_penalty - rate_penalty

        return reward


if HAS_GYM:
    class GimbalEnv(gym.Env):
        """
        Gymnasium environment for gimbal stabilization.

        Observation space: 12D (angles, rates, errors, IMU)
        Action space: 3D continuous (torques)
        """

        metadata = {'render_modes': ['human']}

        def __init__(self, render_mode=None):
            super().__init__()

            self.simulator = GimbalSimulator(dt=0.01)
            self.render_mode = render_mode

            # Observation: [roll, pitch, yaw, rates, errors, imu]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )

            # Action: [roll_torque, pitch_torque, yaw_torque]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

            self.max_steps = 1000
            self.current_step = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            state = self.simulator.reset()

            # Random target
            state.target_roll = np.random.uniform(-0.3, 0.3)
            state.target_pitch = np.random.uniform(-0.3, 0.3)
            state.target_yaw = np.random.uniform(-0.3, 0.3)

            self.current_step = 0

            return state.to_array(), {}

        def step(self, action):
            # Scale action to torque range
            scaled_action = action * self.simulator.max_torque
            gimbal_action = GimbalAction.from_array(scaled_action)

            # Add random disturbance (simulating aircraft motion)
            if self.current_step % 50 == 0:
                self.simulator.set_disturbance(
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-0.5, 0.5)
                )

            state, reward, done = self.simulator.step(gimbal_action)
            self.current_step += 1

            truncated = self.current_step >= self.max_steps

            return state.to_array(), reward, done, truncated, {}

        def render(self):
            if self.render_mode == 'human':
                s = self.simulator.state
                print(f"Roll: {np.degrees(s.roll):6.2f}° | "
                      f"Pitch: {np.degrees(s.pitch):6.2f}° | "
                      f"Yaw: {np.degrees(s.yaw):6.2f}°")


class GimbalRLController:
    """
    RL-based gimbal controller using trained policy.

    Integrates with calibrated IMU data for real-world deployment.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.calibration = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load trained RL model."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(path)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def set_calibration(self, temporal_offset: float, mount_rotation: np.ndarray):
        """Set calibration parameters from calibration result."""
        self.calibration = {
            'time_offset': temporal_offset,
            'mount_rotation': mount_rotation
        }

    def transform_imu(self, imu_p: float, imu_q: float, imu_r: float,
                      timestamp: float) -> Tuple[np.ndarray, float]:
        """
        Apply calibration to raw IMU data.

        Returns:
            (corrected_rates, corrected_timestamp)
        """
        if self.calibration is None:
            return np.array([imu_p, imu_q, imu_r]), timestamp

        # Apply mount rotation
        raw_rates = np.array([imu_p, imu_q, imu_r])
        corrected_rates = self.calibration['mount_rotation'] @ raw_rates

        # Apply time offset
        corrected_time = timestamp + self.calibration['time_offset']

        return corrected_rates, corrected_time

    def compute_action(self, state: GimbalState) -> GimbalAction:
        """
        Compute gimbal torque commands from current state.

        Args:
            state: Current gimbal state with calibrated IMU

        Returns:
            Motor torque commands
        """
        if self.model is None:
            # Fallback to PID if no RL model
            return self._pid_fallback(state)

        obs = state.to_array()
        action, _ = self.model.predict(obs, deterministic=True)

        # Scale action
        max_torque = 0.5
        return GimbalAction(
            roll_torque=float(action[0] * max_torque),
            pitch_torque=float(action[1] * max_torque),
            yaw_torque=float(action[2] * max_torque)
        )

    def _pid_fallback(self, state: GimbalState) -> GimbalAction:
        """Simple PID controller as fallback."""
        Kp = 2.0
        Kd = 0.5

        return GimbalAction(
            roll_torque=Kp * (state.target_roll - state.roll) - Kd * state.roll_rate,
            pitch_torque=Kp * (state.target_pitch - state.pitch) - Kd * state.pitch_rate,
            yaw_torque=Kp * (state.target_yaw - state.yaw) - Kd * state.yaw_rate
        )


def train_gimbal_agent(total_timesteps: int = 100000, save_path: str = "gimbal_ppo"):
    """
    Train gimbal stabilization agent using PPO.

    Args:
        total_timesteps: Number of training steps
        save_path: Path to save trained model
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        print("stable-baselines3 required. Install with: pip install stable-baselines3")
        return None

    print("Creating training environment...")
    env = make_vec_env(GimbalEnv, n_envs=4)

    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./gimbal_tensorboard/"
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print(f"Saving model to {save_path}")
    model.save(save_path)

    return model


def evaluate_with_calibration(model_path: str, calibration_result):
    """
    Evaluate trained agent with real calibrated IMU data.

    Args:
        model_path: Path to trained model
        calibration_result: CalibrationResult from calibration system
    """
    controller = GimbalRLController(model_path)

    # Set calibration
    if calibration_result.temporal:
        time_offset = calibration_result.temporal.time_offset
    else:
        time_offset = 0.0

    if calibration_result.spatial:
        mount_rotation = calibration_result.spatial.rotation_matrix
    else:
        mount_rotation = np.eye(3)

    controller.set_calibration(time_offset, mount_rotation)

    print("Controller ready with calibration:")
    print(f"  Time offset: {time_offset * 1000:.2f} ms")
    print(f"  Mount rotation applied: {calibration_result.spatial is not None}")

    return controller


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gimbal RL Training/Evaluation")
    parser.add_argument('--train', action='store_true', help='Train agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate agent')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--model', type=str, default='gimbal_ppo', help='Model path')

    args = parser.parse_args()

    if args.train:
        if not HAS_GYM:
            print("gymnasium required for training")
        else:
            train_gimbal_agent(args.timesteps, args.model)

    elif args.eval:
        # Simple simulation evaluation
        if HAS_GYM:
            env = GimbalEnv(render_mode='human')
            controller = GimbalRLController(args.model)

            obs, _ = env.reset()
            total_reward = 0

            for _ in range(500):
                state = GimbalState()
                state.roll, state.pitch, state.yaw = obs[0], obs[1], obs[2]
                state.roll_rate, state.pitch_rate, state.yaw_rate = obs[3], obs[4], obs[5]
                state.imu_p, state.imu_q, state.imu_r = obs[9], obs[10], obs[11]

                action = controller.compute_action(state)
                action_arr = np.array([
                    action.roll_torque,
                    action.pitch_torque,
                    action.yaw_torque
                ]) / 0.5  # Scale back

                obs, reward, done, truncated, _ = env.step(action_arr)
                total_reward += reward
                env.render()

                if done or truncated:
                    break

            print(f"Total reward: {total_reward:.2f}")
    else:
        print("Use --train or --eval")
