"""
Reinforcement Learning for System Identification

Uses RL for active and adaptive system identification:
1. IMU bias/noise estimation
2. Gimbal dynamics learning
3. Time delay characterization
4. Aircraft dynamics identification

Key insight: RL can actively choose inputs that maximize
information gain about unknown system parameters.

References:
- "Active Learning for System Identification" (Schreiter et al.)
- "Neural Network Dynamics for Model-Based Deep RL" (Nagabandi et al.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
import threading
import time

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


@dataclass
class IMUNoiseModel:
    """Identified IMU noise parameters."""
    # Gyroscope
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    gyro_noise_density: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.01)  # rad/s/sqrt(Hz)
    gyro_random_walk: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.001)  # rad/s^2/sqrt(Hz)

    # Accelerometer
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2
    accel_noise_density: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.1)  # m/s^2/sqrt(Hz)
    accel_random_walk: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.01)  # m/s^3/sqrt(Hz)

    # Confidence
    samples_used: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            'gyro_bias': self.gyro_bias.tolist(),
            'gyro_noise_density': self.gyro_noise_density.tolist(),
            'gyro_random_walk': self.gyro_random_walk.tolist(),
            'accel_bias': self.accel_bias.tolist(),
            'accel_noise_density': self.accel_noise_density.tolist(),
            'accel_random_walk': self.accel_random_walk.tolist(),
            'samples_used': self.samples_used,
            'confidence': self.confidence
        }


@dataclass
class GimbalDynamicsModel:
    """Identified gimbal dynamics parameters."""
    # Inertia tensor (diagonal approximation)
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.02, 0.015]))  # kg*m^2

    # Damping coefficients
    damping: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))  # Nm/(rad/s)

    # Motor constants
    motor_constant: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))  # Nm/A

    # Friction
    coulomb_friction: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01]))  # Nm

    # Confidence
    samples_used: int = 0
    prediction_rmse: float = float('inf')


@dataclass
class TimeDelayModel:
    """Identified time delay characteristics."""
    mean_delay: float = 0.0  # seconds
    delay_std: float = 0.0  # seconds
    jitter_model: str = "gaussian"  # or "uniform", "exponential"
    burst_probability: float = 0.0  # Probability of burst delivery
    burst_size: float = 0.0  # Average messages per burst


class IMUBiasEstimator:
    """
    Online IMU bias estimation using Allan Variance analysis.

    Can be enhanced with RL for adaptive windowing.
    """

    def __init__(self, sample_rate: float = 200.0):
        self.sample_rate = sample_rate
        self.gyro_samples: List[np.ndarray] = []
        self.accel_samples: List[np.ndarray] = []
        self.stationary_threshold = 0.05  # rad/s - consider stationary below this
        self._lock = threading.Lock()

    def add_sample(self, gyro: np.ndarray, accel: np.ndarray) -> None:
        """Add IMU sample."""
        with self._lock:
            self.gyro_samples.append(gyro.copy())
            self.accel_samples.append(accel.copy())

            # Keep bounded
            max_samples = int(self.sample_rate * 300)  # 5 minutes
            if len(self.gyro_samples) > max_samples:
                self.gyro_samples = self.gyro_samples[-max_samples:]
                self.accel_samples = self.accel_samples[-max_samples:]

    def detect_stationary(self, window_size: int = 100) -> bool:
        """Detect if IMU is stationary (for bias estimation)."""
        with self._lock:
            if len(self.gyro_samples) < window_size:
                return False

            recent = np.array(self.gyro_samples[-window_size:])
            gyro_std = np.std(recent, axis=0)

            return np.all(gyro_std < self.stationary_threshold)

    def estimate_bias(self) -> Optional[IMUNoiseModel]:
        """
        Estimate IMU biases from stationary data.

        Returns:
            IMUNoiseModel if enough stationary data, None otherwise
        """
        with self._lock:
            if len(self.gyro_samples) < 1000:
                return None

            gyro = np.array(self.gyro_samples)
            accel = np.array(self.accel_samples)

            # Find stationary periods
            window = 100
            stationary_mask = np.zeros(len(gyro), dtype=bool)

            for i in range(window, len(gyro)):
                std = np.std(gyro[i-window:i], axis=0)
                if np.all(std < self.stationary_threshold):
                    stationary_mask[i-window:i] = True

            stationary_gyro = gyro[stationary_mask]
            stationary_accel = accel[stationary_mask]

            if len(stationary_gyro) < 500:
                return None

            model = IMUNoiseModel()
            model.gyro_bias = np.mean(stationary_gyro, axis=0)
            model.gyro_noise_density = np.std(stationary_gyro, axis=0)
            model.accel_bias = np.mean(stationary_accel, axis=0) - np.array([0, 0, 9.81])
            model.accel_noise_density = np.std(stationary_accel, axis=0)
            model.samples_used = len(stationary_gyro)
            model.confidence = min(1.0, len(stationary_gyro) / 5000)

            return model

    def allan_variance(self, data: np.ndarray, max_cluster: int = 1000
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Allan Variance for noise characterization.

        Returns:
            (tau, avar) - averaging times and Allan variance
        """
        n = len(data)
        max_m = min(n // 2, max_cluster)

        taus = []
        avars = []

        for m in range(1, max_m, max(1, max_m // 50)):
            tau = m / self.sample_rate

            # Compute Allan variance
            clusters = n // m
            if clusters < 2:
                break

            means = np.array([np.mean(data[i*m:(i+1)*m]) for i in range(clusters)])
            avar = 0.5 * np.mean(np.diff(means)**2)

            taus.append(tau)
            avars.append(avar)

        return np.array(taus), np.array(avars)


class GimbalSystemID:
    """
    System identification for gimbal dynamics.

    Uses input-output data to estimate inertia, damping, and motor parameters.
    Can use RL for active exploration (choosing optimal inputs).
    """

    def __init__(self):
        self.input_history: List[np.ndarray] = []  # Motor torques
        self.output_history: List[np.ndarray] = []  # Angular rates/positions
        self.time_history: List[float] = []
        self._lock = threading.Lock()

    def add_sample(self, torque: np.ndarray, rate: np.ndarray,
                   position: np.ndarray, timestamp: float) -> None:
        """Add input-output sample."""
        with self._lock:
            self.input_history.append(torque.copy())
            self.output_history.append(np.concatenate([position, rate]))
            self.time_history.append(timestamp)

            # Keep bounded
            if len(self.input_history) > 10000:
                self.input_history = self.input_history[-10000:]
                self.output_history = self.output_history[-10000:]
                self.time_history = self.time_history[-10000:]

    def identify(self) -> Optional[GimbalDynamicsModel]:
        """
        Identify gimbal dynamics using least squares.

        Model: I * alpha = tau - b * omega - f * sign(omega)

        Returns:
            GimbalDynamicsModel with identified parameters
        """
        with self._lock:
            if len(self.input_history) < 500:
                return None

            torques = np.array(self.input_history)
            outputs = np.array(self.output_history)
            times = np.array(self.time_history)

        # Extract positions and rates
        positions = outputs[:, :3]
        rates = outputs[:, 3:]

        # Compute accelerations (numerical differentiation)
        dt = np.diff(times)
        dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero
        accels = np.diff(rates, axis=0) / dt[:, np.newaxis]

        # Align arrays
        torques = torques[1:]
        rates = rates[1:]

        # Least squares for each axis: tau = I*alpha + b*omega + f*sign(omega)
        # Rewrite as: tau = [alpha, omega, sign(omega)] @ [I, b, f]

        model = GimbalDynamicsModel()

        for axis in range(3):
            tau = torques[:, axis]
            alpha = accels[:, axis]
            omega = rates[:, axis]
            sign_omega = np.sign(omega)

            # Build regressor matrix
            A = np.column_stack([alpha, omega, sign_omega])

            # Solve least squares
            try:
                params, residuals, _, _ = np.linalg.lstsq(A, tau, rcond=None)
                model.inertia[axis] = max(0.001, params[0])
                model.damping[axis] = max(0, params[1])
                model.coulomb_friction[axis] = max(0, abs(params[2]))
            except:
                pass

        model.samples_used = len(torques)

        # Compute prediction RMSE
        predictions = self._predict(torques, rates, accels, model)
        model.prediction_rmse = np.sqrt(np.mean((torques - predictions)**2))

        return model

    def _predict(self, torques: np.ndarray, rates: np.ndarray,
                 accels: np.ndarray, model: GimbalDynamicsModel) -> np.ndarray:
        """Predict torques given model."""
        predicted = np.zeros_like(torques)
        for axis in range(3):
            predicted[:, axis] = (
                model.inertia[axis] * accels[:, axis] +
                model.damping[axis] * rates[:, axis] +
                model.coulomb_friction[axis] * np.sign(rates[:, axis])
            )
        return predicted


class TimeDelayEstimator:
    """
    Estimate MAVLink time delay characteristics.

    Analyzes the difference between FC timestamps and arrival times.
    """

    def __init__(self):
        self.delays: List[float] = []
        self.inter_arrival: List[float] = []
        self.last_arrival: Optional[float] = None
        self._lock = threading.Lock()

    def add_sample(self, fc_timestamp: float, arrival_time: float) -> None:
        """Add timing sample."""
        with self._lock:
            # Delay = arrival - fc_time (after clock sync)
            delay = arrival_time - fc_timestamp
            if delay > 0:  # Sanity check
                self.delays.append(delay)

            # Inter-arrival time
            if self.last_arrival is not None:
                iat = arrival_time - self.last_arrival
                if 0 < iat < 1.0:  # Reasonable range
                    self.inter_arrival.append(iat)
            self.last_arrival = arrival_time

            # Keep bounded
            if len(self.delays) > 10000:
                self.delays = self.delays[-10000:]
            if len(self.inter_arrival) > 10000:
                self.inter_arrival = self.inter_arrival[-10000:]

    def estimate(self) -> Optional[TimeDelayModel]:
        """Estimate time delay model."""
        with self._lock:
            if len(self.delays) < 100:
                return None

            delays = np.array(self.delays)
            iats = np.array(self.inter_arrival) if self.inter_arrival else np.array([0.01])

        model = TimeDelayModel()
        model.mean_delay = np.mean(delays)
        model.delay_std = np.std(delays)

        # Detect burst patterns
        if len(iats) > 10:
            expected_iat = np.median(iats)
            burst_threshold = expected_iat * 0.3  # Much shorter than expected
            bursts = iats < burst_threshold
            model.burst_probability = np.mean(bursts)

            if model.burst_probability > 0.1:
                # Estimate burst size
                burst_sizes = []
                current_burst = 1
                for is_burst in bursts:
                    if is_burst:
                        current_burst += 1
                    else:
                        if current_burst > 1:
                            burst_sizes.append(current_burst)
                        current_burst = 1
                if burst_sizes:
                    model.burst_size = np.mean(burst_sizes)

        return model


if HAS_GYM:
    class ActiveSystemIDEnv(gym.Env):
        """
        RL environment for active system identification.

        The agent chooses input signals to maximize information
        gain about unknown system parameters.

        Observation: recent input-output history + current parameter estimates
        Action: next input signal to apply
        Reward: reduction in parameter uncertainty
        """

        metadata = {'render_modes': ['human']}

        def __init__(self, true_params: Optional[Dict] = None):
            super().__init__()

            # True system parameters (for simulation)
            self.true_inertia = np.array([0.012, 0.018, 0.014])
            self.true_damping = np.array([0.08, 0.12, 0.10])

            if true_params:
                self.true_inertia = true_params.get('inertia', self.true_inertia)
                self.true_damping = true_params.get('damping', self.true_damping)

            # System ID module
            self.sysid = GimbalSystemID()

            # Simulation state
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            self.dt = 0.01
            self.time = 0.0

            # Spaces
            # Observation: [position(3), velocity(3), est_inertia(3), est_damping(3), uncertainty(1)]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )

            # Action: torque commands (normalized)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

            self.max_torque = 0.5
            self.max_steps = 500
            self.current_step = 0

            # Track parameter estimates
            self.est_inertia = np.ones(3) * 0.01
            self.est_damping = np.ones(3) * 0.1
            self.prev_uncertainty = float('inf')

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            self.time = 0.0
            self.current_step = 0
            self.sysid = GimbalSystemID()
            self.est_inertia = np.ones(3) * 0.01
            self.est_damping = np.ones(3) * 0.1
            self.prev_uncertainty = float('inf')

            return self._get_obs(), {}

        def step(self, action):
            # Scale action to torque
            torque = action * self.max_torque

            # Simulate true system dynamics
            accel = (torque - self.true_damping * self.velocity) / self.true_inertia
            self.velocity += accel * self.dt
            self.position += self.velocity * self.dt
            self.time += self.dt

            # Add measurement noise
            noisy_velocity = self.velocity + np.random.randn(3) * 0.01
            noisy_position = self.position + np.random.randn(3) * 0.001

            # Record for system ID
            self.sysid.add_sample(torque, noisy_velocity, noisy_position, self.time)

            # Attempt system ID
            model = self.sysid.identify()
            if model:
                self.est_inertia = model.inertia
                self.est_damping = model.damping

            # Compute reward: reduction in parameter error
            inertia_error = np.mean((self.est_inertia - self.true_inertia)**2)
            damping_error = np.mean((self.est_damping - self.true_damping)**2)
            uncertainty = inertia_error + damping_error

            # Reward for reducing uncertainty
            reward = self.prev_uncertainty - uncertainty
            self.prev_uncertainty = uncertainty

            # Bonus for exploration (high angular rates)
            exploration_bonus = 0.01 * np.mean(np.abs(self.velocity))
            reward += exploration_bonus

            self.current_step += 1
            truncated = self.current_step >= self.max_steps
            done = uncertainty < 1e-6  # Converged

            return self._get_obs(), reward, done, truncated, {'uncertainty': uncertainty}

        def _get_obs(self):
            uncertainty = np.mean((self.est_inertia - self.true_inertia)**2) + \
                          np.mean((self.est_damping - self.true_damping)**2)

            return np.concatenate([
                self.position,
                self.velocity,
                self.est_inertia,
                self.est_damping,
                [uncertainty]
            ]).astype(np.float32)


class OnlineSystemID:
    """
    Complete online system identification pipeline.

    Integrates IMU bias, gimbal dynamics, and time delay estimation.
    """

    def __init__(self, sample_rate: float = 200.0):
        self.imu_estimator = IMUBiasEstimator(sample_rate)
        self.gimbal_sysid = GimbalSystemID()
        self.delay_estimator = TimeDelayEstimator()

        self.imu_model: Optional[IMUNoiseModel] = None
        self.gimbal_model: Optional[GimbalDynamicsModel] = None
        self.delay_model: Optional[TimeDelayModel] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_interval = 5.0  # seconds

    def start(self) -> None:
        """Start background identification."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("Online System ID started")

    def stop(self) -> None:
        """Stop identification."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("Online System ID stopped")

    def _update_loop(self) -> None:
        """Periodically update models."""
        while self._running:
            time.sleep(self._update_interval)

            # Update IMU model
            new_imu = self.imu_estimator.estimate_bias()
            if new_imu and (self.imu_model is None or
                           new_imu.confidence > self.imu_model.confidence):
                self.imu_model = new_imu
                print(f"IMU model updated: gyro_bias={new_imu.gyro_bias}")

            # Update gimbal model
            new_gimbal = self.gimbal_sysid.identify()
            if new_gimbal and (self.gimbal_model is None or
                               new_gimbal.prediction_rmse < self.gimbal_model.prediction_rmse):
                self.gimbal_model = new_gimbal
                print(f"Gimbal model updated: inertia={new_gimbal.inertia}")

            # Update delay model
            new_delay = self.delay_estimator.estimate()
            if new_delay:
                self.delay_model = new_delay

    def on_imu_sample(self, gyro: np.ndarray, accel: np.ndarray) -> None:
        """Process IMU sample."""
        self.imu_estimator.add_sample(gyro, accel)

    def on_gimbal_sample(self, torque: np.ndarray, rate: np.ndarray,
                         position: np.ndarray, timestamp: float) -> None:
        """Process gimbal sample."""
        self.gimbal_sysid.add_sample(torque, rate, position, timestamp)

    def on_timing_sample(self, fc_timestamp: float, arrival_time: float) -> None:
        """Process timing sample."""
        self.delay_estimator.add_sample(fc_timestamp, arrival_time)

    def get_status(self) -> dict:
        """Get current identification status."""
        return {
            'imu_model': self.imu_model.to_dict() if self.imu_model else None,
            'gimbal_model': {
                'inertia': self.gimbal_model.inertia.tolist(),
                'damping': self.gimbal_model.damping.tolist(),
                'samples': self.gimbal_model.samples_used,
                'rmse': self.gimbal_model.prediction_rmse
            } if self.gimbal_model else None,
            'delay_model': {
                'mean_ms': self.delay_model.mean_delay * 1000,
                'std_ms': self.delay_model.delay_std * 1000,
                'burst_prob': self.delay_model.burst_probability
            } if self.delay_model else None
        }


# Training function for active system ID
def train_active_sysid(timesteps: int = 50000, save_path: str = "sysid_ppo"):
    """Train active system identification agent."""
    if not HAS_GYM:
        print("gymnasium required")
        return None

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        print("stable-baselines3 required")
        return None

    print("Training active system ID agent...")
    env = make_vec_env(ActiveSystemIDEnv, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)

    return model
