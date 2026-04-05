import numpy as np
import gymnasium as gym
from collections import deque

class ActionDisturbance(gym.Wrapper):
    """
    u' = clip(u + d(t), u_min, u_max)
    - impulse / step / sine disturbance on action
    """
    def __init__(self, env, kind="impulse", amp=0.0, t0=200, duration=50, omega=0.05, seed=0):
        super().__init__(env)
        self.kind = kind
        self.amp = float(amp)
        self.t0 = int(t0)
        self.duration = int(duration)
        self.omega = float(omega)
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.k = 0
        return self.env.reset(**kwargs)

    def _disturbance(self):
        if self.amp == 0.0:
            return 0.0

        if self.kind == "impulse":
            return self.amp if self.k == self.t0 else 0.0

        if self.kind == "step":
            return self.amp if self.k >= self.t0 else 0.0

        if self.kind == "window":
            return self.amp if (self.t0 <= self.k < self.t0 + self.duration) else 0.0

        if self.kind == "sine":
            return self.amp * np.sin(self.omega * self.k)

        raise ValueError(f"Unknown kind: {self.kind}")

    def step(self, action):
        d = self._disturbance()

        # action이 (n,) 형태일 수 있으니 브로드캐스트 안전하게
        a = np.array(action, dtype=np.float32)
        a = a + d

        # action space 범위로 clip
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self.k += 1
        return self.env.step(a)


class ActuationDelay(gym.Wrapper):
    """
    Apply fixed-step command delay:
      u_applied(k) = u_cmd(k - d)
    """

    def __init__(self, env, delay_steps: int = 0, u_init=0.0):
        super().__init__(env)
        self.delay_steps = max(int(delay_steps), 0)
        self._u_init = np.array(u_init, dtype=np.float32)
        self._queue = deque()
        self._prev_applied = None

    def _clip_action(self, action):
        a = np.array(action, dtype=np.float32).reshape(self.action_space.shape)
        return np.clip(a, self.action_space.low, self.action_space.high)

    @staticmethod
    def _to_info_value(a: np.ndarray):
        v = np.array(a, dtype=np.float64).reshape(-1)
        return float(v[0]) if v.size == 1 else v.tolist()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._queue.clear()

        u0 = self._clip_action(self._u_init if self._u_init.size > 1 else np.array([float(self._u_init)], dtype=np.float32))
        for _ in range(self.delay_steps):
            self._queue.append(u0.copy())
        self._prev_applied = u0.copy()

        info = dict(info)
        info["delay_steps"] = int(self.delay_steps)
        return obs, info

    def step(self, action):
        u_cmd = self._clip_action(action)
        if self.delay_steps > 0:
            self._queue.append(u_cmd.copy())
            u_applied = self._queue.popleft().copy()
        else:
            u_applied = u_cmd.copy()

        du_applied = u_applied - self._prev_applied
        obs, reward, terminated, truncated, info = self.env.step(u_applied)
        self._prev_applied = u_applied.copy()

        info = dict(info)
        info["u_cmd"] = self._to_info_value(u_cmd)
        info["u_applied"] = self._to_info_value(u_applied)
        info["du_applied"] = self._to_info_value(du_applied)
        info["delay_steps"] = int(self.delay_steps)
        return obs, reward, terminated, truncated, info


class ObservationNoise(gym.Wrapper):
    """
    Add zero-mean Gaussian observation noise to controller-facing observations.
    Termination/dynamics remain based on the true state from inner env.
    """

    def __init__(
        self,
        env,
        seed: int = 0,
        sigma_x: float = 0.0,
        sigma_theta: float = 0.0,
        sigma_xdot: float = 0.0,
        sigma_thetadot: float = 0.0,
    ):
        super().__init__(env)
        self.rng = np.random.default_rng(int(seed))
        self.sigma = np.array([sigma_x, sigma_theta, sigma_xdot, sigma_thetadot], dtype=np.float64)

    def _noise(self, n: int):
        s = self.sigma[:n]
        if np.all(s <= 0.0):
            return np.zeros((n,), dtype=np.float64)
        return self.rng.normal(loc=0.0, scale=s, size=(n,))

    def _apply(self, obs):
        obs_in = np.array(obs)
        obs_true = np.array(obs_in, dtype=np.float64).reshape(-1)
        noise = self._noise(obs_true.size)
        obs_meas = obs_true + noise
        return obs_true, obs_meas, noise, obs_in.dtype

    def observe_true(self, obs_true):
        obs_true = np.array(obs_true, dtype=np.float64).reshape(-1)
        noise = self._noise(obs_true.size)
        return obs_true + noise

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_true, obs_meas, noise, obs_dtype = self._apply(obs)
        info = dict(info)
        info["obs_true"] = obs_true.tolist()
        info["obs_meas"] = obs_meas.tolist()
        info["obs_noise"] = noise.tolist()
        return obs_meas.astype(obs_dtype, copy=False), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_true, obs_meas, noise, obs_dtype = self._apply(obs)
        info = dict(info)
        info["obs_true"] = obs_true.tolist()
        info["obs_meas"] = obs_meas.tolist()
        info["obs_noise"] = noise.tolist()
        return obs_meas.astype(obs_dtype, copy=False), reward, terminated, truncated, info


class TerminationOverride(gym.Wrapper):
    """
    Override termination condition based on configurable state limits.
    """
    def __init__(self, env, theta_limit: float = 0.5, x_limit: float = np.inf):
        super().__init__(env)
        self.theta_limit = float(theta_limit)
        self.x_limit = float(x_limit)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Recompute termination from explicit limits to keep evaluation policy-consistent.
        x = float(obs[0])
        theta = float(obs[1])
        theta_fail = abs(theta) > self.theta_limit
        x_fail = np.isfinite(self.x_limit) and (abs(x) >= self.x_limit)
        terminated = bool((not np.isfinite(obs).all()) or theta_fail or x_fail)
        reward = int(not terminated)

        info = dict(info)
        info["terminated_theta_limit"] = self.theta_limit
        info["terminated_x_limit"] = self.x_limit
        info["terminated_by_theta"] = bool(theta_fail)
        info["terminated_by_x"] = bool(x_fail)
        return obs, reward, terminated, truncated, info


class ForceDisturbance(gym.Wrapper):
    """
    Apply an external generalized force on a given joint DOF (default: cart slider).
    This bypasses action ctrlrange clipping, so amplitude scaling stays meaningful.
    """

    def __init__(
        self,
        env,
        amp: float,
        seed: int = 0,
        kind: str = "impulse",      # "impulse"|"pulse"|"window"
        t0: int = 200,
        duration: int = 1,          # used for "window"
        joint_name: str = "slider",
        random_sign: bool = True,   # sign picked once per episode (reproducible with seed)
        fixed_sign: float | None = None,  # if set, overrides random_sign
    ):
        super().__init__(env)
        self.amp = float(amp)
        self.seed = int(seed)
        self.kind = str(kind)
        self.t0 = int(t0)
        self.duration = int(duration)
        self.joint_name = str(joint_name)
        self.random_sign = bool(random_sign)
        self.fixed_sign = None
        if fixed_sign is not None:
            fs = float(fixed_sign)
            if (not np.isfinite(fs)) or (fs == 0.0):
                raise ValueError("fixed_sign must be a finite non-zero value when provided")
            self.fixed_sign = 1.0 if fs > 0.0 else -1.0

        self._rng = np.random.default_rng(self.seed)
        self._k = 0
        self._dof = None
        self._sign = 1.0

    def _resolve_dof(self):
        # MuJoCo: joint id -> dof address
        m = self.env.unwrapped.model
        try:
            jid = m.joint(self.joint_name).id
        except Exception:
            # fallback: name2id
            jid = m.name2id(self.joint_name, "joint")
        dof = int(m.jnt_dofadr[jid])
        self._dof = dof

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._k = 0

        if self._dof is None:
            self._resolve_dof()

        # clear any leftover applied force
        self.env.unwrapped.data.qfrc_applied[:] = 0.0

        # choose sign once per episode (reproducible)
        if self.fixed_sign is not None:
            self._sign = float(self.fixed_sign)
        elif self.random_sign:
            self._sign = float(self._rng.choice([-1.0, 1.0]))
        else:
            self._sign = 1.0

        return obs, info

    def _force_at_step(self, k: int) -> float:
        if self.amp <= 0.0:
            return 0.0

        if self.kind in ("impulse", "pulse"):
            return self._sign * self.amp if k == self.t0 else 0.0

        if self.kind == "window":
            if self.t0 <= k < self.t0 + self.duration:
                return self._sign * self.amp
            return 0.0

        # default: no force
        return 0.0

    def step(self, action):
        f = self._force_at_step(self._k)

        # apply external generalized force to the cart slider DOF
        self.env.unwrapped.data.qfrc_applied[self._dof] = f

        obs, reward, terminated, truncated, info = self.env.step(action)

        # clear after stepping (prevents accidental persistence)
        self.env.unwrapped.data.qfrc_applied[self._dof] = 0.0

        info = dict(info)
        info["disturb_force"] = float(f)
        info["disturb_step"] = int(self._k)
        info["disturb_sign"] = float(self._sign)

        self._k += 1
        return obs, reward, terminated, truncated, info
