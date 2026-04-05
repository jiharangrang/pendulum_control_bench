import os
import json
import time

import numpy as np
from gymnasium.wrappers import RecordVideo

from envs.wrappers import (
    ActionDisturbance,
    ForceDisturbance,
    TerminationOverride,
    ActuationDelay,
    ObservationNoise,
)
from envs.mujoco_model import make_inverted_pendulum
from controllers.lqr import dlqr
from controllers.mpc import LinearMPC
from controllers.actuator import BoxRateLimiter


ENV_ID = "InvertedPendulum-v5"


def _linearize_fd(env_id: str, seed: int, eps: float):
    env_lin = make_inverted_pendulum()
    env_lin.reset(seed=seed)

    def step_from_state(e, x, u):
        qpos = np.array(x[:2], dtype=np.float64)
        qvel = np.array(x[2:], dtype=np.float64)
        e.unwrapped.set_state(qpos, qvel)
        obs, *_ = e.step(np.array([u], dtype=np.float32))
        return np.array(obs, dtype=np.float64)

    x0 = np.zeros(4, dtype=np.float64)
    u0 = 0.0
    Ad = np.zeros((4, 4), dtype=np.float64)
    Bd = np.zeros((4, 1), dtype=np.float64)

    for i in range(4):
        dx = np.zeros(4, dtype=np.float64)
        dx[i] = eps
        f_p = step_from_state(env_lin, x0 + dx, u0)
        f_m = step_from_state(env_lin, x0 - dx, u0)
        Ad[:, i] = (f_p - f_m) / (2.0 * eps)

    f_p = step_from_state(env_lin, x0, u0 + eps)
    f_m = step_from_state(env_lin, x0, u0 - eps)
    Bd[:, 0] = (f_p - f_m) / (2.0 * eps)

    env_lin.close()
    return Ad, Bd


def _slider_abs_limit(env):
    m = env.unwrapped.model
    try:
        jid = m.joint("slider").id
        return float(np.max(np.abs(m.jnt_range[jid])))
    except Exception:
        return np.inf


def _current_true_obs(env):
    qpos = np.array(env.unwrapped.data.qpos, dtype=np.float64).reshape(-1)
    qvel = np.array(env.unwrapped.data.qvel, dtype=np.float64).reshape(-1)
    return np.array([qpos[0], qpos[1], qvel[0], qvel[1]], dtype=np.float64)


def _obs_from_info_or_true(info, env):
    if isinstance(info, dict) and ("obs_true" in info):
        arr = np.array(info["obs_true"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            return arr[:4]
    return _current_true_obs(env)


def _meas_from_info_or_obs(info, obs):
    if isinstance(info, dict) and ("obs_meas" in info):
        arr = np.array(info["obs_meas"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            return arr[:4]
    return np.array(obs, dtype=np.float64).reshape(-1)[:4]


def _scalar_or_first(v, default=np.nan):
    try:
        arr = np.array(v, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _find_wrapper_instance(env, wrapper_type):
    cur = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None


def run_once(cfg: dict):
    seed = int(cfg["seed"])
    steps = int(cfg["steps"])
    video = bool(cfg["video"])
    video_full_length = bool(cfg.get("video_full_length", False)) and video
    video_tail_steps_cfg = cfg.get("video_tail_steps", None)
    post_term_action_mode = str(cfg.get("post_term_action_mode", "zero")).lower()
    if post_term_action_mode not in ("zero", "last"):
        raise ValueError("post_term_action_mode must be one of: 'zero', 'last'")
    delay_steps = max(int(cfg.get("actuation_delay_steps", 0)), 0)
    scfg = cfg.get("sensor_noise", {})
    sigma_x = float(scfg.get("x", 0.0))
    sigma_theta = float(scfg.get("theta", 0.0))
    sigma_xdot = float(scfg.get("xdot", 0.0))
    sigma_thetadot = float(scfg.get("thetadot", 0.0))

    make_kwargs = {}
    if video:
        # Wider default framing for clearer failure dynamics in saved videos.
        make_kwargs["default_camera_config"] = {"distance": 4.0}
    env = make_inverted_pendulum(render_mode="rgb_array" if video else None, **make_kwargs)
    x_limit_model = _slider_abs_limit(env)
    x_limit_cfg = float(cfg.get("termination_x_limit", np.nan))
    x_limit_term = x_limit_model if np.isnan(x_limit_cfg) else x_limit_cfg
    env = TerminationOverride(
        env,
        theta_limit=float(cfg.get("termination_theta", 0.5)),
        x_limit=float(x_limit_term),
    )

    controller_name = cfg.get("controller", "lqr_fd")
    if controller_name not in ("lqr_fd", "mpc_fd"):
        raise ValueError("controller must be one of: lqr_fd, mpc_fd")

    Q = np.diag(cfg["lqr"]["Q_diag"]).astype(np.float64)
    R = np.array([[cfg["lqr"]["R"]]], dtype=np.float64)
    eps = float(cfg["lqr"]["fd_eps"])
    Ad, Bd = _linearize_fd(ENV_ID, seed=seed, eps=eps)

    # Shared physical actuator limits (applied equally to both controllers).
    acfg = cfg.get("actuator", {})
    u_hw_min = float(env.action_space.low[0])
    u_hw_max = float(env.action_space.high[0])

    u_req_max = float(acfg.get("u_max", u_hw_max))
    if "u_min" in acfg:
        u_req_min = float(acfg["u_min"])
    else:
        u_req_min = -u_req_max

    u_act_max = min(u_hw_max, u_req_max)
    u_act_min = max(u_hw_min, u_req_min)
    if u_act_min > u_act_max:
        raise ValueError(f"Invalid actuator bounds after clipping to hardware: [{u_act_min}, {u_act_max}]")

    du_act_max = float(acfg.get("du_max", np.nan))
    limiter = BoxRateLimiter(u_max=[u_act_max], u_min=[u_act_min], du_max=du_act_max)

    K = None
    mpc = None
    mpc_state_constraint = "none"
    mpc_x_margin = np.nan
    mpc_x_limit_eff = np.nan
    mpc_du_limit_eff = np.nan
    if controller_name == "lqr_fd":
        K, _ = dlqr(Ad, Bd, Q, R)
    else:
        mcfg = cfg.get("mpc", {})

        # MPC input/rate constraints default to the same actuator limits.
        mpc_du_max_cfg = float(mcfg.get("du_max", np.nan))
        mpc_du_max = float(du_act_max) if np.isnan(mpc_du_max_cfg) else mpc_du_max_cfg
        mpc_x_margin = float(mcfg.get("x_margin", 0.0))
        mpc_x_index = int(mcfg.get("x_index", 0))
        state_constraint = str(mcfg.get("state_constraint", "x")).lower()
        mpc_state_constraint = state_constraint
        mpc_du_limit_eff = mpc_du_max

        x_mpc_max = None
        x_mpc_min = None
        if state_constraint in ("x", "state", "position"):
            x_ref_cfg = float(mcfg.get("x_max", np.nan))
            x_ref = float(x_limit_term) if np.isnan(x_ref_cfg) else x_ref_cfg
            if np.isfinite(x_ref):
                x_ref = max(0.0, x_ref - max(0.0, mpc_x_margin))
                x_mpc_max = x_ref
                x_mpc_min = -x_ref
                mpc_x_limit_eff = x_ref

        mpc = LinearMPC(
            Ad=Ad,
            Bd=Bd,
            Q=Q,
            R=R,
            horizon=int(mcfg.get("horizon", 20)),
            u_max=mcfg.get("u_max", u_act_max),
            u_min=mcfg.get("u_min", u_act_min),
            du_max=mpc_du_max,
            x_max=x_mpc_max,
            x_min=x_mpc_min,
            x_index=mpc_x_index,
            solver=str(mcfg.get("solver", "auto")),
            eps_abs=float(mcfg.get("eps_abs", 1e-5)),
            eps_rel=float(mcfg.get("eps_rel", 1e-5)),
            max_iter=int(mcfg.get("max_iter", 4000)),
        )

    if cfg["disturbance"]["enabled"]:
        dcfg = cfg["disturbance"]
        dist_target = dcfg.get("target", cfg.get("disturbance_target", "force"))
        if dist_target == "force":
            fixed_sign_raw = dcfg.get("sign", None)
            fixed_sign = None
            if fixed_sign_raw is not None:
                fs = float(fixed_sign_raw)
                if np.isfinite(fs) and fs != 0.0:
                    fixed_sign = fs
            env = ForceDisturbance(
                env,
                amp=float(dcfg["amp"]),
                seed=seed,
                kind=dcfg["kind"],
                t0=int(dcfg["t0"]),
                duration=int(dcfg["duration"]),
                joint_name="slider",
                random_sign=bool(dcfg.get("random_sign", True)),
                fixed_sign=fixed_sign,
            )
        else:
            env = ActionDisturbance(
                env,
                kind=dcfg["kind"],
                amp=float(dcfg["amp"]),
                t0=int(dcfg["t0"]),
                duration=int(dcfg["duration"]),
                omega=float(dcfg["omega"]),
                seed=seed,
            )

    # Environment-side realism modules:
    # - actuation delay: u_applied(k)=u_cmd(k-d)
    # - sensor noise: controller receives noisy observations
    env = ActuationDelay(env, delay_steps=delay_steps, u_init=float(acfg.get("u_init", 0.0)))
    env = ObservationNoise(
        env,
        seed=seed,
        sigma_x=sigma_x,
        sigma_theta=sigma_theta,
        sigma_xdot=sigma_xdot,
        sigma_thetadot=sigma_thetadot,
    )

    if video:
        env = RecordVideo(
            env,
            video_folder=cfg["video_dir"],
            episode_trigger=lambda ep: ep == 0,
            name_prefix=cfg["run_name"],
        )

    if video_full_length:
        if video_tail_steps_cfg is None:
            video_tail_steps = int(max(steps, 0))
        else:
            video_tail_steps = max(int(video_tail_steps_cfg), 0)
    else:
        video_tail_steps = 0

    try:
        obs, info = env.reset(seed=seed)
        obs_meas = _meas_from_info_or_obs(info, obs)
        obs_true = _obs_from_info_or_true(info, env)

        init = cfg.get("init", {})
        if init:
            qpos = env.unwrapped.data.qpos.copy()
            qvel = env.unwrapped.data.qvel.copy()
            if "x" in init:
                qpos[0] = float(init["x"])
            if "theta" in init:
                qpos[1] = float(init["theta"])
            if "xdot" in init:
                qvel[0] = float(init["xdot"])
            if "thetadot" in init:
                qvel[1] = float(init["thetadot"])
            env.unwrapped.set_state(qpos, qvel)
            obs_true = np.concatenate([qpos, qvel]).astype(np.float64)
            noise_wrapper = _find_wrapper_instance(env, ObservationNoise)
            if noise_wrapper is not None:
                obs_meas = np.array(noise_wrapper.observe_true(obs_true), dtype=np.float64).reshape(-1)
            else:
                obs_meas = obs_true.copy()

        u_max = float(u_act_max)
        u_max_obs = 0.0
        sat_steps = 0
        sat_any = False
        u_prev_cmd = float(np.clip(float(acfg.get("u_init", 0.0)), u_act_min, u_act_max))
        u_prev_applied = float(u_prev_cmd)
        k_term_first = None
        postterm_solver_fail = 0

        log = {
            "obs0": np.array(obs_true, dtype=float).tolist(),
            "obs_meas0": np.array(obs_meas, dtype=float).tolist(),
            "t": [],
            "obs": [],
            "state_true": [],
            "state_meas": [],
            "u_raw": [],
            "u_cmd": [],
            "u_applied": [],
            "du_cmd": [],
            "du_applied": [],
            "action": [],
            "disturb_force": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
        }

        for k in range(steps):
            x_meas_k = np.array(obs_meas, dtype=np.float64).reshape(4, 1)
            x_true_k = np.array(obs_true, dtype=np.float64).reshape(4)

            in_post_term_tail = (k_term_first is not None) and video_full_length and (k > k_term_first)
            if in_post_term_tail:
                u_raw = float(u_prev_cmd) if post_term_action_mode == "last" else 0.0
            else:
                try:
                    if controller_name == "lqr_fd":
                        u_raw = -float((K @ x_meas_k).item())
                    else:
                        u_raw = float(mpc.control(x_meas_k.reshape(-1), u_prev=[u_prev_cmd])[0])
                except RuntimeError:
                    if (k_term_first is not None) and video_full_length:
                        postterm_solver_fail = 1
                        u_raw = float(u_prev_cmd) if post_term_action_mode == "last" else 0.0
                    else:
                        raise

            # u_cmd is controller output after shared actuator constraints.
            u_cmd = float(limiter.project([u_raw], [u_prev_cmd])[0])
            du_cmd = float(u_cmd - u_prev_cmd)
            u_prev_cmd = u_cmd
            action = np.array([u_cmd], dtype=np.float32)

            obs_next, r, terminated, truncated, info = env.step(action)
            dist_f = float(info.get("disturb_force", 0.0))
            u_applied = _scalar_or_first(info.get("u_applied", u_cmd), default=u_cmd)
            du_applied = _scalar_or_first(info.get("du_applied", (u_applied - u_prev_applied)), default=(u_applied - u_prev_applied))
            u_prev_applied = float(u_applied)

            u_abs = abs(float(u_applied))
            u_max_obs = max(u_max_obs, u_abs)
            sat = bool((u_applied >= (u_act_max - 1e-9)) or (u_applied <= (u_act_min + 1e-9)))
            sat_steps += int(sat)
            sat_any = sat_any or sat

            obs_meas_next = _meas_from_info_or_obs(info, obs_next)
            obs_true_next = _obs_from_info_or_true(info, env)

            log["t"].append(k)
            log["obs"].append(np.array(obs_true_next, dtype=float).tolist())
            log["state_true"].append(np.array(x_true_k, dtype=float).tolist())
            log["state_meas"].append(np.array(x_meas_k, dtype=float).reshape(-1).tolist())
            log["u_raw"].append(float(u_raw))
            log["u_cmd"].append(float(u_cmd))
            log["u_applied"].append(float(u_applied))
            log["du_cmd"].append(float(du_cmd))
            log["du_applied"].append(float(du_applied))
            log["action"].append(np.array(action, dtype=float).tolist())
            log["disturb_force"].append(dist_f)
            log["reward"].append(float(r))
            log["terminated"].append(bool(terminated))
            log["truncated"].append(bool(truncated))

            obs_meas = obs_meas_next
            obs_true = obs_true_next

            if (terminated or truncated) and (k_term_first is None):
                k_term_first = int(k)

            if k_term_first is not None:
                if not video_full_length:
                    break
                if (k - k_term_first) >= video_tail_steps:
                    break

        T_run = len(log["action"])
        log["u_max_obs"] = float(u_max_obs)
        log["sat_steps"] = int(sat_steps)
        log["sat_any"] = int(sat_any)
        log["sat_rate"] = float(sat_steps / T_run) if T_run > 0 else 0.0
        log["actuator_u_min"] = float(u_act_min)
        log["actuator_u_max"] = float(u_act_max)
        log["actuator_du_max"] = float(du_act_max)
        log["termination_x_limit"] = float(x_limit_term)
        log["mpc_state_constraint"] = str(mpc_state_constraint)
        log["mpc_x_margin"] = float(mpc_x_margin)
        log["mpc_x_limit_eff"] = float(mpc_x_limit_eff)
        log["mpc_du_limit_eff"] = float(mpc_du_limit_eff)
        log["actuation_delay_steps"] = int(delay_steps)
        log["sensor_noise_x"] = float(sigma_x)
        log["sensor_noise_theta"] = float(sigma_theta)
        log["sensor_noise_xdot"] = float(sigma_xdot)
        log["sensor_noise_thetadot"] = float(sigma_thetadot)
        log["k_term_first"] = float(k_term_first) if k_term_first is not None else np.nan
        log["postterm_solver_fail"] = int(postterm_solver_fail)
        log["video_tail_steps"] = int(video_tail_steps)
        log["post_term_action_mode"] = str(post_term_action_mode)

        return log
    finally:
        env.close()


def main():
    cfg = {
        "run_name": f"fd_compare_debug_{int(time.time())}",
        "seed": 0,
        "steps": 1000,
        "video": False,
        "video_dir": "videos/fd_compare",
        "log_dir": "logs/fd_compare",
        "disturbance_target": "force",
        "controller": "mpc_fd",
        "actuation_delay_steps": 0,
        "sensor_noise": {"x": 0.0, "theta": 0.0, "xdot": 0.0, "thetadot": 0.0},
        "lqr": {"Q_diag": [1.0, 80.0, 1.0, 10.0], "R": 0.1, "fd_eps": 1e-5},
        "actuator": {"du_max": np.nan, "u_init": 0.0},
        "mpc": {"horizon": 20, "solver": "auto", "state_constraint": "x", "x_margin": 0.0},
        "disturbance": {
            "enabled": True,
            "kind": "window",
            "amp": 280.0,
            "t0": 100,
            "duration": 25,
            "omega": 0.05,
        },
    }

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["video_dir"], exist_ok=True)
    log = run_once(cfg)
    outpath = os.path.join(cfg["log_dir"], cfg["run_name"] + ".json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "log": log}, f, indent=2)
    print("OK:", outpath)


if __name__ == "__main__":
    main()
