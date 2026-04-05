import os
import io
import csv
import time
import argparse
from contextlib import redirect_stdout

import numpy as np

from experiments.fd_compare import run_fd_compare as run_mod
from envs.mujoco_model import make_inverted_pendulum


def compute_recovery_time(theta, theta_dot, post_start_idx, dt, theta_tol=0.05236, theta_dot_tol=0.1, hold_steps=5):
    theta = np.asarray(theta)
    theta_dot = np.asarray(theta_dot)
    if post_start_idx >= len(theta):
        return np.nan
    in_band = (np.abs(theta) <= theta_tol) & (np.abs(theta_dot) <= theta_dot_tol)
    post = in_band[post_start_idx:]
    if np.all(post):
        return np.nan
    for i in range(0, len(post) - hold_steps + 1):
        if np.all(post[i:i + hold_steps]):
            return i * dt
    return (len(post) - 1) * dt


def _align_state_with_action(log: dict, T: int):
    obs = np.array(log.get("obs", []), dtype=np.float64)
    if T == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if "obs0" in log and obs.shape[0] >= max(T - 1, 0):
        x0 = np.array(log["obs0"], dtype=np.float64).reshape(1, 4)
        return np.vstack([x0, obs[: max(T - 1, 0)]])
    if obs.shape[0] == T + 1:
        return obs[:-1]
    return obs[:T]


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for s in mask.astype(bool).tolist():
        if s:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _first_true_index(mask: np.ndarray):
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size > 0 else None


def _first_index_consecutive_true(mask: np.ndarray, hold_steps: int):
    hold_steps = max(int(hold_steps), 1)
    run = 0
    for i, v in enumerate(mask.astype(bool).tolist()):
        if v:
            run += 1
            if run >= hold_steps:
                return int(i)
        else:
            run = 0
    return None


def _theta_max_after_first_zero_crossing(theta: np.ndarray, start_idx: int):
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    n = int(theta.shape[0])
    if n <= 1:
        return np.nan
    s = max(int(start_idx), 0)
    if s >= n - 1:
        return np.nan

    z_idx = None
    k0 = max(1, s)
    for k in range(k0, n):
        a = float(theta[k - 1])
        b = float(theta[k])
        if (a == 0.0) or (b == 0.0) or ((a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0)):
            z_idx = int(k)
            break

    if z_idx is None:
        return np.nan
    tail = theta[z_idx:]
    if tail.size == 0:
        return np.nan
    return float(np.max(np.abs(tail)))


def compute_metrics_from_log(
    log: dict,
    u_max: float,
    dt: float,
    steps_cfg: int,
    t0: int,
    duration: int,
    kind: str,
    Q_diag,
    R: float,
    eps_theta: float = 0.05236,
    eps_omega: float = 0.1,
    settle_hold_sec: float = 0.5,
    sat_tol: float = 0.01,
    du_max: float = np.nan,
    du_tol: float = 1e-6,
    theta_limit: float = 0.5,
    x_fail_limit: float = np.inf,
    x_fail_eps: float = 0.01,
    x_fail_hold: int = 3,
):
    obs_post_full = np.array(log.get("obs", []), dtype=np.float64)
    if "u_applied" in log:
        u_full = np.array(log["u_applied"], dtype=np.float64).reshape(-1)
    else:
        u_full = np.array(log.get("action", []), dtype=np.float64).reshape(-1)

    T_full = int(u_full.shape[0])
    x_full = _align_state_with_action(log, T_full)
    terminated_full = np.array(log.get("terminated", [False] * T_full), dtype=bool).reshape(-1)
    truncated_full = np.array(log.get("truncated", [False] * T_full), dtype=bool).reshape(-1)

    # Metrics should be computed on the same effective horizon as standard run:
    # stop at the first terminated/truncated step, even if video_full_length kept recording.
    n0 = min(T_full, x_full.shape[0], obs_post_full.shape[0], terminated_full.shape[0], truncated_full.shape[0])
    u_full = u_full[:n0]
    x_full = x_full[:n0]
    obs_post_full = obs_post_full[:n0]
    terminated_full = terminated_full[:n0]
    truncated_full = truncated_full[:n0]

    obs = obs_post_full
    theta_obs = obs[:, 1] if obs.size else np.array([], dtype=np.float64)
    x_obs = obs[:, 0] if obs.size else np.array([], dtype=np.float64)

    nan_mask = ~np.isfinite(obs).all(axis=1) if obs.size else np.array([], dtype=bool)
    theta_fail_mask = (np.abs(theta_obs) >= float(theta_limit)) if np.isfinite(theta_limit) and theta_obs.size > 0 else np.zeros((n0,), dtype=bool)
    x_near_mask = (
        np.abs(x_obs) >= (float(x_fail_limit) - float(x_fail_eps))
        if np.isfinite(x_fail_limit) and x_obs.size > 0
        else np.zeros((n0,), dtype=bool)
    )

    idx_nan = _first_true_index(nan_mask)
    idx_theta = _first_true_index(theta_fail_mask)
    idx_x = _first_index_consecutive_true(x_near_mask, hold_steps=x_fail_hold) if np.isfinite(x_fail_limit) else None
    idx_env_term = _first_true_index(terminated_full)
    idx_env_trunc = _first_true_index(truncated_full)

    fail_candidates = []
    if idx_nan is not None:
        fail_candidates.append(("nan_fail", idx_nan))
    if idx_theta is not None:
        fail_candidates.append(("theta_fail", idx_theta))
    if idx_x is not None:
        fail_candidates.append(("x_fail", idx_x))
    # Keep environment termination as fallback fail event when no explicit state event was detected.
    if idx_env_term is not None:
        fail_candidates.append(("other_fail", idx_env_term))

    if fail_candidates:
        prio = {"nan_fail": 0, "theta_fail": 1, "x_fail": 2, "other_fail": 3}
        fail_reason, fail_idx = sorted(fail_candidates, key=lambda t: (t[1], prio.get(t[0], 99)))[0]
    else:
        fail_reason, fail_idx = ("time", None)

    end_candidates = []
    if fail_idx is not None:
        end_candidates.append(fail_idx)
    if idx_env_trunc is not None:
        end_candidates.append(idx_env_trunc)
    if end_candidates:
        T = int(min(end_candidates) + 1)
    else:
        T = int(n0)

    u = u_full[:T]
    x = x_full[:T]
    obs_post = obs_post_full[:T]
    terminated = terminated_full[:T]
    truncated = truncated_full[:T]

    theta = x[:, 1] if x.size else np.array([], dtype=np.float64)
    omega = x[:, 3] if x.size else np.array([], dtype=np.float64)
    xpos = x[:, 0] if x.size else np.array([], dtype=np.float64)

    terminated_any = bool(fail_idx is not None)
    success = not terminated_any
    term_step = int(T)
    time_to_term = float(term_step * dt)

    term_reason = str(fail_reason)
    # Event flags are tracked independently (not mutually exclusive).
    # This lets plots show overlap cases (e.g., x and theta both violated).
    fail_theta = int(idx_theta is not None)
    fail_x = int(idx_x is not None)
    fail_nan = int(idx_nan is not None)
    fail_other = int((idx_env_term is not None) and (idx_nan is None) and (idx_theta is None) and (idx_x is None))

    sat_thr = max(float(u_max) - float(sat_tol), 0.0)
    sat_mask = (np.abs(u) >= sat_thr)
    sat_steps = int(sat_mask.sum())
    act_rate_u = float(sat_steps / T) if T > 0 else np.nan
    max_run_u = _max_consecutive_true(sat_mask)

    if T >= 2 and np.isfinite(du_max) and du_max > 0.0:
        du = np.diff(u)
        du_thr = max(float(du_max) - float(du_tol), 0.0)
        du_mask = (np.abs(du) >= du_thr)
        act_rate_du = float(np.mean(du_mask)) if du_mask.size > 0 else np.nan
        max_run_du = _max_consecutive_true(du_mask) if du_mask.size > 0 else np.nan
    else:
        act_rate_du = np.nan
        max_run_du = np.nan

    if np.isfinite(theta_limit) and theta.size > 0:
        min_margin_theta = float(np.min(float(theta_limit) - np.abs(theta)))
    else:
        min_margin_theta = np.nan

    if np.isfinite(x_fail_limit) and xpos.size > 0:
        min_margin_x = float(np.min(float(x_fail_limit) - np.abs(xpos)))
    else:
        min_margin_x = np.nan

    Q = np.diag(np.asarray(Q_diag, dtype=np.float64))
    Ru = float(R)
    if T > 0 and x.shape[0] == T:
        x_cost = np.einsum("bi,ij,bj->b", x, Q, x)
        u_cost = (u ** 2) * Ru
        stage = x_cost + u_cost
        J_emp_sum = float(np.sum(stage))
        J_emp_mean = float(np.mean(stage))
        u_energy_sum = float(np.sum(u ** 2))
        u_energy_mean = float(np.mean(u ** 2))
    else:
        J_emp_sum = np.nan
        J_emp_mean = np.nan
        u_energy_sum = np.nan
        u_energy_mean = np.nan

    duration_eff = 1 if kind in ("impulse", "pulse") else int(duration)
    post_start_idx = max(int(t0 + duration_eff), 0)
    n_steps_post = max(T - post_start_idx, 0)

    post_theta = theta[post_start_idx:] if post_start_idx < len(theta) else np.array([], dtype=np.float64)
    if post_theta.size > 0:
        theta_max_post_all = float(np.max(np.abs(post_theta)))
        theta_rms_post_all = float(np.sqrt(np.mean(post_theta ** 2)))
    elif theta.size > 0:
        theta_max_post_all = float(np.max(np.abs(theta)))
        theta_rms_post_all = float(np.sqrt(np.mean(theta ** 2)))
    else:
        theta_max_post_all = np.nan
        theta_rms_post_all = np.nan

    theta_max_post_succ = theta_max_post_all if success else np.nan
    theta_rms_post_succ = theta_rms_post_all if success else np.nan
    theta_max_post_zero_succ = _theta_max_after_first_zero_crossing(theta, post_start_idx) if success else np.nan

    hold_N = max(1, int(round(settle_hold_sec / dt)))
    recovery_time_succ = compute_recovery_time(
        theta,
        omega,
        post_start_idx,
        dt,
        theta_tol=eps_theta,
        theta_dot_tol=eps_omega,
        hold_steps=hold_N,
    )
    if not success:
        recovery_time_succ = np.nan

    return dict(
        T=T,
        success=int(success),
        terminated_any=int(terminated_any),
        term_reason=term_reason,
        fail_theta=fail_theta,
        fail_x=fail_x,
        fail_nan=fail_nan,
        fail_other=fail_other,
        time_to_term=time_to_term,
        n_steps_post=n_steps_post,
        act_rate_u=act_rate_u,
        sat_rate=act_rate_u,
        sat_steps=sat_steps,
        max_run_u=max_run_u,
        sat_max_run=max_run_u,
        act_rate_du=act_rate_du,
        max_run_du=max_run_du,
        min_margin_theta=min_margin_theta,
        min_margin_x=min_margin_x,
        theta_max_post_all=theta_max_post_all,
        theta_rms_post_all=theta_rms_post_all,
        theta_max_post_succ=theta_max_post_succ,
        theta_max_post_zero_succ=theta_max_post_zero_succ,
        theta_rms_post_succ=theta_rms_post_succ,
        theta_max_post=theta_max_post_succ,
        theta_rms_post=theta_rms_post_succ,
        recovery_time_succ=recovery_time_succ,
        recovery_time=recovery_time_succ,
        u_energy_sum=u_energy_sum,
        u_energy_mean=u_energy_mean,
        u_energy=u_energy_sum,
        J_emp_sum=J_emp_sum,
        J_emp_mean=J_emp_mean,
        J_emp=J_emp_sum,
    )


def make_solver_fail_metrics(term_reason: str = "solver_fail_preterm"):
    return dict(
        T=0,
        success=0,
        terminated_any=1,
        term_reason=str(term_reason),
        fail_theta=0,
        fail_x=0,
        fail_nan=0,
        fail_other=1,
        time_to_term=0.0,
        n_steps_post=0,
        act_rate_u=np.nan,
        sat_rate=np.nan,
        sat_steps=0,
        max_run_u=0,
        sat_max_run=0,
        act_rate_du=np.nan,
        max_run_du=np.nan,
        min_margin_theta=np.nan,
        min_margin_x=np.nan,
        theta_max_post_all=np.nan,
        theta_rms_post_all=np.nan,
        theta_max_post_succ=np.nan,
        theta_max_post_zero_succ=np.nan,
        theta_rms_post_succ=np.nan,
        theta_max_post=np.nan,
        theta_rms_post=np.nan,
        recovery_time_succ=np.nan,
        recovery_time=np.nan,
        u_energy_sum=np.nan,
        u_energy_mean=np.nan,
        u_energy=np.nan,
        J_emp_sum=np.nan,
        J_emp_mean=np.nan,
        J_emp=np.nan,
    )


def save_step_log_csv(log: dict, out_csv: str, dt: float, term_reason: str = ""):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    T = int(len(log.get("t", [])))
    x_true = np.array(log.get("state_true", []), dtype=np.float64)
    if x_true.shape[0] == 0:
        x_true = _align_state_with_action(log, T)
    x_meas = np.array(log.get("state_meas", []), dtype=np.float64)
    if x_meas.shape[0] == 0:
        x_meas = x_true.copy()

    u_raw = np.array(log.get("u_raw", [np.nan] * T), dtype=np.float64).reshape(-1)
    u_cmd = np.array(log.get("u_cmd", log.get("action", [np.nan] * T)), dtype=np.float64).reshape(-1)
    if "u_applied" in log:
        u_applied = np.array(log["u_applied"], dtype=np.float64).reshape(-1)
    else:
        u_applied = np.array(log.get("action", [np.nan] * T), dtype=np.float64).reshape(-1)
    du_cmd = np.array(log.get("du_cmd", [np.nan] * T), dtype=np.float64).reshape(-1)
    du_applied = np.array(log.get("du_applied", [np.nan] * T), dtype=np.float64).reshape(-1)
    disturb = np.array(log.get("disturb_force", [0.0] * T), dtype=np.float64).reshape(-1)
    reward = np.array(log.get("reward", [np.nan] * T), dtype=np.float64).reshape(-1)
    terminated = np.array(log.get("terminated", [False] * T), dtype=bool).reshape(-1)
    truncated = np.array(log.get("truncated", [False] * T), dtype=bool).reshape(-1)

    fieldnames = [
        "k",
        "t_sec",
        "x_true",
        "theta_true",
        "xdot_true",
        "thetadot_true",
        "x_meas",
        "theta_meas",
        "xdot_meas",
        "thetadot_meas",
        "term_reason",
        "x",
        "theta",
        "xdot",
        "thetadot",
        "u_raw",
        "u_cmd",
        "u_applied",
        "du_cmd",
        "du_applied",
        "disturb_force",
        "reward",
        "terminated",
        "truncated",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n = min(
            T,
            x_true.shape[0],
            x_meas.shape[0],
            u_raw.shape[0],
            u_cmd.shape[0],
            u_applied.shape[0],
            du_cmd.shape[0],
            du_applied.shape[0],
            disturb.shape[0],
            reward.shape[0],
            terminated.shape[0],
            truncated.shape[0],
        )
        for k in range(n):
            w.writerow(
                {
                    "k": int(k),
                    "t_sec": float(k * dt),
                    "x_true": float(x_true[k, 0]),
                    "theta_true": float(x_true[k, 1]),
                    "xdot_true": float(x_true[k, 2]),
                    "thetadot_true": float(x_true[k, 3]),
                    "x_meas": float(x_meas[k, 0]),
                    "theta_meas": float(x_meas[k, 1]),
                    "xdot_meas": float(x_meas[k, 2]),
                    "thetadot_meas": float(x_meas[k, 3]),
                    "term_reason": str(term_reason),
                    "x": float(x_true[k, 0]),
                    "theta": float(x_true[k, 1]),
                    "xdot": float(x_true[k, 2]),
                    "thetadot": float(x_true[k, 3]),
                    "u_raw": float(u_raw[k]),
                    "u_cmd": float(u_cmd[k]),
                    "u_applied": float(u_applied[k]),
                    "du_cmd": float(du_cmd[k]),
                    "du_applied": float(du_applied[k]),
                    "disturb_force": float(disturb[k]),
                    "reward": float(reward[k]),
                    "terminated": int(terminated[k]),
                    "truncated": int(truncated[k]),
                }
            )


def _force_sign_for_seed(seed: int, mode: str):
    mode = str(mode).lower()
    if mode == "balanced":
        # With seeds starting at 0, even/odd split gives near 50:50 and exact for even seed counts.
        return 1.0 if (int(seed) % 2 == 0) else -1.0
    if mode == "random":
        return np.nan
    raise ValueError(f"Unknown force-sign mode: {mode}")


def make_cfg(
    controller: str,
    amp: float,
    seed: int,
    steps: int,
    video: bool,
    run_mode: str = "metrics",
    run_id: str = "",
    disturbance_kind: str = "window",
    t0: int = 100,
    duration: int = 25,
    omega: float = 0.05,
    Q_diag=(1.0, 80.0, 1.0, 10.0),
    R=0.1,
    fd_eps=1e-5,
    theta0: float = 0.0,
    mpc_horizon: int = 20,
    mpc_solver: str = "auto",
    mpc_state_constraint: str = "x",
    mpc_x_margin: float = 0.0,
    mpc_du_max: float = np.nan,
    mpc_x_index: int = 0,
    termination_theta: float = 0.5,
    termination_x_limit: float = np.nan,
    actuator_u_max: float = np.nan,
    actuator_u_min: float = np.nan,
    actuator_du_max: float = np.nan,
    actuator_u_init: float = 0.0,
    actuation_delay_steps: int = 0,
    sensor_noise_x: float = 0.0,
    sensor_noise_theta: float = 0.0,
    sensor_noise_xdot: float = 0.0,
    sensor_noise_thetadot: float = 0.0,
    video_full_length: bool = False,
    video_tail_steps: int = 0,
    post_term_action_mode: str = "zero",
    disturbance_sign: float = np.nan,
):
    run_name = f"{controller}_amp{amp:.3f}_seed{seed}_{int(time.time()*1000)}"
    actuator = {
        "du_max": float(actuator_du_max),
        "u_init": float(actuator_u_init),
    }
    if np.isfinite(actuator_u_max):
        actuator["u_max"] = float(actuator_u_max)
    if np.isfinite(actuator_u_min):
        actuator["u_min"] = float(actuator_u_min)

    disturbance_cfg = {
        "enabled": True if amp > 0 else False,
        "kind": disturbance_kind,
        "amp": float(amp),
        "t0": int(t0),
        "duration": int(duration),
        "omega": float(omega),
    }
    if np.isfinite(disturbance_sign) and float(disturbance_sign) != 0.0:
        disturbance_cfg["sign"] = float(disturbance_sign)

    return {
        "run_name": run_name,
        "run_mode": str(run_mode),
        "run_id": str(run_id),
        "seed": int(seed),
        "steps": int(steps),
        "video": bool(video),
        "video_full_length": bool(video_full_length),
        "video_tail_steps": int(video_tail_steps),
        "post_term_action_mode": str(post_term_action_mode),
        "video_dir": "videos/fd_compare",
        "log_dir": "logs/fd_compare",
        "disturbance_target": "force",
        "termination_theta": float(termination_theta),
        "termination_x_limit": float(termination_x_limit),
        "actuation_delay_steps": int(actuation_delay_steps),
        "sensor_noise": {
            "x": float(sensor_noise_x),
            "theta": float(sensor_noise_theta),
            "xdot": float(sensor_noise_xdot),
            "thetadot": float(sensor_noise_thetadot),
        },
        "init": {"theta": float(theta0)},
        "controller": controller,
        "actuator": actuator,
        "lqr": {"Q_diag": list(Q_diag), "R": float(R), "fd_eps": float(fd_eps)},
        "mpc": {
            "horizon": int(mpc_horizon),
            "solver": str(mpc_solver),
            "state_constraint": str(mpc_state_constraint),
            "x_margin": float(mpc_x_margin),
            "du_max": float(mpc_du_max),
            "x_index": int(mpc_x_index),
        },
        "disturbance": disturbance_cfg,
    }


def get_dt_umax_xlimit():
    env = make_inverted_pendulum()
    m = env.unwrapped.model
    dt = float(m.opt.timestep * env.unwrapped.frame_skip)
    u_max = float(env.action_space.high[0])
    try:
        jid = m.joint("slider").id
        x_abs_limit = float(np.max(np.abs(m.jnt_range[jid])))
    except Exception:
        x_abs_limit = np.inf
    env.close()
    return dt, u_max, x_abs_limit


def main():
    parser = argparse.ArgumentParser()
    default_out = "logs/fd_compare/summary_force_fd_compare.csv"
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--mode", type=str, default="metrics", choices=["metrics", "video"], help="metrics: fair aggregate runs, video: render-only runs.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run identifier for traceability in outputs.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--controllers", type=str, default="lqr_fd,mpc_fd")
    parser.add_argument("--amps", type=str, default="250,260,270,280,290")
    parser.add_argument("--video-amp-idx", type=str, default="")
    parser.add_argument("--video-seed", type=int, default=0, help="Single seed used in --mode video.")
    parser.add_argument("--video-tail-sec", type=float, default=1.0, help="Extra seconds to keep recording after first terminated/truncated in --mode video.")
    parser.add_argument(
        "--video-post-term-action",
        type=str,
        default="zero",
        choices=["zero", "last"],
        help="Post-termination action policy for video tail.",
    )
    parser.add_argument(
        "--video-full-length",
        action="store_true",
        help="Deprecated behavior switch. In --mode metrics, ignored; in --mode video, tail length is controlled by --video-tail-sec.",
    )
    parser.add_argument("--disturbance-kind", type=str, default="window")
    parser.add_argument(
        "--force-sign-mode",
        type=str,
        default="balanced",
        choices=["balanced", "random"],
        help="How to assign force disturbance sign across seeds.",
    )
    parser.add_argument("--t0", type=int, default=100)
    parser.add_argument("--duration", type=int, default=25)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--termination-theta", type=float, default=0.5)
    parser.add_argument("--termination-x-limit", type=float, default=np.nan, help="Runtime x-termination limit. Default follows --x-fail-limit/model slider.")
    parser.add_argument(
        "--x-fail-limit",
        type=float,
        default=np.nan,
        help="x-failure threshold in meters. Default=slider limit from model. Use inf to disable.",
    )
    parser.add_argument("--x-fail-eps", type=float, default=0.0, help="x-fail proximity epsilon: fail mask uses |x| >= x_fail_limit - x_fail_eps.")
    parser.add_argument("--x-fail-hold", type=int, default=1, help="Consecutive steps required for x-fail.")

    parser.add_argument("--eps-theta", type=float, default=0.05236)
    parser.add_argument("--eps-omega", type=float, default=0.1)
    parser.add_argument("--settle-hold-sec", type=float, default=0.5)

    parser.add_argument("--sat-tol", type=float, default=0.01, help="Count saturation when |u| >= u_max - sat_tol")
    parser.add_argument("--actuator-u-max", type=float, default=np.nan, help="Optional shared actuator upper bound (clipped by hardware).")
    parser.add_argument("--actuator-u-min", type=float, default=np.nan, help="Optional shared actuator lower bound (clipped by hardware).")
    parser.add_argument("--actuator-du-max", type=float, default=np.nan, help="Shared physical delta-u limit applied to both controllers.")
    parser.add_argument("--actuator-u-init", type=float, default=0.0, help="Initial previous action for rate limiter.")
    parser.add_argument("--actuation-delay-steps", type=int, default=0, help="Input delay d: environment applies u(k-d).")
    parser.add_argument("--sensor-noise-theta-std", type=float, default=0.0, help="Std of Gaussian theta measurement noise.")
    parser.add_argument("--sensor-noise-x-std", type=float, default=0.0, help="Std of Gaussian x measurement noise.")
    parser.add_argument("--sensor-noise-xdot-std", type=float, default=0.0, help="Std of Gaussian xdot measurement noise.")
    parser.add_argument("--sensor-noise-thetadot-std", type=float, default=0.0, help="Std of Gaussian thetadot measurement noise.")
    parser.add_argument("--metric-du-threshold", type=float, default=np.nan, help="Delta-u threshold used only for activation metrics.")
    parser.add_argument("--du-max", type=float, default=np.nan, help="Deprecated alias of --metric-du-threshold.")
    parser.add_argument("--du-tol", type=float, default=1e-6)

    parser.add_argument(
        "--step-log-dir",
        type=str,
        default="logs/fd_compare/steps",
        help="Directory to save per-step CSV logs (default: logs/fd_compare/steps).",
    )
    parser.add_argument("--Q", type=str, default="1,80,1,10")
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--mpc-horizon", type=int, default=20)
    parser.add_argument("--mpc-solver", type=str, default="auto")
    parser.add_argument("--mpc-state-constraint", type=str, default="x", choices=["none", "x"], help="MPC internal state constraint mode.")
    parser.add_argument("--mpc-x-margin", type=float, default=0.0, help="Margin subtracted from x limit for MPC state constraint.")
    parser.add_argument("--mpc-du-max", type=float, default=np.nan, help="Optional MPC-internal du limit override.")
    parser.add_argument("--mpc-x-index", type=int, default=0, help="State index for MPC state bound (x position is 0).")
    args = parser.parse_args()
    mode = str(args.mode).lower()
    run_id = str(args.run_id).strip() or str(int(time.time() * 1000))

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    invalid = [c for c in controllers if c not in ("lqr_fd", "mpc_fd")]
    if invalid:
        raise ValueError(f"Unsupported controllers for fd_compare: {invalid}")

    amps = [float(a.strip()) for a in args.amps.split(",") if a.strip()]
    video_amp_idx = {int(x) for x in args.video_amp_idx.split(",") if x.strip()} if args.video_amp_idx else set()
    Q_diag = tuple(float(x.strip()) for x in args.Q.split(","))

    dt, u_max, x_limit_model = get_dt_umax_xlimit()
    if np.isnan(args.x_fail_limit):
        args.x_fail_limit = float(x_limit_model)
    if np.isnan(args.termination_x_limit):
        args.termination_x_limit = float(args.x_fail_limit)

    metric_du_threshold = float(args.metric_du_threshold)
    if np.isfinite(args.du_max):
        metric_du_threshold = float(args.du_max)
    if not np.isfinite(metric_du_threshold) and np.isfinite(args.actuator_du_max):
        metric_du_threshold = float(args.actuator_du_max)

    if mode == "video" and args.out == default_out:
        args.out = "logs/fd_compare/video_manifest_fd_compare.csv"

    if mode == "metrics":
        if args.video_full_length:
            print("[info] mode=metrics: ignoring --video-full-length.")
        if args.video_amp_idx:
            print("[info] mode=metrics: ignoring --video-amp-idx.")
        if int(args.video_seed) != 0:
            print("[info] mode=metrics: ignoring --video-seed.")
        if float(args.video_tail_sec) != 1.0:
            print("[info] mode=metrics: ignoring --video-tail-sec.")
        if str(args.video_post_term_action).lower() != "zero":
            print("[info] mode=metrics: ignoring --video-post-term-action.")

    video_tail_steps = max(int(round(max(float(args.video_tail_sec), 0.0) / dt)), 0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fieldnames = [
        "run_mode",
        "run_id",
        "run_name",
        "controller",
        "amp",
        "seed",
        "success",
        "terminated_any",
        "term_reason",
        "fail_theta",
        "fail_x",
        "fail_nan",
        "fail_other",
        "T",
        "time_to_term",
        "n_steps_post",
        "u_energy_sum",
        "u_energy_mean",
        "u_energy",
        "J_emp_sum",
        "J_emp_mean",
        "J_emp",
        "act_rate_u",
        "sat_rate",
        "sat_steps",
        "max_run_u",
        "sat_max_run",
        "act_rate_du",
        "max_run_du",
        "min_margin_theta",
        "min_margin_x",
        "theta_max_post_all",
        "theta_rms_post_all",
        "theta_max_post_succ",
        "theta_max_post_zero_succ",
        "theta_rms_post_succ",
        "theta_max_post",
        "theta_rms_post",
        "recovery_time_succ",
        "recovery_time",
        "steps",
        "dt",
        "u_max",
        "t0",
        "duration",
        "kind",
        "disturbance_sign",
        "theta0",
        "termination_theta",
        "termination_x_limit",
        "x_fail_limit",
        "x_fail_eps",
        "x_fail_hold",
        "actuator_u_min",
        "actuator_u_max",
        "actuator_du_max",
        "actuation_delay_steps",
        "sensor_noise_theta_std",
        "sensor_noise_x_std",
        "sensor_noise_xdot_std",
        "sensor_noise_thetadot_std",
        "metric_du_threshold",
        "sat_tol",
        "du_max",
        "mpc_state_constraint",
        "mpc_x_margin",
        "mpc_du_max",
        "mpc_x_limit_eff",
        "k_term_first",
        "postterm_solver_fail",
    ]

    rows = []
    video_rows = []
    if mode == "video":
        selected_amp_idx = video_amp_idx if video_amp_idx else set(range(1, len(amps) + 1))
        seed_values = [int(args.video_seed)]
    else:
        selected_amp_idx = set()
        seed_values = list(range(args.seeds))

    for controller in controllers:
        for amp_idx, amp in enumerate(amps):
            if mode == "video" and ((amp_idx + 1) not in selected_amp_idx):
                continue
            for seed in seed_values:
                video = bool(mode == "video")
                video_full_length = bool(mode == "video")
                disturbance_sign = _force_sign_for_seed(seed=seed, mode=args.force_sign_mode)
                cfg = make_cfg(
                    controller=controller,
                    amp=amp,
                    seed=seed,
                    steps=args.steps,
                    video=video,
                    run_mode=mode,
                    run_id=run_id,
                    video_full_length=video_full_length,
                    video_tail_steps=video_tail_steps if video else 0,
                    post_term_action_mode=str(args.video_post_term_action),
                    disturbance_kind=args.disturbance_kind,
                    t0=args.t0,
                    duration=args.duration,
                    omega=args.omega,
                    Q_diag=Q_diag,
                    R=args.R,
                    fd_eps=args.fd_eps,
                    theta0=args.theta0,
                    mpc_horizon=args.mpc_horizon,
                    mpc_solver=args.mpc_solver,
                    mpc_state_constraint=args.mpc_state_constraint,
                    mpc_x_margin=args.mpc_x_margin,
                    mpc_du_max=args.mpc_du_max,
                    mpc_x_index=args.mpc_x_index,
                    termination_theta=args.termination_theta,
                    termination_x_limit=args.termination_x_limit,
                    actuator_u_max=args.actuator_u_max,
                    actuator_u_min=args.actuator_u_min,
                    actuator_du_max=args.actuator_du_max,
                    actuator_u_init=args.actuator_u_init,
                    actuation_delay_steps=args.actuation_delay_steps,
                    sensor_noise_x=args.sensor_noise_x_std,
                    sensor_noise_theta=args.sensor_noise_theta_std,
                    sensor_noise_xdot=args.sensor_noise_xdot_std,
                    sensor_noise_thetadot=args.sensor_noise_thetadot_std,
                    disturbance_sign=disturbance_sign,
                )
                run_name = str(cfg.get("run_name", ""))

                try:
                    with redirect_stdout(io.StringIO()):
                        log = run_mod.run_once(cfg)
                except RuntimeError as e:
                    if controller != "mpc_fd":
                        raise
                    m = make_solver_fail_metrics(term_reason="solver_fail_preterm")
                    if mode == "video":
                        video_rows.append(
                            {
                                "run_mode": mode,
                                "run_id": run_id,
                                "run_name": run_name,
                                "controller": controller,
                                "amp": float(amp),
                                "seed": int(seed),
                                "success": 0,
                                "term_reason": "solver_fail_preterm",
                                "T": 0,
                                "k_term_first": np.nan,
                                "postterm_solver_fail": 0,
                            }
                        )
                        print(
                            f"[warn] video_solver_fail: controller={controller} amp={amp:.3f} seed={seed} "
                            f"error={type(e).__name__}: {e}"
                        )
                        continue

                    row = {
                        "run_mode": mode,
                        "run_id": run_id,
                        "run_name": run_name,
                        "controller": controller,
                        "amp": amp,
                        "seed": seed,
                        "steps": args.steps,
                        "dt": dt,
                        "u_max": u_max,
                        "t0": args.t0,
                        "duration": args.duration,
                        "kind": args.disturbance_kind,
                        "disturbance_sign": disturbance_sign,
                        "theta0": args.theta0,
                        "termination_theta": args.termination_theta,
                        "termination_x_limit": float(args.termination_x_limit),
                        "x_fail_limit": args.x_fail_limit,
                        "x_fail_eps": args.x_fail_eps,
                        "x_fail_hold": args.x_fail_hold,
                        "actuator_u_min": float(args.actuator_u_min),
                        "actuator_u_max": float(args.actuator_u_max),
                        "actuator_du_max": float(args.actuator_du_max),
                        "actuation_delay_steps": int(args.actuation_delay_steps),
                        "sensor_noise_theta_std": float(args.sensor_noise_theta_std),
                        "sensor_noise_x_std": float(args.sensor_noise_x_std),
                        "sensor_noise_xdot_std": float(args.sensor_noise_xdot_std),
                        "sensor_noise_thetadot_std": float(args.sensor_noise_thetadot_std),
                        "metric_du_threshold": metric_du_threshold,
                        "sat_tol": args.sat_tol,
                        "du_max": metric_du_threshold,
                        "mpc_state_constraint": str(args.mpc_state_constraint),
                        "mpc_x_margin": float(args.mpc_x_margin),
                        "mpc_du_max": float(args.mpc_du_max),
                        "mpc_x_limit_eff": np.nan,
                        "k_term_first": np.nan,
                        "postterm_solver_fail": 0,
                    }
                    row.update(m)
                    rows.append(row)
                    print(
                        f"[warn] solver_fail_preterm: controller={controller} amp={amp:.3f} seed={seed} "
                        f"error={type(e).__name__}: {e}"
                    )
                    continue

                m = compute_metrics_from_log(
                    log=log,
                    u_max=u_max,
                    dt=dt,
                    steps_cfg=args.steps,
                    t0=args.t0,
                    duration=args.duration,
                    kind=args.disturbance_kind,
                    Q_diag=Q_diag,
                    R=args.R,
                    eps_theta=args.eps_theta,
                    eps_omega=args.eps_omega,
                    settle_hold_sec=args.settle_hold_sec,
                    sat_tol=args.sat_tol,
                    du_max=metric_du_threshold,
                    du_tol=args.du_tol,
                    theta_limit=args.termination_theta,
                    x_fail_limit=args.x_fail_limit,
                    x_fail_eps=args.x_fail_eps,
                    x_fail_hold=args.x_fail_hold,
                )

                if args.step_log_dir:
                    step_csv = os.path.join(args.step_log_dir, f"{controller}_amp{amp:.3f}_seed{seed}.csv")
                    save_step_log_csv(log, step_csv, dt=dt, term_reason=str(m.get("term_reason", "")))

                if mode == "video":
                    video_rows.append(
                        {
                            "run_mode": mode,
                            "run_id": run_id,
                            "run_name": run_name,
                            "controller": controller,
                            "amp": float(amp),
                            "seed": int(seed),
                            "success": int(m.get("success", 0)),
                            "term_reason": str(m.get("term_reason", "")),
                            "T": int(m.get("T", 0)),
                            "k_term_first": float(log.get("k_term_first", np.nan)),
                            "postterm_solver_fail": int(log.get("postterm_solver_fail", 0)),
                        }
                    )
                    continue

                row = {
                    "run_mode": mode,
                    "run_id": run_id,
                    "run_name": run_name,
                    "controller": controller,
                    "amp": amp,
                    "seed": seed,
                    "steps": args.steps,
                    "dt": dt,
                    "u_max": u_max,
                    "t0": args.t0,
                    "duration": args.duration,
                    "kind": args.disturbance_kind,
                    "disturbance_sign": disturbance_sign,
                    "theta0": args.theta0,
                    "termination_theta": args.termination_theta,
                    "termination_x_limit": float(log.get("termination_x_limit", args.termination_x_limit)),
                    "x_fail_limit": args.x_fail_limit,
                    "x_fail_eps": args.x_fail_eps,
                    "x_fail_hold": args.x_fail_hold,
                    "actuator_u_min": float(log.get("actuator_u_min", args.actuator_u_min)),
                    "actuator_u_max": float(log.get("actuator_u_max", args.actuator_u_max)),
                    "actuator_du_max": float(log.get("actuator_du_max", args.actuator_du_max)),
                    "actuation_delay_steps": int(log.get("actuation_delay_steps", args.actuation_delay_steps)),
                    "sensor_noise_theta_std": float(log.get("sensor_noise_theta", args.sensor_noise_theta_std)),
                    "sensor_noise_x_std": float(log.get("sensor_noise_x", args.sensor_noise_x_std)),
                    "sensor_noise_xdot_std": float(log.get("sensor_noise_xdot", args.sensor_noise_xdot_std)),
                    "sensor_noise_thetadot_std": float(log.get("sensor_noise_thetadot", args.sensor_noise_thetadot_std)),
                    "metric_du_threshold": metric_du_threshold,
                    "sat_tol": args.sat_tol,
                    "du_max": metric_du_threshold,
                    "mpc_state_constraint": str(log.get("mpc_state_constraint", args.mpc_state_constraint)),
                    "mpc_x_margin": float(log.get("mpc_x_margin", args.mpc_x_margin)),
                    "mpc_du_max": float(log.get("mpc_du_limit_eff", args.mpc_du_max)),
                    "mpc_x_limit_eff": float(log.get("mpc_x_limit_eff", np.nan)),
                    "k_term_first": float(log.get("k_term_first", np.nan)),
                    "postterm_solver_fail": int(log.get("postterm_solver_fail", 0)),
                }
                row.update(m)
                rows.append(row)

    if mode == "metrics":
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"OK: wrote {len(rows)} rows -> {args.out}")
    else:
        video_manifest_fields = [
            "run_mode",
            "run_id",
            "run_name",
            "controller",
            "amp",
            "seed",
            "success",
            "term_reason",
            "T",
            "k_term_first",
            "postterm_solver_fail",
        ]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=video_manifest_fields)
            w.writeheader()
            w.writerows(video_rows)
        print(f"OK: wrote {len(video_rows)} video rows -> {args.out}")


if __name__ == "__main__":
    main()
