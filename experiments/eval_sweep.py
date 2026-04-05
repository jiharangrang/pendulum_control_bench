# experiments/eval_sweep.py
import os
import io
import csv
import time
import argparse
from contextlib import redirect_stdout

import numpy as np

# experiments/run.py 안에 run_once(cfg)가 "top-level"로 있어야 import 가능
import experiments.run as run_mod
from envs.mujoco_model import make_inverted_pendulum


def compute_recovery_time(theta, theta_dot, post_start_idx, dt,
                          theta_tol=0.05236, theta_dot_tol=0.1, hold_steps=5):
    """
    disturbance 끝난 뒤(post_start_idx부터) 상태가
    (|theta|<=tol AND |theta_dot|<=tol) 밴드로 '복귀'해 hold_steps 연속 유지하는 데 걸린 시간.
    밴드 밖으로 한 번도 나가지 않았다면 NaN(정의 불가) 반환.
    """
    theta = np.asarray(theta)
    theta_dot = np.asarray(theta_dot)

    if post_start_idx >= len(theta):
        return np.nan

    in_band = (np.abs(theta) <= theta_tol) & (np.abs(theta_dot) <= theta_dot_tol)
    post = in_band[post_start_idx:]

    # 애초에 밴드 밖으로 안 나갔으면 "회복"이라는 사건이 없음
    if np.all(post):
        return np.nan

    # hold_steps 연속 True 되는 최초 시점 찾기
    for i in range(0, len(post) - hold_steps + 1):
        if np.all(post[i:i+hold_steps]):
            return i * dt

    # 끝까지 못 돌아오면: (원하면) 전체 post 길이로 표시
    return (len(post) - 1) * dt


def compute_metrics_from_log(
    log: dict,
    u_max: float,
    dt: float,
    t0: int,
    duration: int,
    kind: str,
    Q_diag,
    R: float,
    eps_theta: float = 0.05236,
    eps_omega: float = 0.1,
    settle_hold_sec: float = 0.5,
):
    obs = np.array(log["obs"], dtype=np.float64)                 # usually (T,4) of post-step states
    u = np.array(log["action"], dtype=np.float64).reshape(-1)    # (T,)
    terminated = np.array(log["terminated"], dtype=bool)

    # Align x_k with u_k as best as possible across log formats.
    # - If run.py provides obs0 + post-step obs, use x_k = [obs0, obs[:-1]]
    # - If obs already has length T+1, use x_k = obs[:-1]
    # - Else fall back to obs as-is (approximation).
    T = int(u.shape[0])
    if T == 0:
        x = np.zeros((0, 4), dtype=np.float64)
    elif "obs0" in log and obs.shape[0] >= max(T - 1, 0):
        x0 = np.array(log["obs0"], dtype=np.float64).reshape(1, 4)
        x = np.vstack([x0, obs[: max(T - 1, 0)]])
    elif obs.shape[0] == T + 1:
        x = obs[:-1]
    else:
        x = obs[:T]

    theta = x[:, 1] if x.size else np.array([], dtype=np.float64)
    omega = x[:, 3] if x.size else np.array([], dtype=np.float64)

    # 성공/실패: terminated만 실패로 본다 (TimeLimit truncated는 실패 아님)
    terminated_any = bool(np.any(terminated))
    success = not terminated_any

    # 포화 판단: |u(t)| >= u_max - eps
    eps = 1e-9
    sat_mask = (np.abs(u) >= (u_max - eps))
    sat_steps = int(sat_mask.sum())
    sat_rate = float(sat_steps / T) if T > 0 else 0.0

    # 연속 포화 최대 길이 (step 수)
    sat_max_run = 0
    run = 0
    for s in sat_mask.tolist():
        if s:
            run += 1
            if run > sat_max_run:
                sat_max_run = run
        else:
            run = 0

    # 입력 에너지
    u_energy = float(np.sum(u ** 2))

    # Empirical LQR cost using the same (Q,R) used for controller design.
    Q = np.diag(np.asarray(Q_diag, dtype=np.float64))
    Ru = float(R)
    if T > 0 and x.shape[0] == T:
        x_cost = np.einsum("bi,ij,bj->b", x, Q, x)
        u_cost = (u ** 2) * Ru
        J_emp = float(np.sum(x_cost + u_cost))
    else:
        J_emp = float("nan")

    # 외란 종료 시점 이후 구간 정의 (마지막 외란 스텝 + 1)
    duration_eff = 1 if kind in ("impulse", "pulse") else int(duration)
    post_start_idx = int(t0 + duration_eff)
    post_start_idx = max(post_start_idx, 0)

    post_theta = theta[post_start_idx:] if post_start_idx < len(theta) else np.array([])
    post_omega = omega[post_start_idx:] if post_start_idx < len(omega) else np.array([])

    if post_theta.size:
        theta_max_post = float(np.max(np.abs(post_theta)))
        theta_rms_post = float(np.sqrt(np.mean(post_theta ** 2)))
    elif theta.size:
        theta_max_post = float(np.max(np.abs(theta)))
        theta_rms_post = float(np.sqrt(np.mean(theta ** 2)))
    else:
        theta_max_post = float("nan")
        theta_rms_post = float("nan")

    hold_N = max(1, int(round(settle_hold_sec / dt)))
    recovery_time = compute_recovery_time(
        theta, omega, post_start_idx, dt,
        theta_tol=eps_theta, theta_dot_tol=eps_omega, hold_steps=hold_N
    )

    return dict(
        T=T,
        success=int(success),
        terminated_any=int(terminated_any),
        u_energy=u_energy,
        J_emp=J_emp,
        sat_rate=sat_rate,
        sat_steps=sat_steps,
        sat_max_run=int(sat_max_run),
        theta_max_post=theta_max_post,
        theta_rms_post=theta_rms_post,
        recovery_time=recovery_time,
    )


def make_cfg(controller: str, amp: float, seed: int, steps: int, video: bool,
             disturbance_kind: str = "impulse", t0: int = 200, duration: int = 50, omega: float = 0.05,
             Q_diag=(1.0, 80.0, 1.0, 10.0), R=0.1, fd_eps=1e-5, theta0: float = 0.0):
    # run.py가 요구하는 cfg 키를 맞춤
    run_name = f"{controller}_amp{amp:.3f}_seed{seed}_{int(time.time()*1000)}"

    cfg = {
        "run_name": run_name,
        "seed": int(seed),
        "steps": int(steps),
        "video": bool(video),
        "video_dir": "videos",
        "log_dir": "logs",
        "disturbance_target": "force",
        "init": {
            "theta": float(theta0),
        },
        "controller": controller,  # "random", "lqr_theory", "lqr_fd"
        "lqr": {
            "Q_diag": list(Q_diag),
            "R": float(R),
            "fd_eps": float(fd_eps),
        },
        "disturbance": {
            "enabled": True if amp > 0 else False,
            "kind": disturbance_kind,
            "amp": float(amp),
            "t0": int(t0),
            "duration": int(duration),
            "omega": float(omega),
        },
    }
    return cfg


def get_dt_and_umax():
    env = make_inverted_pendulum()
    m = env.unwrapped.model
    dt = float(m.opt.timestep * env.unwrapped.frame_skip)
    u_max = float(env.action_space.high[0])
    env.close()
    return dt, u_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="logs/summary.csv")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--controllers", type=str, default="lqr_theory,lqr_fd")
    parser.add_argument("--amps", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--video-amp-idx", type=str, default="",
                        help="1-based indices into --amps; videos saved for those amps (seed 0 only).")
    parser.add_argument("--disturbance-kind", type=str, default="impulse")
    parser.add_argument("--t0", type=int, default=200)
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--theta0", type=float, default=0.0)

    # 정착 기준
    parser.add_argument("--eps-theta", type=float, default=0.05236)
    parser.add_argument("--eps-omega", type=float, default=0.1)
    parser.add_argument("--settle-hold-sec", type=float, default=0.5)

    # LQR 가중치
    parser.add_argument("--Q", type=str, default="1,80,1,10")
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--fd-eps", type=float, default=1e-5)

    args = parser.parse_args()

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    amps = [float(a.strip()) for a in args.amps.split(",") if a.strip()]
    video_amp_idx = {int(x) for x in args.video_amp_idx.split(",") if x.strip()} if args.video_amp_idx else set()
    Q_diag = tuple(float(x.strip()) for x in args.Q.split(","))

    dt, u_max = get_dt_and_umax()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fieldnames = [
        "controller", "amp", "seed",
        "success", "terminated_any", "T",
        "u_energy", "J_emp", "sat_rate", "sat_steps", "sat_max_run",
        "theta_max_post", "theta_rms_post", "recovery_time",
        "steps", "dt", "u_max",
        "t0", "duration", "kind",
        "theta0",
    ]

    rows = []

    # run.py 안에서 프린트(K, Ad/Bd 등) 많이 할 수 있으니 sweep에서는 기본적으로 묵음 처리
    for controller in controllers:
        for amp_idx, amp in enumerate(amps):
            for seed in range(args.seeds):
                video = (amp_idx + 1 in video_amp_idx) and (seed == 0)
                cfg = make_cfg(
                    controller=controller,
                    amp=amp,
                    seed=seed,
                    steps=args.steps,
                    video=video,
                    disturbance_kind=args.disturbance_kind,
                    t0=args.t0,
                    duration=args.duration,
                    omega=args.omega,
                    Q_diag=Q_diag,
                    R=args.R,
                    fd_eps=args.fd_eps,
                    theta0=args.theta0,
                )

                # stdout 억제(원하면 redirect_stdout 제거하면 됨)
                with redirect_stdout(io.StringIO()):
                    log = run_mod.run_once(cfg)

                m = compute_metrics_from_log(
                    log,
                    u_max=u_max,
                    dt=dt,
                    t0=args.t0,
                    duration=args.duration,
                    kind=args.disturbance_kind,
                    Q_diag=Q_diag,
                    R=args.R,
                    eps_theta=args.eps_theta,
                    eps_omega=args.eps_omega,
                    settle_hold_sec=args.settle_hold_sec,
                )

                row = {
                    "controller": controller,
                    "amp": amp,
                    "seed": seed,
                    "steps": args.steps,
                    "dt": dt,
                    "u_max": u_max,
                    "t0": args.t0,
                    "duration": args.duration,
                    "kind": args.disturbance_kind,
                    "theta0": args.theta0,
                }
                row.update(m)
                rows.append(row)

    # CSV 저장
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
