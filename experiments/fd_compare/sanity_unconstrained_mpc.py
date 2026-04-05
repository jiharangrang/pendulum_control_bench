import argparse

import gymnasium as gym
import numpy as np

from controllers.lqr import dlqr
from controllers.mpc import LinearMPC


ENV_ID = "InvertedPendulum-v5"


def linearize_fd(env_id: str, seed: int, eps: float):
    env_lin = gym.make(env_id)
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


def rollout_linear(Ad, Bd, policy_fn, x0, steps):
    x = np.array(x0, dtype=np.float64).reshape(-1)
    xs = [x.copy()]
    us = []
    for _ in range(steps):
        u = float(policy_fn(x))
        us.append(u)
        x = Ad @ x + Bd.reshape(-1) * u
        xs.append(x.copy())
    return np.array(xs), np.array(us)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--Q", type=str, default="1,80,1,10")
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--u-max", type=float, default=100.0)
    parser.add_argument("--solver", type=str, default="auto")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--x-scale", type=float, default=0.05)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--tol-u", type=float, default=1e-3)
    parser.add_argument("--tol-x", type=float, default=1e-3)
    args = parser.parse_args()

    Q_diag = tuple(float(v.strip()) for v in args.Q.split(","))
    Q = np.diag(Q_diag).astype(np.float64)
    R = np.array([[args.R]], dtype=np.float64)

    Ad, Bd = linearize_fd(ENV_ID, seed=args.seed, eps=args.fd_eps)
    K, _ = dlqr(Ad, Bd, Q, R)
    mpc = LinearMPC(
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        R=R,
        horizon=args.horizon,
        u_max=args.u_max,
        u_min=-args.u_max,
        solver=args.solver,
    )

    rng = np.random.default_rng(args.seed)
    max_u_err = 0.0
    mean_u_err = 0.0

    for _ in range(args.n_samples):
        x = rng.normal(loc=0.0, scale=args.x_scale, size=(4,))
        u_lqr = -float((K @ x.reshape(4, 1)).item())
        u_mpc = float(mpc.control(x)[0])
        err = abs(u_lqr - u_mpc)
        max_u_err = max(max_u_err, err)
        mean_u_err += err
    mean_u_err /= float(args.n_samples)

    x0 = np.array([0.02, 0.05, 0.0, 0.0], dtype=np.float64)
    xs_lqr, us_lqr = rollout_linear(
        Ad,
        Bd,
        policy_fn=lambda x: -float((K @ x.reshape(4, 1)).item()),
        x0=x0,
        steps=args.rollout_steps,
    )
    xs_mpc, us_mpc = rollout_linear(
        Ad,
        Bd,
        policy_fn=lambda x: float(mpc.control(x)[0]),
        x0=x0,
        steps=args.rollout_steps,
    )

    max_rollout_u_err = float(np.max(np.abs(us_lqr - us_mpc)))
    max_rollout_x_err = float(np.max(np.abs(xs_lqr - xs_mpc)))

    pass_u = max(max_u_err, max_rollout_u_err) <= args.tol_u
    pass_x = max_rollout_x_err <= args.tol_x
    passed = pass_u and pass_x

    print("[sanity] FD linear model:", ENV_ID)
    print("[sanity] N (horizon):", args.horizon)
    print("[sanity] u_max for MPC:", args.u_max)
    print("[sanity] max |u_lqr - u_mpc| over random states:", max_u_err)
    print("[sanity] mean |u_lqr - u_mpc| over random states:", mean_u_err)
    print("[sanity] max rollout |u_lqr - u_mpc|:", max_rollout_u_err)
    print("[sanity] max rollout |x_lqr - x_mpc|:", max_rollout_x_err)
    print("[sanity] tolerance u:", args.tol_u, "tolerance x:", args.tol_x)
    print("[sanity] RESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
