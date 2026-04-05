import argparse
import io
import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

from experiments.fd_compare import run_fd_compare as run_mod
from experiments.fd_compare.eval_sweep_fd_compare import get_dt_umax_xlimit, make_cfg


def _parse_qdiag(s: str):
    vals = [float(v.strip()) for v in s.split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("--Q must have 4 comma-separated numbers, e.g. 1,80,1,10")
    return tuple(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", type=float, default=280.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--disturbance-kind", type=str, default="window")
    parser.add_argument("--t0", type=int, default=100)
    parser.add_argument("--duration", type=int, default=25)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--termination-theta", type=float, default=0.5)
    parser.add_argument("--Q", type=str, default="1,80,1,10")
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--fd-eps", type=float, default=1e-5)
    parser.add_argument("--mpc-horizon", type=int, default=20)
    parser.add_argument("--mpc-solver", type=str, default="auto")
    parser.add_argument(
        "--u-kind",
        type=str,
        default="raw",
        choices=("raw", "applied"),
        help="raw: controller output before env clip, applied: action sent to env",
    )
    parser.add_argument("--out-csv", type=str, default="logs/fd_compare/u_compare_amp280.csv")
    parser.add_argument("--out-png", type=str, default="plots/fd_compare/u_diff_amp280.png")
    args = parser.parse_args()

    qdiag = _parse_qdiag(args.Q)

    cfg_lqr = make_cfg(
        controller="lqr_fd",
        amp=args.amp,
        seed=args.seed,
        steps=args.steps,
        video=False,
        disturbance_kind=args.disturbance_kind,
        t0=args.t0,
        duration=args.duration,
        omega=args.omega,
        Q_diag=qdiag,
        R=args.R,
        fd_eps=args.fd_eps,
        theta0=args.theta0,
        mpc_horizon=args.mpc_horizon,
        mpc_solver=args.mpc_solver,
        termination_theta=args.termination_theta,
    )
    cfg_mpc = make_cfg(
        controller="mpc_fd",
        amp=args.amp,
        seed=args.seed,
        steps=args.steps,
        video=False,
        disturbance_kind=args.disturbance_kind,
        t0=args.t0,
        duration=args.duration,
        omega=args.omega,
        Q_diag=qdiag,
        R=args.R,
        fd_eps=args.fd_eps,
        theta0=args.theta0,
        mpc_horizon=args.mpc_horizon,
        mpc_solver=args.mpc_solver,
        termination_theta=args.termination_theta,
    )

    with redirect_stdout(io.StringIO()):
        log_lqr = run_mod.run_once(cfg_lqr)
    with redirect_stdout(io.StringIO()):
        log_mpc = run_mod.run_once(cfg_mpc)

    if args.u_kind == "raw":
        u_lqr = np.array(log_lqr.get("u_raw", []), dtype=np.float64)
        u_mpc = np.array(log_mpc.get("u_raw", []), dtype=np.float64)
    else:
        u_lqr = np.array(log_lqr.get("u_applied", []), dtype=np.float64)
        u_mpc = np.array(log_mpc.get("u_applied", []), dtype=np.float64)

    n = int(min(len(u_lqr), len(u_mpc)))
    if n <= 0:
        raise RuntimeError("No control data found in logs.")

    u_lqr = u_lqr[:n]
    u_mpc = u_mpc[:n]
    k = np.arange(n, dtype=np.int64)
    delta_u = u_mpc - u_lqr
    abs_delta_u = np.abs(delta_u)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)

    np.savetxt(
        args.out_csv,
        np.column_stack([k, u_lqr, u_mpc, delta_u, abs_delta_u]),
        delimiter=",",
        header="k,u_lqr,u_mpc,delta_u,abs_delta_u",
        comments="",
    )

    _, _, dt = None, None, None
    dt, _, _ = get_dt_umax_xlimit()
    sim_sec = n * dt
    max_abs = float(np.max(abs_delta_u))
    mean_delta = float(np.mean(delta_u))
    mean_abs = float(np.mean(abs_delta_u))

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(k, delta_u, label="u_mpc - u_lqr", linewidth=2.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("step k")
    ax.set_ylabel("delta_u")
    ax.set_title(f"Control Input Difference ({sim_sec:.1f}s, {n} steps, amp={args.amp:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    txt = (
        f"max|delta_u|={max_abs:.3e}\n"
        f"mean(delta_u)={mean_delta:.3e}\n"
        f"mean|delta_u|={mean_abs:.3e}"
    )
    ax.text(
        0.015,
        0.985,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)

    print(f"OK: saved csv -> {args.out_csv}")
    print(f"OK: saved plot -> {args.out_png}")
    print(
        "stats:",
        f"max_abs={max_abs:.3e}, mean={mean_delta:.3e}, mean_abs={mean_abs:.3e}, steps={n}, sim_sec={sim_sec:.2f}",
    )


if __name__ == "__main__":
    main()
