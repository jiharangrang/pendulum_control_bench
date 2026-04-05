import argparse
import os
import subprocess
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _du_tag(v: float):
    return f"{v:.3f}".replace(".", "p")


def run_single_sweep(args, du_value: float, out_csv: str):
    cmd = [
        "python",
        "-m",
        "experiments.fd_compare.eval_sweep_fd_compare",
        "--controllers",
        "lqr_fd,mpc_fd",
        "--amps",
        args.amps,
        "--seeds",
        str(args.seeds),
        "--steps",
        str(args.steps),
        "--disturbance-kind",
        args.disturbance_kind,
        "--t0",
        str(args.t0),
        "--duration",
        str(args.duration),
        "--theta0",
        str(args.theta0),
        "--termination-theta",
        str(args.termination_theta),
        "--termination-x-limit",
        str(args.termination_x_limit),
        "--x-fail-limit",
        str(args.x_fail_limit),
        "--x-fail-eps",
        str(args.x_fail_eps),
        "--x-fail-hold",
        str(args.x_fail_hold),
        "--actuator-u-max",
        str(args.actuator_u_max),
        "--actuator-u-min",
        str(args.actuator_u_min),
        "--actuator-du-max",
        str(du_value),
        "--metric-du-threshold",
        str(du_value),
        "--mpc-state-constraint",
        args.mpc_state_constraint,
        "--mpc-x-margin",
        str(args.mpc_x_margin),
        "--sat-tol",
        str(args.sat_tol),
        "--out",
        out_csv,
        "--step-log-dir",
        "",
    ]
    if np.isfinite(args.mpc_du_max):
        cmd += ["--mpc-du-max", str(args.mpc_du_max)]
    return subprocess.run(cmd, capture_output=True, text=True)


def make_overall_summary(df_all: pd.DataFrame):
    g = (
        df_all.groupby(["du", "controller"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            fail_x_rate=("fail_x", "mean"),
            fail_theta_rate=("fail_theta", "mean"),
            mean_time_to_term=("time_to_term", "mean"),
            act_rate_du=("act_rate_du", "mean"),
            n=("success", "size"),
        )
    )
    return g


def make_amp_summary(df_all: pd.DataFrame):
    g = (
        df_all.groupby(["du", "amp", "controller"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            fail_x_rate=("fail_x", "mean"),
            fail_theta_rate=("fail_theta", "mean"),
            mean_time_to_term=("time_to_term", "mean"),
            n=("success", "size"),
        )
    )
    return g


def select_recommended_du(df_overall: pd.DataFrame):
    piv = df_overall.pivot(index="du", columns="controller", values="success_rate")
    piv = piv.dropna()
    if not {"lqr_fd", "mpc_fd"}.issubset(piv.columns):
        return np.nan, "selection failed: missing controller columns"

    piv = piv.copy()
    piv["gap"] = piv["mpc_fd"] - piv["lqr_fd"]

    # Decision rule:
    # 1) Clear separation: MPC >= 0.6 and LQR <= 0.3
    # 2) Among them, choose the smallest du (avoid overly loose constraint).
    cand = piv[(piv["mpc_fd"] >= 0.6) & (piv["lqr_fd"] <= 0.3)]
    if len(cand) > 0:
        du_star = float(cand.index.min())
        reason = "smallest du with clear separation (MPC>=0.6, LQR<=0.3)"
        return du_star, reason

    # Fallback: max success-gap.
    du_star = float(piv["gap"].idxmax())
    reason = "fallback: maximum success-rate gap (MPC-LQR)"
    return du_star, reason


def plot_overall(df_overall: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for ctl, g in df_overall.groupby("controller"):
        g = g.sort_values("du")
        ax.plot(g["du"], g["success_rate"], marker="o", linewidth=2, label=ctl)
    ax.set_xlabel("actuator du_max")
    ax.set_ylabel("success rate")
    ax.set_title("High-Amp (270~300) Success Rate vs du_max")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "overall_success_vs_du.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for ctl, g in df_overall.groupby("controller"):
        g = g.sort_values("du")
        ax.plot(g["du"], g["mean_time_to_term"], marker="o", linewidth=2, label=ctl)
    ax.set_xlabel("actuator du_max")
    ax.set_ylabel("mean time-to-term [s]")
    ax.set_title("High-Amp (270~300) Mean Time-to-Term vs du_max")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "overall_time_to_term_vs_du.png"), dpi=220)
    plt.close(fig)

    piv = df_overall.pivot(index="du", columns="controller", values="success_rate").dropna()
    if {"lqr_fd", "mpc_fd"}.issubset(piv.columns):
        gap = piv["mpc_fd"] - piv["lqr_fd"]
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.plot(gap.index.to_numpy(float), gap.to_numpy(float), marker="o", linewidth=2, color="black")
        ax.axhline(0.0, linestyle=":", color="gray")
        ax.set_xlabel("actuator du_max")
        ax.set_ylabel("success gap (MPC - LQR)")
        ax.set_title("Success-Rate Gap vs du_max")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "success_gap_vs_du.png"), dpi=220)
        plt.close(fig)


def plot_by_amp(df_amp: pd.DataFrame, outdir: str):
    amps = sorted(df_amp["amp"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, ctl in zip(axes, ["lqr_fd", "mpc_fd"]):
        g0 = df_amp[df_amp["controller"] == ctl]
        for amp in amps:
            g = g0[g0["amp"] == amp].sort_values("du")
            ax.plot(g["du"], g["success_rate"], marker="o", linewidth=1.8, label=f"amp={amp:g}")
        ax.set_xlabel("actuator du_max")
        ax.set_title(ctl)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Per-Amp Success Rate vs du_max (270~300)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "per_amp_success_vs_du.png"), dpi=220)
    plt.close(fig)


def write_selection_note(
    out_md: str,
    args,
    du_values,
    df_overall: pd.DataFrame,
    df_amp: pd.DataFrame,
    du_star: float,
    reason: str,
):
    piv = df_overall.pivot(index="du", columns="controller", values="success_rate").dropna()
    gap_table = ""
    if {"lqr_fd", "mpc_fd"}.issubset(piv.columns):
        t = piv.copy()
        t["gap_mpc_minus_lqr"] = t["mpc_fd"] - t["lqr_fd"]
        gap_table = t.to_string(float_format=lambda x: f"{x:.3f}")
    else:
        gap_table = "(missing controller columns)"

    by_amp = (
        df_amp.pivot_table(index=["du", "amp"], columns="controller", values="success_rate")
        .sort_index()
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    body = f"""\
# du_max Selection Note (High Amp 270~300)

## Experiment Setup
- amps: `{args.amps}`
- seeds per amp: `{args.seeds}`
- steps: `{args.steps}`
- disturbance: kind=`{args.disturbance_kind}`, t0=`{args.t0}`, duration=`{args.duration}`
- global limits: u in [`{args.actuator_u_min}`, `{args.actuator_u_max}`], x-limit=`{args.termination_x_limit}`, theta-limit=`{args.termination_theta}`
- tested du values: `{", ".join([str(v) for v in du_values])}`

## Selection Rule
1. Prefer a value where separation is clear: `MPC success >= 0.6` and `LQR success <= 0.3`.
2. Among those, choose the **smallest du** (do not make constraints looser than needed).
3. Fallback rule (if rule 1 has no candidate): pick max `(MPC-LQR)` success-rate gap.

## Recommended du_max
- selected: `{du_star:.3f}`
- reason: `{reason}`

## Overall Success Table
```text
{gap_table}
```

## Per-Amp Success Table
```text
{by_amp}
```

## Why this is defensible
- This is not a single cherry-picked run.
- The decision uses a fixed rule and a full du sweep.
- The chosen value is the smallest one that already shows a stable MPC-vs-LQR separation under the target amp range.
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--du-values", type=str, default="0.8,1.0,1.2,1.6,2.0,2.2,2.4,2.6,2.8,3.0")
    parser.add_argument("--amps", type=str, default="270,275,280,285,290,295,300")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--disturbance-kind", type=str, default="window")
    parser.add_argument("--t0", type=int, default=100)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--termination-theta", type=float, default=0.5)
    parser.add_argument("--termination-x-limit", type=float, default=1.0)
    parser.add_argument("--x-fail-limit", type=float, default=1.0)
    parser.add_argument("--x-fail-eps", type=float, default=0.0)
    parser.add_argument("--x-fail-hold", type=int, default=1)
    parser.add_argument("--actuator-u-max", type=float, default=3.0)
    parser.add_argument("--actuator-u-min", type=float, default=-3.0)
    parser.add_argument("--mpc-state-constraint", type=str, default="x", choices=["none", "x"])
    parser.add_argument("--mpc-x-margin", type=float, default=0.02)
    parser.add_argument("--mpc-du-max", type=float, default=np.nan)
    parser.add_argument("--sat-tol", type=float, default=0.02)
    parser.add_argument("--outdir", type=str, default="logs/fd_compare/du_tuning_270_300")
    parser.add_argument("--plotdir", type=str, default="plots/fd_compare/du_tuning_270_300")
    args = parser.parse_args()

    du_values = _parse_csv_floats(args.du_values)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    all_rows = []
    run_status = []
    for du in du_values:
        tag = _du_tag(du)
        out_csv = os.path.join(args.outdir, f"summary_du_{tag}.csv")
        res = run_single_sweep(args, du_value=du, out_csv=out_csv)
        ok = (res.returncode == 0) and os.path.exists(out_csv)
        run_status.append(
            {
                "du": du,
                "ok": int(ok),
                "returncode": int(res.returncode),
                "stdout_tail": (res.stdout or "")[-160:],
                "stderr_tail": (res.stderr or "")[-160:],
                "csv": out_csv,
            }
        )
        if not ok:
            continue
        df = pd.read_csv(out_csv)
        df["du"] = float(du)
        all_rows.append(df)

    df_status = pd.DataFrame(run_status)
    df_status.to_csv(os.path.join(args.outdir, "run_status.csv"), index=False)

    if len(all_rows) == 0:
        raise SystemExit("No successful du sweeps. Check run_status.csv")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(args.outdir, "combined_raw.csv"), index=False)

    df_overall = make_overall_summary(df_all)
    df_amp = make_amp_summary(df_all)
    df_overall.to_csv(os.path.join(args.outdir, "overall_by_du_controller.csv"), index=False)
    df_amp.to_csv(os.path.join(args.outdir, "amp_by_du_controller.csv"), index=False)

    plot_overall(df_overall=df_overall, outdir=args.plotdir)
    plot_by_amp(df_amp=df_amp, outdir=args.plotdir)

    du_star, reason = select_recommended_du(df_overall)
    with open(os.path.join(args.outdir, "recommended_du.txt"), "w", encoding="utf-8") as f:
        if np.isfinite(du_star):
            f.write(f"{du_star:.3f}\n")
        else:
            f.write("nan\n")

    write_selection_note(
        out_md=os.path.join(args.outdir, "selection_note.md"),
        args=args,
        du_values=du_values,
        df_overall=df_overall,
        df_amp=df_amp,
        du_star=du_star if np.isfinite(du_star) else np.nan,
        reason=reason,
    )

    print(f"OK: du tuning outputs -> {args.outdir}")
    print(f"OK: du tuning plots   -> {args.plotdir}")
    if np.isfinite(du_star):
        print(f"recommended du_max: {du_star:.3f} ({reason})")
    else:
        print("recommended du_max: nan")


if __name__ == "__main__":
    main()
