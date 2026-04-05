import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

X_DODGE = 0.0
Y_DODGE_FRAC = 0.0
MARKER_ALPHA = 0.55
MARKER_EDGE_ALPHA = 0.80


def ordered_amps_from_df(df: pd.DataFrame):
    return [float(a) for a in df["amp"].drop_duplicates().tolist()]


def select_amps_by_indices(amps_ordered, idx_arg: str):
    if not idx_arg:
        return list(amps_ordered)
    idxs = [int(x.strip()) for x in idx_arg.split(",") if x.strip()]
    selected = []
    n = len(amps_ordered)
    for i in idxs:
        if 1 <= i <= n:
            selected.append(float(amps_ordered[i - 1]))
    return list(dict.fromkeys(selected))


def _ensure_columns(df: pd.DataFrame):
    if "term_reason" not in df.columns:
        if "success" in df.columns:
            df["term_reason"] = np.where(df["success"] == 1, "time", "other_fail")
        else:
            df["term_reason"] = "other_fail"

    for col in ["fail_theta", "fail_x", "fail_nan", "fail_other"]:
        if col not in df.columns:
            df[col] = 0

    if "fail_theta" in df.columns and df["fail_theta"].sum() == 0:
        df.loc[df["term_reason"] == "theta_fail", "fail_theta"] = 1
    if "fail_x" in df.columns and df["fail_x"].sum() == 0:
        df.loc[df["term_reason"] == "x_fail", "fail_x"] = 1
    if "fail_nan" in df.columns and df["fail_nan"].sum() == 0:
        df.loc[df["term_reason"] == "nan_fail", "fail_nan"] = 1
    if "fail_other" in df.columns and df["fail_other"].sum() == 0:
        df.loc[df["term_reason"].isin(["other_fail"]) , "fail_other"] = 1

    # Canonical metric aliases for robust aggregation.
    if "theta_max_post_cens" not in df.columns:
        if "theta_max_post_all" in df.columns:
            df["theta_max_post_cens"] = df["theta_max_post_all"]
        else:
            df["theta_max_post_cens"] = df.get("theta_max_post", np.nan)

    if "theta_rms_post_cens" not in df.columns:
        if "theta_rms_post_all" in df.columns:
            df["theta_rms_post_cens"] = df["theta_rms_post_all"]
        else:
            df["theta_rms_post_cens"] = df.get("theta_rms_post", np.nan)

    if "theta_max_post_succ" not in df.columns:
        df["theta_max_post_succ"] = df.get("theta_max_post", np.nan)

    if "theta_max_post_zero_succ" not in df.columns:
        df["theta_max_post_zero_succ"] = np.nan

    if "theta_rms_post_succ" not in df.columns:
        df["theta_rms_post_succ"] = df.get("theta_rms_post", np.nan)

    if "recovery_time_succ" not in df.columns:
        df["recovery_time_succ"] = df.get("recovery_time", np.nan)

    if "act_rate_u" not in df.columns:
        df["act_rate_u"] = df.get("sat_rate", np.nan)

    if "max_run_u" not in df.columns:
        df["max_run_u"] = df.get("sat_max_run", np.nan)

    if "act_rate_du" not in df.columns:
        df["act_rate_du"] = np.nan

    if "max_run_du" not in df.columns:
        df["max_run_du"] = np.nan

    if "min_margin_x" not in df.columns:
        df["min_margin_x"] = np.nan

    if "u2_mean" not in df.columns:
        if "u_energy_mean" in df.columns:
            df["u2_mean"] = df["u_energy_mean"]
        else:
            df["u2_mean"] = np.nan
    if "u2_fixed_sum" not in df.columns:
        df["u2_fixed_sum"] = np.nan

    if "j_stage_mean" not in df.columns:
        if "J_emp_mean" in df.columns:
            df["j_stage_mean"] = df["J_emp_mean"]
        else:
            df["j_stage_mean"] = np.nan

    return df


def agg_by_controller_amp(df: pd.DataFrame):
    metrics = [
        "act_rate_u",
        "act_rate_du",
        "max_run_u",
        "max_run_du",
        "min_margin_x",
        "theta_max_post_cens",
        "theta_rms_post_cens",
        "theta_max_post_succ",
        "theta_max_post_zero_succ",
        "theta_rms_post_succ",
        "recovery_time_succ",
        "u2_fixed_sum",
        "u2_mean",
        "j_stage_mean",
    ]

    out_rows = []
    for (controller, amp), g in df.groupby(["controller", "amp"]):
        g = g.copy()
        g_succ = g[g["success"] == 1]
        row = {
            "controller": controller,
            "amp": float(amp),
            "n_total": int(len(g)),
            "n_success": int(len(g_succ)),
            "coverage": float(len(g_succ) / len(g)) if len(g) > 0 else np.nan,
            "success_rate": float(g["success"].mean()) if len(g) > 0 else np.nan,
            # Use mutually-exclusive first failure reason for plotting failure-rate curves.
            "fail_theta_rate": float((g["term_reason"] == "theta_fail").mean()) if len(g) > 0 else np.nan,
            "fail_x_rate": float((g["term_reason"] == "x_fail").mean()) if len(g) > 0 else np.nan,
            "fail_nan_rate": float((g["term_reason"] == "nan_fail").mean()) if len(g) > 0 else np.nan,
            "fail_other_rate": float(g["term_reason"].isin(["other_fail", "solver_fail_preterm"]).mean()) if len(g) > 0 else np.nan,
            "mean_time_to_term": float(g["time_to_term"].mean()) if "time_to_term" in g.columns and len(g) > 0 else np.nan,
        }

        for m in metrics:
            if m in g.columns:
                row[f"{m}_all_mean"] = float(g[m].mean()) if len(g) > 0 else np.nan
                row[f"{m}_all_std"] = float(g[m].std(ddof=0)) if len(g) > 0 else np.nan
                row[f"{m}_succ_mean"] = float(g_succ[m].mean()) if len(g_succ) > 0 else np.nan
                row[f"{m}_succ_std"] = float(g_succ[m].std(ddof=0)) if len(g_succ) > 0 else np.nan
            else:
                row[f"{m}_all_mean"] = np.nan
                row[f"{m}_all_std"] = np.nan
                row[f"{m}_succ_mean"] = np.nan
                row[f"{m}_succ_std"] = np.nan

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(["controller", "amp"])


def _controller_linestyle(controller: str):
    if str(controller) == "lqr_fd":
        return "-"
    if str(controller) == "mpc_fd":
        return "--"
    return "-"


def _succ_linestyle(controller: str):
    if str(controller) == "lqr_fd":
        return "-"
    if str(controller) == "mpc_fd":
        return (0, (12, 8))
    return "-"


def _censored_linestyle():
    return (0, (2, 2))


def _controller_color(controller: str):
    cmap = {
        "lqr_fd": "#1f77b4",
        "mpc_fd": "#ff7f0e",
    }
    return cmap.get(str(controller), "#4c4c4c")


def _with_alpha(color, alpha: float):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, float(alpha))


def _amp_series_color(idx: int):
    palette = [
        "#ff7f0e",  # orange
        "#1f77b4",  # blue
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    return palette[int(idx) % len(palette)]


def _status_color(name: str):
    cmap = {
        "succ": "#2ca02c",
        "censored": "#d62728",
    }
    return cmap.get(str(name), "#4c4c4c")


def _reason_color(reason: str):
    cmap = {
        "theta_fail": "#d62728",
        "x_fail": "#ff7f0e",
        "nan_fail": "#9467bd",
        "other_fail": "#7f7f7f",
    }
    return cmap.get(str(reason), "#4c4c4c")


def _apply_y_dodge(y, idx: int, n_total: int, frac: float = Y_DODGE_FRAC):
    y = np.asarray(y, dtype=float)
    if n_total <= 1:
        return y
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y
    span = float(np.nanmax(finite) - np.nanmin(finite))
    scale = span if span > 0.0 else max(abs(float(np.nanmean(finite))), 1e-6)
    offset = (idx - (n_total - 1) / 2.0) * float(frac) * scale
    return y + offset


def _plot_metric_lines(ax, df_agg: pd.DataFrame, col: str, ylabel: str, title: str = "", style: str = "-"):
    controllers = sorted(df_agg["controller"].dropna().unique().tolist())
    n = len(controllers)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dodge = X_DODGE
    for i, controller in enumerate(controllers):
        g = df_agg[df_agg["controller"] == controller].sort_values("amp")
        x = g["amp"].to_numpy(dtype=float)
        y = _apply_y_dodge(g[col].to_numpy(dtype=float), i, n)
        if n > 1:
            x = x + (i - (n - 1) / 2.0) * dodge
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            linestyle=_controller_linestyle(controller) if style == "-" else style,
            color=_controller_color(controller),
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor=_with_alpha(_controller_color(controller), MARKER_ALPHA),
            markeredgecolor=_with_alpha(_controller_color(controller), MARKER_EDGE_ALPHA),
            markeredgewidth=0.8,
            label=controller,
        )
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_outcome(df_agg: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    _plot_metric_lines(axes[0], df_agg, "success_rate", "success rate", title="Outcome Rate")
    axes[0].legend()

    reason_cols = [
        ("fail_theta_rate", "theta_fail"),
        ("fail_x_rate", "x_fail"),
        ("fail_nan_rate", "nan_fail"),
        ("fail_other_rate", "other_fail"),
    ]
    controllers = sorted(df_agg["controller"].dropna().unique().tolist())
    n = len(controllers)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dodge = X_DODGE
    for i, controller in enumerate(controllers):
        g = df_agg[df_agg["controller"] == controller].sort_values("amp")
        x = g["amp"].to_numpy(dtype=float)
        if n > 1:
            x = x + (i - (n - 1) / 2.0) * dodge
        for col, name in reason_cols:
            y = _apply_y_dodge(g[col].to_numpy(dtype=float), i, n)
            axes[1].plot(
                x,
                y,
                marker=markers[i % len(markers)],
                linestyle=_controller_linestyle(controller),
                color=_reason_color(name),
                linewidth=1.6,
                markersize=5.0,
                markerfacecolor=_with_alpha(_reason_color(name), MARKER_ALPHA),
                markeredgecolor=_with_alpha(_reason_color(name), MARKER_EDGE_ALPHA),
                markeredgewidth=0.8,
                label=f"{controller}:{name}",
            )
    axes[1].set_xlabel("disturbance amp")
    axes[1].set_ylabel("failure reason rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_constraint_rate(df_agg: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)

    _plot_metric_lines(axes[0], df_agg, "act_rate_u_all_mean", "u act rate", title="Constraint Activation")
    _plot_metric_lines(axes[1], df_agg, "act_rate_du_all_mean", "du act rate")
    axes[1].set_xlabel("disturbance amp")
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_constraint_run(df_agg: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    _plot_metric_lines(axes[0], df_agg, "max_run_u_all_mean", "max run u [steps]", title="Constraint Max Run")
    _plot_metric_lines(axes[1], df_agg, "max_run_du_all_mean", "max run du [steps]")
    axes[1].set_xlabel("disturbance amp")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_succ_vs_censored(
    df_agg: pd.DataFrame,
    metric_succ: str,
    metric_cens: str,
    ylabel: str,
    title: str,
    outpath: str,
    color_mode: str = "status",
):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_succ_vs_censored_on_ax(
        ax=ax,
        df_agg=df_agg,
        metric_succ=metric_succ,
        metric_cens=metric_cens,
        ylabel=ylabel,
        title=title,
        color_mode=color_mode,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_succ_vs_censored_on_ax(
    ax,
    df_agg: pd.DataFrame,
    metric_succ: str,
    metric_cens: str,
    ylabel: str,
    title: str,
    color_mode: str = "status",
):
    controllers = sorted(df_agg["controller"].dropna().unique().tolist())
    n = len(controllers)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dodge = X_DODGE
    for i, controller in enumerate(controllers):
        g = df_agg[df_agg["controller"] == controller].sort_values("amp")
        x = g["amp"].to_numpy(dtype=float)
        y_succ = _apply_y_dodge(g[metric_succ].to_numpy(dtype=float), i, n)
        y_cens = _apply_y_dodge(g[metric_cens].to_numpy(dtype=float), i, n)
        if n > 1:
            x = x + (i - (n - 1) / 2.0) * dodge
        mk = markers[i % len(markers)]
        if color_mode == "controller":
            c_succ = _controller_color(controller)
            c_cens = _controller_color(controller)
        else:
            c_succ = _status_color("succ")
            c_cens = _status_color("censored")
        ax.plot(
            x,
            y_succ,
            marker=mk,
            linestyle=_succ_linestyle(controller),
            color=c_succ,
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor=_with_alpha(c_succ, MARKER_ALPHA),
            markeredgecolor=_with_alpha(c_succ, MARKER_EDGE_ALPHA),
            markeredgewidth=0.8,
            label=f"{controller} succ",
        )
        ax.plot(
            x,
            y_cens,
            marker=mk,
            linestyle=_censored_linestyle(),
            color=c_cens,
            linewidth=1.1,
            markersize=4.8,
            markerfacecolor=_with_alpha(c_cens, MARKER_ALPHA * 0.35),
            markeredgecolor=_with_alpha(c_cens, MARKER_EDGE_ALPHA),
            markeredgewidth=1.0,
            alpha=0.8,
            label=f"{controller} censored",
        )

    ax.set_xlabel("disturbance amp")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    return ax


def _single_seed_u_series_until_lqr_fail(step_log_dir: str, amp: float, seed: int = 0, k_start: int = 105):
    if not step_log_dir or (not os.path.isdir(step_log_dir)):
        return None

    lqr_p = _step_csv_path(step_log_dir, "lqr_fd", float(amp), int(seed))
    mpc_p = _step_csv_path(step_log_dir, "mpc_fd", float(amp), int(seed))
    if not (os.path.exists(lqr_p) and os.path.exists(mpc_p)):
        return None

    lqr = pd.read_csv(lqr_p)
    mpc = pd.read_csv(mpc_p)
    n = min(len(lqr), len(mpc))
    if n <= k_start:
        return None

    term = np.where(lqr["terminated"].to_numpy(dtype=bool)[:n])[0]
    k_fail = int(term[0]) if term.size > 0 else (n - 1)
    k_end = min(k_fail, n - 1)
    if k_end < k_start:
        return None

    lqr_u = lqr["u_applied"] if "u_applied" in lqr.columns else lqr["u_raw"]
    mpc_u = mpc["u_applied"] if "u_applied" in mpc.columns else mpc["u_raw"]
    lqr_u = lqr_u.to_numpy(dtype=float)[:n]
    mpc_u = mpc_u.to_numpy(dtype=float)[:n]

    k_vals = np.arange(int(k_start), int(k_end) + 1, dtype=int)

    return {
        "k": k_vals,
        "lqr_u": lqr_u[k_vals],
        "mpc_u": mpc_u[k_vals],
        "du": (mpc_u - lqr_u)[k_vals],
        "lqr_u2": np.square(lqr_u[k_vals]),
        "mpc_u2": np.square(mpc_u[k_vals]),
        "k_fail": float(k_fail),
    }


def plot_u_energy_with_post_series(
    df_agg: pd.DataFrame,
    outpath: str,
    step_log_dir: str,
    amp_for_series: float = 280.0,
    seed_for_series: int = 0,
    k_start: int = 105,
    k_len: int = 20,
):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11.2), sharex=False)

    n_eval = max(int(k_len), 1)
    k_end = int(k_start) + n_eval - 1
    metric_succ = "u2_fixed_sum_succ_mean"
    metric_cens = "u2_fixed_sum_all_mean"
    ylabel = f"sum(u^2), k={int(k_start)}..{k_end}"
    title = f"Control Effort (fixed-window sum, n={n_eval})"
    if (metric_succ not in df_agg.columns) or (metric_cens not in df_agg.columns):
        metric_succ = "u2_mean_succ_mean"
        metric_cens = "u2_mean_all_mean"
        ylabel = "mean(u^2)"
        title = "Control Effort (succ vs all)"
    elif (not np.isfinite(df_agg[metric_succ]).any()) and (not np.isfinite(df_agg[metric_cens]).any()):
        metric_succ = "u2_mean_succ_mean"
        metric_cens = "u2_mean_all_mean"
        ylabel = "mean(u^2)"
        title = "Control Effort (succ vs all)"

    _plot_succ_vs_censored_on_ax(
        ax=axes[0],
        df_agg=df_agg,
        metric_succ=metric_succ,
        metric_cens=metric_cens,
        ylabel=ylabel,
        title=title,
        color_mode="controller",
    )

    s = _single_seed_u_series_until_lqr_fail(
        step_log_dir=step_log_dir,
        amp=float(amp_for_series),
        seed=int(seed_for_series),
        k_start=int(k_start),
    )
    if s is None:
        axes[1].text(0.5, 0.5, "No step logs for requested amp/seed.", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("delta u")
        axes[1].grid(True, alpha=0.3)
        axes[2].text(0.5, 0.5, "No step logs for requested amp/seed.", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("u")
        axes[2].grid(True, alpha=0.3)
    else:
        k = s["k"]
        axes[1].plot(
            k,
            s["du"],
            color="#444444",
            linestyle="-",
            linewidth=1.8,
            label="u_mpc - u_lqr",
        )
        axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
        axes[1].axvline(
            float(s["k_fail"]),
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label=f"LQR fail k={s['k_fail']:.1f}",
        )
        axes[1].set_title(f"Post-disturbance delta u (amp={amp_for_series:g}, seed={int(seed_for_series)}, k>={int(k_start)})")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("delta u")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        axes[2].plot(
            k,
            s["lqr_u"],
            color=_controller_color("lqr_fd"),
            linestyle="-",
            linewidth=1.8,
            label="lqr_fd u",
        )
        axes[2].plot(
            k,
            s["mpc_u"],
            color=_controller_color("mpc_fd"),
            linestyle="--",
            linewidth=1.8,
            label="mpc_fd u",
        )
        axes[2].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
        axes[2].axvline(
            float(s["k_fail"]),
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label=f"LQR fail k={s['k_fail']:.1f}",
        )
        axes[2].set_title(f"Post-disturbance u (amp={amp_for_series:g}, seed={int(seed_for_series)}, k>={int(k_start)})")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("u")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_theta_max_post(df_agg: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    controllers = sorted(df_agg["controller"].dropna().unique().tolist())
    n = len(controllers)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dodge = X_DODGE

    for i, controller in enumerate(controllers):
        g = df_agg[df_agg["controller"] == controller].sort_values("amp")
        x = g["amp"].to_numpy(dtype=float)
        y_row1_succ = _apply_y_dodge(g["theta_max_post_succ_succ_mean"].to_numpy(dtype=float), i, n)
        y_row2 = _apply_y_dodge(g["theta_max_post_zero_succ_succ_mean"].to_numpy(dtype=float), i, n)
        if n > 1:
            x = x + (i - (n - 1) / 2.0) * dodge
        mk = markers[i % len(markers)]

        # Row 1: success-only theta max post.
        axes[0].plot(
            x,
            y_row1_succ,
            marker=mk,
            linestyle=_succ_linestyle(controller),
            color=_controller_color(controller),
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor=_with_alpha(_controller_color(controller), MARKER_ALPHA),
            markeredgecolor=_with_alpha(_controller_color(controller), MARKER_EDGE_ALPHA),
            markeredgewidth=0.8,
            label=f"{controller} succ",
        )

        # Row 2: post-zero-crossing max theta (success only).
        axes[1].plot(
            x,
            y_row2,
            marker=mk,
            linestyle=_succ_linestyle(controller),
            color=_controller_color(controller),
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor=_with_alpha(_controller_color(controller), MARKER_ALPHA),
            markeredgecolor=_with_alpha(_controller_color(controller), MARKER_EDGE_ALPHA),
            markeredgewidth=0.8,
            label=f"{controller} post-zero succ",
        )

    axes[0].set_ylabel("theta max post [rad]")
    axes[0].set_title("Theta Max Post")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_ylabel("theta max after first zero [rad]")
    axes[1].set_xlabel("disturbance amp")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_recovery(df_agg: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    controllers = sorted(df_agg["controller"].dropna().unique().tolist())
    n = len(controllers)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dodge = X_DODGE
    for i, controller in enumerate(controllers):
        g = df_agg[df_agg["controller"] == controller].sort_values("amp")
        x = g["amp"].to_numpy(dtype=float)
        y = _apply_y_dodge(g["recovery_time_succ_succ_mean"].to_numpy(dtype=float), i, n)
        if n > 1:
            x = x + (i - (n - 1) / 2.0) * dodge
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            linestyle=_controller_linestyle(controller),
            color=_controller_color(controller),
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor=_with_alpha(_controller_color(controller), MARKER_ALPHA),
            markeredgecolor=_with_alpha(_controller_color(controller), MARKER_EDGE_ALPHA),
            markeredgewidth=0.8,
            label=f"{controller} succ",
        )

    ax.set_xlabel("disturbance amp")
    ax.set_ylabel("recovery time [s] (success only)")
    ax.set_title("Recovery Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _step_csv_path(step_log_dir: str, controller: str, amp: float, seed: int):
    return os.path.join(step_log_dir, f"{controller}_amp{amp:.3f}_seed{seed}.csv")


def _read_step_pair(step_log_dir: str, amp: float, seed: int):
    lqr_p = _step_csv_path(step_log_dir, "lqr_fd", amp, seed)
    mpc_p = _step_csv_path(step_log_dir, "mpc_fd", amp, seed)
    if not (os.path.exists(lqr_p) and os.path.exists(mpc_p)):
        return None
    lqr = pd.read_csv(lqr_p)
    mpc = pd.read_csv(mpc_p)
    if (len(lqr) <= 0) or (len(mpc) <= 0):
        return None
    return lqr.copy(), mpc.copy()


def _disturbance_window_from_df(df: pd.DataFrame):
    if len(df) == 0 or "t0" not in df.columns:
        return None
    t0 = int(df["t0"].iloc[0])
    kind = str(df["kind"].iloc[0]) if "kind" in df.columns else "window"
    duration = int(df["duration"].iloc[0]) if "duration" in df.columns else 1
    if kind in ("impulse", "pulse"):
        duration = 1
    return t0, max(duration, 1)


def _augment_u2_fixed_sum_from_step_logs(df: pd.DataFrame, step_log_dir: str, k_start: int, k_len: int):
    if len(df) == 0 or (not step_log_dir):
        return df

    k0 = max(int(k_start), 0)
    n_eval = max(int(k_len), 1)
    vals = []
    for _, r in df.iterrows():
        controller = str(r["controller"])
        amp = float(r["amp"])
        seed = int(r["seed"])
        p = _step_csv_path(step_log_dir, controller, amp, seed)
        if not os.path.exists(p):
            vals.append(np.nan)
            continue

        try:
            s = pd.read_csv(p)
        except Exception:
            vals.append(np.nan)
            continue

        if "u_applied" in s.columns:
            u = s["u_applied"].to_numpy(dtype=float)
        elif "u_raw" in s.columns:
            u = s["u_raw"].to_numpy(dtype=float)
        else:
            vals.append(np.nan)
            continue

        if k0 >= len(u):
            vals.append(0.0)
            continue
        seg = u[k0:min(len(u), k0 + n_eval)]
        vals.append(float(np.sum(np.square(seg))))

    out = df.copy()
    out["u2_fixed_sum"] = vals
    return out


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


def _augment_theta_max_post_zero_from_step_logs(df: pd.DataFrame, step_log_dir: str):
    if len(df) == 0 or (not step_log_dir):
        return df
    if "theta_max_post_zero_succ" in df.columns and df["theta_max_post_zero_succ"].notna().any():
        return df

    vals = []
    for _, r in df.iterrows():
        if int(r.get("success", 0)) != 1:
            vals.append(np.nan)
            continue

        controller = str(r["controller"])
        amp = float(r["amp"])
        seed = int(r["seed"])
        p = _step_csv_path(step_log_dir, controller, amp, seed)
        if not os.path.exists(p):
            vals.append(np.nan)
            continue

        try:
            s = pd.read_csv(p)
        except Exception:
            vals.append(np.nan)
            continue

        if "theta" not in s.columns:
            vals.append(np.nan)
            continue

        t0 = int(r.get("t0", 0))
        duration = int(r.get("duration", 1))
        kind = str(r.get("kind", "window"))
        duration_eff = 1 if kind in ("impulse", "pulse") else duration
        post_start_idx = max(t0 + duration_eff, 0)
        theta = s["theta"].to_numpy(dtype=float)
        vals.append(_theta_max_after_first_zero_crossing(theta, start_idx=post_start_idx))

    df = df.copy()
    df["theta_max_post_zero_succ"] = vals
    return df


def _recovery_start_index(
    df_step: pd.DataFrame,
    t0: int,
    duration: int,
    dt: float,
    theta_tol: float,
    theta_dot_tol: float,
    settle_hold_sec: float,
):
    if len(df_step) == 0:
        return None
    if ("theta" not in df_step.columns) or ("thetadot" not in df_step.columns):
        return None

    hold_steps = max(1, int(round(float(settle_hold_sec) / float(dt))))
    post_start_idx = max(int(t0 + duration), 0)

    theta = df_step["theta"].to_numpy(dtype=float)
    theta_dot = df_step["thetadot"].to_numpy(dtype=float)
    if post_start_idx >= len(theta):
        return None

    in_band = (np.abs(theta) <= float(theta_tol)) & (np.abs(theta_dot) <= float(theta_dot_tol))
    post = in_band[post_start_idx:]
    if np.all(post):
        return None

    for i in range(0, len(post) - hold_steps + 1):
        if np.all(post[i:i + hold_steps]):
            return int(post_start_idx + i)
    return None


def _plot_recovery_marker(axes, df_step: pd.DataFrame, idx: int, dt: float, color, marker: str):
    if idx is None or idx < 0 or idx >= len(df_step):
        return

    t_rec = float(df_step["k"].iloc[idx]) * float(dt) if "k" in df_step.columns else float(idx) * float(dt)
    u_series = df_step["u_applied"] if "u_applied" in df_step.columns else df_step.get("u_raw")
    if u_series is None:
        return

    yvals = [
        float(df_step["theta"].iloc[idx]),
        float(df_step["thetadot"].iloc[idx]),
        float(df_step["x"].iloc[idx]),
        float(u_series.iloc[idx]),
    ]
    for ax, y in zip(axes, yvals):
        ax.scatter(
            [t_rec],
            [y],
            marker=marker,
            s=34,
            color=_with_alpha(color, MARKER_ALPHA),
            edgecolors=_with_alpha(color, MARKER_EDGE_ALPHA),
            linewidths=0.7,
            zorder=6,
            alpha=1.0,
        )


def plot_timeseries_multi_channel(
    step_log_dir: str,
    outdir: str,
    amps,
    seed: int,
    dt: float,
    u_max: float,
    theta_limit: float,
    x_limit: float,
    disturb_win,
    recovery_theta_tol: float,
    recovery_theta_dot_tol: float,
    recovery_hold_sec: float,
):
    amps = sorted(float(a) for a in amps)
    available = []
    for amp in amps:
        pair = _read_step_pair(step_log_dir, amp, seed)
        if pair is not None:
            available.append((amp, pair[0], pair[1]))

    if not available:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.6), sharex=True)

    for i, (amp, lqr, mpc) in enumerate(available):
        t_lqr = lqr["k"].to_numpy(dtype=float) * float(dt)
        t_mpc = mpc["k"].to_numpy(dtype=float) * float(dt)
        color = _amp_series_color(i)

        axes[0].plot(t_lqr, lqr["theta"], color=color, linestyle="-", linewidth=1.3, alpha=0.9, label=f"LQR amp={amp:g}")
        axes[0].plot(t_mpc, mpc["theta"], color=color, linestyle="--", linewidth=1.5, alpha=0.9, label=f"MPC amp={amp:g}")

        axes[1].plot(t_lqr, lqr["thetadot"], color=color, linestyle="-", linewidth=1.3, alpha=0.9)
        axes[1].plot(t_mpc, mpc["thetadot"], color=color, linestyle="--", linewidth=1.5, alpha=0.9)

        axes[2].plot(t_lqr, lqr["x"], color=color, linestyle="-", linewidth=1.3, alpha=0.9)
        axes[2].plot(t_mpc, mpc["x"], color=color, linestyle="--", linewidth=1.5, alpha=0.9)

        lqr_u = lqr["u_applied"] if "u_applied" in lqr.columns else lqr["u_raw"]
        mpc_u = mpc["u_applied"] if "u_applied" in mpc.columns else mpc["u_raw"]
        axes[3].plot(t_lqr, lqr_u, color=color, linestyle="-", linewidth=1.3, alpha=0.9)
        axes[3].plot(t_mpc, mpc_u, color=color, linestyle="--", linewidth=1.5, alpha=0.9)

        if disturb_win is not None:
            t0, duration = disturb_win
            rec_lqr = _recovery_start_index(
                lqr,
                t0=t0,
                duration=duration,
                dt=dt,
                theta_tol=recovery_theta_tol,
                theta_dot_tol=recovery_theta_dot_tol,
                settle_hold_sec=recovery_hold_sec,
            )
            rec_mpc = _recovery_start_index(
                mpc,
                t0=t0,
                duration=duration,
                dt=dt,
                theta_tol=recovery_theta_tol,
                theta_dot_tol=recovery_theta_dot_tol,
                settle_hold_sec=recovery_hold_sec,
            )
            _plot_recovery_marker(axes, lqr, rec_lqr, dt=dt, color=color, marker="D")
            _plot_recovery_marker(axes, mpc, rec_mpc, dt=dt, color=color, marker="o")

    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("thetadot [rad/s]")
    axes[2].set_ylabel("x [m]")
    axes[3].set_ylabel("u")
    axes[3].set_xlabel("time [s]")

    if np.isfinite(theta_limit):
        axes[0].axhline(float(theta_limit), color="black", linestyle=":", linewidth=1.0)
        axes[0].axhline(-float(theta_limit), color="black", linestyle=":", linewidth=1.0)
    if np.isfinite(x_limit):
        axes[2].axhline(float(x_limit), color="black", linestyle=":", linewidth=1.0)
        axes[2].axhline(-float(x_limit), color="black", linestyle=":", linewidth=1.0)
    axes[3].axhline(float(u_max), color="black", linestyle=":", linewidth=1.0)
    axes[3].axhline(-float(u_max), color="black", linestyle=":", linewidth=1.0)

    if disturb_win is not None:
        t0, duration = disturb_win
        t1 = t0 * dt
        t2 = (t0 + duration) * dt
        for ax in axes:
            ax.axvspan(t1, t2, color="gray", alpha=0.12)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"State/Input Timeseries for Selected Amps (seed={seed})")
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    rec_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=6,
            markerfacecolor=_with_alpha("black", MARKER_ALPHA),
            markeredgecolor=_with_alpha("black", MARKER_EDGE_ALPHA),
            label="LQR recovery point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=_with_alpha("black", MARKER_ALPHA),
            markeredgecolor=_with_alpha("black", MARKER_EDGE_ALPHA),
            label="MPC recovery point",
        ),
    ]
    axes[0].legend(handles=line_handles + rec_handles, labels=line_labels + [h.get_label() for h in rec_handles], fontsize=7, ncol=2)

    fig.tight_layout()
    out = os.path.join(outdir, f"u_timeseries_all_amps_seed{seed}.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_timeseries_single_amp(
    step_log_dir: str,
    outdir: str,
    amp: float,
    seed: int,
    dt: float,
    u_max: float,
    theta_limit: float,
    x_limit: float,
    disturb_win,
    recovery_theta_tol: float,
    recovery_theta_dot_tol: float,
    recovery_hold_sec: float,
):
    pair = _read_step_pair(step_log_dir, amp, seed)
    if pair is None:
        return None

    lqr, mpc = pair
    t_lqr = lqr["k"].to_numpy(dtype=float) * float(dt)
    t_mpc = mpc["k"].to_numpy(dtype=float) * float(dt)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.4), sharex=True)
    axes[0].plot(t_lqr, lqr["theta"], label="lqr theta", linestyle="-")
    axes[0].plot(t_mpc, mpc["theta"], label="mpc theta", linestyle="--")

    axes[1].plot(t_lqr, lqr["thetadot"], label="lqr thetadot", linestyle="-")
    axes[1].plot(t_mpc, mpc["thetadot"], label="mpc thetadot", linestyle="--")

    axes[2].plot(t_lqr, lqr["x"], label="lqr x", linestyle="-")
    axes[2].plot(t_mpc, mpc["x"], label="mpc x", linestyle="--")

    lqr_u = lqr["u_applied"] if "u_applied" in lqr.columns else lqr["u_raw"]
    mpc_u = mpc["u_applied"] if "u_applied" in mpc.columns else mpc["u_raw"]
    axes[3].plot(t_lqr, lqr_u, label="lqr u", linestyle="-")
    axes[3].plot(t_mpc, mpc_u, label="mpc u", linestyle="--")

    if disturb_win is not None:
        t0, duration = disturb_win
        rec_lqr = _recovery_start_index(
            lqr,
            t0=t0,
            duration=duration,
            dt=dt,
            theta_tol=recovery_theta_tol,
            theta_dot_tol=recovery_theta_dot_tol,
            settle_hold_sec=recovery_hold_sec,
        )
        rec_mpc = _recovery_start_index(
            mpc,
            t0=t0,
            duration=duration,
            dt=dt,
            theta_tol=recovery_theta_tol,
            theta_dot_tol=recovery_theta_dot_tol,
            settle_hold_sec=recovery_hold_sec,
        )
        _plot_recovery_marker(axes, lqr, rec_lqr, dt=dt, color="C0", marker="D")
        _plot_recovery_marker(axes, mpc, rec_mpc, dt=dt, color="C1", marker="o")

    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("thetadot [rad/s]")
    axes[2].set_ylabel("x [m]")
    axes[3].set_ylabel("u")
    axes[3].set_xlabel("time [s]")

    if np.isfinite(theta_limit):
        axes[0].axhline(float(theta_limit), color="black", linestyle=":", linewidth=1.0)
        axes[0].axhline(-float(theta_limit), color="black", linestyle=":", linewidth=1.0)
    if np.isfinite(x_limit):
        axes[2].axhline(float(x_limit), color="black", linestyle=":", linewidth=1.0)
        axes[2].axhline(-float(x_limit), color="black", linestyle=":", linewidth=1.0)
    axes[3].axhline(float(u_max), color="black", linestyle=":", linewidth=1.0)
    axes[3].axhline(-float(u_max), color="black", linestyle=":", linewidth=1.0)

    if disturb_win is not None:
        t0, duration = disturb_win
        t1 = t0 * dt
        t2 = (t0 + duration) * dt
        for ax in axes:
            ax.axvspan(t1, t2, color="gray", alpha=0.12)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Timeseries (amp={amp:.3f}, seed={seed})")
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    rec_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=6,
            markerfacecolor=_with_alpha("black", MARKER_ALPHA),
            markeredgecolor=_with_alpha("black", MARKER_EDGE_ALPHA),
            label="LQR recovery point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=_with_alpha("black", MARKER_ALPHA),
            markeredgecolor=_with_alpha("black", MARKER_EDGE_ALPHA),
            label="MPC recovery point",
        ),
    ]
    axes[0].legend(handles=line_handles + rec_handles, labels=line_labels + [h.get_label() for h in rec_handles])

    fig.tight_layout()
    out = os.path.join(outdir, f"u_timeseries_amp{amp:.3f}_seed{seed}.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/fd_compare/summary_force_fd_compare.csv")
    parser.add_argument("--outdir", type=str, default="plots/fd_compare")
    parser.add_argument(
        "--step-log-dir",
        type=str,
        default="logs/fd_compare/steps",
        help="Directory for per-step CSV logs (default: logs/fd_compare/steps).",
    )
    parser.add_argument("--u-seed", type=int, default=0, help="Seed to use for timeseries plot")
    parser.add_argument("--u-amp", type=float, default=None, help="Optional single amp plot")
    parser.add_argument(
        "--u-amp-idx",
        type=str,
        default="",
        help="1-based amp indices for multi-amp timeseries, e.g. 1,3,5. Default: all amps.",
    )
    parser.add_argument("--recovery-theta-tol", type=float, default=0.05236, help="Theta threshold for recovery marker.")
    parser.add_argument("--recovery-thetadot-tol", type=float, default=0.1, help="Theta-dot threshold for recovery marker.")
    parser.add_argument("--recovery-hold-sec", type=float, default=0.5, help="Required in-band hold time for recovery marker.")
    parser.add_argument("--u-energy-amp", type=float, default=280.0, help="Amp to plot u^2 post-disturbance timeseries on row 2.")
    parser.add_argument("--u-energy-seed", type=int, default=0, help="Seed to plot on u_energy rows 2/3.")
    parser.add_argument("--u-energy-k-start", type=int, default=105, help="Start step k for u^2 post-disturbance timeseries.")
    parser.add_argument("--u-energy-k-len", type=int, default=20, help="Fixed window length for row-1 control effort: sum_{k=k_start}^{k_start+len-1} u_k^2.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["amp"] = df["amp"].astype(float)
    df["success"] = df["success"].astype(int)
    df = _ensure_columns(df)
    df = _augment_u2_fixed_sum_from_step_logs(
        df,
        step_log_dir=args.step_log_dir,
        k_start=int(args.u_energy_k_start),
        k_len=int(args.u_energy_k_len),
    )
    df = _augment_theta_max_post_zero_from_step_logs(df, step_log_dir=args.step_log_dir)

    df_agg = agg_by_controller_amp(df)
    df_agg.to_csv(os.path.join(args.outdir, "summary_agg.csv"), index=False)

    # 1) Outcome
    plot_outcome(df_agg, os.path.join(args.outdir, "success_rate.png"))

    # 2) Constraint activation
    plot_constraint_rate(df_agg, os.path.join(args.outdir, "sat_rate.png"))

    # 3) Constraint max run
    plot_constraint_run(df_agg, os.path.join(args.outdir, "sat_max_run.png"))

    # 4) theta max
    plot_theta_max_post(
        df_agg=df_agg,
        outpath=os.path.join(args.outdir, "theta_max_post.png"),
    )

    # 5) theta rms (succ vs censored)
    plot_succ_vs_censored(
        df_agg=df_agg,
        metric_succ="theta_rms_post_succ_succ_mean",
        metric_cens="theta_rms_post_cens_all_mean",
        ylabel="theta RMS post [rad]",
        title="Theta RMS Post (succ vs censored)",
        outpath=os.path.join(args.outdir, "theta_rms_post.png"),
    )

    # 6) recovery (succ only + coverage)
    plot_recovery(df_agg, os.path.join(args.outdir, "recovery_time.png"))

    # 7) u energy (row1: fixed-window sum(u^2), row2/3: post-disturbance timeseries for selected amp)
    plot_u_energy_with_post_series(
        df_agg=df_agg,
        outpath=os.path.join(args.outdir, "u_energy.png"),
        step_log_dir=args.step_log_dir,
        amp_for_series=float(args.u_energy_amp),
        seed_for_series=int(args.u_energy_seed),
        k_start=int(args.u_energy_k_start),
        k_len=int(args.u_energy_k_len),
    )

    # 8) empirical stage cost (succ vs all)
    plot_succ_vs_censored(
        df_agg=df_agg,
        metric_succ="j_stage_mean_succ_mean",
        metric_cens="j_stage_mean_all_mean",
        ylabel="mean stage cost",
        title="Empirical Cost (succ vs all)",
        outpath=os.path.join(args.outdir, "J_emp.png"),
        color_mode="controller",
    )

    # 9) timeseries (theta/x/u)
    dt = float(df["dt"].iloc[0]) if "dt" in df.columns and len(df) > 0 else 1.0
    u_max = float(df["u_max"].iloc[0]) if "u_max" in df.columns and len(df) > 0 else 3.0
    theta_limit = float(df["termination_theta"].iloc[0]) if "termination_theta" in df.columns and len(df) > 0 else np.nan
    x_limit = float(df["x_fail_limit"].iloc[0]) if "x_fail_limit" in df.columns and len(df) > 0 else np.nan
    disturb_win = _disturbance_window_from_df(df)

    u_plot_msg = "timeseries plot skipped (step logs not found)."
    if args.step_log_dir:
        msgs = []
        amps_ordered = ordered_amps_from_df(df)
        amps_for_multi = select_amps_by_indices(amps_ordered, args.u_amp_idx)

        out_multi = plot_timeseries_multi_channel(
            step_log_dir=args.step_log_dir,
            outdir=args.outdir,
            amps=amps_for_multi,
            seed=args.u_seed,
            dt=dt,
            u_max=u_max,
            theta_limit=theta_limit,
            x_limit=x_limit,
            disturb_win=disturb_win,
            recovery_theta_tol=args.recovery_theta_tol,
            recovery_theta_dot_tol=args.recovery_thetadot_tol,
            recovery_hold_sec=args.recovery_hold_sec,
        )
        if out_multi is not None:
            msgs.append(f"multi-channel timeseries saved: {out_multi}")

        if args.u_amp is not None:
            out_single = plot_timeseries_single_amp(
                step_log_dir=args.step_log_dir,
                outdir=args.outdir,
                amp=float(args.u_amp),
                seed=args.u_seed,
                dt=dt,
                u_max=u_max,
                theta_limit=theta_limit,
                x_limit=x_limit,
                disturb_win=disturb_win,
                recovery_theta_tol=args.recovery_theta_tol,
                recovery_theta_dot_tol=args.recovery_thetadot_tol,
                recovery_hold_sec=args.recovery_hold_sec,
            )
            if out_single is not None:
                msgs.append(f"single-amp timeseries saved: {out_single}")

        if msgs:
            u_plot_msg = " ".join(msgs)

    print(f"OK: plots saved to {args.outdir}/ and summary_agg.csv written. {u_plot_msg}")


if __name__ == "__main__":
    main()
