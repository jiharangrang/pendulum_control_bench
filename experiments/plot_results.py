# experiments/plot_results.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def agg_by_controller_amp(df: pd.DataFrame):
    out_rows = []
    for (controller, amp), g in df.groupby(["controller", "amp"]):
        success_rate = g["success"].mean()

        # 성공한 rollout 기준 평균 (실패 포함시키면 지표 해석이 애매해짐)
        g_succ = g[g["success"] == 1]

        out_rows.append({
            "controller": controller,
            "amp": amp,
            "success_rate": success_rate,
            "mean_u_energy": g_succ["u_energy"].mean() if len(g_succ) else np.nan,
            "mean_J_emp": g_succ["J_emp"].mean() if len(g_succ) else np.nan,
            "mean_sat_rate": g_succ["sat_rate"].mean() if len(g_succ) else np.nan,
            "mean_sat_max_run": g_succ["sat_max_run"].mean() if len(g_succ) else np.nan,
            "mean_theta_max_post": g_succ["theta_max_post"].mean() if len(g_succ) else np.nan,
            "mean_theta_rms_post": g_succ["theta_rms_post"].mean() if len(g_succ) else np.nan,
            "mean_recovery_time": g_succ["recovery_time"].mean() if len(g_succ) else np.nan,
            "n": len(g),
            "n_success": len(g_succ),
        })
    return pd.DataFrame(out_rows).sort_values(["controller", "amp"])


def plot_metric(df_agg: pd.DataFrame, metric: str, ylabel: str, outpath: str):
    plt.figure()
    for controller, g in df_agg.groupby("controller"):
        g = g.sort_values("amp")
        plt.plot(g["amp"], g[metric], marker="o", label=controller)
    plt.xlabel("disturbance amp")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/summary.csv")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # 타입 정리
    df["amp"] = df["amp"].astype(float)
    df["success"] = df["success"].astype(int)

    df_agg = agg_by_controller_amp(df)
    df_agg.to_csv(os.path.join(args.outdir, "summary_agg.csv"), index=False)

    plot_metric(df_agg, "success_rate", "success rate", os.path.join(args.outdir, "success_rate.png"))
    plot_metric(df_agg, "mean_sat_rate", "mean saturation rate (success only)", os.path.join(args.outdir, "sat_rate.png"))
    plot_metric(df_agg, "mean_sat_max_run", "mean max saturation run length [steps] (success only)", os.path.join(args.outdir, "sat_max_run.png"))
    plot_metric(df_agg, "mean_theta_max_post", "mean max |theta| after disturbance [rad] (success only)", os.path.join(args.outdir, "theta_max_post.png"))
    plot_metric(df_agg, "mean_theta_rms_post", "mean theta RMS after disturbance [rad] (success only)", os.path.join(args.outdir, "theta_rms_post.png"))
    plot_metric(df_agg, "mean_recovery_time", "mean recovery time after disturbance [s] (success only)", os.path.join(args.outdir, "recovery_time.png"))
    plot_metric(df_agg, "mean_u_energy", "mean sum(u^2) (success only)", os.path.join(args.outdir, "u_energy.png"))
    plot_metric(df_agg, "mean_J_emp", "mean empirical LQR cost J (success only)", os.path.join(args.outdir, "J_emp.png"))

    print(f"OK: plots saved to {args.outdir}/ and summary_agg.csv written.")


if __name__ == "__main__":
    main()
