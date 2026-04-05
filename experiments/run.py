import os, json, time
import numpy as np
from gymnasium.wrappers import RecordVideo

from envs.wrappers import ActionDisturbance, ForceDisturbance, TerminationOverride
from envs.mujoco_model import make_inverted_pendulum

from controllers.lqr import dlqr, zoh_discretize, build_theory_continuous_matrices, get_mujoco_basic_params


ENV_ID = "InvertedPendulum-v5"

def run_once(cfg: dict):
    seed = int(cfg["seed"])
    steps = int(cfg["steps"])
    video = bool(cfg["video"])

    env = make_inverted_pendulum(render_mode="rgb_array" if video else None)
    env = TerminationOverride(env, theta_limit=float(cfg.get("termination_theta", 0.5)))

        # ---- LQR controller setup ----
    controller_name = cfg.get("controller", "random")  # "random", "lqr_theory", "lqr_fd"
    K = None

    if controller_name == "lqr_theory":
        params = get_mujoco_basic_params(env)
        Ac, Bc_u = build_theory_continuous_matrices(
            M=params["M"], m=params["m"], l=params["l"], Iyy=params["Iyy"], gear=params["gear"]
        )
        Ad, Bd = zoh_discretize(Ac, Bc_u, params["dt"])
        print("[LQR theory] Ad =\n", Ad)
        print("[LQR theory] Bd =\n", Bd)

        Q = np.diag(cfg["lqr"]["Q_diag"]).astype(np.float64)
        R = np.array([[cfg["lqr"]["R"]]], dtype=np.float64)

        K, _ = dlqr(Ad, Bd, Q, R)
        print("[LQR theory] K =", K)

    elif controller_name == "lqr_fd":
        # FD 선형화는 (교란 wrapper 영향 없이) 별도 env로 수행하는 게 안전
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
        eps = cfg["lqr"]["fd_eps"]

        Ad = np.zeros((4, 4), dtype=np.float64)
        Bd = np.zeros((4, 1), dtype=np.float64)

        for i in range(4):
            dx = np.zeros(4); dx[i] = eps
            f_p = step_from_state(env_lin, x0 + dx, u0)
            f_m = step_from_state(env_lin, x0 - dx, u0)
            Ad[:, i] = (f_p - f_m) / (2 * eps)

        f_p = step_from_state(env_lin, x0, u0 + eps)
        f_m = step_from_state(env_lin, x0, u0 - eps)
        Bd[:, 0] = (f_p - f_m) / (2 * eps)

        env_lin.close()
        print("[LQR FD] Ad =\n", Ad)
        print("[LQR FD] Bd =\n", Bd)

        Q = np.diag(cfg["lqr"]["Q_diag"]).astype(np.float64)
        R = np.array([[cfg["lqr"]["R"]]], dtype=np.float64)

        K, _ = dlqr(Ad, Bd, Q, R)
        print("[LQR FD] K =", K)

    # 최종 K 로그 (random 모드는 K 없음)
    if K is not None:
        print("[LQR] K =", K)

    # 교란 주입
    if cfg["disturbance"]["enabled"]:
        dcfg = cfg["disturbance"]
        dist_target = dcfg.get("target", cfg.get("disturbance_target", "force"))  # "action" or "force"
        if dist_target == "force":
            env = ForceDisturbance(
                env,
                amp=float(dcfg["amp"]),
                seed=seed,
                kind=dcfg["kind"],
                t0=int(dcfg["t0"]),
                duration=int(dcfg["duration"]),
                joint_name="slider",
            )
        else:
            env = ActionDisturbance(
                env,
                kind=dcfg["kind"],
                amp=float(dcfg["amp"]),
                t0=int(dcfg["t0"]),
                duration=int(dcfg["duration"]),
                omega=float(dcfg["omega"]),
                seed=seed
            )

    # 영상은 첫 에피소드만 저장
    if video:
        env = RecordVideo(
            env,
            video_folder=cfg["video_dir"],
            episode_trigger=lambda ep: ep == 0,
            name_prefix=cfg["run_name"]
        )

    obs, info = env.reset(seed=seed)

    init = cfg.get("init", {})
    if init:
        qpos = env.unwrapped.data.qpos.copy()
        qvel = env.unwrapped.data.qvel.copy()

        # qpos = [x, theta], qvel = [xdot, thetadot]
        if "x" in init:        qpos[0] = float(init["x"])
        if "theta" in init:    qpos[1] = float(init["theta"])
        if "xdot" in init:     qvel[0] = float(init["xdot"])
        if "thetadot" in init: qvel[1] = float(init["thetadot"])

        env.unwrapped.set_state(qpos, qvel)

        # 관측 obs도 동기화
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

    print("[init] theta0 =", env.unwrapped.data.qpos[1])

    u_max = float(env.action_space.high[0])
    u_max_obs = 0.0
    sat_steps = 0
    sat_any = False

    log = {
        "obs0": np.array(obs, dtype=float).tolist(),
        "t": [],
        "obs": [],
        "action": [],
        "disturb_force": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
    }

    for k in range(steps):
        if controller_name == "random":
            action = env.action_space.sample()
        else:
            x = np.array(obs, dtype=np.float64).reshape(4, 1)
            u = -float(K @ x)  # scalar
            u = float(np.clip(u, env.action_space.low[0], env.action_space.high[0]))
            action = np.array([u], dtype=np.float32)

        u_scalar = float(np.array(action, dtype=np.float32).reshape(-1)[0])
        u_abs = abs(u_scalar)
        u_max_obs = max(u_max_obs, u_abs)
        eps = 1e-9
        sat = (u_abs >= (u_max - eps))
        sat_steps += int(sat)
        sat_any = sat_any or sat

        obs, r, terminated, truncated, info = env.step(action)
        dist_f = float(info.get("disturb_force", 0.0))

        log["t"].append(k)
        log["obs"].append(np.array(obs, dtype=float).tolist())
        log["action"].append(np.array(action, dtype=float).tolist())
        log["disturb_force"].append(dist_f)
        log["reward"].append(float(r))
        log["terminated"].append(bool(terminated))
        log["truncated"].append(bool(truncated))

        if terminated or truncated:
            break

    T_run = len(log["action"])
    log["u_max_obs"] = float(u_max_obs)
    log["sat_steps"] = int(sat_steps)
    log["sat_any"] = int(sat_any)
    log["sat_rate"] = float(sat_steps / T_run) if T_run > 0 else 0.0

    env.close()
    return log

def main():
    cfg = {
        "run_name": f"debug_{int(time.time())}",
        "seed": 0,
        "steps": 1000,
        "video": True,
        "video_dir": "videos",
        "log_dir": "logs",
        "disturbance": {
            "enabled": True,
            "kind": "impulse",   # impulse / step / window / sine
            "amp": 10,
            "t0": 200,
            "duration": 50,
            "omega": 0.05,
        },
         "controller": "lqr_fd",  # "random", "lqr_theory", "lqr_fd"
        "lqr": {
            "Q_diag": [1.0, 80.0, 1.0, 10.0],
            "R": 0.1,
            "fd_eps": 1e-5,
        },
    }

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["video_dir"], exist_ok=True)

    log = run_once(cfg)

    outpath = os.path.join(cfg["log_dir"], cfg["run_name"] + ".json")
    with open(outpath, "w") as f:
        json.dump({"cfg": cfg, "log": log}, f, indent=2)

    print("OK:", outpath)

if __name__ == "__main__":
    main()
