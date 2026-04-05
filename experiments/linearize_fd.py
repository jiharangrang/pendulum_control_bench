import numpy as np
import gymnasium as gym

ENV_ID = "InvertedPendulum-v5"

def step_from_state(env, x, u):
    # obs = [qpos, qvel]라고 가정 (앞에서 확인해야 함)
    qpos = np.array(x[:2], dtype=np.float64)
    qvel = np.array(x[2:], dtype=np.float64)

    env.unwrapped.set_state(qpos, qvel)
    obs, r, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
    return np.array(obs, dtype=np.float64)

def main():
    env = gym.make(ENV_ID)
    obs0, _ = env.reset(seed=0)

    x0 = np.zeros(4, dtype=np.float64)  # 직립 근처 평형 가정
    u0 = 0.0

    eps_x = 1e-5
    eps_u = 1e-5

    n = 4
    m = 1
    Ad = np.zeros((n, n), dtype=np.float64)
    Bd = np.zeros((n, m), dtype=np.float64)

    fx0 = step_from_state(env, x0, u0)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps_x
        f_plus  = step_from_state(env, x0 + dx, u0)
        f_minus = step_from_state(env, x0 - dx, u0)
        Ad[:, i] = (f_plus - f_minus) / (2 * eps_x)

    for j in range(m):
        du = eps_u
        f_plus  = step_from_state(env, x0, u0 + du)
        f_minus = step_from_state(env, x0, u0 - du)
        Bd[:, j] = (f_plus - f_minus) / (2 * eps_u)

    print("Ad=\n", Ad)
    print("Bd=\n", Bd)

    env.close()

if __name__ == "__main__":
    main()
