import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0)  # 시작상태 노이즈 최소화
obs, _ = env.reset(seed=0)

m = env.unwrapped.model
d = env.unwrapped.data

# id 찾기 (이름은 문서에 slider/hinge로 명시)
slider_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slider")
hinge_jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

cart_bid = m.jnt_bodyid[slider_jid]
pole_bid = m.jnt_bodyid[hinge_jid]

M = float(m.body_mass[cart_bid])
m_pole = float(m.body_mass[pole_bid])

# hinge의 월드 좌표(조인트 앵커)와 pole body의 COM 월드 좌표 차이 = hinge->COM 거리
hinge_pos = d.xanchor[hinge_jid].copy()
pole_com  = d.xipos[pole_bid].copy()
L = float(np.linalg.norm(pole_com - hinge_pos))

print("obs = [x, theta, xdot, thetadot] =", obs)
print("cart body name:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, cart_bid), "M =", M)
print("pole body name:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, pole_bid), "m =", m_pole)
print("hinge->COM L =", L)

# 참고: dt, 입력 범위도 같이 출력
fs = env.unwrapped.frame_skip
dt = m.opt.timestep * fs
print("dt =", dt)
print("ctrlrange =", m.actuator_ctrlrange)

env.close()
