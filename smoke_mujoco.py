import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

ENV_ID = "InvertedPendulum-v5"

def main():
    env = gym.make(ENV_ID, render_mode="rgb_array")

    # 매 에피소드 영상 저장 (처음엔 무조건 저장해서 확인)
    env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda ep: ep == 0,  # 0번 에피소드만 저장
    name_prefix="smoke"
)


    obs, info = env.reset(seed=0)

    for _ in range(600):
        action = env.action_space.sample()  # 아직 제어기 없으니 랜덤
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"OK: video saved under ./videos (MUJOCO_GL={os.environ.get('MUJOCO_GL')})")

if __name__ == "__main__":
    main()
