# xvfb-run -s "-screen 0 1400x900x24" python3 record.py

from stable_baselines3 import PPO

from kitchen_env import KitchenEnv
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers.flatten_observation import FlattenObservation

env = KitchenEnv(render_mode="rgb_array")
env = FlattenObservation(env)
env = RecordVideo(env, video_folder="./recorded_video", name_prefix="test-video")

model = PPO.load("ppo_kitchen_env")
obs, info = env.reset(seed=42)
env.start_video_recorder()

for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    env.render()
    print(f"Step {_}")

env.close_video_recorder()
env.close()
