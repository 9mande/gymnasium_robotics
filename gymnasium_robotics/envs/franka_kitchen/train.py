import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from kitchen_env import KitchenEnv
from gymnasium.wrappers.flatten_observation import FlattenObservation

env = KitchenEnv()
env = FlattenObservation(env)

env = make_vec_env(lambda: env, n_envs=1)

tensorboard_log_dir = "./ppo_kitchen_tensorboard/"


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir,device="cuda")

start_time = time.time()

model.learn(total_timesteps=100000)

end_time = time.time()
training_duration = end_time - start_time


model.save("ppo_kitchen_env")

print(f"Training completed in {training_duration:.2f} seconds")

# 763.57 seconds for 100352 steps