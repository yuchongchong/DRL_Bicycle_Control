import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from env import Position_control_td3_complex_line

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = log_dir+"best_model"
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 20 episodes
              mean_reward = np.mean(y[-20:])

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
        return True

env = Position_control_td3_complex_line(render=0)
env.rl = 1
env.record_flag = 1
log_dir = "model/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=env.max_step_num, log_dir=log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

model = TD3(
    "MlpPolicy",                                # MlpPolicy定义策略网络为MLP网络
    env=env,
    gamma=0.99,
    learning_rate=0.001,
    batch_size=256,
    buffer_size=1000000,
    action_noise=action_noise,
    #policy_kwargs=policy_kwargs,
    target_policy_noise=0.1,
    target_noise_clip=0.3,
    learning_starts=3000,
    train_freq=(1, "step"),
    gradient_steps=-1,
    policy_delay=2,
    seed=42,
    verbose=0                              # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
)

model.learn(total_timesteps=1650*500,callback = callback)



