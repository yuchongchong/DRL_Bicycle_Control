import time
from stable_baselines3 import TD3
from env import Attitude_control
model = TD3.load("model/best_model.zip")

env = Attitude_control(render=1)
env.max_step_num = 400
env.record_flag = 1
env.test = 1
env.rl = 1
while True:
    state = env.reset()

    while True:

        action = model.predict(state)
        # agent与环境进行一步交互
        next_state, reward, done, _ = env.step(action)
        state = next_state
        time.sleep(0.01)
        if done:
            break

