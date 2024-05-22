import time
from stable_baselines3 import TD3
from env import Position_control_td3_complex_line
model = TD3.load("model/position_rl_stanley.zip")

env = Position_control_td3_complex_line(render=1)

env.max_step_num = 1500
env.record_flag = 1
env.stanley = 1
for episode in range(500):
    state = env.reset()

    while 1:

        action = model.predict(state)
        # agent与环境进行一步交互
        next_state, reward, done, _ = env.step(action)
        state = next_state
        time.sleep(0.01)
        if done:
            break

