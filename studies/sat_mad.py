from environments.factory import rooms
import random
from gym.wrappers import FrameStack

env = rooms(n_agents=2)
env = FrameStack(env, num_stack=3)
state, *_ = env.reset()

for i in range(1000):
    state, *_ = env.step([random.randint(0, 9), random.randint(0, 9)])
    env.render()