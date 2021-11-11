from environments.factory import make
import random
from gym.wrappers import FrameStack

n_agents = 4
env = make('DirtyFactory-v0', n_agents=n_agents)
env = FrameStack(env, num_stack=3)
state, *_ = env.reset()

for i in range(1000):
    state, *_ = env.step([env.unwrapped.action_space.sample() for _ in range(n_agents)])
    env.render()