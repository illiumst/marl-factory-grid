import gym
import glob
from environments.policy_adaption.natural_rl_environment.imgsource import *
from environments.policy_adaption.natural_rl_environment.natural_env import *

if __name__ == "__main__":
    env = make('SpaceInvaders-v0', 'color')  # gravitar, breakout, MsPacman, Space Invaders
    play.play(env, zoom=4)