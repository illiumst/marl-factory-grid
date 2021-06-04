import gym
import glob
from environments.policy_adaption.natural_rl_environment.imgsource import *
from environments.policy_adaption.natural_rl_environment.natural_env import *

if __name__ == "__main__":
    imgsource = 'video'
    env = gym.make('SpaceInvaders-v0')  # gravitar, breakout, MsPacman, Space Invaders
    shape2d = env.observation_space.shape[:2]
    print(shape2d)

    if imgsource == 'video':
        imgsource = RandomVideoSource(shape2d, ['/Users/romue/PycharmProjects/EDYS/environments/policy_adaption/natural_rl_environment/videos/stars.mp4'])
    elif imgsource == "color":
        imgsource = RandomColorSource(shape2d)
    elif imgsource == "noise":
        imgsource = NoiseSource(shape2d)
    wrapped_env = ReplaceBackgroundEnv(
        #env, BackgroundMattingWithColor((144, 72, 17)), imgsource
        env, BackgroundMattingWithColor((0, 0, 0)), imgsource
    )
    env = wrapped_env


    env.reset()

    state, *_ = env.step(env.action_space.sample())
    play.play(wrapped_env, zoom=4)