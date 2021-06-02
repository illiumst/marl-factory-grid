import gym

env_dict = gym.envs.registration.registry.env_specs.copy()


for env in env_dict:
    if 'CarRacingColor3-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
#from environments.domain_adaption.natural_rl_environment import natural_env
import environments.domain_adaption.car_racing_variants


env = gym.make('CarRacingColor3-v0')
env.seed(666)

while True:
    ob = env.reset()
    done = False
    step = 0
    while not done and 0 <= step <= 500:
        ob, reward, done, _ = env.step(env.action_space.sample())
        step += 1
        env.render()