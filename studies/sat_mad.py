from environments.factory import make
import salina
import torch
from gym.wrappers import FrameStack


class MyAgent(salina.TAgent):
    def __init__(self):
        super(MyAgent, self).__init__()

    def forward(self, t, **kwargs):
        self.set(('timer', t), torch.tensor([t]))


if __name__ == '__main__':
    n_agents = 1
    env = make('DirtyFactory-v0', n_agents=n_agents)
    env = FrameStack(env, num_stack=3)
    env.reset()
    agent = MyAgent()
    workspace = salina.Workspace()
    agent(workspace, t=0, n_steps=10)

    print(workspace)


    for i in range(1000):
        state, *_ = env.step([env.unwrapped.action_space.sample() for _ in range(n_agents)])
        #env.render()