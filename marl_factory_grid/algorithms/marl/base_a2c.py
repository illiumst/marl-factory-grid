import numpy as np; import torch as th; import scipy as sp; import gym
import os; from collections import deque; import matplotlib.pyplot as plt
from tqdm import tqdm

# RLLab Magic for calculating the discounted return G(t) = R(t) + gamma * R(t-1)
# cf. https://github.com/rll/rllab/blob/ba78e4c16dc492982e648f117875b22af3965579/rllab/misc/special.py#L107
cumulate_discount = lambda x, gamma: sp.signal.lfilter([1], [1, - gamma], x[::-1], axis=0)[::-1]

class Net(th.nn.Module):
  def __init__(self, shape, activation, lr):
    super().__init__()
    self.net =  th.nn.Sequential(*[ layer
      for io, a in zip(zip(shape[:-1], shape[1:]), [activation] * (len(shape)-2) + [th.nn.Identity] )
        for layer in [th.nn.Linear(*io), a()]])
    self.optimizer =  th.optim.Adam(self.net.parameters(), lr=lr)

class ValueNet(Net):
  def __init__(self, obs_dim, hidden_sizes=[64,64], activation=th.nn.Tanh, lr=1e-3):
    super().__init__([obs_dim] + hidden_sizes + [1], activation, lr)
  def forward(self, obs): return self.net(obs)
  def loss(self, states, returns): return ((returns - self(states))**2).mean()

class PolicyNet(Net):
  def __init__(self, obs_dim, act_dim, hidden_sizes=[64,64], activation=th.nn.Tanh, lr=3e-4):
    super().__init__([obs_dim] + hidden_sizes + [act_dim], activation, lr)
    self.distribution = lambda obs: th.distributions.Categorical(logits=self.net(obs))

  def forward(self, obs, act=None, det=False):
    """Given an observation: Returns policy distribution and probablilty for a given action
      or Returns a sampled action and its corresponding probablilty"""
    pi = self.distribution(obs)
    if act is not None: return pi, pi.log_prob(act)
    act = self.net(obs).argmax() if det else pi.sample()  # sample from the learned distribution
    return act, pi.log_prob(act)

  def loss(self, states, actions, advantages):
    _, logp = self.forward(states, actions)
    loss = -(logp * advantages).mean()
    return loss


class PolicyGradient:
  """ Autonomous agent using vanilla policy gradient. """
  def __init__(self, env, seed=42,  gamma=0.99, agent_id=0, act_dim=None, obs_dim=None):
    self.env = env; self.gamma = gamma;                       # Setup env and discount
    th.manual_seed(seed);np.random.seed(seed)  # Seed Torch, numpy and gym
    # Keep track of previous rewards and performed steps to calcule the mean Return metric
    self._episode, self.ep_returns, self.num_steps = [], deque(maxlen=100), 0
    # Get observation and action shapes
    if not obs_dim:
      obs_size = env.observation_space.shape if len(env.state.entities.by_name("Agents")) == 1 \
        else env.observation_space[agent_id].shape # Single agent case vs. multi-agent case
      obs_dim = np.prod(obs_size);
    if not act_dim: act_dim = env.action_space[agent_id].n
    self.vf = ValueNet(obs_dim)             # Setup Value Network (Critic)
    self.pi = PolicyNet(obs_dim, act_dim)   # Setup Policy Network (Actor)

  def step(self, obs):
    """ Given an observation, get action and probs from policy and values from critic"""
    with th.no_grad(): (a, _), v = self.pi(obs), self.vf(obs)
    self._episode.append((None,None,None,v))
    return a.numpy()

  def policy(self, obs, det=True): return self.pi(obs, det=det)[0].numpy()

  def finish_episode(self):
    """Process self._episode & reset self.env, Returns (s,a,G,V)-Tuple and new inital state"""
    s, a, R, V = (np.array(e) for e in zip(*self._episode)) # Get trajectories from rollout
    self.ep_returns.append(sum(R)); self._episode = []      # Add epoisode return to buffer & reset
    return (s,a,R,V)                                        # state, action, Return, Value Tensors

  def train(self, states, actions, returns, advantages):        # Update policy weights
    self.pi.optimizer.zero_grad(); self.vf.optimizer.zero_grad()# Reset optimizer
    states = states.flatten(1,-1)                               # Reduce dimensionality to rollout_dim x input_dim
    policy_loss = self.pi.loss(states, actions, advantages)     # Calculate Policy loss
    policy_loss.backward(); self.pi.optimizer.step()            # Apply Policy loss
    value_loss = self.vf.loss(states, returns)                  # Calculate Value loss
    value_loss.backward(); self.vf.optimizer.step()             # Apply Value loss