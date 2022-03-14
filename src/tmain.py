from Env2 import Env2

from RLAgents.QLearningAgent import QLearningAgent
from RLAgents.MonteCarloAgent import MonteCarloAgent
from RLAgents.MonteCarloTabAgent import MonteCarloTabAgent
import torch
import sys

def test(env, agent= None,maxit=1000):
  # load the mountain car problem
  if agent is not None:
    agent.reset(env)
    with torch.no_grad():
      agent.compute_policy(env, max_iterations=maxit, debug=True, alpha=0.0001)
print(sys.getrecursionlimit())

env = Env2()
agent = MonteCarloTabAgent(env)
test(env, agent, 1000)

print('Finish')