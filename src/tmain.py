from Env2 import Env2

from RLAgents.QLearningAgent import QLearningAgent
from RLAgents.MonteCarloAgent import MonteCarloAgent
from RLAgents.MonteCarloTabAgent import MonteCarloTabAgent
from RLAgents.QLearningTab import QLearningAgentTab
from RLAgents.NNAgent import NNAgent
import torch
import sys

def evaluate(env, agent, nb_tests=100):

  winning = 0
  for _ in range(nb_tests):
    state = env.reset()
    done = False
    while not done:
      tstate = torch.tensor(state, dtype=torch.float64)
      action, option = agent.get_policy(tstate, env)
      state, r,_, done,_ = env.step(action, option)
    
    if r == env.max_reward:
      winning += 1

  print("sur {nbtests} test l agent a réussi {n} fois".format(nbtests=nb_tests, n=winning))

def test(env, agent= None,maxit=1000, grad=True):
  # load the mountain car problem
  if agent is not None:
    agent.reset(env)
    agent.compute_policy(env, max_iterations=maxit, debug=True, alpha=0.00001)
    evaluate(env, agent)


print(sys.getrecursionlimit())

env = Env2(200)
agent = NNAgent(env)
try:
  test(env, agent, 10000)
except Exception as e:
  env.print()
  print(env.current_statement)
  print(e)

print('Finish')