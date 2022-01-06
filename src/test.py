from BasicEnv import BasicEnv

from RLAgents.QLearningAgent import QLearningAgent
from RLAgents.MonteCarloAgent import MonteCarloAgent
import torch
import sys
def evaluate(env, policy, file_name):

    env.init_tests()
    state = env.reset()

    done = False
    infos = None
    while not done:
        action = policy.get_policy(state, env)
        state,_,done,infos = env.step(action)

    f = open(file_name, 'w')
    f.write("------------------------\n")
    f.write(str(env.algorithm)+'\n')
    f.write("------------------------\n")
    f.write("codage : " + str(infos['state']) + '\n')
    f.write(str(infos['Errors']) + '\n')
    f.close()

def test(env, agent= None,maxit=1000):
  # load the mountain car problem
  if agent is not None:
    agent.reset()
    with torch.no_grad():
      agent.compute_policy(env, max_iterations=maxit, debug=True, alpha=0.0001)
  evaluate(env,agent, 'algorithm.alg')
agents = {
  'QLearningAgent':lambda: QLearningAgent(),
  'MonteCarloAgent':lambda: MonteCarloAgent()
  }
if len(sys.argv) < 2 or sys.argv[1] not in agents:
  print("Please specify one of the following arguments : ")
  for agentName in agents:
    print(agentName)
  sys.exit(1)
agent = agents[sys.argv[1]]()
env = BasicEnv(horizon=30)
test(env, agent, 1000)