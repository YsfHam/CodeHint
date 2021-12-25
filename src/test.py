from BasicEnv import BasicEnv
import pprint
import torch
import statement
from QLearningAgent import QLearningAgent

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
      agent.compute_policy(env, max_iterations=maxit, debug=True)
  evaluate(env,agent, 'algorithm.alg')
agent = QLearningAgent()
env = BasicEnv(horizon=100)
test(env, agent, 500)