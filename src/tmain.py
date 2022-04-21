from Env2 import Env2

from RLAgents.QLearningAgent import QLearningAgent
from RLAgents.QLearningTab import QLearningAgentTab
from RLAgents.NNAgent import NNAgent
from RLAgents.QMC import QMC
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

def test(env, agent, alpha_, maxit=1000):
  agent.reset(env)
  agent.compute_policy(env, max_iterations=maxit, debug=True, alpha=alpha_)
  evaluate(env, agent)


if len(sys.argv) < 2:
  print("Spécifier les arguments de l'entrainement")
  print("usage {nom} options".format(nom=sys.argv[0]))
  print("option est une suite d'options de la forme option=valeur")
  print("option: algo, horizon, iter")
  print("Valeur pour algo : QLearningAgent, QLearningAgentTab, NNAgent, QMC")
  exit(-1)

options_values_lambdas = {
  'QLearningAgent':lambda env:QLearningAgent(env),
  'QLearningAgentTab':lambda env:QLearningAgentTab(env),
  'NNAgent': lambda env: NNAgent(env),
  'QMC': lambda env: QMC(env),
  'horizon':lambda x: int(x),
  'iter': lambda x: int(x),
  'alpha': lambda x: float(x)
}
options = ['algo', 'horizon', 'iter']
options_values = {}
for i in range(1, len(sys.argv)):
  option, value = sys.argv[i].split('=')
  if option not in options:
    print('option {op} est inconnue'.format(op=option))
    exit(-1)
  if option != 'algo':
    options_values[option] = options_values_lambdas[option](value)
  else:
    options_values[option] = options_values_lambdas[value]

options_default_values = {
  'horizon': 50,
  'iter': 1000,
  'alpha': 0.0001
}

for k in options_default_values:
  if k not in options_values:
    options_values[k] = options_default_values[k]


print(options_values)

env = Env2(options_values['horizon'])
agent = options_values['algo'](env)
try:
  test(env, agent, options_values['alpha'], options_values['iter'])
except Exception as e:
  env.print()
  print(env.current_statement)
  print(e)
  e.with_traceback(None)

print('Finish')