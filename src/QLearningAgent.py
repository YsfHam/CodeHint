import torch

basehash = hash

#Integer Hash Table
class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles   

class QLearningAgent:
    def __init__(self):
        self.max_size =  4096 * 8
        self.num_tilings = 8
        self.tiling_dim = 8


        # Initialize index hash table (IHT) for tile coding.
        self.iht = IHT(self.max_size)

        self.reset()

    def reset(self):
        self.parameters = torch.zeros(self.max_size)
    
    def epsilon_greedy_action(self, env, state, epsilon):
        #begin your code here
        r = torch.rand((1,)).item()
        if r < epsilon:
            return env.actions_space.sample()
    
        return self.get_policy(state, env)

    def featurize(self, state, env):
        res = torch.zeros((env.actions_space.n, self.max_size))

        for action in range(env.actions_space.n):
            for index in tiles(self.iht, self.num_tilings, [], list(state) + [action]):
                res[action, index] = 1

        return res

    def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        alpha = alpha / self.num_tilings

        tot_rewards = 0
        epsilon = base_epsilon
        if debug:
            print("Training policy...")
        for m in range(max_iterations):
            done = False
            tot_reward = 0.0

            state = env.reset()
            action = self.epsilon_greedy_action(env, state, epsilon)
            infos = None
            while not done:
                newstate, reward, done, infos = env.step(action)                
                #parameter update
                # if state is final
                # -> .... update (final case)
                # else#
                # -> .... update (non final case)
                newaction = self.epsilon_greedy_action(env, newstate, epsilon)
                stateQ = self.get_q(state, env)
                newstateQ = self.get_q(newstate, env)
                stateFeaturized = self.featurize(state, env)
                if reward == 100: print("bingo")
                if done:
                    self.parameters += alpha * (reward - stateQ[action]) * stateFeaturized[action]
                else:
                    self.parameters += alpha * (reward + gamma * torch.max(newstateQ) - stateQ[action]) * stateFeaturized[action]

                state, action = newstate, newaction
                tot_reward += reward
                
            tot_rewards += tot_reward

            if debug and ((m+1)%100 == 0):
                avg = tot_rewards / (m+1)
                print("****************************")
                print("success_rate : ", infos['success_rate'])
                print(m+1, avg, epsilon)
                print("------------------------")
                print(env.algorithm)
                print("------------------------")
                print("codage : ", infos['state'])
                print(infos['Errors'])
                print("****************************")
            
            if epsilon > 0:
                epsilon -= base_epsilon/max_iterations
        if debug:
            print("Training finished")
            print("****************************")
            print("success_rate : ", infos['success_rate'])
            print("------------------------")
            print(env.algorithm)
            print("------------------------")
            print("codage : ", infos['state'])
            print(infos['Errors'])
            print("****************************")

    def get_q(self, state, env):
        return self.featurize(state, env) @ self.parameters

    def get_policy(self, state, env):
        return torch.argmax(self.get_q(state, env)).item()