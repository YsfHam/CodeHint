from .Agent import LearningAgent
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, nin, nhidden, nout, activ):
        super(MLP, self).__init__()
        self.linearin = nn.Linear(nin, nhidden)
        self.linearout = nn.Linear(nhidden,nout)
        self.activation = activ
        
        self.initialize_parameters()

    def initialize_parameters(self):
        #pass
        nn.init.orthogonal_(self.linearin.weight)
        nn.init.zeros_(self.linearin.bias)
        
        nn.init.orthogonal_(self.linearout.weight)
        nn.init.zeros_(self.linearout.bias)


    def forward(self, x):
        x = self.linearin(x)
        x = self.linearout(self.activation(x))
        return x
        
class NNAgent(LearningAgent):
    def __init__(self, env):
        super().__init__()
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda:0")
        self.actionsnet = MLP(env.state_len,50,env.action_space.n, nn.ReLU()).to(self.device)
        self.optionsnet = MLP(env.state_len+1,100,env.options_space.n, nn.ReLU()).to(self.device)
        self.reset(env)

    def reset(self, env):
        self.actionsnet.initialize_parameters()
        self.optionsnet.initialize_parameters()

    def get_actions_q(self, state, env):
        return self.actionsnet(state)
    def get_options_q(self, state, action, env):
        return self.optionsnet(torch.cat((state, action)))

    def compute_policy(self, env, max_iterations, gamma=0.99, debug=False, alpha= 1e-3, base_epsilon=0.8):
    
        a_optimizer = torch.optim.Adam(self.actionsnet.parameters(), lr=alpha)
        o_optimizer = torch.optim.Adam(self.optionsnet.parameters(), lr=alpha)
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=alpha, momentum=0.9)

        tot_rewards = 0.0

        epsilon= base_epsilon
        self.actionsnet.train()
        self.optionsnet.train()
        if debug:
            print("Training policy...")
        for m in range(max_iterations):
            done = False
            state = env.reset()
            tot_reward = 0.0


            while not done:

                # example conversion from gym/numpy  to torch 
                tstate = torch.tensor(state, dtype=torch.float, device=self.device)

                # from state, generate action, next_state and reward
                action, option = self.epsilon_greedy_action(env, tstate, epsilon)
                taction = torch.tensor([action], dtype=torch.float, device=self.device)
                newstate, actions_reward, options_reward, done, infos = env.step(action, option)
                tnewstate = torch.tensor(newstate, dtype=torch.float, device=self.device)
                newaction, newoption = self.epsilon_greedy_action(env, tnewstate, epsilon)
                tnewaction = torch.tensor([newaction], dtype=torch.float, device=self.device)
            
                #compute error (with bellman estimation)
                # don't forget to disconnect estimation from derivation graph
                #... your code here
                if not done:
                    a_u = actions_reward + gamma * torch.max(self.actionsnet(tnewstate))
                    o_u = options_reward + gamma * torch.max(self.optionsnet(torch.cat((tnewstate, tnewaction))))
                    a_u = a_u.detach().clone()
                    o_u = o_u.detach().clone()
                else:
                    a_u = actions_reward
                    o_u = options_reward
                a_err = torch.square(a_u - self.actionsnet(tstate)[action])
                o_err = torch.square(o_u - self.optionsnet(torch.cat((tstate, taction)))[option])

                #print(loss, estimation, self.net(tstate)[action], self.net(tstate).requires_grad)
                # compute gradient of error, update parameters, reset gradients
                a_err.backward()
                a_optimizer.step()
                a_optimizer.zero_grad()

                o_err.backward()
                o_optimizer.step()
                o_optimizer.zero_grad()
            
                #update state/tot_rewards/...
                state, action, option = newstate, newaction, newoption

                tot_reward += options_reward + actions_reward

            tot_rewards += tot_reward

            if debug and ((m+1)%100 == 0):
                print(m+1, tot_reward, tot_rewards/m, epsilon) #, self.parameters)
                print(env.algorithm)

            
            if epsilon > 0:
                epsilon -= base_epsilon / max_iterations
        if debug:
            print("Training finished")
            print("****************************")
            print("success_rate : ", infos['success_rate'])
            print("------------------------")
            print(env.algorithm)
            print("------------------------")
            print(env.infos)
            print("------------------------")
            print(env.state_infos)
            print("------------------------")
            print("****************************")