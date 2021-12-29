from .FuncApproxAgent import FuncApproxAgent

class MonteCarloAgent(FuncApproxAgent):
     def compute_policy(self, env, gamma=0.9, max_iterations=1000000, base_epsilon=0.8, alpha=0.2, debug=False):
        alpha = alpha * self.num_tilings

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
            states_actions = [(state, action)]
            while not done:
                state, reward, done, infos = env.step(action)                
                action = self.epsilon_greedy_action(env, state, epsilon)
                states_actions.append((state, action))
                tot_reward += reward

            for state, action in states_actions:
                self.parameters += alpha * (tot_reward - self.get_q(state, env)[action]) * self.featurize(state, env)[action]
                
            tot_rewards += tot_reward

            if debug and ((m+1)%100 == 0):
                avg = tot_rewards / (m+1)
                print("****************************")
                print("success_rate : ", infos['success_rate'])
                print(m+1, avg, epsilon)
                print("------------------------")
                print(env.algorithm)
                print(infos['Algo'])
                print("------------------------")
                print("state infos : ", infos['state'])
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
