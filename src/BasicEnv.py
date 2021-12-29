import statement
import myExceptions

from gym.spaces import Discrete
import time
import random
import torch
class BasicEnv:
    def __init__(self, horizon=30):
        self.algorithm = statement.StatementsBlock()
        self.instructions = {'poser':statement.FunctionStatement('poser', lambda x: x + 1),\
            'retirer':statement.FunctionStatement('retirer', lambda x: x - 1)}

        self.current_statement = None
        self.nb_steps = 0
        self.horizon = horizon
        self.infos = {'Errors':[], 'Algo':'', 'success_rate':0, 'tests':None, 'state': None}
        self.rand = random.Random()
        self.init_tests()
        #(codage, nbposer, nbretirer, nbSi, nbcndPos, nbcndNeg, nbSinon)
        self.desired_state = [7, 0, 1, 1, 0, 1, 0]
        '''
        (0, 0) ajouter poser
        (0, 1) ajouter retirer
        (0, 2) ajouter un block
        (0, 3) ajouter une condition pos
        (0, 4) ajouter une condition neg
        (0, 5) ajouter un block Si
        (0, 6) ajouter un block Sinon

        (1, 0) aller en haut
        (1, 1) aller à gauche
        (1, 2) aller à droite

        (2, 0) enlever poser
        (2, 1) enlever retirer
        (2, 2) enlever un block
        (2, 3) enlever une condition pos
        (2, 4) enlever une condition neg
        (2, 5) enlever un block Si
        (2, 6) enlever un block Sinon
        '''
        self.actions = [(0, n) for n in range(7)] + [(1, n) for n in range(3)] + [(2, n) for n in range(7)] 
        self.actions_space = Discrete(len(self.actions))

    def reset(self):
        self.algorithm = statement.StatementsBlock()
        self.current_statement = self.algorithm
        #(codage, nbposer, nbretirer, nbSi, nbcndPos, nbcndNeg, nbSinon)
        self.state = {'codage':'', 'poser':0, 'retirer':0, 'Si':0, 'CndPos':0, 'CndNeg':0, 'Sinon':0}     
        self.nb_steps = 0
        return (-1, 0, 0, 0, 0, 0, 0)
        
    '''step return new state, reward, done'''
    def step(self, action):
        if not self.actions_space.contains(action):
            raise Exception("action not found", action, "but ", self.actions_space.n)
        self.nb_steps += 1
        reward = -1
        done = False
        action_detail = self.actions[action]
        if action_detail in self.possible_actions():
            self.perform_action(action_detail)
            self.state['codage'] = self.algorithm.encode()
            success, reward = self.test_algorithm()
            success_rate = success / len(self.tests_io) * 100
            self.infos['success_rate'] = success_rate
            done = success_rate == 100

        self.infos['state'] = self.state
        self.infos['Algo'] = str(self.algorithm)

        state_num_val = [self.state[k] for k in self.state]
        state_num_val[0] = int(state_num_val[0], base=2) if state_num_val[0] != '' else -1
        if done:
            self.init_tests()
            t1 = torch.tensor(state_num_val)
            t2 = torch.tensor(self.desired_state)
            diff = torch.abs(t1 - t2).sum().item()
            if diff == 0:
                reward = 100

        done = done or self.nb_steps == self.horizon
        return state_num_val, reward, done, self.infos

    def test_algorithm(self):
        success = 0
        int_reward = 0
        self.infos['Errors'] = []
        for input, output in self.tests_io:
            try: 
                res = self.algorithm.evaluate(input)
                diff = abs(res - output)
                if diff == 0:
                    success += 1
                    int_reward += 1
                else:
                    int_reward -= diff / len(self.tests_io)
            except myExceptions.AgentError as e:
                self.infos['Errors'].append(e)
                int_reward -= 1 / len(self.tests_io)

        return success, int_reward

    def possible_actions(self):
        current_node_type = type(self.current_statement)

        if current_node_type == statement.FunctionStatement:
            return [(1, 0)]
        if current_node_type == statement.ConditionStatement:
            if self.current_statement.ifBlock is None:
                return [(0, 5), (1, 0)]
            pa = [(1, 1), (1, 0), (2, 5)]
            pa += [(0, 6)] if self.current_statement.elseBlock is None else [(1, 2), (2, 6)]
            return pa
        if current_node_type == statement.StatementsBlock:
            if self.current_statement.statement is None:
                return [(0, 0), (0, 1), (0, 3), (0, 4)]
            
            if type(self.current_statement.statement) == statement.ConditionStatement:
                cndS = self.current_statement.statement
                pa = [(1, 1)]
                pa += [(2, 3), (2, 4)] if cndS.ifBlock is None and cndS.elseBlock is None else []
                return pa
            pa = [] if self.current_statement == self.algorithm else [(1, 0)]
            pa += [(2, 0), (2, 1)]
            if self.current_statement.statementsBlock is None:
                pa += [(0, 2)]
            else:
                block = self.current_statement.statementsBlock
                pa += [(1, 2)]
                pa += [] if block.statement is not None else [(2, 2)]
            return pa

        raise myExceptions.EnvironmentError("Error while computing possible actions")

    def perform_action(self, action_detail):
        action_class, action_code = action_detail
        # action d'ajout
        if action_class == 0 and action_code == 0: 
            self.current_statement.add(self.instructions['poser'])
            self.state['poser'] += 1
        elif action_class == 0 and action_code == 1: 
            self.current_statement.add(self.instructions['retirer'])
            self.state['retirer'] += 1
        elif action_class == 0 and action_code == 2: 
            self.current_statement.add(statement.StatementsBlock())
        elif action_class == 0 and action_code == 3: 
            self.current_statement.add(statement.ConditionStatement("est_vide", lambda x : x == 0))
            self.state['CndPos'] += 1
        elif action_class == 0 and action_code == 4: 
            self.current_statement.add(statement.ConditionStatement("non est_vide", lambda x : x > 0))
            self.state['CndNeg'] += 1
        elif action_class == 0 and action_code == 5: 
            self.current_statement.add(True)
            self.state['Si'] += 1
        elif action_class == 0 and action_code == 6: 
            self.current_statement.add(False)
            self.state['Sinon'] += 1

        #actions d'enlèvement
        elif action_class == 2 and action_code == 0: 
            if self.current_statement.statement.name == "poser":
                self.state['poser'] -= 1
                self.current_statement.statement = None
        elif action_class == 2 and action_code == 1: 
            if self.current_statement.statement.name == "retirer":
                self.state['retirer'] -= 1
                self.current_statement.statement = None
        elif action_class == 2 and action_code == 2: 
            self.current_statement.statementsBlock = None
        elif action_class == 2 and action_code in [3, 4]: 
            cndStatus = 'CndPos' if self.current_statement.statement.name == 'est_vide' else 'CndNeg'
            self.current_statement.statement = None
            self.state[cndStatus] -= 1
        elif action_class == 2 and action_code == 5: 
            self.current_statement.ifBlock = None
            self.state['Si'] -= 1
        elif action_class == 2 and action_code == 6: 
            self.current_statement.elseBlock = None
            self.state['Sinon'] -= 1

        #actions de déplacement
        elif action_class == 1 and action_code == 0: self.current_statement = self.current_statement.parent
        elif action_class == 1 and action_code == 1:
            if type(self.current_statement) == statement.StatementsBlock:
                self.current_statement = self.current_statement.statement
            elif type(self.current_statement) == statement.ConditionStatement:
                self.current_statement = self.current_statement.ifBlock
        elif action_class == 1 and action_code == 2:
            if type(self.current_statement) == statement.StatementsBlock:
                self.current_statement = self.current_statement.statementsBlock
            elif type(self.current_statement) == statement.ConditionStatement:
                self.current_statement = self.current_statement.elseBlock
        
    def init_tests(self):
        inputs = self.rand.sample(range(5), k=4)
        self.tests_io = [(x, x - 1) if x >= 1 else (x, x) for x in inputs]
        self.infos['tests'] = self.tests_io
