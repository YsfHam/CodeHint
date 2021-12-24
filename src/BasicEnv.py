import statement

from gym.spaces import Discrete
import torch
class BasicEnv:
    def __init__(self):
        self.algorithm = statement.StatementsBlock()
        self.instructions = {'poser':statement.ElementaryStatement('poser'),\
            'retirer':statement.ElementaryStatement('retirer')}

        self.current_statement = self.algorithm
        '''
        (0,0) ajouter poser
        (0,1) ajouter retirer
        (0, 3, 0) ajouter une branche
        (0, 3, 1) ajouter Si
        (0, 3, 2) ajouter sinon
        (0, 4, 0) ajouter une condition pos
        (0, 4, 1) ajouter une condition neg
        (1, 0) go down
        (1, 1) go up

        '''
        
    '''step return new state, reward, done'''
    def step(self, action):
        if not self.action_space.contains(action):
            raise Exception("action not found")

        action_detail = self.actions[action]
        if self.islegal(action_detail):
            return self.perform_action(action_detail)

        return self.current_state, -1, False

    def illegal(self, action_detail):
        las = self.legal_actions_set()
        return action_detail in las
        
