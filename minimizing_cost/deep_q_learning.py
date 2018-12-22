# -*- coding: utf-8 -*-


############################################################### Import Libraries

import numpy as np


############################ Implementing deep q-learning with experience replay

class DQN(object):
    
### Initializing parameters and variables of dqn
    
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
        
        

    