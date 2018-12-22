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
        
        
### Building Memory in experience replay (last come, first go)   

    def remember(self, transition, game_over):                 ## start adding each move to memory list
        self.memory.append([transition, game_over])           
        ## circulate momory if it grows above 100
        if len(self.memory) > self.max_memory:  
            del self.memory[0]                                  
            
            
### Building two batches of 10 inputs and 10 targets by extracting 10 transitions

    def get_batch(self, model, batch_size = 10):                                ## get keras model
    
        len_memory = len(self.memory)  ## memory from init
        
        ## TRANSITION - [current_state, action played, reward, next state ]
        num_inputs = self.memory[0][0][0].shape[1]                              ## get the current state np.shape (3)
        num_outputs = model.output_shape[-1]                                    ## (5) actions
                              
        ## BATCH OF INPUTS [num_users, rate_data, server_temp]      
        ## BATCH OF TARGETS [action1-5]             
                           ## either len_memory if it's less than 10 
        inputs = np.zeros((min(len_memory,batch_size) , num_inputs))   ## initialize with 10 rows for each state and 3 columns
        targets = np.zeros((min(len_memory,batch_size) , num_outputs)) ## initialize with 10 rows for each state and 5 columns
        
        ## Pick 10 random transitions from memory and update arrays on inputs and targets to create memory batch
        
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory,batch_size))):            
            current_state, action, reward, next_state =  self.memory[idx][0]    ## extract 3 variables from memory
            game_over = self.memory[idx][1]                                     ## state of game
            inputs[i] = current_state                                           ## place current state into array
            targets[i] = model.predict(current_state)[0]                        ## place predictions into target array
            Q_sa = np.max(model.predict(next_state)[0])                         ## highest rewarded action   
            if game_over:
                targets[i, action] = reward            ## make sure that if AI failed it remembers it                         
            else:
                targets[i, action] = reward + self.discount * Q_sa              ## for backpropagation - reward + discounted action
         
        ## return both arrays    
        return inputs, targets        
                
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        