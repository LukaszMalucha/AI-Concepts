# -*- coding: utf-8 -*-


################################################### Import Libraries & .py files

import os
import numpy as np
import random as rn

import environment
import brain
import deep_q_learning

### Reproducibility seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


######################################################### Setting the Parameters

## Exploration parameter 30%
epsilon = 0.3

number_actions = 5

## Boundry that separates dircetion of temperature change (+ or -)
direction_boundary = (number_actions - 1) / 2

number_epochs = 1000
max_memory = 3000
batch_size = 512

## Temperature change between two consecutive actions
temperature_step = 1.5


####################################################### Building the Environment

env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)


############################################################# Building the Brain

brain = brain.Brain(learning_rate = 0.0001, number_actions = number_actions)


#################################################### Building DQ Learning Object

dqn = deep_q_learning.DQN(max_memory = max_memory, discount = 0.9)


#################################################################### Choose Mode

train = True


#################################################################### Training AI

env.train = train
model = brain.model

early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

## epoch - 5 month, timestep(iteration)- 1min

if (env.train):
    for epoch in range(1, number_epochs):
        total_reward = 0 
        loss = 0.                 ## initialize loss as a float
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        
        ## start training in specific state
        current_state, _, _ = env.observe()
        
        ## initialize variable to loop over iterations
        timestep = 0
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
    
### playing the next action by Exploration  - 30% of a time
            if np.random.rand() < epsilon:            ## TRICK - CREATE 30% OF A TIME
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step   

### playing the next action by DQN Inference - 70% of a time
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                # updating energy 
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step 

### updating environment for the next state                                   ## index of the month
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward     
            
### storing transition in memory  
            dqn.remember([current_state, action, reward, next_state], game_over)

### Gathering two separate batches  of inputs and targets - AI and noAI

            inputs, targets = dqn.get_batch(model, batch_size = batch_size)

### Compute the Loss over the two whole batches of inputs and targets

            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
################################################ PRINTING RESULTS FOR EACH EPOCH  

        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))  


        # EARLY STOPPING
        if (early_stopping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count >= patience):
                print("Early Stopping")
                break        
                          
            
##################################################################### SAVE MODEL         

        model.save("model.h5")     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    






















