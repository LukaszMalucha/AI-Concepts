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

if (env.train):
    for epoch in range(1, number_epochs):
        total_reward = 0 
        loss = 0.                 ## initialize loss as a float
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        ## start training in specific state
        current_state, _, _ = env.observe()
        timestep
    






















