# -*- coding: utf-8 -*-

############################################################### Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import random


############################################################# Setting Parameters

N = 10000       ## total number of rounds (customers connecting to website)
d = 9           ## number of strategies


############################################################ Creating Simulation

conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]       ## 9 strategies and conversion rates unknown to AI
X = np.array(np.zeros([N,d]))                                                   ## initiate array of 10000 rows and 9 columns with zeros                                 


## Update succesfull plan with "1"
for i in range(N):
    for j in range(d):                                                          ## Bernoulli distribution
        if np.random.rand() <= conversion_rates[j]:
            X[i,j] = 1
            
            
############################### Implementing Random Strategy vs Thomson Sampling

## For each strategy i take a random draw from the following distribution

strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

for n in range(0, N):
    # Random Strategy
    strategy_rs = random.randrange(d)            ## select random 0-8 strategy
    strategies_selected_rs.append(strategy_rs)   
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





























