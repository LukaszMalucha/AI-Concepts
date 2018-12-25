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

for n in range(0, N):           ## for each round
    # Random Strategy
    strategy_rs = random.randrange(d)            ## select random 0-8 strategy
    strategies_selected_rs.append(strategy_rs)   ## append to list of random strategies
    reward_rs = X[n, strategy_rs]                ## compare selected action with "real life simulation" X and get assigned reward
    total_reward_rs += reward_rs                 ## get total reward
    
    # Thomson Sampling
    strategy_ts = 0
    max_random = 0
    for i in range(0, d):                        ## for each strategy
        ## compare how many times till now that strategy recieved 1 or 0 to get the Random Draw
        random_beta = random.betavariate(numbers_of_rewards_1[i] +1, numbers_of_rewards_0[i] +1)
        # update random beta for each strategy
        if random_beta > max_random:       
            max_random = random_beta
            strategy_ts = i 
            
    reward_ts = X[n, strategy_ts]    ## compare selected action with "real life simulation" X and get assigned reward 
    # update number of rewards
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
    ## append to list of ts strategies    
    strategies_selected_ts.append(strategy_ts)
    ## accumulate total ts rewards
    total_reward_ts += reward_ts
    
        
####################################### Compute the Absolute and Relative Return    
    
absolute_return = (total_reward_ts - total_reward_rs)*100    ## each customer converion = 100 USD
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
    
print("Absolute Return: {:.0f} $".format(absolute_return))    
print("Relative Return: {:.0f} %".format(relative_return))       
    
    
    
########################################### Plotting the Histogram of Selections

plt.hist(strategies_selected_ts) 
plt.title("Histogram of Selections")
plt.xlabel('Strategy')
plt.ylabel('Number of times the strategy was aselected')
plt.show()
    
    
