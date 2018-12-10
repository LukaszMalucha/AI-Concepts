# -*- coding: utf-8 -*-

################################## Q-Learing implementation for Warehouse Flows ###################################################


### Importing Libraries
import numpy as np


########################################## Setting up gamma and alpha parameters


## Discount Factor

gamma = 0.75

## Learning Rate

alpha = 0.9



####################################################### Defining the Environment


### Define the states - in a form of dictionary, locations to state

location_to_state = {"A":0, "B":1,"C":2, "D":3, 
                     "E":4, "F":5, "G":6, "H":7,
                     "I":8, "J":9, "K":10, "L":11}


### Define the actions - possible spots where the robot can go

actions = [0,1,2,3,4,5,6,7,8,9,10,11]

### Define the rewards

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])



####################################################### AI Solution Architecture


########### Initializing the Q-Values - matrix of 12x12 zeros

Q = np.array(np.zeros([12,12]))

########################### Implementing Q-Learning process

for i in range(1000):                                   ## repeat 100 times
    current_state  = np.random.randint(0,12)            ## intitate in random spot
    playable_actions = []                               ## possible actions for current state in a form of a list                   
    for j in range(12):                                 ## For the eeach row
        if R[current_state, j] > 0:                     ## if value in reward matrix[x,y] is '1'
            playable_actions.append(j)                  ## add 'y' coordinate to playable actions 
    next_state = np.random.choice(playable_actions)     ## play random action from the list as a next step
            
## BELMANN EQUATION    
## reward for action we played in current state                          ## empty spot will be returned by argmax function
    TD = R[current_state, next_state] +        gamma *      Q[next_state, np.argmax(Q[next_state, ])]        - Q[current_state, next_state]
                                         ## discount factor
                                                             ## Matrix of Q-values[correct row, column that has the highest value]

    ## Update Q-Value by adding temporal difference times learning rate
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD 



########################################################## Going into Production


def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location                               ## initiaite variable as a start location
    while (next_location != ending_location):                       ## while loop as iterations amount not precised
        starting_state = location_to_state[starting_location]       ## get the starting location index
        next_state = np.argmax(Q[starting_state,])                  ## get a column corresponding tot hte highest value
        
    













