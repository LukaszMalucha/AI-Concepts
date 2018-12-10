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
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])


