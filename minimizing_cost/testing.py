# -*- coding: utf-8 -*-


################################################### Import Libraries & .py files

import os
import numpy as np
import random as rn
from keras.models import load_model
import environment


### Reproducibility seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


######################################################### Setting the Parameters

number_actions = 5

## Boundry that separates dircetion of temperature change (+ or -)
direction_boundary = (number_actions - 1) / 2

## Temperature change between two consecutive actions
temperature_step = 1.5


####################################################### Building the Environment

env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)


############################################################## LOADING the Brain

model = load_model("model.h5")


#################################################################### Choose Mode

train = False


########################################################### Running a Simulation

env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
    current_state = next_state

            
################################################ PRINTING RESULTS FOR EACH EPOCH  

print("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
 