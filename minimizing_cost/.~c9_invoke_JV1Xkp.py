# -*- coding: utf-8 -*-


############################################################### Import Libraries

import numpy as np



##################################################### Building Environment Class


class Environment(object):
        
### INITIALIZING ENVIRONMENT VARIABLES

    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]  ## 12 months
        self.initial_month = initial_month   ## specify starting month                    
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]    ## get the temperature in the chosen month
        self.optimal_temperature = optimal_temperature         ## optimal server temperature
        self.min_temperature = -20                             ## min server operation temp                          
        self.max_temperature = 80                              ## max server operation temp 
        
        self.min_number_users = 10                             ## min amount of users on server
        self.max_number_users = 100
        self.max_update_users = 5                              ## max users fluctuation per minute
        
        self.min_rate_data = 20                                ## min data transfer
        self.max_rate_data = 300 
        self.max_update_data = 10                              ## max data flucturation per minute 
        
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users       ## users updated through iterations
        
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data             ## data transfer updated through 
        
        ### common temperature as a basepoint (from formula)
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        ### AI-regulated server
        self.temperature_ai = self.intrinsic_temperature
        
        ### Non-AI server - cooling systems brings itautomatically to the middle value
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        
        ### initialize values for competition
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        
        ## Environment initialization variables
        self.reward = 0.0
        self.game_over = 0      # (Boolean)
        self.train = 1          # (1-train mode, 0-test mode)
        
        
        
### UPDATING ENVIRONMENT AFTER THE AI ACTION

    def update_env(self, direction, energy_ai, month):    ## direction of temperature change(1,-1), energy spent, actual month
    
### Getting Reward
        
        ## energy spent when there's no ai each minute
        energy_noai = 0             
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai   ## energy for heating up
            
            self.temperature_noai = self.optimal_temperature[0]                 ## energy spent and temperature back to normal
   
        elif (self.temperature_noai > self.optimal_temperature[1]):             ## cooling down
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            
            self.temperature_noai = self.optimal_temperature[1]
        
        
        ## compute the reward - ai energy savings
        self.reward = energy_noai - energy_ai
        
        ## scaling reward as recommended
        self.reward = 1e-3 * self.reward
        
        
### Get the next state

        ## update atmospheric temperature - month specified in a method 
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        
        ## update number of users (range -5 to +5)
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        
        ## make sure that users are between 10-100
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users  
            
        ## update data transmission rate    
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)

        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data  
            
            
        ## compute the delta of Intransic Temperature - common part of ai and no-ai simulations
        past_intrinsic_temperature = self.intrinsic_temperature
        ## as per formula new temperature:
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        
        ## delta during the specific minute(iteration)
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        
        ## Delta of temperature caused by AI (direction - cooling down or heating up)
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai
            
        ## Update new server temperature with AI
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        
        ## Server with No AI
        self.temperature_noai += delta_intrinsic_temperature
        
        
        
### Getting ENDGAME - server goes to un-operational temperatures - below -20 or above 80
    
        if (self.temperature_ai < self.min_temperature):
            ## check if in training mode
            if (self.train == 1):
                self.game_over = 1
            ## if normal mode then force AI to  bring down temperature to optimal lower bound   
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        elif (self.temperature_ai > self.max_temperature):
            ## check if in training mode
            if (self.train == 1):
                self.game_over = 1
            ## if normal mode then force AI to  bring down temperature to optimal lower bound   
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]         
                
### Updating scores

        ## total energy spent by AI        
        self.total_energy_ai += energy_ai
        
        ## energy send by no AI
        self.total_energy_noai += energy_noai
        
        
### Scaling the next state  - for neural network (normalization 0-1)

        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data ) / (self.max_rate_data - self.min_rate_data)
                              
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data ])        
        
        
### Return the next state, the reward and game over
        
        return next_state, self.reward, self.game_over
        
        
        
        
        
        
        
            

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                     
             
        
        