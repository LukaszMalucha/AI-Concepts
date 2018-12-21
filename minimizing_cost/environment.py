# -*- coding: utf-8 -*-


############################################################### Import Libraries

import numpy as np



##################################################### Building Environment Class


class Environment(object):
        
### Initializing environment variables
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

    def update_env(self, direction, energy_ai, month):
    
### Getting Reward

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                     
             
        
        