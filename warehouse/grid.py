
import numpy as np

## Discount Factor

gamma = 0.75

## Learning Rate

alpha = 0.9
        
full_grid = []

for n in range(0,64):
    grid = [0] * 64
    for i,j in enumerate(grid):     
        if n % 8 == 0:
            if i == n+1 or i == n+8 or i == n-8:
                grid[i] = 1
        elif (n + 1) % 8 == 0:
            if i == n+8 or i == n-1 or i == n-8:
                grid[i] = 1 
        else:
            if i == n+1 or i == n+8 or i == n-1 or i == n-8:
                grid[i] = 1
    full_grid.append(grid)            
            
R = np.array(full_grid)







def route(starting_location, ending_location, desert_storm_1, desert_storm_2, desert_storm_3, desert_storm_4, reward_grid):
    R_new = np.copy(reward_grid)                                             
    R_new[ending_location, ending_location] = 100            
    Q = np.array(np.zeros([64,64]))

    for i in range(10000):                                   
        current_state  = np.random.randint(0,64)            
        playable_actions = []                                               
        for j in range(64): 
            if j not in [desert_storm_1, desert_storm_2, desert_storm_3, desert_storm_4]:                                
                if R_new[current_state, j] > 0:
                    playable_actions.append(j)                 
        next_state = np.random.choice(playable_actions)                  
        TD = R_new[current_state, next_state] +  gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]                                      
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD 
        
                                 
    route = [starting_location]
    next_location = starting_location                               
    while (next_location != ending_location):                                            
        next_location = np.argmax(Q[starting_location,])               
        route.append(next_location)                                
        starting_location = next_location                           
    return route

q = route(0,20,26,32,12,24,R)









