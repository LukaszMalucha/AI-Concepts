# -*- coding: utf-8 -*-


############################################################### Import Libraries

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


################################################################# Building Brain


class Brain(object):
    
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        
        self.learning_rate = learning_rate        
        states = Input(shape = (3,))                                            ## three rows, one column
        x = Dense(units = 64, activation = 'sigmoid')(states)                   ## add states to create full connection with previous layer
        y = Dense(units = 32, activation = 'sigmoid')(x)
        q_values = Dense(units = number_actions, activation = 'softmax')(y)     ## output layer        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
        
        
        
        

