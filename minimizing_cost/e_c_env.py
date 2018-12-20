# -*- coding: utf-8 -*-

## Environment Parameters:
#- average atmosphere temp over a month                 C 
#- optimal server temperature                           18-24oC  
#- minimum operational server temp                      -20oC
#- maximum operational server temp                      80oC
#- minimum server users                                 10
#- maximum server users                                 100
#- maximum users fluctuation per min                    ~5
#- minimum server transmission rate(Kbps)               20
#- maximum server transmission rate(Kbps)               300
#- maximum transmission fluctuation rate per min(Kbps)  10 

## Variable units
##### cost for both simulations
#- server temperature                           oC / at given minute
#- server users                                 users/ at given minute
#- data transmission                            kbps / at given minute
##### two separate simulations 
#- energy spend by AI                           units / at given minute      - first simulation
#- energy spent by integrated cooling system    units / at given minute      - second simulation



## Assumption 1 - temperature can be approximated thrugh M.L. Regression:
#    server temp = b0  +  b1 x atmospheric temp  +  b2 x number of users  + b3 x data transmission rate
#    b1,b2,b3 > 0 

## simplification
#  server temp = atmospheric temp + 1.25 x number of users + 1.25 x data transmission rate

## Assumption 2- energy spend by one of the systems within one minute can be approx through linear regression 
## function of the server absolute temperature change:
    
#Et = |DeltaTt| = |Tt+1 - Tt| (30oC - 27oC) ~ 3j  



######## STATE INPUTS

# (server temp, number of users, rate of data transmission) at time t


######## AI ACTIONS

#0 - AI cools down by 3.0oC
#1 - AI cools down by 1.5oC
#2 - no action
#3 - AI heats up by 1.5oC
#4 - AI heats up by 3.0oC


######## REWARD - difference of energy consumption between AI decision and integrated system 
#Reward = E(noAI) - E(AI)
#Reward = |Tt(noAI)| - |Tt(AI)|






























