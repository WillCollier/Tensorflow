#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""
import numpy as np
import matplotlib.pyplot as plt
import time

#%%


"""
Reinforcement learning with Q-learning

This is different to many other types of machine learning, and has many 
applications in training agents (AI) to interact with environments such as 
games. Rather than feeding the machine learning model millions of exampes
we let the model our model come up with its's own models by exploring the
environemnt.



Environemnt --> In reinforcement learning tasks we have the notion of the 
environment, This is what the agent will explore. An example would be when 
training  an AI to play a game of mario the level is the environment

Agent --> THis is the entity that is exploring the environment. Our agent will 
onteract and take different actions withint the environment.

State --> At all teim the agent will be ina stae. The state simply tells us 
about the status of the agent. An example is the location. Changings location
chanes the state.

Action --> Any interaction between the agent and the environment would
be considered an action. For example, moving or jumping etc.  An action does 
not have to change the state of the agent Doing nothing is also an action!

Reward --> every actions that our agent takes witll result in a reward of some
magnitude (+ve or -ve) The goal will be for the gent to maximise it's reward 
in it's environemtn. Sometimes the reward is simple, performing an action to 
generate a postive reward. If it results in losing score or dying, then this 
would be attributed a negative reward.

Determinign hte reward functions is the most important part of reinforcement
learning.

"""

"""

Q-learning


Technique in reinforcement leearnign is called Q-learning

Simple yet pwoerful technique in machine learning that involed learning a 
matrix of action-reward values. This matrix is referred to as a Q-Table or
Q-Matrix. This matrix has a shape (number of possible states, number of 
possible actions) where each value at matrix[n,m] represents teh agents 
expected reward given they are in state n, and take action m. The Q-learning
algorithm defines the way we updaet the values in the matrix and decide what 
action for each state. After successful learning the agent makes decisions 
based on the Q-Matrix values for a given state.

Have to be able to avoid local minima.
Therefore need to be able to explore the environment randomly, and therefore
avoid the local minima/maxima to update the Q-table


Updating Q-values

a -> alpha -> learning rate - makes sure you don't over-updatet the Q-table
y -> gamma -> discount factor -> How much weight is place on the current action
Q[state, action] = Q[state, action] + 
                    a * (reward + y * max(Q[newState,:]) - Q[state,action])

"""

# %%
# Using Open AI Gym
# Developed for prgrammers to practise machine learning using unique 
# envionments

import gym

# Sets up an environment
env = gym.make('FrozenLake-v0')

print(env.observation_space.n) #number of states
print(env.action_space.n) #number of actions

env.reset() #set environment to default state

#get a random action
action = env.action_space.sample()

# take action, to see how things change
observation, reward, done, info = env.step(action)

# render a gui
env.render()


# %%


env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Start the Q table with all zeros
Q = np.zeros((STATES,ACTIONS))


EPISODES = 30000 #Number of times to run form hte beginning
MAX_STEPS = 100 #Max nuber of steps allowed in the environment

LEARNING_RATE = 0.81 #Learning rate
GAMMA = 0.96


"""
# picking an action
# we either pick an action with a random decision OR
# we use the Q-table value to find the best action

# Define a constant for the probability of selecting a random action
# high epsilon, high chance of randomly choosing (0.9 --> 90% chance of random)
# can reduce epsilon as training takes place to slowly put together the optimal 
# solution
epsilon = 0.9
# code to pick an action
if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample() #take a random action
else:
    action = np.argmax(Q[state, :]) #use the Q-table to select the best action

# Update Q Values

Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * 
                                                       np.max(Q[new_state, :]) 
                                                       - Q[state,action])
"""

RENDER = False

epsilon = 0.9

# 

rewards = []

for episode in range(EPISODES):
    
    state = env.reset()
    for _ in range(MAX_STEPS):
        
        if RENDER:
            env.render()
            
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #take a random action
        else:
            action = np.argmax(Q[state, :]) #use the Q-table to select the best action
        
        next_state, reward, done, _ = env.step(action)
        
        if reward == 0.0:
            env.render()
        
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA  
                            * np.max(Q[next_state, :]) - Q[state,action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break #reached goal
        
        
print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}")
# and now we can see the Q values

# 


def get_average(values):
    return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))
    
plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

