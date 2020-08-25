#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

#%%

"""
Calculating clustering!
--> Hidden Markov Models
For a given input of starting on a hot or cold day
predict the temperature of a given day
"""

import tensorflow_probability as tfp


tfd = tfp.distributions

#Simply weather model

#cold day => 1, hot day => 0
#80% chance of starting on a cold day
#categorical distribution:
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

#cold day has 30% chance of being followed by hot day
# hot day has 20% chance of being followed by a cold day
#modelled as:
    
transition_distribution = tfd.Categorical(probs=[[0.6, 0.4],
                                                 [0.3,0.7]])

# Temperature is defined as a normal distribution
# cold day mu 0, 5 sigma
# hot day mu 15 sigma 10
# loc = mu scale = sigma in function call
observation_distribution = tfd.Normal(loc=[0., 15.], scale =[5., 10.])

# combine into a week long model

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
    )

# partially defined tensor
mean = model.mean()


with tf.compat.v1.Session() as sess:
    print(mean.numpy())

model.log_prob(tf.zeros(shape=[7]))