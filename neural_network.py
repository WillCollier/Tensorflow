#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


#%%

"""
Create a neural network
Using the fashion sample from mnist
"""

# select the dataset from mnist
fashion_mnist = keras.datasets.fashion_mnist

# load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#  Pre-process the data
# max pixel value = 255.
# normalise the images (nueral nets start with weights and biases between 0 and 1)

train_images = train_images.copy() / 255.
test_images = test_images.copy() / 255.


# print("Max values is: {}".format(train_images.max()))

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# deep learning consists of layers with the nodes chained together (Sequential, from left to right, features through nodes to labels)
# start by flattening the images which are 28x28 pixels
# add dense layers, 128 neurons, with a relu acitvation function <0 = 0, >0 = value
# final 10 layers returns logits array wth length 10 (10 different items in the inputs, each node is a class of item)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

# To compile the model there must be other settings
#  define the loss function (decribes how well the neural net is doing, and is minimised)
# Define the optimiser, which is how it updates teh model based on the loss function and the data
# Metrics - Used to monitor the trianing and testing steps --> using accuracy which uses teh % succesful

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=7)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: {}'.format(test_acc))

# if model accuracy is > test accuracy then the model was overfit
# Change some of the inputs, and see what improves the accuracy (number of neurons etc)


# make predictions -> each image has a probability of being one of the classes
predictions = model.predict(test_images)

# turn predictions into a dataframe
predictions_df = pd.DataFrame(data=predictions, columns=class_names)


def plot_image(predictions_array, true_label, img):
    """
    Parameters
    ----------
    predictions_array : dataframe
        predicted probabilities for each possible class for a test image.
    true_label : int
        The correct class for the item.
    img : Array
        The image which the neural network has attempted to identify.

    Returns
    -------
    Imshow of the img, with title of true, and xlabel of prediction.

    """    

    
    probability = int(predictions_array.max().round(3) * 100)
    item = class_names[predictions_array.argmax()]
    
    item_true = class_names[true_label]
    
    plt.imshow(img)
    plt.xlabel(f"Guess: {item} at {probability}%")
    plt.title(f"Expected to be {item_true}")

    return

def plot_value_array(predictions):
    """
    Parameters
    ----------
    predictions : dataframe
        The probability predicted for each of the classes for one test image.

    Returns
    -------
    Bar plot of values.

    """
    predictions.plot.bar()
    
    return


def get_val():
    
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
            else:
                print("Try again")

# j=0
# while j<= 5:
#     i = get_val()
j = (np.random.random(5) * 10000).round()
for i in j.astype(int):
    plt.figure()
    plt.subplot(1,2,1)
    plot_image(predictions_df.loc[i], test_labels[i], test_images[i])
    plt.subplot(1,2,2)
    plot_value_array(predictions_df.loc[i])
    plt.tight_layout()
    plt.show()
    # j+=1

