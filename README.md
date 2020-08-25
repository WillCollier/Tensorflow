# Tensorflow
Learning how to use Scikit-learn for a variety of tasks.

Here is a list detailing what is contained within each file.
Each file is also commented.

### Linear regression.py

Uses the titanic dataset from google.
Using linear regression between the nine features to estimate probability of survival.
linear regression is fitting a line of best fit into the hyper-plane.

Tensowflow returns probability of events, rather than a yes/no answer. 

### Classification

A DNNClassifier classifier algorithm which is effective for smaller datasets. 
It creates a hyperplane in the features to seperate the different classes.

The Iris dataset has three classes (the target) and four features (sepal length/width, petal length/width) 
The classes are broken into 0, 1 and 2, with the classes list to translate back into strings.

The data is split into a training sample and a test sample. 
The training sample trains the network. 
The testing sample tests the accuracy. 
Finally the accuracy and an example prection is shown.


### hiddenMarkov.py

This is a clustering algorithm. However, it uses probabilities to make predictions.

For example, weather. Given the weather on one day, make predictions as to future days.
Here, given it's a hot or cold day, and then estimates the temperature.

### neural_network.py
Import the fashion sample from mnist.
This script identifies the type of clothing given the image.
However, unlike scikit learn, it returns the probability of the image being each of the potential classifications rather than just the best guess.

### convolutional_network.py
Using the CIFAR10 dataset, identify different types of vehicle.
Can train on a dataset.
OR can use a pretrained network.


### RNN_language_processing.py & RNN_shakespeare.py
Process language within a recurring neural network.
An RNN avoids the flaws of bag of words and word embedding attempting to find "context".
This example uses movie reviews and attempts to qualify if they are positive or negative.

Shakespeare attempts to use shakespeares writing to then make a play.


### Q_learning.py
This is a method used for AI in games.
It creates a Q matrix.
This gives the highest reward to the best option at each stage in a game. i.e. move, etc.






