#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""
import numpy as np
import tensorflow as tf
import os

#%%






# RNN play generator
# train on a set of sequences from another play
# predicts the most likely next character
# then feed back into the model to continue generating hte next character

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print('Length of text: {} charcters'.format(len(text)))
print(text[:250])


vocab = sorted(set(text))
# creating the mapping from unique characters to integers
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)


print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

# convert numeric values to text
def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))


# 

# Need to create training examples
# Can't just input the whole play
# Set a sequence length
# have an output thats shifted right by one character


seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples/targets
# convert everything to integers as a dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# 

# Now the seqeuences are 101 characters
#  split into input and output (0:100, 1:101)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
    
# maps the sequences through the function
# stored as the dataset
dataset = sequences.map(split_input_target)


# 

for x,y in dataset.take(2):
    print("Example")
    print("Input")
    print(int_to_text(x))
    print("Output")
    print(int_to_text(y))


# 

# Finally make the training batches

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
# This is the size to shuffle the dataset
# TF data is designed to work with possibly infinite sequences,
# so doesn't shuffle the entire sequence in memory
# Instead is maintains a buffer in which it shuffle elements
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    # Set None as we don't know how long the sequence will be
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                           batch_input_shape = [batch_size, None]),
        #return sequences, returns the intermediate stage at each step
        # so that you can see what the model is seeing as it goes 
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        # amounts of nodes equal to the vocab size, so any character can be added
        tf.keras.layers.Dense(vocab_size)
        ])

    return model


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
# Setting up the model so that later on it can be rebuilt for a batch_size of 1

# 

# Now we have to create our own loss function
#  The model will output a shaped tensor which represents the probabiilty
# distribution for each character at each timestampl in the batch
# A trick for deciding this, is to look at sample input and outputs of the 
# untrained model -> therefore find out what the model is actually returning

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)

# 64 arrays (one for each entry in the batch)
#len() 100, as this is the length of the string essenitally number of it/s
# 65 unique characters so final array is probability of each

pred = example_batch_predictions[0]
print(len(pred))
print(pred)

time_pred = pred[0]
print(len(time_pred))
print(time_pred)

# Need our own loss function to deal with this form of output

# 

# Determine the predicted charcter we need to sample from
# the output distribution -> selected by probability
sampled_indices = tf.random.categorical(pred, num_samples=1)

# reshape the array to onvert all of hte integers to numbers to see the actual charcters
samples_indices = np.reshape(sampled_indices, (1,- 1))[0]
predicted_chars= int_to_text(samples_indices)

print(predicted_chars)

# 

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)




# 

# set up checkpoints to save outputs
checkpoint_dir = './training_checkpoints'
# name of checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# 

# Training the model

history = model.fit(data, epochs=40, callbacks=[checkpoint_callback])

# 
# Rebuild the model, using batch size of 1 to feed one piece of text at a time
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# 

# use the last saved checkpoint file
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Can load any of the checkpoints
# checkpoint_num = 10
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_"+str(checkpoint_num)))
# model.build(tf.TensorShape([1, None]))

# 
# Now the model has been reset

def generate_text(model, start_string):
    
    # Number of charcters to generate
    num_generate = 800

    # Cnverting the start string into numbers
    input_eval = [char2idx[s] for s in start_string]
    # nests the list
    input_eval = tf.expand_dims(input_eval, 0)
    
    # Empty string for storing results
    text_generated = []
    
    # Low temperatures result in more predictable text
    # Higher temp more surprising
    # Experiment to find the best value
    temperature = 1
    
    # Here batch_size = 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        
        # remove batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # using a categorical distribution to predict the character returned by model
        predictions = predictions/ temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # We pass along the predicted charcter as the next input into the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
        
    return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))


