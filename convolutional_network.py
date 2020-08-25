#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_datasets as tfds

#%%

"""
Deep computer vision

"""


from tensorflow.keras import datasets, layers, models


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images/255. , test_images/255.

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']


IMG_INDEX = 2

plt.figure()
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()



plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i][:,:,0], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()





# Make the model
# Add the convolutional layers defining the number of filters and how big the filters are. (Plus the activation function)
# Add pooling layers between, using 2x2 samples and a stride of 2 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Prints the definition of the model
# So far this is the convolutional base, which is for extracting features
# Needs to be added into dense layers to classify based on the features
model.summary()

# Add the neural net
# Final layer has 10 components, one for each of the labels
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Prints the definition of the model
# Now 10 neurons out to determine the predicted class
model.summary()


# Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)



"""
Hard to have a very accurate conv network without millions of input images.
One solution to this is a pretrained network.
Or augment each image, to increase the database of images (i.e. rotation, mirroring, zoom, stretch etc)
"""


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creatures a data generator object that transforms images

datagen  = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Pick an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img)
img = img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'): #This loops forever until it is broken, saving images locally
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i+=1
    if i>4:
        break
plt.show()



"""
A pretrained network
Fine-tune the last few layers of a premade network
Base layers are good for generalised image traits, such as edges
"""

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True, )

print(raw_train)
print(raw_validation)
print(raw_test)


get_label_name = metadata.features['label'].int2str #function object to get the data labels

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# However images are all different sizes
# Therefore they need to be rescaled

IMG_SIZE = 160 #makes images 160x160 pixels

def format_example(image, label):
    """
    returns an image that is resizes to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label



train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(get_label_name(label))


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# INclude top is whether you include the classifier from the network already --> because retraining, no!
# As it was traind on 1000 classes
# Weights are a specific save of the weights on the net

base_model.summary()
# Using this can see the shape of the output from the base model
# The shape is 32,5,5,1280 from the original 1,160,160,3 which we enter
# 32 different layers of filters/features

for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

# Freezing the base
# Do not want to retrain all of teh weights and biases!
# Therefore fix the learned values
base_model.trainable = False

base_model.summary()
# Now the params are all non-trainable

# Now add our own classifier
# Average pooling layer will average the entire 5x5 area of each feature map 
# and return a 1280 element vector per filter
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# Add a prediction layer which is a single dense neuron (as there are only 2 classes to predict)
prediction_layer = keras.layers.Dense(1)

# combine the different layers into a model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer])


model.summary()
# Now there are trainable parameters aagin

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']

print(acc)

# Save and load models after the training has been carried out
model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# Then do model.predict etc to predict the cat vs dog etc (Can then add to display images as before)

