#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""

import tensorflow as tf
import pandas as pd


#%%
"""
CLASSIFICATION --> The IRIS dataset

This is also in my Scikit-learn testing
4 features
3 classifications
"""
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolour', 'Viriginica']


train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    
    
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    

train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()

train.shape
    
    
def input_fn(features, labels, training=True, batch_size=256):
    #Encode the dataframe into a tf.data.Dataset which tensorflow requires
    #if training=True then the data order is randomised
    #batch size is how many elements(rows) given to the model per iteration
    #rewritten compared to above for the dataset provided
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)



my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)




"""
Can use liner Classifier or Deep Neural
Predicts probability of being within one of the classes
"""
#Deep neural net with 2 hidden layers iwith 30 and 10 nodes
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    #Two hidden layers of 30 and 10 nodes respectively
    hidden_units=[30,10],
    #The model must choose between three classes
    n_classes=3)

#lambda creates the function - one line function call
#defines a function that returns a function because in def, theres no embedded function
#steps is similar to epoch -> but just looks at 5000
#Want loss to be low --> and prints output so that you can check on the status of the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y),
    steps=5000)


eval_result = classifier.evaluate(input_fn = lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


#predicting based on any given flower

def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels
    return  tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

predict = {}

# print("Type numeric values as prompted")
# for feature in features:
#     valid = True
#     while valid:
#         val = input(feature+": ")
#         if not val.isdigit(): valid = False
        
#     predict[feature] = [float(val)]
    
vals = ['7.9', '6.2', '22.5', '11.3']

for i in range(len(features)):
    predict[features[i]] = [float(vals[i])]
  
    
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1]
    
    }
    
predictions = classifier.predict(input_fn = lambda: input_fn(predict_x))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Predictions is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))
