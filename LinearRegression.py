#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: William Collier
"""
import tensorflow as tf
import pandas as pd
from IPython.display import clear_output

#%%

"""
Linear regression

This is trained on the titanic dataset
The dataset icludes passengers from the titanic
There are 9 features 
The target is probability of survival
"""

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#plotting examples
# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')

# combine dftrain and y_train, grouped by column 'sex'. Then find the mean survived, and plot, and change an axis label
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('pct survive')



CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

#Feed the features into tensorflow

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #get a list of all unique values from a given column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))




def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
    #Encode the dataframe into a tf.data.Dataset which tensorflow requires
    #if shuffle=True then the data order is randomised
    #number of epochs is the number of times the data is reentered into the model
    #batch size is how many elements(rows) given to the model per iteration
    
    def input_fn():
        ds= tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_fn

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
        
        
#Print some useful outputs, the feature keys, the features from a cloumn, and some labels      
for feature_batch, label_batch in train_input_fn().take(1):    
    print('Some feature keys: {}'.format(list(feature_batch.keys())))
    print()
    print('A batch of class: {}'.format(feature_batch['class'].numpy()))
    print()
    print('A batch of labels: {}'.format(label_batch.numpy()))
    
    

#Print the values from a column, ie. ages
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()    
#Or column which would have strings
gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()



linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result= linear_est.evaluate(eval_input_fn)

clear_output()
print(result)


age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

print(dfeval.loc[8])
print(y_eval.loc[8])
print(pred_dicts[8]['probabilities'][1])


probs.plot(kind='hist', bins=20, title='predicted probabilities')








