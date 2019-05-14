import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'

def estimator_fn(run_config, params):
    
    feature_columns = [tf.feature_column.numeric_column(key='sepal_length'),
                    tf.feature_column.numeric_column(key='sepal_width'),
                    tf.feature_column.numeric_column(key='petal_length'),
                    tf.feature_column.numeric_column(key='petal_width')]


    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 10],
                                      n_classes=3,
                                      config=run_config)


def train_input_fn(training_dir, params):
    # read input file iris_train.csv .
    input_file = os.path.join(training_dir, 'iris_train.csv')
    df_iris_train = pd.read_csv(input_file, header=0, engine="python")

    # convert categorical target attribute 'species'  from  strings to integers
    df_iris_train['species'] = df_iris_train['species'].map({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2})

    # extract numpy data from a DataFrame
    labels = df_iris_train['species'].values

    features = {
        'sepal_length': df_iris_train['sepal_length'].values,
        'sepal_width': df_iris_train['sepal_width'].values,
        'petal_length': df_iris_train['petal_length'].values,
        'petal_width': df_iris_train['petal_width'].values
    }

    return features, labels


def eval_input_fn(training_dir, params):

    # read input file iris_test.csv .
    input_file = os.path.join(training_dir, 'iris_test.csv')
    df_iris_test = pd.read_csv(input_file, header=0, engine="python")

    # convert categorical target attribute 'species'  from  strings to integers
    df_iris_test['species'] = df_iris_test['species'].map({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2})

    # extract numpy data from a DataFrame
    labels = df_iris_test['species'].values

    features = {
        'sepal_length': df_iris_test['sepal_length'].values,
        'sepal_width': df_iris_test['sepal_width'].values,
        'petal_length': df_iris_test['petal_length'].values,
        'petal_width': df_iris_test['petal_width'].values
    }


    return features, labels

def serving_input_fn(params):

    feature_spec = {
        'sepal_length': tf.FixedLenFeature(dtype=tf.float32, shape=[1]),
        'sepal_width': tf.FixedLenFeature(dtype=tf.float32, shape=[1]),
        'petal_length': tf.FixedLenFeature(dtype=tf.float32, shape=[1]),
        'petal_width': tf.FixedLenFeature(dtype=tf.float32, shape=[1])
    }

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()