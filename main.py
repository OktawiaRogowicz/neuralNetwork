from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import logging
import matplotlib.pyplot as plt

from IPython.display import clear_output
from six.moves import urllib


def create():
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    CATEROGICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    feature_columns = []
    for feature_name in CATEROGICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(
            tf.feature_column.sequence_categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    print(feature_columns)

    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_echos=1, shuffle=False)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    print(result['accuracy'])
    print(result)

    result = list(linear_est.predict(eval_input_fn))
    print(dfeval.loc[0])
    print(result[0]['probabilities'][1])
    print(y_eval.loc[0])


def make_input_fn(data_df, label_df, num_echos=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_echos)
        return ds

    return input_function


def create2():
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    train_path = tf.keras.utils.get_file("iris_training.csv",
                                         "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file("iris_test.csv",
                                        "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    train_y = train.pop('Species')
    test_y = test.pop('Species')

    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3)
    classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)
    eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
    print(eval_result)

    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    predict = {}

    print("Please type numeric values as prompted.")
    for feature in features:
        valid = True
        while valid:
            val = input(feature + ": ")
            if not val.isdigit():
                valid = False
        predict[feature] = [float(val)]

    predictions = classifier.predict(input_fn=lambda: input_fn2(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))


def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def input_fn2(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # create()
    create2()
