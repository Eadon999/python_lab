import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# since the model expects a single feature vector of size 784 #convert from (28,28) to 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

'''parse function to be used. This function is needed to do the preprocessing of data like reshaping ,converting to tensors from numpy arrays ,one-hot encoding ,etc.'''


def _parse_and_preprocess(x, y):
    x = tf.cast(x, tf.float32)
    # cast to float32 as the weights are float32.
    y = tf.cast(y, tf.int32)  # cast to tensor of int32
    return (dict({'image': x}),
            y)  # return tuple of dict of feature # name with key as provided in the feature column and label.


##define the function that feeds the data to the model .
def train_input_fn(x_train, y_train, batch_size=2, epoch=3):
    ##Here we are using dataset API.
    '''
    take the data from tensor_slices i.e. an array of data-points in simple words.
    '''
    dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]))
    dataset = dataset.shuffle(buffer_size=2)

    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(
        epoch)
    dataset_iterator = dataset.make_one_shot_iterator()
    return dataset_iterator.get_next()


if __name__ == '__main__':
    iter = train_input_fn(x_train, y_train)
    with tf.Session() as sess:
        for i in range(3):
            while True:
                value = sess.run(iter)
                print(value[0], value[1])
            print("=================")
