import numpy as np
import tensorflow as tf


class TFDataProcesser:
    """For user tensorflow training needs data structure"""

    def __init__(self, batch_size, buffer_size, drop_remainder=False):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder

    def data2tf_data_set(self, features, labels, is_large_numpy=False):
        """Note that if tensors contains a NumPy array, and eager execution is not enabled, the values will be
        embedded in the graph as one or more tf.constant operations. For large datasets (> 1 GB), this can waste memory
        and run into byte limits of graph serialization. If tensors contains one or more large NumPy arrays, please set
        the parameter as:True.
        Tips:If tensors contains one or more large NumPy arrays, please set the parameter as:True.Must initialize firstly
        before your loop code
        """
        if not is_large_numpy:
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = dataset.shuffle(self.buffer_size).batch(batch_size=self.batch_size,
                                                              drop_remainder=self.drop_remainder).repeat()
            dataset = dataset.make_one_shot_iterator()
            iteration_element = dataset.get_next()
            return iteration_element
        else:
            # #=============tensors contains one or more large NumPy arrays===============
            # data slices
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            # shuffle batched repeat
            dataset = dataset.shuffle(self.buffer_size).batch(batch_size=self.batch_size,
                                                              drop_remainder=self.drop_remainder).repeat()
            iterator = dataset.make_initializable_iterator()
            iteration_element = iterator.get_next()
            return iterator, iteration_element

    def tfrecord2tf_data_set(self, tf_files):
        dataset = tf.data.TFRecordDataset(filenames=[tf_files])
        # Parse each line.
        dataset = dataset.map(self._parse_line, num_parallel_calls=50)
        print(dataset)
        dataset = dataset.shuffle(self.buffer_size).batch(batch_size=self.batch_size,
                                                          drop_remainder=self.drop_remainder).repeat()
        dataset = dataset.make_one_shot_iterator()
        iteration_element = dataset.get_next()
        return iteration_element

    def _parse_line(self, serialized):
        features = tf.parse_single_example(
            serialized=serialized,
            features={})

        # dense_tensor = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
        # print(dense_tensor)
        label = features['unpadded_class']


if __name__ == '__main__':
    features = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    labels = [1, 2, 3, 4, 5]

    EPOCHS = 20
    BATCH_SIZE = 2
    BUFFER_SIZE = 10
    NUM_BATCHES = 3
    is_large_numpy = False
    processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    data_iteration = processer.data2tf_data_set(features=features, labels=labels, is_large_numpy=is_large_numpy)
    with tf.Session() as sess:
        for epoch in range(EPOCHS):
            print("==============Start epoch:{}===============".format(epoch))
            for batch in range(NUM_BATCHES):
                value = sess.run(data_iteration)
                print(value)
            print("==============Finish epoch:{}!===============".format(epoch))
    """++++++++++++++++++Large numpy condition++++++++++++++++++++"""
    is_large_numpy = True
    features = np.array(features)
    labels = np.array(labels)
    # build dataset
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    iterator, data_element = processer.data2tf_data_set(features=features, labels=labels, is_large_numpy=is_large_numpy)
    with tf.Session() as sess:
        # must initialize before loop
        sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                                  labels_placeholder: labels})
        for epoch in range(EPOCHS):
            print("==============Start epoch:{}===============".format(epoch))
            for batch in range(NUM_BATCHES):
                value = sess.run(data_iteration)
                print(value)
            print("==============Finish epoch:{}!===============".format(epoch))
    """++++++++++++++++++tf record++++++++++++++++++++"""
    is_large_numpy = False
    tf_path = r"C:\Users\Administrator\.keras\datasets\fsns.tfrec"
    # build dataset
    processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    data_iteration = processer.tfrecord2tf_data_set(tf_path)
    with tf.Session() as sess:
        for epoch in range(EPOCHS):
            print("==============Start epoch:{}===============".format(epoch))
            for batch in range(NUM_BATCHES):
                value = sess.run(data_iteration)
                print(value)
            print("==============Finish epoch:{}!===============".format(epoch))
