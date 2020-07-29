import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import gensim

"""
Visualize word2vec result
tensorflow==1.15.0
"""


def load_trained_word_embedding(embedding_file, limit, binary):
    """
    load the file of word2vec trained file
    :param embedding_file: word2vec file path
    :param limit: load limit the number of word
    :param binary: binary mode:True Or False
    :return:
    """
    s_t = time.time()
    embedding_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, limit=limit, binary=binary)
    print("load vector file use:{}".format((time.time() - s_t) / 60))
    return embedding_vectors


def word_vector_visualizer(embedding_result, output_path, embedding_dim):
    """
    visualize the word vector result by tensorboard function of tensorflow
    :param embedding_result: word2vec word vectors
    :param output_path: tensorflow output file of tensorboard, it is necessary
    :param embedding_dim: embedding dimension
    :return:
    """
    meta_file = "w2v_visual_metadata.tsv"  # necessary file for tensorboard
    placeholder = np.zeros((len(embedding_result.wv.index2word), embedding_dim))
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(embedding_result.wv.index2word):
            placeholder[i] = embedding_result.get_vector(word)
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '' or word == '<\s>':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2v_visual_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2v_visual_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2v_visual_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
    print('visit the web address:{http://localhost:6006/}, to login tensorboard web! Web may be different by yourself')


if __name__ == "__main__":
    word2vec_file = "./word2vec.w2v"
    tensorboard_file = "./embedding_visualize_file"
    embedding_dim = 50
    embedding_result = load_trained_word_embedding(word2vec_file, None, True)
    word_vector_visualizer(embedding_result=embedding_result, output_path=tensorboard_file, embedding_dim=embedding_dim)
