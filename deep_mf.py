import matplotlib as mp
import numpy as np
import random
from scipy.spatial.distance import cosine
import tensorflow as tf
import time


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)

    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h1 = tf.nn.dropout(h1, p_keep_hidden)

    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

def train_model(input_counts, output_counts, hidden_units=128, batch_size=100):

    num_users, num_items = input_counts.shape
    # Balance pos and neg output weight in cross entropy
    pos_weight = num_users * num_items / np.count_nonzero(input_counts)

    # Initialize input + output placeholders
    X = tf.placeholder(tf.float32, [None, num_items])
    Y = tf.placeholder(tf.float32, [None, num_items])

    # Initialize weights
    w_h1 = init_weights([num_items, hidden_units])
    w_h2 = init_weights([hidden_units, hidden_units])
    # w_o == item vectors
    w_o = init_weights([hidden_units, num_items])

    # Dropout probabilities
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    py_x = model(X, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden)

    # Define cost and training procedure
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(py_x, Y, pos_weight))
    train_op = tf.train.AdagradOptimizer(0.1).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        for i in range(1000):
            avg_loss = 0
            for start, end in zip(range(0, num_users, batch_size), range(batch_size - 1, num_users, batch_size)):
                _, loss = sess.run([train_op, cost], feed_dict={X: input_counts[start:end],
                                                                Y: output_counts[start:end],
                                                                p_keep_input: 0.8,
                                                                p_keep_hidden: 0.5})
                avg_loss += loss / (num_users / batch_size)

            if i % 10 == 0:
                print('%i iterations finished' % i)
                print('cross entropy: %f' % avg_loss)
        vecs =  w_o.eval()

    return vecs.T

