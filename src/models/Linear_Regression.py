import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
    def __init__(self, input_size, batch_size):
        self.input_size = input_size
        self.batch_size = batch_size

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.variable_scope('regression_layer'):
            self.add_regression_layer()

        with tf.name_scope('compute_cost'):
            self.compute_mse_cost()

        with tf.name_scope('train_step'):
            self.add_train_step()

    def add_input_layer(self):
        self.xs = tf.placeholder(
            tf.float64,
            [None, self.input_size],
            name='xs'
        )
        self.ys = tf.placeholder(
            tf.float64,
            [None, 1],
            name='ys'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            name='learning_rate'
        )

    def add_regression_layer(self):
        weights = tf.Variable(tf.random_normal(
            [self.input_size, 1],
            stddev=0.5,
            dtype=tf.float64
        ), name='weights')
        biases = tf.Variable(tf.random_normal(
            [1],
            stddev=0.5,
            dtype=tf.float64
        ), name='biases')

        self.prediction = tf.matmul(self.xs, weights) + biases

    def compute_mse_cost(self):
        self.error = tf.reduce_mean(tf.square(tf.subtract(
            self.ys,
            self.prediction
        )))

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.error)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
