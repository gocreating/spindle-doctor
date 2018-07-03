import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
    def __init__(self, input_size, step_size, class_size, batch_size):
        self.input_size = input_size
        self.step_size = step_size
        self.class_size = class_size
        self.batch_size = batch_size
        self._regression_layer_count = 0

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.variable_scope('regression_layers'):
            self.add_regression_layers()

        with tf.name_scope('compute_cost'):
            self.compute_entropy_cost()

        with tf.variable_scope('output_layer'):
            self.add_output_layer()

        with tf.name_scope('train_step'):
            self.add_train_step()

    def add_input_layer(self):
        self.xs = tf.placeholder(
            tf.float64,
            [None, self.step_size, self.input_size],
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

    def stack_regression_layer(
        self,
        inputs, # (batch_size, origin_size)
        origin_size,
        target_size
    ):
        weights = tf.Variable(tf.random_normal(
            [origin_size, target_size],
            dtype=tf.float64,
            name='weights_{0}'.format(self._regression_layer_count)
        ))
        biases = tf.Variable(tf.random_normal(
            [target_size],
            dtype=tf.float64,
            name='biases_{0}'.format(self._regression_layer_count)
        ))
        combination = tf.nn.sigmoid(
            tf.matmul(
                inputs,
                weights
            ) + biases
        )
        self._regression_layer_count = self._regression_layer_count + 1
        return combination

    def add_regression_layers(self):
        self.reshaped_xs = tf.reshape(self.xs, [self.batch_size, self.step_size * self.input_size])
        layer = self.stack_regression_layer(
            self.reshaped_xs,
            self.step_size * self.input_size,
            128
        )
        layer = self.stack_regression_layer(
            layer,
            128,
            64
        )
        layer = self.stack_regression_layer(
            layer,
            64,
            32
        )
        layer = self.stack_regression_layer(
            layer,
            32,
            16
        )
        self.prediction_logits = self.stack_regression_layer(
            layer,
            16,
            self.class_size
        )
        self.prediction = tf.reshape(tf.argmax(self.prediction_logits, axis=1), [-1, 1])

    def compute_entropy_cost(self):
        logits = [self.prediction_logits]
        targets = tf.unstack(tf.cast(self.ys, dtype=tf.int32), axis=1)
        weights = [tf.ones([self.batch_size], tf.float64)]
        self.entropy = tf.contrib.legacy_seq2seq.sequence_loss(
            logits,
            targets=targets,
            weights=weights
        )

    def add_output_layer(self):
        correct_prediction = tf.equal(
            tf.cast(self.ys, tf.int64),
            self.prediction
        )

        # shape = [batch_size]
        accuracy_per_row = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=1)
        self.accuracy = tf.reduce_mean(accuracy_per_row)

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.error)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.entropy)
