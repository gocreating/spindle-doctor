import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

class Model:
    def __init__(self, step_size, hidden_size, embedding_size, symbol_size, layer_depth, batch_size, dropout_rate, rnn_unit, mse_weights, initial_centroids):
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.symbol_size = self.cluster_size = symbol_size
        self.layer_depth = layer_depth
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.mse_weights = mse_weights
        self.rnn_unit = rnn_unit
        self.initial_centroids = initial_centroids

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.name_scope('kmeans_layer'):
            self.add_kmeans_layer()
            self.expand_assignments()

        with tf.variable_scope('embedding_layer'):
            self.add_embedding_layer()

        with tf.variable_scope('output_layer'):
            self.add_output_layer()

        with tf.name_scope('compute_cost'):
            self.compute_entropy_cost()
            self.compute_mse_cost()

        with tf.name_scope('train_step'):
            self.add_train_step()

    def add_input_layer(self):
        self.xs = tf.placeholder(
            tf.float64,
            [None, self.step_size],
            name='xs'
        )
        self.ys = tf.placeholder(
            tf.float64,
            [None, self.step_size],
            name='ys'
        )
        self.feed_previous = tf.placeholder(
            tf.bool,
            name='feed_previous'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            name='learning_rate'
        )
        self.mse_weights = tf.constant(self.mse_weights)

    def add_kmeans_layer(self):
        data_points = self.xs[:, 0]
        self.centroids = tf.Variable(self.initial_centroids)
        self.centroids.assign(self.initial_centroids)
        duplicated_xs = tf.transpose(tf.reshape(tf.tile(data_points, [self.cluster_size]), [self.cluster_size, self.batch_size]))
        distance_map = tf.abs(duplicated_xs - self.centroids)

        # assignment phase
        self.assignments = tf.argmin(distance_map, axis=1)

        # update phase
        self.masks = tf.cast(tf.one_hot(self.assignments, self.cluster_size), tf.bool)
        self.updated_centroids = tf.map_fn(
            # if mask contains only False, nan will be returned
            (lambda mask: tf.reduce_mean(tf.boolean_mask(data_points, mask))),
            tf.transpose(self.masks),
            dtype=tf.float64,
            parallel_iterations=self.cluster_size
        )
        # ref: https://stackoverflow.com/questions/42043488/replace-nan-values-in-tensorflow-tensor
        self.centroids = tf.where(tf.is_nan(self.updated_centroids), self.centroids, self.updated_centroids)

    def expand_assignments(self):
        # map raw value into classes
        self.expanded_assignments = tf.cast(tf.transpose(tf.stack([
            tf.argmin(tf.abs(tf.transpose(tf.reshape(tf.tile(self.xs[:, i], [self.cluster_size]), [self.cluster_size, self.batch_size])) - self.centroids), axis=1)
            for i in range(self.step_size)
        ])), tf.int32)

        # a = [
        #     tf.argmin(tf.abs(tf.transpose(tf.reshape(tf.tile(self.xs[:, i], [self.cluster_size]), [self.cluster_size, self.batch_size])) - self.centroids), axis=1)
        #     for i in range(self.step_size)
        # ]
        # b = tf.stack(a)
        # c = tf.transpose(b)
        # d = tf.cast(c, tf.int32)
        # self.expanded_assignments = d
        # print('=========================')
        # print(a)
        # print(b)
        # print(c)
        # print(d)
        # print(self.expanded_assignments)
        # print('=========================')


    def rnn_cell(self, output_size):
        cell = None
        if self.rnn_unit == 'LSTM':
            cell = rnn.BasicLSTMCell(
                num_units=output_size,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.nn.sigmoid,
                reuse=tf.get_variable_scope().reuse
            )
        elif self.rnn_unit == 'GRU':
            cell = rnn.GRUCell(
                num_units=output_size,
                activation=tf.nn.sigmoid,
                reuse=tf.get_variable_scope().reuse
            )
        elif self.rnn_unit == 'BASIC-RNN':
            cell = rnn.BasicRNNCell(
                num_units=output_size,
                activation=tf.nn.sigmoid,
                reuse=tf.get_variable_scope().reuse
            )
        return rnn.DropoutWrapper(cell, input_keep_prob=1 - self.dropout_rate)

    def add_embedding_layer(self):
        self.enc_inp = tf.unstack(self.expanded_assignments, axis=1)
        self.dec_inp = [np.zeros(self.batch_size, dtype=np.int) for t in range(self.step_size)]
        layers = rnn.MultiRNNCell(
            [self.rnn_cell(self.hidden_size) for i in range(self.layer_depth)],
            state_is_tuple=True
        )
        self.dec_outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.enc_inp,
            self.dec_inp,
            layers,
            self.symbol_size,
            self.symbol_size,
            self.embedding_size,
            output_projection=None,
            feed_previous=self.feed_previous
        )

    def add_output_layer(self):
        # shape = [batch_size, step_size, symbol_size]
        self.outputs = tf.stack(self.dec_outputs, axis=1)

        # shape = [batch_size, step_size]
        self.prediction = tf.cast(tf.argmax(self.outputs, axis=2), tf.int32)
        # self.prediction = tf.argmax(self.outputs, axis=2)
        correct_prediction = tf.equal(self.expanded_assignments, self.prediction)

        # shape = [batch_size]
        accuracy_per_row = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=1)
        self.accuracy = tf.reduce_mean(accuracy_per_row)

    def compute_entropy_cost(self):
        current_batch_size = tf.shape(self.outputs)[0]
        logits = tf.unstack(tf.cast(self.outputs, tf.float32), axis=1)
        targets = tf.unstack(self.expanded_assignments, axis=1)
        weights = [
            tf.ones([current_batch_size], tf.float32)
            for _ in range(self.step_size)
        ]
        self.error = tf.contrib.legacy_seq2seq.sequence_loss(
            logits,
            targets=targets,
            weights=weights
        )

    def compute_mse_cost(self):
        self.restored_prediction = tf.reshape(
            tf.map_fn(
                lambda level: self.mse_weights[level],
                tf.reshape(self.prediction, [-1]),
                dtype=tf.float64,
            ),
            [self.batch_size, -1]
        )
        # self.restored_ys = tf.reshape(
        #     tf.map_fn(
        #         lambda level: self.mse_weights[level],
        #         tf.reshape(self.ys, [-1]),
        #         dtype=tf.float64,
        #     ),
        #     [self.batch_size, -1]
        # )
        self.mse_error = tf.reduce_mean(tf.square(tf.subtract(
            self.ys,
            self.restored_prediction
        )))

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.error)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
        # self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.error)
