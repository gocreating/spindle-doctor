import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
    def __init__(self, step_size, input_size, batch_size, dropout_rate, cluster_size, initial_centroids):
        self.step_size = step_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.cluster_size = cluster_size
        self.initial_centroids = initial_centroids
        self._rnn_layer_count = 0

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.name_scope('kmeans_layer'):
            self.add_kmeans_layer()
            self.expand_assignments()

        with tf.variable_scope('autoencoder_layer'):
            self.add_autoencoder_layer()

        with tf.variable_scope('evaluation_layer'):
            self.add_evaluation_layer()

        with tf.name_scope('compute_cost'):
            # self.compute_mse_cost()
            self.compute_entropy_cost()

        with tf.name_scope('train_step'):
            self.add_train_step()

    def add_input_layer(self):
        self.xs = tf.placeholder(
            tf.float64,
            [None, self.step_size],
            name='xs'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            name='learning_rate'
        )

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
        ])), tf.float64)

    def lstm_cell(self, num_units):
        cell = rnn.BasicLSTMCell(
            num_units=num_units,
            forget_bias=1.0,
            state_is_tuple=True,
            activation=tf.nn.sigmoid
            # reuse=tf.get_variable_scope().reuse
        )
        return rnn.DropoutWrapper(cell, input_keep_prob=1 - self.dropout_rate)

    # return shape=(batch_size, output_step_size * output_feature_size)
    def stack_rnn_layer(
        self,
        inputs, # (batch_size, input_step_size * input_feature_size)
        input_feature_size,
        output_feature_size,
        input_step_size,
        output_step_size,
        scope
    ):
        inputs = tf.reshape(inputs, [-1, input_step_size, input_feature_size])
        # cell_outputs.shape = (batch_size, input_step_size, input_feature_size)
        cell_outputs, cell_states = tf.nn.dynamic_rnn(
            self.lstm_cell(output_feature_size),
            inputs,
            dtype=tf.float64,
            scope=scope
        )

        origin_size = input_step_size * output_feature_size
        target_size = output_step_size * output_feature_size
        flattened_cell_outputs = tf.reshape(
            cell_outputs,
            [-1, origin_size]
        )
        weights = tf.Variable(tf.random_normal(
            [origin_size, target_size],
            dtype=tf.float64,
            name='weights_{0}'.format(self._rnn_layer_count)
        ))
        biases = tf.Variable(tf.random_normal(
            [target_size],
            dtype=tf.float64,
            name='biases_{0}'.format(self._rnn_layer_count)
        ))
        combination = tf.nn.sigmoid(
            tf.matmul(
                flattened_cell_outputs,
                weights
            ) + biases
        )
        return combination
        # return tf.reshape(combination, [-1, output_step_size, num_units])

    def add_autoencoder_layer(self):
        # self.enc_1 = self.stack_rnn_layer(self.xs,     1, 32, self.step_size, 8, 'enc_1')
        # self.enc_2 = self.stack_rnn_layer(self.enc_1, 32, 16,                 8, 4, 'enc_2')
        # self.dec_1 = self.stack_rnn_layer(self.enc_2, 16, 32,                 4, 8, 'dec_1')
        # self.dec_2 = self.stack_rnn_layer(self.dec_1, 32,  1,                 8, self.step_size, 'dec_2')

        self.enc_1 = self.stack_rnn_layer(self.expanded_assignments,     1, 32, self.step_size,  16, 'enc_1')
        self.enc_2 = self.stack_rnn_layer(self.enc_1, 32, 16,              16,  8, 'enc_2')
        self.enc_3 = self.stack_rnn_layer(self.enc_2, 16,  8,               8,  4, 'enc_3')
        self.dec_1 = self.stack_rnn_layer(self.enc_3,  8, 16,               4,  8, 'dec_1')
        self.dec_2 = self.stack_rnn_layer(self.dec_1, 16, 32,               8,  16, 'dec_2')
        self.dec_3 = self.stack_rnn_layer(self.dec_2, 32,  self.cluster_size,              16,  self.step_size, 'dec_3')

    def add_evaluation_layer(self):
        self.prediction_logits = tf.reshape(
            self.dec_3,
            [-1, self.step_size, self.cluster_size]
        )
        self.prediction = tf.argmax(self.prediction_logits, axis=2)

    # def compute_mse_cost(self):
    #     self.mse_error = tf.reduce_mean(tf.square(tf.subtract(
    #         self.xs,
    #         self.prediction
    #     )))

    def compute_entropy_cost(self):
        # self.entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.one_hot(tf.cast(self.expanded_assignments, tf.int32), self.cluster_size),
        #     logits=self.prediction_logits
        # ))

        logits = tf.unstack(self.prediction_logits, axis=1)
        targets = tf.unstack(self.prediction, axis=1)
        weights = [
            tf.ones([self.batch_size], tf.float64)
            for _ in range(self.step_size)
        ]
        self.entropy = tf.contrib.legacy_seq2seq.sequence_loss(
            logits,
            targets=targets,
            weights=weights
        )

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.entropy)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.entropy)
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.entropy)
