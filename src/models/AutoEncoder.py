import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
    def __init__(self, step_size, input_size, batch_size, dropout_rate):
        self.step_size = step_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self._rnn_layer_count = 0

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.variable_scope('autoencoder_layer'):
            self.add_autoencoder_layer()

        with tf.variable_scope('output_layer'):
            self.add_output_layer()

        with tf.name_scope('compute_cost'):
            self.compute_mse_cost()
            # self.compute_entropy_cost()

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

        self.enc_1 = self.stack_rnn_layer(self.xs,     1, 32, self.step_size,  16, 'enc_1')
        self.enc_2 = self.stack_rnn_layer(self.enc_1, 32, 16,              16,  8, 'enc_2')
        self.enc_3 = self.stack_rnn_layer(self.enc_2, 16,  8,               8,  4, 'enc_3')
        self.dec_1 = self.stack_rnn_layer(self.enc_3,  8, 16,               4,  8, 'dec_1')
        self.dec_2 = self.stack_rnn_layer(self.dec_1, 16, 32,               8,  16, 'dec_2')
        self.dec_3 = self.stack_rnn_layer(self.dec_2, 32,  1,              16,  self.step_size, 'dec_3')

    def add_output_layer(self):
        # self.prediction = self.dec_2
        self.prediction = tf.reshape(
            self.dec_3,
            [-1, self.step_size]
        )

    def compute_entropy_cost(self):
        def ms_error(labels, logits):
            return tf.square(tf.subtract(labels, logits))

        current_batch_size = tf.shape(self.prediction)[0]
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.prediction],
            [self.xs],
            [tf.ones([current_batch_size], dtype=tf.float64)],
            average_across_timesteps=True,
            softmax_loss_function=ms_error
        )
        self.mse_error = tf.reduce_mean(losses)

    def compute_mse_cost(self):
        self.mse_error = tf.reduce_mean(tf.square(tf.subtract(
            self.xs,
            self.prediction
        )))

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.error)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse_error)
