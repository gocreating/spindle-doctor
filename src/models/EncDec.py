import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

class Model:
    def __init__(self, step_size, input_size, hidden_size, output_size, layer_depth, batch_size, dropout_rate, feed_previous):
        self.step_size = step_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_depth = layer_depth
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.feed_previous = feed_previous

        with tf.name_scope('input_layer'):
            self.add_input_layer()

        with tf.variable_scope('rnn_encoder_layer'):
            self.add_rnn_encoder_layer()

        with tf.variable_scope('rnn_decoder_layer'):
            self.add_rnn_decoder_layer()

        with tf.variable_scope('output_layer'):
            self.add_output_layer()

        with tf.name_scope('compute_cost'):
            self.compute_mse_cost()
            # self.compute_entropy_cost()

        if not feed_previous:
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
            [None, self.step_size, self.output_size],
            name='ys'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            name='learning_rate'
        )

    def lstm_cell(self, output_size):
        cell = rnn.BasicLSTMCell(
            num_units=output_size,
            forget_bias=1.0,
            state_is_tuple=True,
            # activation=tf.nn.sigmoid,
            activation=tf.nn.tanh,
            reuse=tf.get_variable_scope().reuse
        )
        return rnn.DropoutWrapper(cell, input_keep_prob=1 - self.dropout_rate)

    def add_rnn_encoder_layer(self):
        layers = rnn.MultiRNNCell(
            [self.lstm_cell(self.hidden_size) for _ in range(self.layer_depth)],
            state_is_tuple=True
        )
        _, self.final_enc_states = tf.nn.dynamic_rnn(
            layers,
            self.xs,
            dtype=tf.float64
        )

    def add_rnn_decoder_layer(self):
        # https://stackoverflow.com/questions/36994067/no-feed-previous-argument-for-tensorflow-basic-rnn-seq2seq-function
        weights = tf.Variable(tf.random_normal(
            [self.hidden_size, self.output_size],
            # mean=0.5,
            stddev=0.5,
            dtype=tf.float64
        ))
        biases = tf.Variable(tf.random_normal(
            [self.output_size],
            # mean=0.5,
            stddev=0.5,
            dtype=tf.float64
        ))
        def inference_loop_function(prev, _):
            return tf.matmul(prev, weights) + biases

        loop_function = inference_loop_function if self.feed_previous else None
        layers = rnn.MultiRNNCell(
            [self.lstm_cell(self.hidden_size) for i in range(self.layer_depth)],
            state_is_tuple=True
        )
        outputs, self.cell_states = legacy_seq2seq.rnn_decoder(
            tf.unstack(self.ys, axis=1),
            self.final_enc_states,
            layers,
            loop_function=loop_function
        )

        self.cell_outputs = tf.stack(outputs, axis=1)

    def add_output_layer(self):
        # self.cell_outputs shape = (batch_size, step_size, hidden_size)
        # extracted_outputs shape = (batch_size, step_size * hidden_size)
        extracted_outputs = tf.reshape(
            self.cell_outputs,
            [-1, self.step_size * self.hidden_size]
        )
        weights = tf.Variable(tf.random_normal(
            [self.step_size * self.hidden_size, self.step_size * self.output_size],
            # mean=0.5,
            stddev=0.5,
            dtype=tf.float64
        ))
        biases = tf.Variable(tf.random_normal(
            [self.step_size * self.output_size],
            # mean=0.5,
            stddev=0.5,
            dtype=tf.float64
        ))

        # shape = (batch_size, step_size, output_size)
        linear_combination_of_outputs = tf.reshape(
            tf.matmul(extracted_outputs, weights) + biases,
            [-1, self.step_size, self.output_size]
        )
        self.prediction = linear_combination_of_outputs
        self.prediction = tf.nn.sigmoid(linear_combination_of_outputs)
        # self.prediction = tf.nn.tanh(linear_combination_of_outputs)
        # self.prediction = tf.nn.softplus(linear_combination_of_outputs)

    #
    # def compute_entropy_cost(self):
    #     def ms_error(labels, logits):
    #         return tf.square(tf.subtract(labels, logits))
    #     # def ms_error(y_target, y_prediction):
    #     #     return tf.square(tf.subtract(y_target, y_prediction))
    #
    #     current_batch_size = tf.shape(self.prediction)[0]
    #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #         [tf.reshape(self.prediction, [-1])],
    #         [tf.reshape(self.ys, [-1])],
    #         [tf.ones([current_batch_size], dtype=tf.float64)],
    #         average_across_timesteps=True,
    #         softmax_loss_function=ms_error
    #     )
    #     self.error = tf.reduce_mean(losses)

    def compute_mse_cost(self):
        self.error = tf.losses.mean_squared_error(
            self.ys[:, 0, 0],
            self.prediction[:, 0, 0]
        )

    def add_train_step(self):
        # self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.error)
        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.error)
