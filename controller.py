import tensorflow as tf
from collections import deque

from environment import *

GAME_HEIGHT = 210
GAME_WIDTH = 160

LEARNING_RATE = 0.001

class DQN(object):
     def __init__(self):
        # self.game = Env()
        # self.input_height, self.input_width, self.channel = self.game.observation.shape
        self.action_n = self.game.action_n
        self.lr = LEARNING_RATE
        self.gamma = 0.1
        self.memory_size = 500
        self.gamma = 0.9
        self.epsilon = 0.9
        self.replace_target_iter = 300
        self.memory_size = 500
        self.batch_size = 32

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, GAME_HEIGHT * GAME_WIDTH * 2 + 2)) # s, a, r, s_next

        self.build_network()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []

        self.replay_buffer = deque()

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(self, x):
        tf.summary.histogram(x.op.name + '/activations', x)
        tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

    def build_network(self):
        self.s = tf.placeholder(tf.float32, [None, GAME_HEIGHT, GAME_WIDTH],
                                name='state')
        self.s_next = tf.placeholder(tf.float32, [None, GAME_HEIGHT, GAME_WIDTH],
                                     name='next_state')
        self.r = tf.placeholder(tf.float32, [None, ], name='reward')
        self.a = tf.placeholder(tf.float32, [None, self.action_n], name='action')

        # evaluation net
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('conv1') as scope:
                kernel = self._variable_with_weight_decay('weights',
                                                          shape=[5, 5, 3, 64],
                                                          stddev=5e-2,
                                                          wd=None)
                conv = tf.nn.conv2d(self.s, kernel, [1, 1, 1, 1], padding='SAME')
                biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(pre_activation, name=scope.name)
                self._activation_summary(conv1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')

            with tf.variable_scope('conv2') as scope:
                kernel = self._variable_with_weight_decay('weights',
                                                          shape=[5, 5, 64, 64],
                                                          stddev=5e-2,
                                                          wd=None)
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name=scope.name)
                self._activation_summary(conv2)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool2')

            with tf.variable_scope('local') as scope:
                reshape = tf.reshape(pool2, [1, -1])
                dim = reshape.get_shape()[1].value
                weights = self._variable_with_weight_decay('weights', [dim, 192], stddev=0.04, wd=0.004)
                biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
                local = tf.relu(tf.multiply(reshape, weights) + biases, name=scope.name)
                self._activation_summary(local)

            with tf.variable_scope('q') as scope:
                weights = self._variable_with_weight_decay('weights', [192, self.action_n],
                                                           stddev=1/192.0, wd=None)
                biases = self._variable_on_cpu('biases', [self.action_n], tf.constant_initializer(0.0))
                self.q_eval = tf.add(tf.matmul(local, weights), biases, name=scope.name)
                self._activation_summary(self.q_eval)

        # target net
        with tf.variable_scope('target_net'):
            with tf.variable_scope('conv1') as scope:
                kernel = self._variable_with_weight_decay('weights',
                                                          shape=[5, 5, 3, 64],
                                                          stddev=5e-2,
                                                          wd=None)
                conv = tf.nn.conv2d(self.s_next, kernel, [1, 1, 1, 1], padding='SAME')
                biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(pre_activation, name=scope.name)
                self._activation_summary(conv1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')

            with tf.variable_scope('conv2') as scope:
                kernel = self._variable_with_weight_decay('weights',
                                                          shape=[5, 5, 64, 64],
                                                          stddev=5e-2,
                                                          wd=None)
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name=scope.name)
                self._activation_summary(conv2)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool2')

            with tf.variable_scope('local') as scope:
                reshape = tf.reshape(pool2, [1, -1])
                dim = reshape.get_shape()[1].value
                weights = self._variable_with_weight_decay('weights', [dim, 192], stddev=0.04, wd=0.004)
                biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
                local = tf.relu(tf.multiply(reshape, weights) + biases, name=scope.name)
                self._activation_summary(local)

            with tf.variable_scope('q_next') as scope:
                weights = self._variable_with_weight_decay('weights', [192, self.action_n],
                                                           stddev=1/192.0, wd=None)
                biases = self._variable_on_cpu('biases', [self.action_n], tf.constant_initializer(0.0))
                self.q_next = tf.add(tf.matmul(local, weights), biases, name=scope.name)
                self._activation_summary(self.q_next)

        with tf.variable_scope('q_target') as scope:
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='max_q_next')
            self.q_target = tf.stop_gradient()
        with tf.variable_scope('loss') as scope:
            self.loss = tf.reduce_mean(self.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train') as scope:
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_data(self, s, a, r, s_next):
        self.replay_buffer.append([s, a, r, s_next])
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def train_network(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            # update target network
            self.sess.run(self.replace_target_op)

        batch_memory = np.random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in batch_memory]
        action_batch = [data[1] for data in batch_memory]
        reward_batch = [data[2] for data in batch_memory]
        state_next_batch = [data[3] for data in batch_memory]

        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.s: state_batch,
                                    self.a: action_batch,
                                    self.r: reward_batch,
                                    self.s_next: state_next_batch
                                })

        self.cost_his.append(cost)

        self.learn_step_counter += 1


    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.action_n)
        return action

