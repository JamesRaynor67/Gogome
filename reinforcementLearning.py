import numpy as np
import tensorflow as tf
from gomoku import Game
import os


BOARD_SIZE = 15
BOARD_STATE_NUM = 9
MAX_EPISODES = 3
MAX_EP_STEPS = 400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 5
BATCH_SIZE = 32

ENV_NAME = 'gomoku'

###############################  DDPG  ####################################

class DDPG(object):
	def __init__(self, a_dim, s_dim, save_path):
		self.state_memory = np.zeros((MEMORY_CAPACITY, BOARD_STATE_NUM*2+1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		self.reward_memory = np.zeros((MEMORY_CAPACITY, 1), dtype=np.float32)
		self.pointer = 0
		self.sess = tf.Session()
		self.a_replace_counter, self.c_replace_counter = 0, 0

		self.a_dim, self.s_dim = a_dim, s_dim
		self.S = tf.placeholder(tf.float32, [None, BOARD_STATE_NUM, BOARD_SIZE, BOARD_SIZE], 's')
		self.S_ = tf.placeholder(tf.float32, [None, BOARD_STATE_NUM, BOARD_SIZE, BOARD_SIZE], 's_')
		self.R = tf.placeholder(tf.float32, [None, 1], 'r')

		with tf.variable_scope('Actor'):
			self.a = self._build_a(self.S, scope='eval', trainable=True)
			a_ = self._build_a(self.S_, scope='target', trainable=False)
		with tf.variable_scope('Critic'):
			# assign self.a = a in memory when calculating q for td_error,
			# otherwise the self.a is from Actor when updating Actor
			q = self._build_c(self.S, self.a, scope='eval', trainable=True)
			q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

		# networks parameters
		self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
		self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
		self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
		self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

		q_target = self.R + GAMMA * q_
		# in the feed_dic for the td_error, the self.a should change to actions in memory
		td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
		self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

		a_loss = - tf.reduce_mean(q)    # maximize the q
		self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

		self.sess.run(tf.global_variables_initializer())

		self.save_path = save_path
		tf.summary.scalar("loss", a_loss)
		self.merged_summary_op = tf.summary.merge_all()
		self.summaryWriter = tf.summary.FileWriter(self.save_path, self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

	def choose_action(self, s):
		return self.sess.run(self.a, {self.S: s[np.newaxis, :]})

	def learn(self):
		# hard replace parameters
		if self.a_replace_counter % REPLACE_ITER_A == 0:
			self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
		if self.c_replace_counter % REPLACE_ITER_C == 0:
			self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
		self.a_replace_counter += 1; self.c_replace_counter += 1

		indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
		b_sm = self.state_memory[indices, :]
		b_rm = self.reward_memory[indices]
		bs = b_sm[:, :BOARD_STATE_NUM]
		ba = b_sm[:, BOARD_STATE_NUM: BOARD_STATE_NUM + 1]
		bs_ = b_sm[:, -1 * BOARD_STATE_NUM:]
		br = b_rm

		self.sess.run(self.atrain, {self.S: bs})
		self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

	def store_transition(self, s, a, r, s_):
		print s.shape, a.shape, s_.shape
		transition = np.concatenate((s, a, s_), axis=0)
		reward = r
		index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
		self.state_memory[index, :] = transition
		self.reward_memory[index, :] = reward
		self.pointer += 1

	def _build_a(self, s, scope, trainable):
		with tf.variable_scope(scope):
			l1 = tf.layers.conv2d(inputs=s, filters = 1, kernel_size=(7,7), strides=(1, 1), padding='same', data_format='channels_first', activation=tf.nn.relu, name='l1', trainable=trainable)
			l2 = tf.layers.conv2d(inputs=l1, filters = 1, kernel_size=(5,5), strides=(1, 1), padding='same', data_format='channels_first', activation=tf.nn.tanh, name='l2', trainable=trainable)
			return l2

	def _build_c(self, s, a, scope, trainable):
		with tf.variable_scope(scope):
			tmp_repeated_a = tf.layers.conv2d(inputs=a, filters = 9, kernel_size=(1,1), strides=(1, 1), padding='same', data_format='channels_first', name='tmp', trainable=trainable)
			# s_a = tf.concat([s, tmp_repeated_a], 0)
			s_a = s + tmp_repeated_a
			l1 = tf.layers.conv2d(inputs=s_a, filters = 1, kernel_size=(7,7), strides=(1, 1), padding='valid', data_format='channels_first', activation=tf.nn.tanh, name='l2', trainable=trainable)
			l2 = tf.layers.conv2d(inputs=l1, filters = 1, kernel_size=(5,5), strides=(1, 1), padding='valid', data_format='channels_first', activation=tf.nn.tanh, name='l3', trainable=trainable)
			l3 = tf.layers.conv2d(inputs=l2, filters = 1, kernel_size=(5,5), strides=(1, 1), padding='valid', data_format='channels_first', activation=tf.nn.tanh, name='l4', trainable=trainable)
			return tf.layers.dense(l3, 1, trainable=trainable)  # Q(s,a)

	def save_model(self, learn_step_counter):
		saver = tf.train.Saver()
		saver.save(self.sess, os.path.join(self.save_path, 'my_final_model'), global_step=learn_step_counter)

###############################  training  ####################################
if __name__ == '__main__':
	game = Game()
	s_dim = BOARD_STATE_NUM
	a_dim = 1
	PRINT = False
	done = False
	ddpg = DDPG(a_dim, s_dim, save_path='/home/hong/workspace/gomoku')
	learn_step_counter = 0
	for i in range(MAX_EPISODES):
		done = False
		s = game.startInit()
		ep_reward = 0
		step = 0
		while done is False:
			print 'step == ', step
			if PRINT:
				game.printBoard()

			# Add exploration noise
			a = ddpg.choose_action(s)
			x = a.argmax() / BOARD_SIZE
			y = a.argmax() % BOARD_SIZE
			s_, r, done, info = game.putStone(x, y, game.nextTrun)
			step += 1
			learn_step_counter += 1
			a = np.zeros((1, BOARD_SIZE, BOARD_SIZE))
			a[0,x,y] = 1
			# print 'action x,y == ', x, y
			# print a
			ddpg.store_transition(s, a, r, s_)

			if ddpg.pointer > MEMORY_CAPACITY:
				ddpg.learn()

			s = s_
			ep_reward += r
			if done:
				print('Episode:', i, ' Reward: %i' % int(ep_reward))

			if i % 1000 == 0:
				game.writeRecord('Record' + str(i))
	ddpg.save_model(learn_step_counter)