import theano
import theano.tensor as T
import numpy as np
import string
import random

class RNN(object):
	def __init__(self,vocab_size, x, hidden_layer_size=512, activation=T.tanh):

		self.activation = activation
		
		
		self.x = x

		self.h0 = theano.shared(value=np.zeros(hidden_layer_size), name="h0", borrow=True)

		self.bh = theano.shared(value=np.zeros(hidden_layer_size), name="bh", borrow=True)
		self.bo = theano.shared(value=np.zeros(vocab_size), name="bo", borrow=True)

		self.Uf = theano.shared(value=np.random.uniform(low=-0.08, high=0.08, size=hidden_layer_size), name='Uf', borrow=True)
		self.Vf = theano.shared(value=np.random.uniform(low=-0.08, high=0.08, size=hidden_layer_size), name='Vf', borrow=True)
		self.Wf = theano.shared(value=np.random.uniform(low=-0.08, high=0.08, size=vocab_size), name="Wf", borrow=True)
		self.Wo = theano.shared(value=np.random.uniform(low=-0.08, high=0.08, size=(hidden_layer_size, vocab_size)), name="Wo", borrow=True)

		self.params = [self.Uf, self.Vf, self.Wf, self.Wo, self.h0, self.bh, self.bo]

		def time_step(char, prev_hidden):
			h_t = self.activation(T.dot(T.dot(self.Wf, char)*(T.outer(self.Vf, self.Uf)+self.bh),prev_hidden))
			y_t = T.dot(h_t, self.Wo) + self.bo

			return [h_t, y_t]

		[self.h, self.y_pred], _ = theano.scan(time_step,
										sequences = self.x,
										outputs_info=[self.h0, None])


		self.L1_norm = abs(self.Wf.sum()) + abs(self.Wo.sum())
		self.L2_norm = (self.Wf ** 2).sum() + (self.Wo ** 2).sum()

		self.prob = T.nnet.softmax(self.y_pred)
		self.y_out = T.argmax(self.prob, axis=-1)
		self.loss = lambda y: -T.mean(T.nnet.categorical_crossentropy(self.prob, y))

class RNN_Wrapper(object):
	def __init__(self, data_input_file, learning_rate=0.01, L1_lambda=0.0, L2_lambda=0.0, sequence_len=50):
		#hyper-parameters
		self.learning_rate = learning_rate
		self.sequence_len = sequence_len
		self.L1_lambda = L1_lambda
		self.L2_lambda = L2_lambda
		self.__read_data(data_input_file)		



		self.y = T.imatrix()
		self.x = T.matrix()

		self.RNN = RNN(vocab_size=len(self.vocab_map), x=self.x)
		self.predict_prob = theano.function(inputs=[self.x],
											outputs=self.RNN.prob)
		self.predict = theano.function(inputs=[self.x],
										outputs=self.RNN.y_out)

		self.cost = self.RNN.loss(self.y) + (self.L1_lambda * self.RNN.L1_norm) + (self.L2_lambda * self.RNN.L2_norm)

		self.gparams = [T.grad(self.cost, param) for param in self.RNN.params]

		self.updates = [(param, param + self.learning_rate * gparam) for param, gparam in zip(self.RNN.params, self.gparams)]

		self.get_cost = theano.function(inputs=[self.x, self.y],
										outputs=self.cost,
										updates=self.updates,
										allow_input_downcast=True)



	def __read_data(self, data_input_file):
		self.__create_vocab_mappings(data_input_file)
		with open(data_input_file, 'r') as f:

			self.seqs = np.rollaxis(np.dstack([self.seq_2_mat(seq) for seq in iter(lambda: f.read(self.sequence_len), '')][0:-1]), 2)
			self.next_step = np.zeros(self.seqs.shape)
			for i, matrix in enumerate(self.seqs):
				self.next_step[i, 0:-1, :] = self.seqs[i, 1:, :]
				try:
					self.next_step[i, -1, :] = self.seqs[i+1, 0, :]
				except IndexError:
					#Except the index out of bounds error that gets caused on the last iteration and just break from the loop
					break
			self.next_step[-1, -1, :] = self.one_hot("")




	def __create_vocab_mappings(self, data_input_file):
	    #create mapping from char -> int
	    vocab_mapping = {}
	    i = 0
	    with open(data_input_file, 'r') as f:
	        while True:
	            char = f.read(1)
	            if not char:
	                break
	            else:
	                if char not in vocab_mapping:
	                    vocab_mapping[char] = i
	                    i += 1
	    vocab_mapping[""] = i
	    self.vocab_map = vocab_mapping

	    #create mapping from int -> char
	    self.int_map = [None]*len(self.vocab_map)
	    for char, num in self.vocab_map.iteritems():
	    	self.int_map[num] = char

	def train(self, breakPoint):
		#Super simple training
		for i in xrange(len(self.seqs)):
			print("Cost: " + str(self.get_cost(self.seqs[i], self.next_step[i])) + "\tMinibatch: " + str(i))
			if(i==breakPoint):
				break



	def sample(self, length=500, primeText=None):
		if(primeText == None):
			primeText = [random.choice(string.letters)]
		else:
			primeText = [primeText]
		for i in xrange(length):
			pred = self.predict_prob(self.seq_2_mat("".join(primeText)))
			index = self.sample_dist(pred[-1])[0]
			primeText.append(self.int_map[index])
		print("".join(primeText))


	def one_hot(self, char, dtype=theano.config.floatX):
	    i = self.vocab_map[char]
	    tmp = np.zeros(len(self.vocab_map), dtype=dtype)
	    tmp[i] = 1
	    return tmp

	def seq_2_mat(self, chars, dtype=theano.config.floatX):
	    return np.array([self.one_hot(x, dtype=dtype) for x in chars])

	def reverse_one_hot(self, sequence):
		return ''.join([self.int_map[np.argmax(row)] for row in sequence])
	def sample_dist(self, dist):
		rand_num = random.random()
		cur_sum = 0.0
		for i, num in enumerate(dist):
			if(rand_num > cur_sum and rand_num < num + cur_sum):
				return (i, num)
			cur_sum += num





