import theano
import theano.tensor as T
import numpy as np

class RNN(object):
	def __init__(self,vocab_size, x, hidden_layer_size=512, activation=T.tanh):

		self.activation = activation
		
		
		self.x = x

		self.h0 = theano.shared(value=np.random.rand(hidden_layer_size), name="h0", borrow=True)

		self.Uf = theano.shared(value=np.random.rand(hidden_layer_size), name='Uf', borrow=True)
		self.Vf = theano.shared(value=np.random.rand(hidden_layer_size), name='Vf', borrow=True)
		self.Wf = theano.shared(value=np.random.rand(vocab_size), name="Wf", borrow=True)
		self.Wo = theano.shared(value=np.random.rand(hidden_layer_size, vocab_size), name="Wo", borrow=True)

		self.params = [self.Uf, self.Vf, self.Wf, self.Wo]

		def time_step(char, prev_hidden):
			h_t = self.activation(T.dot(T.dot(self.Wf, char)*T.outer(self.Vf, self.Uf),prev_hidden))
			y_t = T.dot(h_t, self.Wo)

			return [h_t, y_t]

		[self.h, self.y_pred], _ = theano.scan(time_step,
										sequences = self.x,
										outputs_info=[self.h0, None])

		self.prob = T.nnet.softmax(self.y_pred)
		self.y_out = T.argmax(self.prob, axis=-1)
		self.loss = lambda y: -T.mean(T.nnet.categorical_crossentropy(self.prob, y))

class RNN_Wrapper(object):
	def __init__(self, data_input_file, learning_rate=0.01, sequence_len=50):
		#hyper-parameters
		self.learning_rate = learning_rate
		self.sequence_len = sequence_len

		self.__read_data(data_input_file)		



		self.y = T.imatrix()
		self.x = T.matrix()

		self.RNN = RNN(vocab_size=len(self.vocab_map), x=self.x)
		self.predict_prob = theano.function(inputs=[self.x],
											outputs=self.RNN.prob)
		self.predict = theano.function(inputs=[self.x],
										outputs=self.RNN.y_out)

		self.cost = self.RNN.loss(self.y)

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

	def train(self):
		#Super simple training
		for i in xrange(len(self.seqs)):
			print(self.get_cost(self.seqs[i], self.next_step[i]))





	def one_hot(self, char, dtype=theano.config.floatX):
	    i = self.vocab_map[char]
	    tmp = np.zeros(len(self.vocab_map), dtype=dtype)
	    tmp[i] = 1
	    return tmp

	def seq_2_mat(self, chars, dtype=theano.config.floatX):
	    return np.array([self.one_hot(x, dtype=dtype) for x in chars])

	def reverse_one_hot(self, sequence):
		return ''.join([self.int_map[np.argmax(row)] for row in sequence])




