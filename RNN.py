import datetime
import theano
import theano.tensor as T
import numpy as np
import string
import random
import cPickle as pickle

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

		self.updates = {}
		for param in self.params:
			init = np.zeros(param.get_value(borrow=True).shape,
				dtype=theano.config.floatX)
			self.updates[param] = theano.shared(init)

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

class LSTM(object):
    def __init__(self, input_size, x, hidden_layer_size, activation=T.tanh):
        self.activation = activation
        self.x = x
        self.input_size = input_size
        self.output_size = input_size
        
        #first hidden layer values
        self.h0 = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="h0", borrow=True)
        
        #first memory cell values
        self.C0 = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="C0", borrow=True)
        
        """
        Weight matricies are initiated uniformly between -.08 and .08 (Karpathy 15) 
        Bias vectors are initiated to 0
        
        Conventions
        ===========
        W_ -> weight matrix for the input 
        U_ -> weight matrix for the previous hidden layer
        b_ -> bias for the given gate
        
        
        TODO
        ====
        -Concatenate all W_, U_, and b_ into one matrix/vector for efficiency
        -Implement the ability to have depth
        """ 
        
        #Values for forget gate
        self.Wf = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, input_size)).astype(theano.config.floatX), name="Wf", borrow=True)
        self.Uf = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, hidden_layer_size)).astype(theano.config.floatX), name="Uf", borrow=True)
        self.bf = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="bf", borrow=True)
        
        #Values for input gate
        self.Wi = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, input_size)).astype(theano.config.floatX), name="Wi", borrow=True)
        self.Ui = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, hidden_layer_size)).astype(theano.config.floatX), name="Ui", borrow=True)
        self.bi = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="bi", borrow=True)
        
        #Values for output gate
        self.Wo = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size,input_size)).astype(theano.config.floatX), name="Wo", borrow=True)
        self.Uo = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, hidden_layer_size)).astype(theano.config.floatX), name="Uo", borrow=True)
        self.bo = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="bo", borrow=True)
        
        #Values for the memory cell
        self.Wc = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, input_size)).astype(theano.config.floatX), name="Wc", borrow=True)
        self.Uc = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, hidden_layer_size)).astype(theano.config.floatX), name="Uc", borrow=True)
        self.bc = theano.shared(value=np.zeros(shape=(hidden_layer_size,), dtype=theano.config.floatX), name="bc", borrow=True)
        
        #Overall Output Weights (used for prediction)
        self.Wp = theano.shared(value=np.random.uniform(low=-0.8, high=0.8, size=(hidden_layer_size, self.output_size)).astype(theano.config.floatX), name="Wp", borrow=True)
        self.bp = theano.shared(value=np.zeros(shape=(self.output_size,), dtype=theano.config.floatX), name="bp", borrow=True)
        
        
        self.params = [self.Wf, self.Uf, self.bf, self.Wi, self.Ui, self.bi,
                      self.Wo, self.Uo, self.bo, self.Wc, self.Uc, self.bc,
                      self.Wp, self.bp]
        self.updates = {}
        for param in self.params:
        	init = np.zeros(param.get_value(borrow=True).shape,
        		dtype=theano.config.floatX)
        	self.updates[param] = theano.shared(init)
        
        def time_step(x, prev_hidden, prev_cell):
            f_t = T.nnet.sigmoid(T.dot(self.Wf, x) + T.dot(self.Uf, prev_hidden) + self.bf)
            i_t = T.nnet.sigmoid(T.dot(self.Wi, x) + T.dot(self.Ui, prev_hidden) + self.bi)
            c_tild_t = T.tanh(T.dot(self.Wc, x) + T.dot(self.Uc, prev_hidden) + self.bc)
            c_t = (i_t * c_tild_t) + (f_t * prev_cell)
            o_t = T.nnet.sigmoid(T.dot(self.Wo, x) + T.dot(self.Uo, prev_hidden) + self.bo)
            h_t = o_t*T.tanh(c_t)
            y_t = T.dot(h_t, self.Wp) + self.bp
            return [h_t, c_t, y_t]
        
        [self.h, self.C, self.y_pred], _ = theano.scan(time_step,
                                      sequences=self.x,
                                      outputs_info=[self.h0, self.C0, None])
        
        self.L1_norm = abs(self.Wf.sum()) + abs(self.Wi.sum()) + abs(self.Wo.sum()) + \
                        abs(self.Wc.sum()) + abs(self.Wp.sum()) + \
                        abs(self.Uf.sum()) + abs(self.Ui.sum()) + abs(self.Uo.sum()) + \
                        abs(self.Uc.sum())
        
        self.L2_norm = (self.Wf ** 2).sum() + (self.Wi ** 2).sum() + (self.Wo ** 2).sum() + \
                        (self.Wc ** 2).sum() + (self.Wp ** 2).sum() + \
                        (self.Uf ** 2).sum() + (self.Ui ** 2).sum() + (self.Uo ** 2).sum() + \
                        (self.Uc ** 2).sum()
        
        self.prob = T.nnet.softmax(self.y_pred)
        self.y_out = T.argmax(self.prob, axis=-1)
        self.loss = lambda y: T.mean(T.nnet.categorical_crossentropy(self.prob, y))

class RNN_Wrapper(object):
	def __init__(self, data_input_file, learning_rate=10**-.5, L1_lambda=0.0, L2_lambda=0.0,
	 sequence_len=50, hidden_layer_size=512, flavor="LSTM", initial_momentum=0.5, final_momentum=0.9, momentum_switchover=5,
	 n_epochs=100):
		#hyper-parameters
		self.learning_rate = learning_rate
		self.sequence_len = sequence_len
		self.L1_lambda = L1_lambda
		self.L2_lambda = L2_lambda
		self.initial_momentum = initial_momentum
		self.final_momentum = final_momentum
		self.momentum_switchover = momentum_switchover
		self.n_epochs = n_epochs

		self.cost_over_time = list()

		self.__read_data(data_input_file)	





		self.y = T.imatrix()
		self.x = T.matrix(dtype='float32')
		if(flavor == "RNN"):
			self.RNN = RNN(vocab_size=len(self.vocab_map), x=self.x, hidden_layer_size=hidden_layer_size)
		elif(flavor == "LSTM"):
			self.RNN = LSTM(input_size=len(self.vocab_map), x=self.x, hidden_layer_size=hidden_layer_size)
		self.predict_prob = theano.function(inputs=[self.x],
											outputs=self.RNN.prob)
		self.predict = theano.function(inputs=[self.x],
										outputs=self.RNN.y_out)

		self.cost = self.RNN.loss(self.y) + (self.L1_lambda * self.RNN.L1_norm) + (self.L2_lambda * self.RNN.L2_norm)

		
		self.mom = T.scalar('mom', dtype=theano.config.floatX)


		self.gparams = [T.clip(T.grad(self.cost, param),-5,5) for param in self.RNN.params]

		self.updates = theano.compat.python2x.OrderedDict() #Behavior undefined with regular dict
		for param, gparam in zip(self.RNN.params, self.gparams):
			weight_update = self.RNN.updates[param]
			upd = (self.mom * weight_update) - (self.learning_rate * gparam)
			self.updates[weight_update] = upd
			self.updates[param] = param + upd

		#self.updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(self.RNN.params, self.gparams)]

		self.index = T.lscalar('index')

		self.shared_x = theano.shared(np.asarray(self.seqs))
		self.shared_y = theano.shared(np.asarray(self.next_step, dtype=np.int32))

		self.get_cost = theano.function(inputs=[self.index, self.mom],
										outputs=self.cost,
										updates=self.updates,
										givens={
											self.x: self.shared_x[self.index],
											self.y: self.shared_y[self.index]
										},
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
	
	'''
	def save(self, path='./'):
		f_name = path + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.pkl'

		f = open(f_name, 'wb')
		state = self.__getstate()
		pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()

	def __getstate(self):
		params = self.__getparams() 
		weights = [w.get_value() for w in self.RNN.params]
		return (params, weights)


	
	def __setstate(self, state):
		params, weights = state
		self.set_params(**params)
		self.
	'''
	def train(self):
		epoch = 0
		while(epoch < self.n_epochs):

			for i in xrange(len(self.seqs)):
				if(epoch > self.momentum_switchover):
					effective_momentum = self.final_momentum
				else:
					effective_momentum = self.initial_momentum
				cost = self.get_cost(i, effective_momentum)
				self.cost_over_time.append(cost)
				if(i%10 == 0):
					print("Cost:\t%f\tMinibatch:\t%d\tEpoch:\t%d"%(cost,i,epoch))

			epoch += 1
			self.sample(primeText="To be")


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






