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
			print(prev_hidden.type)
			h_t = self.activation(T.dot(T.dot(self.Wf, char)*T.outer(self.Vf, self.Uf),prev_hidden))
			y_t = T.dot(h_t, self.Wo)

			return [h_t, y_t]

		[self.h, self.y_pred], _ = theano.scan(time_step,
										sequences = self.x,
										outputs_info=[self.h0, None])
		print(self.h.type)
		print(self.y_pred.type)
		self.prob = T.nnet.softmax(self.y_pred)
		print(self.prob.type)
		self.y_out = T.argmax(self.prob, axis=-1)
		print(self.y_out.type)
		self.loss = lambda y: -T.mean(T.nnet.categorical_crossentropy(self.prob, y))
		#self.loss = lambda y: -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])


def create_vocab_mapping(file):
    vocab_mapping = {}
    i = 0
    with open(file, 'r') as f:
        while True:
            char = f.read(1)
            if not char:
                break
            else:
                if char not in vocab_mapping:
                    vocab_mapping[char] = i
                    i += 1
    return vocab_mapping


def one_hot(v_map, char, dtype=theano.config.floatX):
    i = v_map[char]
    tmp = np.zeros(len(v_map), dtype=dtype)
    tmp[i] = 1
    return tmp

def seq_2_mat(chars, vocab_map, dtype=theano.config.floatX):
    return np.matrix(map(lambda x: one_hot(vocab_map, x, dtype=dtype), chars))



