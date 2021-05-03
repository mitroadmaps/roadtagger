import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm


def uniform(shape, scale=0.05, name=None):
	"""Uniform init."""
	initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
	return tf.get_variable(name, shape=shape, initializer = tf.initializer.random_uniform(minval=-scale, maxval = scale), dtype=tf.float32)
	#return tf.Variable(initial, name=name, dtype=tf.float32)


def glorot(shape, name=None):
	"""Glorot & Bengio (AISTATS 2010) init."""
	#init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	#initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.get_variable(name, shape = shape, initializer = tf.glorot_uniform_initializer(), dtype=tf.float32)
	#return tf.Variable(initial, name=name)


def zeros(shape, name=None):
	"""All zeros."""
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)

	#return tf.Variable(initial, name=name)


def ones(shape, name=None):
	"""All ones."""
	initial = tf.ones(shape, dtype=tf.float32)
	return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(1.0), dtype=tf.float32)

	#return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
	"""Wrapper for tf.matmul (sparse vs dense)."""
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res



def create_conv_layer(name, input_tensor, in_channels, out_channels, is_training = True, activation='relu', kx = 3, ky = 3, stride_x = 2, stride_y = 2, batchnorm=False, padding='VALID', add=None, deconv = False):
	if deconv == False:
		input_tensor = tf.pad(input_tensor, [[0, 0], [kx/2, kx/2], [kx/2, kx/2], [0, 0]], mode="CONSTANT")


	weights = tf.get_variable(name+'weights', shape=[kx, ky, in_channels, out_channels],
			initializer=tf.truncated_normal_initializer(stddev=np.sqrt(0.02 / kx / ky / in_channels)),
			dtype=tf.float32
	)
	biases = tf.get_variable(name+'biases', shape=[out_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

	

	if deconv == False:
		t = tf.nn.conv2d(input_tensor, weights, [1, stride_x, stride_y, 1], padding=padding)
		s = tf.nn.bias_add(t, biases)

	else:
		batch = tf.shape(input_tensor)[0]
		size = tf.shape(input_tensor)[1]


		print(input_tensor)
		print(tf.transpose(weights,perm=[0,1,3,2]))



		t = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights,perm=[0,1,3,2]),[batch, size * stride_x, size * stride_y, out_channels], [1, stride_x, stride_y, 1],
				padding='SAME', data_format = "NHWC")
		
		# t = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights,perm=[0,1,3,2]),tf.tensor([batch, size * stride_x, size * stride_y, out_channels]), [1, stride_x, stride_y, 1],
		# 		padding='SAME', data_format = "NHWC")
		

		s = tf.nn.bias_add(t, biases)

	if add is not None: # res
		s = s + add 

	if batchnorm:
		print("use batchnorm ", name)
		n = batch_norm(s, decay = 0.95, center=True, scale=True, updates_collections=None, is_training=is_training)
	else:
		n = s 

	if activation == 'relu':
			return tf.nn.relu(n), weights, biases
	elif activation == 'sigmoid':
			return tf.nn.sigmoid(n), weights, biases
	elif activation == 'tanh':
			return tf.nn.tanh(n), weights, biases
	elif activation == 'linear':
			return n, weights, biases



# 
# input_tensor ->   A x N x H (N is the number of vertices, H is the number of feature )

def create_gcn_layer_basic(name, input_tensor, graph_structure, input_dim, output_dim, dropout = None, activation = 'tanh'):
	# create weights/bias
	with tf.variable_scope(name):
		weights = glorot([input_dim, output_dim], name='weights')
		bias = zeros([output_dim], name = 'bias')


	# dropout 
	if dropout is not None:
		x = tf.nn.dropout(input_tensor, 1 - dropout)
	else:
		x = input_tensor


	# convolve
	pre_sup = dot(x, weights)
	output = dot(graph_structure, pre_sup, sparse=True)

	# bias 
	output = output + bias 


	# todo normalize 


	if activation == 'relu':
		return tf.nn.relu(output)
	elif activation == 'sigmoid':
		return tf.nn.sigmoid(output)
	elif activation == 'tanh':
		return tf.nn.tanh(output)
	elif activation == 'linear':
		return output


def create_gcn_layer_GRU_generic_one_fc(name, input_tensor, graph_structures, input_dim, output_dim, dropout = None, activation = tf.nn.leaky_relu, reuse=True):
	# create weights/bias

	graph_num = len(graph_structures) # should be >= 1 

	with tf.variable_scope(name + 'fc_vars', reuse=tf.AUTO_REUSE):

		weights = []
		bias = []
		tensor_list = []


		for i in xrange(graph_num):
			weights.append(glorot([input_dim, input_dim], name='weights_fc'+str(i)))
			bias.append(zeros([input_dim], name = 'bias'+str(i)))

			print(input_dim*i, input_dim*(i+1))

			input_tensor_i = input_tensor[:,input_dim*i:input_dim*(i+1)]

			input_tensor_i_ = tf.nn.tanh(dot(input_tensor_i, weights[i]) + bias[i])

			input_tensor_i_ = dot(graph_structures[i], input_tensor_i_, sparse = True)

			tensor_list.append(input_tensor_i_)

		input_tensor = tf.concat(tensor_list, axis = 1)

	with tf.variable_scope(name + '_vars', reuse=tf.AUTO_REUSE):

		ba = zeros([input_dim * graph_num], name = 'biasA')
		bz = zeros([input_dim * graph_num], name = 'biasZ')
		br = zeros([input_dim * graph_num], name = 'biasR')
		bh = zeros([input_dim * graph_num], name = 'biasH')

		weightsWZ = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsWZ')
		weightsWR = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsWR')
		weightsWH = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsWH')

		weightsUZ = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsUZ')
		weightsUR = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsUR')
		weightsUH = glorot([input_dim * graph_num, input_dim * graph_num], name='weightsUH')

		x = input_tensor
		a = x + ba

		z = tf.nn.sigmoid(dot(a, weightsWZ) + dot(x, weightsUZ) + bz)

		r = tf.nn.sigmoid(dot(a, weightsWR) + dot(x, weightsUR) + br)


		h_ = tf.nn.tanh(dot(a, weightsWH) + dot(tf.multiply(r, x), weightsUH) + bh )

		h = tf.multiply((1-z),x) + tf.multiply(z, h_)

	return h









