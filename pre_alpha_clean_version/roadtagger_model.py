import roadtagger_tf_common_layer as common
#import tf_common_layer_before20190225 as common

import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm
import random
import pickle 
import scipy.ndimage as nd 
import scipy 
import math
import svgwrite
from svgwrite.image import Image as svgimage
from PIL import Image
import resnet 
image_size = 384 


class DeepRoadMetaInfoModel():
	def __init__(self, sess, cnn_type="simple", gnn_type="simple", loss_func = "L2", number_of_gnn_layer= 4, lane_number_weight = 1.0, parking_weight = 1.0, biking_weight = 1.0, roadtype_weight = 1.0, noLeftRight = False, reuse = True, GRU=False, use_old_tf_common_layer = False, use_batchnorm = False, stage=None, homogeneous_loss_factor = 1.0):
		global common 

		# if use_old_tf_common_layer == True :
		# 	import tf_common_layer_before20190225 as common
		# else:
		# 	import tf_common_layer as common

		self.stage = stage 
		self.use_batchnorm = use_batchnorm
		self.sess = sess 
		self.loss_func = loss_func 
		self.cnn_type = cnn_type 
		self.gnn_type = gnn_type
		self.GRU = GRU
		self.reuse = reuse 

		self.lane_number_weight = lane_number_weight
		self.parking_weight = parking_weight
		self.biking_weight = biking_weight
		self.roadtype_weight = roadtype_weight
		self.noLeftRight = noLeftRight 

		self.number_of_gnn_layer = number_of_gnn_layer
		

		self.homogeneous_loss_factor = homogeneous_loss_factor

		self.Build(image_size = 384)

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=30)

		self.saver_best1 = tf.train.Saver(max_to_keep=3)
		self.saver_best2 = tf.train.Saver(max_to_keep=3)
		self.saver_best3 = tf.train.Saver(max_to_keep=3)
		self.saver_best4 = tf.train.Saver(max_to_keep=3)
		self.saver_best5 = tf.train.Saver(max_to_keep=3)


		print("network summary! ")
		print(tf.trainable_variables())



		pass 
	# simple
	def _buildCNN(self, raw_inputs, dropout = None, feature_size=126, encoder_dropout = None, is_training = True,  batchnorm=False):

		conv1, _, _ = common.create_conv_layer('cnn_l1', raw_inputs, 3, 8, kx = 5, ky = 5, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, 8, 16, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)
		conv3, _, _ = common.create_conv_layer('cnn_l3', conv2, 16, 32, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)   # 48*48*32 
		conv4, _, _ = common.create_conv_layer('cnn_l4', conv3, 32, 32, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)   # 48*48*32 
		conv5, _, _ = common.create_conv_layer('cnn_l5', conv4, 32, 32, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)   # 24*24*32
		conv6, _, _ = common.create_conv_layer('cnn_l6', conv5, 32, 64, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)   # 12*12*64
		conv7, _, _ = common.create_conv_layer('cnn_l7', conv6, 64, 64, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm)   # 6*6*64
				 
		dense0 = tf.reshape(conv7,[-1, 6*6*64])
		
		if encoder_dropout is not None:
			dense0 = tf.nn.dropout(dense0, 1-encoder_dropout)

		self.cnn_saver = tf.train.Saver(max_to_keep=40)


		dense1 = tf.layers.dense(inputs=dense0, units=256, activation=tf.nn.relu)
		dense2 = tf.layers.dense(inputs=dense1, units=feature_size)

		if dropout is not None:
			dense2 = tf.nn.dropout(dense2, 1-dropout)

		return dense2, dense0 #  64 features 

	# simple2 11+3 layer 
	def _buildCNN2(self, raw_inputs, dropout = None, feature_size=126, encoder_dropout = None, is_training = True,  batchnorm=False):
		conv1, _, _ = common.create_conv_layer('cnn_l1', raw_inputs, 3, 16, kx = 5, ky = 5, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, 16, 16, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv3, _, _ = common.create_conv_layer('cnn_l3', conv2, 16, 32, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		conv4, _, _ = common.create_conv_layer('cnn_l4', conv3, 32, 32, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv5, _, _ = common.create_conv_layer('cnn_l5', conv4, 32, 64, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		
		conv6, _, _ = common.create_conv_layer('cnn_l6', conv5, 64, 64, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv7, _, _ = common.create_conv_layer('cnn_l7', conv6, 64, 128, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 24*24*128 

		conv8, _, _ = common.create_conv_layer('cnn_l8', conv7, 128, 128, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv9, _, _ = common.create_conv_layer('cnn_l9', conv8, 128, 128, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 12*12*128 
		
		conv10, _, _ = common.create_conv_layer('cnn_l10', conv9, 128, 128, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv11, _, _ = common.create_conv_layer('cnn_l11', conv10, 128, 128, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 6*6*128 
				 
		dense0 = tf.reshape(conv11,[-1, 6*6*128])

		if encoder_dropout is not None:
			dense0 = tf.nn.dropout(dense0, 1-encoder_dropout)

		self.cnn_saver = tf.train.Saver(max_to_keep=40)

		return self._buildCNN2_readout(dense0, dropout = dropout, feature_size = feature_size), dense0 


	def _buildCNN2_readout(self, dense0, dropout = None, feature_size=126):
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		dense1 = tf.layers.dense(inputs=dense0, units=1024, activation=tf.nn.relu)
		dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
		dense3 = tf.layers.dense(inputs=dense2, units=feature_size)

		if dropout is not None:
			dense3 = tf.nn.dropout(dense3, 1-dropout)

		return dense3 #  64 features 


	# low resolution net  128 x 128 resolution
	def _buildCNN3(self, raw_inputs, dropout = None, feature_size=126, encoder_dropout = None, is_training = True,  batchnorm=False):

		raw_inputs = tf.image.resize_images(raw_inputs, [128,128])


		conv1, _, _ = common.create_conv_layer('cnn_l1', raw_inputs, 3, 16, kx = 5, ky = 5, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, 16, 16, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv3, _, _ = common.create_conv_layer('cnn_l3', conv2, 16, 32, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		conv4, _, _ = common.create_conv_layer('cnn_l4', conv3, 32, 32, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv5, _, _ = common.create_conv_layer('cnn_l5', conv4, 32, 64, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm)
		
		conv6, _, _ = common.create_conv_layer('cnn_l6', conv5, 64, 64, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv7, _, _ = common.create_conv_layer('cnn_l7', conv6, 64, 128, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 8*8*128 

		conv8, _, _ = common.create_conv_layer('cnn_l8', conv7, 128, 128, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv9, _, _ = common.create_conv_layer('cnn_l9', conv8, 128, 256, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 4*4*256
		
		conv10, _, _ = common.create_conv_layer('cnn_l10', conv9, 256, 256, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm)
		conv11, _, _ = common.create_conv_layer('cnn_l11', conv10, 256, 512, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 2*2*512
				 
		dense0 = tf.reshape(conv11,[-1, 2*2*512])

		if encoder_dropout is not None:
			dense0 = tf.nn.dropout(dense0, 1-encoder_dropout)

		self.cnn_saver = tf.train.Saver(max_to_keep=40)

		return self._buildCNN2_readout(dense0, dropout = dropout, feature_size = feature_size), dense0 


	def _buildCNN3_readout(self, dense0, dropout = None, feature_size=126):
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		dense1 = tf.layers.dense(inputs=dense0, units=1024, activation=tf.nn.relu)
		dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
		dense3 = tf.layers.dense(inputs=dense2, units=feature_size)

		if dropout is not None:
			dense3 = tf.nn.dropout(dense3, 1-dropout)

		return dense3 #  64 features




	def _buildResNet18(self, raw_inputs, is_training = True, dropout= None, feature_size=126, encoder_dropout = None):

		dense0 = resnet.resnet18plus(raw_inputs, is_training= is_training)

		if encoder_dropout is not None:
			dense0 = tf.nn.dropout(dense0, 1-encoder_dropout)




		self.cnn_saver = tf.train.Saver(max_to_keep=40)

		return self._buildCNN2_readout(dense0, dropout = dropout, feature_size = feature_size), dense0 


	# simple4  based on simple 2    11+3 layer   use average pooling, concat high resolution features to low resolution features 
	def _buildCNN4(self, raw_inputs, dropout = None, feature_size=126, encoder_dropout = None, is_training = True,  batchnorm=False):
		conv1, _, _ = common.create_conv_layer('cnn_l1', raw_inputs, 3, 16, kx = 5, ky = 5, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 192*192*16
		conv2, _, _ = common.create_conv_layer('cnn_l2', conv1, 16, 16, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm) # 192*192*16
		conv3, _, _ = common.create_conv_layer('cnn_l3', conv2, 16, 32, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 96*96*32
		conv4, _, _ = common.create_conv_layer('cnn_l4', conv3, 32, 32, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm) # 96*96*32
		conv5, _, _ = common.create_conv_layer('cnn_l5', conv4, 32, 64, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 48*48*64
		
		conv6, _, _ = common.create_conv_layer('cnn_l6', conv5, 64, 64, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm) # 48*48*64
		conv7, _, _ = common.create_conv_layer('cnn_l7', conv6, 64, 128, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 24*24*128 

		conv8, _, _ = common.create_conv_layer('cnn_l8', conv7, 128, 128, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm) # 24*24*128 
		conv9, _, _ = common.create_conv_layer('cnn_l9', conv8, 128, 256, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 12*12*256
		
		conv10, _, _ = common.create_conv_layer('cnn_l10', conv9, 256, 256, kx = 3, ky = 3, stride_x = 1, stride_y = 1,is_training = is_training, batchnorm = batchnorm) # 12*12*256
		conv11, _, _ = common.create_conv_layer('cnn_l11', conv10, 256, 512, kx = 3, ky = 3, stride_x = 2, stride_y = 2,is_training = is_training, batchnorm = batchnorm) # 6*6*512
				 

		avg_conv11 = tf.reduce_mean(conv11, axis=[1, 2], keepdims=True) # 512
		avg_conv8 = tf.reduce_mean(conv8, axis=[1, 2], keepdims=True) # 128
		avg_conv6 = tf.reduce_mean(conv6, axis=[1, 2], keepdims=True) # 64
		avg_conv4 = tf.reduce_mean(conv4, axis=[1, 2], keepdims=True) # 32
		avg_conv2 = tf.reduce_mean(conv2, axis=[1, 2], keepdims=True) # 16

		feature = tf.concat([avg_conv11, avg_conv8, avg_conv6, avg_conv4, avg_conv2], axis=3)

		dense0 = tf.reshape(feature,[-1, 512+128+64+32+16])

		if encoder_dropout is not None:
			dense0 = tf.nn.dropout(dense0, 1-encoder_dropout)

		self.cnn_saver = tf.train.Saver(max_to_keep=40)

		return self._buildCNN4_readout(dense0, dropout = dropout, feature_size = feature_size), dense0 


	def _buildCNN4_readout(self, dense0, dropout = None, feature_size=126):
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		dense1 = tf.layers.dense(inputs=dense0, units=1024, activation=tf.nn.relu)
		dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
		dense3 = tf.layers.dense(inputs=dense2, units=feature_size)

		if dropout is not None:
			dense3 = tf.nn.dropout(dense3, 1-dropout)

		return dense3 #  64 features 




	# def _buildGCN(self, input_features,dropout = None, target_dim = 5):

	# 	gcn1 = common.create_gcn_layer_2('gcn1',input_features, self.graph_structure, 64, 128)
	# 	gcn2 = common.create_gcn_layer_2('gcn2',gcn1, self.graph_structure, 128, 128)
	# 	gcn3 = common.create_gcn_layer_2('gcn3',gcn2, self.graph_structure, 128, 128)
	# 	gcn4 = common.create_gcn_layer_2('gcn4',gcn3, self.graph_structure, 128, 128)

	# 	# unroll ?
	# 	gcn5 = common.create_gcn_layer_basic('gcn5',gcn4, self.graph_structure, 128, 128)
	# 	gcn6 = common.create_gcn_layer_basic('gcn6',gcn5, self.graph_structure, 128, 128)
	# 	gcn7 = common.create_gcn_layer_basic('gcn7',gcn6, self.graph_structure, 128, 128, dropout = dropout)

		
	# 	dense1 = tf.layers.dense(inputs=gcn7, units=128, activation=tf.nn.relu)
	# 	dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
	# 	dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
	# 	dense4 = tf.layers.dense(inputs=dense3, units=8, activation=tf.nn.relu)
	# 	dense5 = tf.layers.dense(inputs=dense4, units=target_dim)


	# 	return dense5 

	# def _buildGCN(self, input_features,dropout = None, target_dim = 14, loop = 4):

	# 	x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.relu)
	# 	x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)

	# 	gcn1 = common.create_gcn_layer_2('gcn1',x, self.graph_structure, 128, 128)
	# 	gcn2 = common.create_gcn_layer_2('gcn2',gcn1, self.graph_structure, 128, 128)
	# 	gcn3 = common.create_gcn_layer_2('gcn3',gcn2, self.graph_structure, 128, 128)
	# 	gcn4 = common.create_gcn_layer_2('gcn4',gcn3, self.graph_structure, 128, 128, dropout = dropout)

	# 	#gcn_loop = gcn4

	# 	gcn_loop = [gcn4]
	# 	# unroll ?
	# 	for i in xrange(loop):
	# 		gcn_loop.append(common.create_gcn_layer_2('gcn_loop',gcn_loop[i], self.graph_structure, 128, 128))
		
	# 	# skip layer 
	# 	concat_layer = tf.concat([x, gcn_loop[loop]], axis = 1)

	# 	dense1 = tf.layers.dense(inputs=concat_layer, units=64, activation=tf.nn.relu)
	# 	dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.relu)
	# 	dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
	# 	dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
	# 	dense5 = tf.layers.dense(inputs=dense4, units=target_dim)


	# 	return dense5 


	def _buildGCN(self, input_features,dropout = None, target_dim = 14, loop = 4):
		reuse = self.reuse 

		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.leaky_relu)
		x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)

		x_ = x 
		# unroll ?
		loop = self.number_of_gnn_layer

		print("Number of GNN layer ", loop)

		for i in xrange(loop):
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.leaky_relu)
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
			

			if self.GRU==True:
				print("use gru")
				x_ = common.create_gcn_layer_GRU('gcn_loop', x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)

			else:
				if reuse :
					x_ = common.create_gcn_layer_2('gcn_loop', x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
				else:
					x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
				
		# skip layer 
		concat_layer = tf.concat([x, x_], axis = 1)

		#concat_layer = x_ # no skip layer 

		dense1 = tf.layers.dense(inputs=concat_layer, units=64, activation=tf.nn.leaky_relu)
		dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.leaky_relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
		#dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
		dense5 = tf.layers.dense(inputs=dense2, units=target_dim)


		return dense5


	def _buildGCNRawGraph(self, input_features,dropout = None, target_dim = 14, loop = 4):
		reuse = self.reuse 

		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.leaky_relu)
		x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)

		x_ = x 
		# unroll ?
		loop = self.number_of_gnn_layer

		print("Number of GNN layer ", loop)

		for i in xrange(loop):
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.leaky_relu)
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
			
			if self.GRU==True:
				print("use gru")
				x_ = common.create_gcn_layer_GRU_one_more_fc('gcn_loop', x_, self.graph_structure_fully_connected, 128, 128, activation = tf.nn.tanh)

			else:

				if reuse :
					x_ = common.create_gcn_layer_2('gcn_loop', x_, self.graph_structure_fully_connected, 128, 128, activation = tf.nn.tanh)
				else:
					x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure_fully_connected, 128, 128, activation = tf.nn.tanh)
				
		# skip layer 
		concat_layer = tf.concat([x, x_], axis = 1)

		#concat_layer = x_ # no skip layer 

		dense1 = tf.layers.dense(inputs=concat_layer, units=64, activation=tf.nn.leaky_relu)
		dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.leaky_relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
		#dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
		dense5 = tf.layers.dense(inputs=dense2, units=target_dim)


		return dense5


	def _buildGCNRoadGeneric(self, input_features, graphs, dropout=None, target_dim=14, loop = 4):
		reuse = self.reuse 

		num_graphs = len(graphs)

		x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.leaky_relu)
		x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)

		x_gnn = tf.concat([x for i in xrange(num_graphs)], axis = 1)

		loop = self.number_of_gnn_layer
		print("Number of GNN layer ", loop)


		for i in xrange(loop):
			print("use gru generic")
			x_gnn = common.create_gcn_layer_GRU_generic_one_fc('gcn_loop', x_gnn, graphs, 128, 128, activation = tf.nn.tanh)


		concat_layer = tf.concat([x, x_gnn], axis = 1)


		dense1 = tf.layers.dense(inputs=concat_layer, units=64*num_graphs, activation=tf.nn.leaky_relu)
		dense2 = tf.layers.dense(inputs=dense1, units=32*num_graphs, activation=tf.nn.leaky_relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
		#dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
		dense5 = tf.layers.dense(inputs=dense2, units=target_dim)


		return dense5


	def _buildGCNRoadExtractionBD(self, input_features,dropout = None, target_dim = 14, loop = 4):
		reuse = self.reuse 

		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.leaky_relu)
		x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)

		x_ = x 
		# unroll ?
		loop = self.number_of_gnn_layer

		print("Number of GNN layer ", loop)

		for i in xrange(loop):
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.leaky_relu)
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
			
			if self.GRU==True:
				print("use gru")
				x_ = common.create_gcn_layer_GRU_bidirectional_one_fc('gcn_loop', x_, self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2, 128, 128, activation = tf.nn.tanh)

			else:
				exit()
					
		# skip layer 
		concat_layer = tf.concat([x, x_], axis = 1)

		#concat_layer = x_ # no skip layer 

		dense1 = tf.layers.dense(inputs=concat_layer, units=64, activation=tf.nn.leaky_relu)
		dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.leaky_relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
		#dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
		dense5 = tf.layers.dense(inputs=dense2, units=target_dim)


		return dense5

	def _buildGCNRawGraph_RoadExtractionBD(self, input_features,dropout = None, target_dim = 14, loop = 4):
		reuse = self.reuse 

		#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

		x = tf.layers.dense(inputs=input_features, units=128, activation=tf.nn.leaky_relu)
		x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)

		x_ = x 
		# unroll ?
		loop = self.number_of_gnn_layer

		print("Number of GNN layer ", loop)

		for i in xrange(loop):
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.leaky_relu)
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
			
			if self.GRU==True:
				print("use gru")
				x_ = common.create_gcn_layer_GRU_one_more_fc('gcn_loop', x_, self.graph_structure_fully_connected, 128, 128, activation = tf.nn.tanh)

			else:
				exit()
					
		# skip layer 
		# concat_layer = tf.concat([x, x_], axis = 1)


		# x__ = concat_layer

		# x__ = tf.layers.dense(inputs=x__, units=256, activation=tf.nn.leaky_relu)
		# x__ = tf.layers.dense(inputs=x__, units=128, activation=tf.nn.leaky_relu)

		x__ = x
		for i in xrange(loop):
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.leaky_relu)
			#x_ = common.create_gcn_layer_2('gcn_loop'+str(i), x_, self.graph_structure, 128, 128, activation = tf.nn.tanh)
			
			if self.GRU==True:
				print("use gru")
				x__ = common.create_gcn_layer_GRU_bidirectional_one_fc('gcn_loop2', x__, self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2, 128, 128, activation = tf.nn.tanh)

			else:
				exit()
				
		concat_layer = tf.concat([x, x_, x__], axis = 1)

		#concat_layer = x_ # no skip layer 

		dense1 = tf.layers.dense(inputs=concat_layer, units=64, activation=tf.nn.leaky_relu)
		dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.leaky_relu)
		#dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
		#dense4 = tf.layers.dense(inputs=dense3, units=32, activation=tf.nn.relu)
		dense5 = tf.layers.dense(inputs=dense2, units=target_dim)

		return dense5

	
	# build the gcn 
	def Build(self, image_size = 384, image_channel = 2):

		self.image_size = image_size
		self.image_channel = 3

		self.lr = tf.placeholder(tf.float32, shape=[])
		self.graph_structure = tf.sparse_placeholder(tf.float32)

		self.graph_structure_fully_connected = tf.sparse_placeholder(tf.float32)
		self.graph_structure_decomposed_dir1 = tf.sparse_placeholder(tf.float32)
		self.graph_structure_decomposed_dir2 = tf.sparse_placeholder(tf.float32)
		self.graph_structure_auxiliary = tf.sparse_placeholder(tf.float32)


		self.per_node_raw_inputs = tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size, self.image_channel])
		
		self.target = tf.placeholder(tf.int32, shape = [None,6])
		self.target_mask = tf.placeholder(tf.float32, shape = [None])

		self.lane_number_balance = tf.placeholder(tf.float32, shape = [None])
		self.parking_balance = tf.placeholder(tf.float32, shape = [None])
		self.biking_balance = tf.placeholder(tf.float32, shape = [None])
		self.roadtype_balance = tf.placeholder(tf.float32, shape = [None])

		self.global_loss_mask = tf.placeholder(tf.float32, shape = [None])
		self.homogeneous_loss_mask =  tf.placeholder(tf.float32, shape = [None])

		self.dropout = tf.placeholder(tf.float32, shape = [])

		self.heading_vector = tf.placeholder(tf.float32, shape = [None,2])
		self.intersectionFeatures = tf.placeholder(tf.float32, shape = [None, 64])

		self.is_training = tf.placeholder(tf.bool)

		self.nonIntersectionNodeNum = tf.placeholder(tf.int32, shape= [])


		self.node_dropout_mask = tf.placeholder(tf.float32, shape = [None, 62])
		self.node_dropout_gradient_mask = tf.placeholder(tf.float32, shape = [None, 62])


		if self.gnn_type != "none":

			
			if self.cnn_type == "simple":
				self.node_feature,_ = self._buildCNN(self.per_node_raw_inputs,dropout = None, encoder_dropout=self.dropout, feature_size=62, batchnorm = self.use_batchnorm, is_training = self.is_training) # memory issue (to be fixed)
			elif self.cnn_type == "simple2":
				self.node_feature,_ = self._buildCNN2(self.per_node_raw_inputs,dropout = None, encoder_dropout=self.dropout, feature_size=62, batchnorm = self.use_batchnorm, is_training = self.is_training) # memory issue (to be fixed)
			elif self.cnn_type == "simple3":
				self.node_feature,_ = self._buildCNN3(self.per_node_raw_inputs,dropout = None, encoder_dropout=self.dropout, feature_size=62, batchnorm = self.use_batchnorm, is_training = self.is_training) # memory issue (to be fixed)
			elif self.cnn_type == "simple4":
				self.node_feature,_ = self._buildCNN4(self.per_node_raw_inputs,dropout = None, encoder_dropout=self.dropout, feature_size=62, batchnorm = self.use_batchnorm, is_training = self.is_training) # memory issue (to be fixed)
			
			elif self.cnn_type == "resnet18":
				self.node_feature = tf.nn.dropout(resnet.resnet(self.per_node_raw_inputs, is_training = self.is_training, feature_size=62),1-self.dropout)

			if self.stage == 2:

				self.node_feature_intermediate = tf.placeholder(tf.float32, shape = [None, 62])


			else:
				self.node_feature_intermediate = self.node_feature


			# node feature whole node drop out 


			self.node_feature_intermediate_ = tf.multiply(self.node_feature_intermediate, self.node_dropout_mask)

			# stop_gradient
			self.node_feature_intermediate_ = tf.stop_gradient(tf.multiply(self.node_feature_intermediate_, self.node_dropout_gradient_mask)) + tf.multiply(self.node_feature_intermediate_, 1.0-self.node_dropout_gradient_mask)
			



			self.node_feature = tf.concat([self.node_feature_intermediate_, self.heading_vector], axis = 1)
			
			


			self.node_feature = tf.concat([self.node_feature, self.intersectionFeatures], axis = 0)


			print("GNN type", self.gnn_type)

			if self.gnn_type == "simple":
				self._output = self._buildGCN(self.node_feature,self.dropout, target_dim = 16)  # 6 + 4 * 2
			
			elif self.gnn_type == "RawGraph":
				self._output = self._buildGCNRawGraph(self.node_feature,self.dropout, target_dim = 16)
				pass
			elif self.gnn_type == "RoadExtraction":
				self._output = self._buildGCN(self.node_feature,self.dropout, target_dim = 16)  # 6 + 4 * 2
				print("TODO", self.gnn_type)
				exit() 

			elif self.gnn_type == "RoadExtractionBD":
				self._output = self._buildGCNRoadExtractionBD(self.node_feature,self.dropout, target_dim = 16)
				pass
			elif self.gnn_type == "RawGraphRoadExtractionBD":
				self._output = self._buildGCNRawGraph_RoadExtractionBD(self.node_feature,self.dropout, target_dim = 16)
								    
				pass 
			elif self.gnn_type == "RBDplusRaw":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2, self.graph_structure_fully_connected],  self.dropout, target_dim = 16)
			elif self.gnn_type == "RBD":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2],  self.dropout, target_dim = 16)
			elif self.gnn_type == "RBDplusAux":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2, self.graph_structure_auxiliary],  self.dropout, target_dim = 16)
			elif self.gnn_type == "RBDplusRawplusAux":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure_decomposed_dir1, self.graph_structure_decomposed_dir2, self.graph_structure_fully_connected, self.graph_structure_auxiliary],  self.dropout, target_dim = 16)
			elif self.gnn_type == "Raw":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure_fully_connected],  self.dropout, target_dim = 16)
			elif self.gnn_type == "Road":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure],  self.dropout, target_dim = 16)
			elif self.gnn_type == "RoadplusAux":
				self._output = self._buildGCNRoadGeneric(self.node_feature, [self.graph_structure, self.graph_structure_auxiliary],  self.dropout, target_dim = 16)
			
			else:
				print("TODO", self.gnn_type)
				exit() 

			self._output_whole_graph = self._output
			self._output = self._output[0:self.nonIntersectionNodeNum,:]


		else:
			print(self.cnn_type)
			
			if self.cnn_type == "simple":
				self._output, _ = self._buildCNN(self.per_node_raw_inputs,dropout = None,feature_size=16, encoder_dropout=self.dropout, batchnorm = self.use_batchnorm, is_training = self.is_training) # memory issue (to be fixed)
			elif self.cnn_type == "simple2":
				print("???2")
				self._output, _ = self._buildCNN2(self.per_node_raw_inputs,dropout = None,feature_size=16, encoder_dropout=self.dropout, batchnorm = self.use_batchnorm, is_training = self.is_training)
			elif self.cnn_type == "simple3":
				print("???3")
				self._output, _ = self._buildCNN3(self.per_node_raw_inputs,dropout = None,feature_size=16, encoder_dropout=self.dropout, batchnorm = self.use_batchnorm, is_training = self.is_training)
			elif self.cnn_type == "simple4":
				print("???3")
				self._output, _ = self._buildCNN4(self.per_node_raw_inputs,dropout = None,feature_size=16, encoder_dropout=self.dropout, batchnorm = self.use_batchnorm, is_training = self.is_training)
			
			elif self.cnn_type == "resnet18":
				self._output, _ = self._buildResNet18(self.per_node_raw_inputs,dropout = None,is_training = self.is_training, feature_size=16, encoder_dropout=self.dropout)

			self.real_output = self._output 

			if self.stage == 2:
				self.node_feature_intermediate = tf.placeholder(tf.float32, shape = [None, 16])
				self._output = self.node_feature_intermediate
			else:
				self.node_feature_intermediate = self._output 

				#self._output = resnet.resnet(self.per_node_raw_inputs, is_training = self.is_training, feature_size=16)
		#self._output = tf.slice(self._output,[0,0], [self.nonIntersectionNodeNum, 14])

		self._output_unstacks = tf.unstack(self._output, axis = 1)
		self._output_unstacks_reshape = []

		for x in self._output_unstacks:
			#print(x, tf.reshape(x, shape=[-1,1]))
			self._output_unstacks_reshape.append(tf.reshape(x, shape=[-1,1]))
		
		if self.gnn_type != "none":
			self._output_unstacks_whole_graph = tf.unstack(self._output_whole_graph, axis = 1)
			self._output_unstacks_whole_graph_reshape = []

			for x in self._output_unstacks_whole_graph:
				#print(x, tf.reshape(x, shape=[-1,1]))
				self._output_unstacks_whole_graph_reshape.append(tf.reshape(x, shape=[-1,1]))
			


			self._output_lane_number_whole_graph = tf.nn.softmax(tf.concat(self._output_unstacks_whole_graph_reshape[0:6], axis = 1))
			self._output_left_park_whole_graph = tf.nn.softmax(tf.concat(self._output_unstacks_whole_graph_reshape[6:8], axis = 1))
			self._output_left_bike_whole_graph = tf.nn.softmax(tf.concat(self._output_unstacks_whole_graph_reshape[8:10], axis = 1))
			self._output_roadtype_whole_graph = tf.nn.softmax(tf.concat(self._output_unstacks_whole_graph_reshape[14:16], axis = 1))




		self._output_lane_number = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[0:6], axis = 1))
		self._output_left_park = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[6:8], axis = 1))
		self._output_left_bike = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[8:10], axis = 1))
		self._output_right_bike = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[10:12], axis = 1))
		self._output_right_park = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[12:14], axis = 1))
		self._output_roadtype = tf.nn.softmax(tf.concat(self._output_unstacks_reshape[14:16], axis = 1))

		self.__output_lane_number = tf.concat(self._output_unstacks_reshape[0:6], axis = 1)
		self.__output_left_park = tf.concat(self._output_unstacks_reshape[6:8], axis = 1)
		self.__output_left_bike = tf.concat(self._output_unstacks_reshape[8:10], axis = 1)
		self.__output_right_bike = tf.concat(self._output_unstacks_reshape[10:12], axis = 1)
		self.__output_right_park = tf.concat(self._output_unstacks_reshape[12:14], axis = 1)
		self.__output_roadtype = tf.concat(self._output_unstacks_reshape[14:16], axis = 1)

		self._target_unstacks = tf.unstack(self.target, axis = 1)

		if self.gnn_type != "none":
			self.lane_number_balance_ = tf.multiply(self.lane_number_balance, self.global_loss_mask)
			self.roadtype_balance_ = tf.multiply(self.roadtype_balance, self.global_loss_mask)
			self.parking_balance_ = tf.multiply(self.parking_balance, self.global_loss_mask)
			self.biking_balance_ = tf.multiply(self.biking_balance, self.global_loss_mask)
		else:
			self.lane_number_balance_ = self.lane_number_balance
			self.roadtype_balance_ = self.roadtype_balance
			self.parking_balance_ = self.parking_balance
			self.biking_balance_ = self.biking_balance



		# homogeneous_loss

		


		self.loss_lane_number = self.lane_number_weight * tf.reduce_mean(tf.multiply(self.lane_number_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[0], logits = self.__output_lane_number)))
		#self.loss_lane_number = 1.0/3.0 * tf.reduce_mean(tf.multiply(self.lane_number_balance, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[0], logits = self._output_lane_number)))
		
		self.loss_left_park = self.parking_weight * tf.reduce_mean(tf.multiply(self.parking_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[1], logits = self.__output_left_park)))
		self.loss_left_bike = self.biking_weight * tf.reduce_mean(tf.multiply(self.biking_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[2], logits = self.__output_left_bike)))
		self.loss_right_bike = self.biking_weight * tf.reduce_mean(tf.multiply(self.biking_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[3], logits = self.__output_right_bike)))
		self.loss_right_park = self.parking_weight * tf.reduce_mean(tf.multiply(self.parking_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[4], logits = self.__output_right_park)))
		self.loss_roadtype = self.roadtype_weight * tf.reduce_mean(tf.multiply(self.roadtype_balance_, tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_unstacks[5], logits = self.__output_roadtype)))
		
		if self.noLeftRight == False:
			self.loss = self.loss_lane_number + self.loss_left_park + self.loss_left_bike + self.loss_right_bike + self.loss_right_park + self.loss_roadtype
		else:
			self.loss = self.loss_lane_number + self.loss_left_park + self.loss_left_bike + self.loss_roadtype



		if self.gnn_type != "none":

			self.softmax_output_concat = tf.concat([self._output_lane_number_whole_graph, self._output_roadtype_whole_graph, self._output_left_park_whole_graph, self._output_left_bike_whole_graph], axis = 1)

			diff = tf.square(self.softmax_output_concat - tf.sparse_tensor_dense_matmul(self.graph_structure, self.softmax_output_concat))

			#diff = diff[0:self.nonIntersectionNodeNum,:]

			diff = tf.reduce_mean(diff, axis = 1)

			self.homogeneous_loss = tf.reduce_mean(tf.multiply(diff, self.homogeneous_loss_mask))
		else:
			self.homogeneous_loss = self.loss



		if self.gnn_type != "none":
			loss_addon = self.homogeneous_loss * self.homogeneous_loss_factor 
		else:
			loss_addon = 0 




		#self.loss += tf.losses.get_regularization_loss()

		#self.loss = tf.reduce_mean(tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.target, logits = self.output), self.target_mask))
		

		#self.loss = tf.reduce_mean(tf.multiply(tf.square(self.output-self.target), self.target_mask)) 

		# self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss+tf.losses.get_regularization_loss())

		# self.train_lane_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_lane_number+tf.losses.get_regularization_loss())
		# self.train_type_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_roadtype+tf.losses.get_regularization_loss())
		# self.train_bike_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_left_bike+tf.losses.get_regularization_loss())

		if self.stage == 2 and self.gnn_type == "none":
			self.loss_fake = tf.reduce_mean(self.real_output)

			self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_fake)

			self.train_lane_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_fake)
			self.train_type_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_fake)
			self.train_bike_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_fake)

		else:
			self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss + loss_addon)

			self.train_lane_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_lane_number + loss_addon)
			self.train_type_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_roadtype + loss_addon)
			self.train_bike_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_left_bike + loss_addon)


		self.summary_loss = []
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)
		self.total_train_loss =  tf.placeholder(tf.float32)
		self.test_acc_lane = tf.placeholder(tf.float32)
		self.test_acc_road_type = tf.placeholder(tf.float32)
		self.test_acc_biking = tf.placeholder(tf.float32)
		self.test_acc_parking = tf.placeholder(tf.float32)
		self.test_acc_overall = tf.placeholder(tf.float32)

		self.train_homogeneous_loss = tf.placeholder(tf.float32)
		self.test_homogeneous_loss = tf.placeholder(tf.float32)



	
		self.summary_loss.append(tf.summary.scalar('loss/test', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('loss/train', self.train_loss))

		self.summary_loss.append(tf.summary.scalar('loss/total_train', self.total_train_loss))

		self.summary_loss.append(tf.summary.scalar('homogeneous_loss/train', self.train_homogeneous_loss))
		self.summary_loss.append(tf.summary.scalar('homogeneous_loss/test', self.test_homogeneous_loss))

		self.summary_loss.append(tf.summary.scalar('acc/overall', self.test_acc_overall))
		self.summary_loss.append(tf.summary.scalar('acc/lane', self.test_acc_lane))
		self.summary_loss.append(tf.summary.scalar('acc/road_type', self.test_acc_road_type))
		self.summary_loss.append(tf.summary.scalar('acc/biking', self.test_acc_biking))
		self.summary_loss.append(tf.summary.scalar('acc/parking', self.test_acc_parking))


		self.merged_summary = tf.summary.merge_all()

		

		pass


	def Train(self, roadNetwork, learning_rate = 0.001, train_op = None, batch_size = None, use_drop_node = True, train_gnn_only=False):

		r,m = roadNetwork.GetNodeDropoutMask(use_drop_node, batch_size, stop_gradient = train_gnn_only)


		feed_dict = {
			self.graph_structure: roadNetwork.GetGraphStructure(),
			self.graph_structure_decomposed_dir1: roadNetwork.tf_spares_graph_structure_direction1,
			self.graph_structure_decomposed_dir2: roadNetwork.tf_spares_graph_structure_direction2,
			self.graph_structure_fully_connected: roadNetwork.tf_spares_graph_structure_fully_connected,
			self.graph_structure_auxiliary:roadNetwork.tf_spares_graph_structure_auxiliary,
			self.lr: learning_rate,
			self.per_node_raw_inputs: roadNetwork.GetImages(batch_size),
			self.target : roadNetwork.GetTarget(batch_size),
			self.nonIntersectionNodeNum: roadNetwork.nonIntersectionNodeNum,
			self.heading_vector: roadNetwork.GetHeadingVector(),
			self.intersectionFeatures: roadNetwork.GetIntersectionFeatures(),
			self.target_mask : roadNetwork.GetTargetMask(batch_size),
			self.lane_number_balance : roadNetwork.Get_lane_number_balance(batch_size),
			self.parking_balance : roadNetwork.Get_parking_balance(batch_size),
			self.biking_balance : roadNetwork.Get_biking_balance(batch_size),
			self.roadtype_balance : roadNetwork.Get_roadtype_balance(batch_size),
			self.node_dropout_mask: r,
			self.node_dropout_gradient_mask: m,
			self.global_loss_mask: roadNetwork.GetGlobalLossMask(128),
			self.homogeneous_loss_mask: roadNetwork.GetHomogeneousLossMask(),
			self.dropout: 0.3,
			self.is_training:True
		}

		if train_op is None:
			train_op = self.train_op

		return self.sess.run([self.loss, self._output_lane_number, self._output_roadtype, self.loss_lane_number, self.loss_left_park, self.loss_left_bike, self.loss_right_bike, self.loss_right_park, self.loss_roadtype,  train_op, self.homogeneous_loss], feed_dict = feed_dict)

	def Evaluate(self, roadNetwork, batch_size = None):
		r,m = roadNetwork.GetNodeDropoutMask(False, batch_size)



		feed_dict = {
			self.graph_structure: roadNetwork.GetGraphStructure(),
			self.graph_structure_decomposed_dir1: roadNetwork.tf_spares_graph_structure_direction1,
			self.graph_structure_decomposed_dir2: roadNetwork.tf_spares_graph_structure_direction2,
			self.graph_structure_fully_connected: roadNetwork.tf_spares_graph_structure_fully_connected,
			self.graph_structure_auxiliary:roadNetwork.tf_spares_graph_structure_auxiliary,
			self.per_node_raw_inputs: roadNetwork.GetImages(batch_size),
			self.target : roadNetwork.GetTarget(batch_size),
			self.nonIntersectionNodeNum: roadNetwork.nonIntersectionNodeNum,
			self.heading_vector: roadNetwork.GetHeadingVector(use_random=False),
			self.intersectionFeatures: roadNetwork.GetIntersectionFeatures(),
			self.target_mask : roadNetwork.GetTargetMask(batch_size),
			self.lane_number_balance : roadNetwork.Get_lane_number_balance(batch_size),
			self.parking_balance : roadNetwork.Get_parking_balance(batch_size),
			self.biking_balance : roadNetwork.Get_biking_balance(batch_size),
			self.roadtype_balance : roadNetwork.Get_roadtype_balance(batch_size),
			self.node_dropout_mask: r,
			self.node_dropout_gradient_mask: m,
			self.global_loss_mask: roadNetwork.GetGlobalLossMask(None),
			self.homogeneous_loss_mask: roadNetwork.GetHomogeneousLossMask(),
			self.dropout: 0.0,
			self.is_training:False
		}

		return self.sess.run([self.loss, self._output_lane_number, self._output_left_park, self._output_left_bike, self._output_right_bike, self._output_right_park, self._output_roadtype, self._output_unstacks_reshape, self.homogeneous_loss], feed_dict = feed_dict)


	def GetIntermediateNodeFeature(self, roadNetwork,st,ed, batch_size = None):
		feed_dict = {
			self.per_node_raw_inputs: roadNetwork.GetImages(batch_size)[st:ed,:,:,:],
			self.dropout: 0.0,
			self.is_training:False
		}

		return self.sess.run([self.node_feature_intermediate], feed_dict = feed_dict)


	def EvaluateWithIntermediateNodeFeature(self, roadNetwork, node_feature_intermediate, batch_size = None):
		r,m = roadNetwork.GetNodeDropoutMask(False, batch_size)

		feed_dict = {
			self.graph_structure: roadNetwork.GetGraphStructure(),
			self.graph_structure_decomposed_dir1: roadNetwork.tf_spares_graph_structure_direction1,
			self.graph_structure_decomposed_dir2: roadNetwork.tf_spares_graph_structure_direction2,
			self.graph_structure_fully_connected: roadNetwork.tf_spares_graph_structure_fully_connected,
			self.graph_structure_auxiliary:roadNetwork.tf_spares_graph_structure_auxiliary,
			self.node_feature_intermediate: node_feature_intermediate,
			self.target : roadNetwork.GetTarget(batch_size),
			self.nonIntersectionNodeNum: roadNetwork.nonIntersectionNodeNum,
			self.heading_vector: roadNetwork.GetHeadingVector(use_random=False),
			self.intersectionFeatures: roadNetwork.GetIntersectionFeatures(),
			self.target_mask : roadNetwork.GetTargetMask(batch_size),
			self.lane_number_balance : roadNetwork.Get_lane_number_balance(batch_size),
			self.parking_balance : roadNetwork.Get_parking_balance(batch_size),
			self.biking_balance : roadNetwork.Get_biking_balance(batch_size),
			self.roadtype_balance : roadNetwork.Get_roadtype_balance(batch_size),
			self.node_dropout_mask: r,
			self.node_dropout_gradient_mask: m,
			self.global_loss_mask: roadNetwork.GetGlobalLossMask(None),
			self.homogeneous_loss_mask: roadNetwork.GetHomogeneousLossMask(),
			self.dropout: 0.0,
			self.is_training:False
		}

		return self.sess.run([self.loss, self._output_lane_number, self._output_left_park, self._output_left_bike, self._output_right_bike, self._output_right_park, self._output_roadtype, self._output_unstacks_reshape, self.homogeneous_loss], feed_dict = feed_dict)


	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def saveModelBest(self, saver, path):
		saver.save(self.sess, path)



	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

	def saveCNNModel(self,path):
		self.cnn_saver.save(self.sess, path)

	def restoreCNNModel(self, path):
		self.cnn_saver.restore(self.sess, path)


	def addLog(self, test_loss, train_loss, test_acc_parking = 0, test_acc_biking = 0, test_acc_overall= 0, test_acc_lane = 0, test_acc_road_type = 0, test_homogeneous_loss = 0, train_homogeneous_loss = 0, total_train_loss = 0 ):
		return self.sess.run(self.merged_summary , feed_dict = {self.total_train_loss : total_train_loss, self.train_homogeneous_loss: train_homogeneous_loss, self.test_homogeneous_loss: test_homogeneous_loss, self.test_loss:test_loss, self.train_loss: train_loss, self.test_acc_biking:test_acc_biking, self.test_acc_lane:test_acc_lane, self.test_acc_overall:test_acc_overall, self.test_acc_parking:test_acc_parking, self.test_acc_road_type:test_acc_road_type})


	def dumpWeights(self):
		variables_names = [v.name for v in tf.trainable_variables()]
		values = self.sess.run(variables_names)
		for k, v in zip(variables_names, values):
			# print("Variable: ", k)
			# print("Shape: ", v.shape)
			# print(np.amin(v), np.amax(v), np.mean(v), np.std(v))

			if np.isnan(np.amin(v)) or np.isnan(np.amax(v)) or np.isnan(np.mean(v)):
				print(np.amin(v), np.amax(v), np.mean(v), np.std(v))
				print("Variable: ", k)
				print("Shape: ", v.shape)
				return False 

		return True 




