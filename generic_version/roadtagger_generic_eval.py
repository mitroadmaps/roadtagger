import json
import sys  
from roadtagger_generic_network import RoadNetwork, SubRoadNetwork
import numpy as np 
import scipy.ndimage 
from roadtagger_generic_model import RoadTaggerModel
import random
import tensorflow as tf 
import math
import code 
from time import time, sleep 
from roadtagger_generic_graph_loader import myRoadNetworkLoader

if __name__ == "__main__":
	config = json.load(open(sys.argv[1], "r"))

	target_shape = config["target_shape"]

	training_networks = []

	for folder in config["dataset_eval"]:
		print("loading... ", folder)
		network = myRoadNetworkLoader(folder + "/graph.json", folder, target_shape=target_shape)
		network.annotation_filter_for_light_poles()
		training_networks.append(network)


	for gid in range(len(training_networks)):
		graph_size = training_networks[gid].graphsize()
		#graph_size = 2048
		input_region = training_networks[gid].SampleSubRoadNetwork(graph_size = graph_size, reseed=True)

		# stage 1 
		node_feature = np.zeros((input_region.nonIntersectionNodeNum, 62))

		with tf.Session() as sess:
			model = RoadTaggerModel(sess, number_of_gnn_layer = config["propagation_step"], target_shape=target_shape)
			model.restoreModel(sys.argv[2])

			batch_size = 256 

			st = 0

			while st < input_region.nonIntersectionNodeNum:
				print(st)
				ed = min(st+batch_size, input_region.nonIntersectionNodeNum)

				t0 = time() 
				node_feature[st:ed,:] = model.GetIntermediateNodeFeature(input_region, st, ed)[0]
				#time_cnn += time() - t0 

				st += batch_size

		# stage 2
		tf.reset_default_graph()

		with tf.Session() as sess:
			model = RoadTaggerModel(sess, number_of_gnn_layer = config["propagation_step"], target_shape=target_shape)
			model.restoreModel(sys.argv[2])

			outputs = model.EvaluateWithIntermediateNodeFeature(input_region, node_feature)

			input_region.VisualizeResultLight(outputs[1], "output_light_%d.png" % gid)
			input_region.VisualizeResult(outputs[1], "output_%d.png" % gid)
			input_region.VisualizeResult(outputs[1], "output_4x_%d.png" % gid, imgname = "/sat_16384.png", size=16384, scale=3)

		tf.reset_default_graph()





