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
	#validation_networks = []

	for folder in config["dataset_train"]:
		print("loading... ", folder)
		network = myRoadNetworkLoader(folder + "/graph.json", folder, target_shape=target_shape)
		network.annotation_filter_for_light_poles()
		training_networks.append(network)

	# for folder in config["dataset_eval"]:
	# 	print("loading... ", folder)
	# 	network = myRoadNetworkLoader(folder + "/graph.json", folder, target_shape=target_shape)
	# 	network.annotation_filter_for_light_poles()
	# 	validation_networks.append(network)


	preload_graph = None 
	preload_graph_num = config["subgraph_batch"] # just for testing 
	step = 0 
	lr = config["learningrate"]
	sloss = 0 
	with tf.Session(config=tf.ConfigProto()) as sess:
		model = RoadTaggerModel(sess, number_of_gnn_layer = config["propagation_step"], target_shape=target_shape)

		t0 = time()
		while True:
			# sample preload graph 
			if preload_graph is None or step % config["subgraph_reload_steps"] == 0:
				preload_graph = []

				for i in range(preload_graph_num):
					preload_graph.append(random.choice(training_networks).SampleSubRoadNetwork(config["subgraph_size"]))


			train_subgraph = random.choice(preload_graph)

			items = model.Train(train_subgraph, train_op = model.train_op, learning_rate=lr)
			sloss += items[0]

			#if step % 10 == 0:
			#	print(step, items[0])

			#console = code.InteractiveConsole(locals())
			#console.interact()
			# exit()

			step += 1

			if step % 200 == 0:
				print("average loss ", sloss/200.0, "at step", step)
				sloss = 0.0 

			if step % 1000 == 0 or (step<1000 and step % 200 == 0):
				print("save model to backup")
				model.saveModel(config["model_save_folder"] + "/backup")

				if step <= 1000:
					print("training 200 iterations spent", time()-t0, "seconds")
				else:
					print("training 1000 iterations spent", time()-t0, "seconds")
				t0 = time() 


			if step % 10000 == 0:
				model.saveModel(config["model_save_folder"] + "/model%d" % step)


			if step in config["learningrate_decay_at_step"]:
				lr = lr * config["learningrate_decay"]























