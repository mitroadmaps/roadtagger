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

# example graph loader
class myRoadNetwork():
	def __init__(self, jsonGraphFile, tileFolder, target_shape=[2]):
		self.roadnetwork = RoadNetwork()

		jsongraph = json.load(open(jsonGraphFile,"r"))

		for nid in range(len(jsongraph["nodes"])):
			loc = jsongraph["nodes"][nid]
			loc = (loc[0],loc[1])

			self.roadnetwork.nodes[loc] = [nid, []]
			self.roadnetwork.nid2loc[nid] = loc 
			self.roadnetwork.node_num += 1 

		for edge in jsongraph["edges"]:
			self.roadnetwork.AddEdge(edge[0], edge[1])

		# annotation 

		self.roadnetwork.target_shape = target_shape
		self.roadnetwork.annotation = {}

		for nid in range(len(jsongraph["nodes"])):
			loc = jsongraph["nodes"][nid]
			loc = (loc[0],loc[1])

			self.roadnetwork.annotation[nid] = {}
			self.roadnetwork.annotation[nid]["degree"] = len(self.roadnetwork.node_degree[nid])
			self.roadnetwork.annotation[nid]["remove"] = 0
			

			heading_vector_lat = 0 
			heading_vector_lon = 0

			if len(self.roadnetwork.node_degree[nid]) > 2:
				heading_vector_lat = 0 
				heading_vector_lon = 0
			elif len(self.roadnetwork.node_degree[nid]) == 1:
				loc1 = self.roadnetwork.nid2loc[nid]
				loc2 = self.roadnetwork.nid2loc[self.roadnetwork.node_degree[nid][0]]

				dlat = loc1[0] - loc2[0]
				dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]))

				l = np.sqrt(dlat*dlat + dlon * dlon)

				dlat /= l
				dlon /= l 

				heading_vector_lat = dlat 
				heading_vector_lon = dlon 

			elif len(self.roadnetwork.node_degree[nid]) == 2:
				loc1 = self.roadnetwork.nid2loc[self.roadnetwork.node_degree[nid][1]]
				loc2 = self.roadnetwork.nid2loc[self.roadnetwork.node_degree[nid][0]]

				dlat = loc1[0] - loc2[0]
				dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]))

				l = np.sqrt(dlat*dlat + dlon * dlon)

				dlat /= l
				dlon /= l 

				heading_vector_lat = dlat 
				heading_vector_lon = dlon 

			self.roadnetwork.annotation[nid]["heading_vector"] = [heading_vector_lat, heading_vector_lon]

		dim = len(target_shape)

		self.roadnetwork.targets = np.zeros((len(jsongraph["nodes"]), len(target_shape)), dtype=np.int32)
		self.roadnetwork.mask = np.ones((len(jsongraph["nodes"])), dtype=np.float32)

		for nid in range(len(jsongraph["nodes"])):
			for i in range(dim):
				self.roadnetwork.targets[nid,i] = jsongraph["nodelabels"][nid][i]

		self.roadnetwork.preload_img = None 

		self.roadnetwork.config = {}
		self.roadnetwork.config["folder"] = tileFolder

	def SampleSubRoadNetwork(self,graph_size = 256):
		return SubRoadNetwork(self.roadnetwork, graph_size = graph_size, search_mode = random.randint(0,3))


if __name__ == "__main__":
	config = json.load(open(sys.argv[1], "r"))

	target_shape = config["target_shape"]

	training_networks = []

	for folder in config["dataset_train"]:
		print("loading... ", folder)
		network = myRoadNetwork(folder + "/graph.json", folder, target_shape=target_shape)
		training_networks.append(network)


	preload_graph = None 
	preload_graph_num = config["subgraph_batch"] # just for testing 
	step = 0 
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

			items = model.Train(train_subgraph, train_op = model.train_op, learning_rate=0.0001)
			sloss += items[0]

			if step % 10 == 0:
				print(step, items[0])

			#console = code.InteractiveConsole(locals())
			#console.interact()
			# exit()

			step += 1

			if step % 100 == 0:
				print("average loss ", sloss/100.0, "at step", step)
				sloss = 0.0 

			if step % 1000 == 0 or (step<1000 and step % 100 == 0):
				print("save model to backup")
				model.saveModel(config["model_save_folder"] + "/backup")

				if step <= 1000:
					print("training 100 iterations spent", time()-t0, "seconds")
				else:
					print("training 1000 iterations spent", time()-t0, "seconds")
				t0 = time() 

























