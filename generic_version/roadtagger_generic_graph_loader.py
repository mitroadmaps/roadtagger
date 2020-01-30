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
class myRoadNetworkLoader():
	def __init__(self, jsonGraphFile, tileFolder, target_shape=[2]):
		self.roadnetwork = RoadNetwork()

		cfg = json.load(open(tileFolder+"/config.json","r"))
		self.roadnetwork.region = cfg["region"]

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

		self.jsongraph = jsongraph

	def annotation_filter_for_light_poles(self):
		hasdata = {}

		total = 0 
		remove = 0 

		jsongraph = self.jsongraph

		for nid in range(len(jsongraph["nodes"])):
			loc = jsongraph["nodes"][nid]

			k = (int(int(loc[0]*111111.0) / 100), int(int(loc[1]*111111.0) / 100))

			if jsongraph["nodelabels"][nid][0] > 0.5:
				hasdata[k] = True

		for nid in range(len(jsongraph["nodes"])):
			loc = jsongraph["nodes"][nid]

			k = (int(int(loc[0]*111111.0) / 100), int(int(loc[1]*111111.0) / 100))
			total += 1
			if k not in hasdata: 
				self.roadnetwork.annotation[nid]["remove"] = 1
				remove += 1 

		print("remove %d nodes from %d nodes" % (remove, total))



	def graphsize(self):
		return len(self.roadnetwork.annotation.keys())
	def SampleSubRoadNetwork(self,graph_size = 256, reseed=False):
		return SubRoadNetwork(self.roadnetwork, graph_size = graph_size, search_mode = random.randint(0,3), reseed=reseed)

