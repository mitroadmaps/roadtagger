import numpy as np

import random
import pickle 
import scipy.ndimage as nd 
import scipy 
import math
import svgwrite
from svgwrite.image import Image as svgimage
from PIL import Image
import json 
import tensorflow as tf 
import cv2
import scipy.sparse as sp 
from subprocess import Popen
import os

from rtree import index

import opengm 

from abc import ABC, abstractmethod 


class RoadNetwork():
	def __init__(self, target_dim = 1, loss_func = "L2"):
		self.edges = []
		self.node_degree = {}
		self.nodes = {}
		self.nid2loc = {}
		self.node_num = 0 
		self.region = None 
		self.image_file = None 
		self.images = None 
		self.target = None 
		self.target_mask = None 
		self.sat_image = None 
		self.loss_func = loss_func

		self.spares_graph_structure = None 
		self.tf_spares_graph_structure = None 

		pass 

	def AddNode(self, lat, lon, target):
		if (lat,lon) in self.nodes:
			return self.nodes[(lat,lon)][0]
		else:
			self.nodes[(lat,lon)] = (self.node_num, target) 
			self.nid2loc[self.node_num] = (lat,lon)
			self.node_num += 1  

			return self.nodes[(lat,lon)][0]

	def AddEdge(self, n1, n2):
		self.edges.append((n1,n2))


		if n1 in self.node_degree:
			if n2 not in self.node_degree[n1]:
				self.node_degree[n1].append(n2)
		else:
			self.node_degree[n1] = [n2]

		if n2 in self.node_degree:
			if n1 not in self.node_degree[n2]:
				self.node_degree[n2].append(n1)
		else:
			self.node_degree[n2] = [n1]


	def Dump(self):
		print("")
		print("#### Nodes (%d) ####" % len(self.nodes))
		for k,v in self.nodes.iteritems():
			print(k,v)


		print("")
		print("#### Edges (%d) ####" % len(self.edges))
		for edge in self.edges:
			print(edge[0], edge[1])

	def DumpToFile(self, filename):
		pickle.dump(self, open(filename,"wb"))


	def GetGraphStructure(self):
		if self.tf_spares_graph_structure is None:
			if self.spares_graph_structure is None:
				self.spares_graph_structure = {}
				self.spares_graph_structure['indices'] = []
				self.spares_graph_structure['values'] = []
				self.spares_graph_structure['shape'] = [len(self.nodes), len(self.nodes)]

				for edge in self.edges:
					self.spares_graph_structure['indices'].append([edge[0], edge[1]])
					self.spares_graph_structure['indices'].append([edge[1], edge[0]])

					self.spares_graph_structure['values'].append(1.0)
					self.spares_graph_structure['values'].append(1.0)


				for i in range(self.node_num):
					self.spares_graph_structure['indices'].append([i, i])
					self.spares_graph_structure['values'].append(1.0)



			self.tf_spares_graph_structure = tf.SparseTensorValue(self.spares_graph_structure['indices'], self.spares_graph_structure['values'], self.spares_graph_structure['shape']) 

		return self.tf_spares_graph_structure

	def GetImages(self):
		if self.images is None:
			#
			print("loading images...")
			self.images = np.zeros((len(self.nodes), image_size, image_size, 2), dtype = np.float32)

			img = scipy.ndimage.imread(self.image_file).astype(np.float32)/255.0
			img_road = scipy.ndimage.imread(self.road_image_file).astype(np.float32)/255.0

			print(np.shape(img))

			for k,v in self.nodes.iteritems():
				nid = v[0]

				lat = k[0] / 111111.0 
				lon = k[1] / 111111.0

				iy = int((lon - self.region[1]) * 111111.0 * 2.0 * math.cos(math.radians(self.region[0])))
				ix = int((self.region[2] - lat) * 111111.0 * 2.0)

				#print(ix,iy)

				self.images[nid,:,:,0] = img[ix-64:ix+64, iy-64:iy+64]
				self.images[nid,:,:,1] = img_road[ix-64:ix+64, iy-64:iy+64]


			print("done!")

		return self.images

	def GetTarget(self):
		if self.target is None:
			if self.loss_func == "L2":
				self.target = np.zeros((len(self.nodes),1), dtype = np.float32)

				for k,v in self.nodes.iteritems():
					nid = v[0]

					label = 0

					self.target[nid][0] = v[1]

			else:


				self.target = np.zeros((len(self.nodes)), dtype = np.int)

				for k,v in self.nodes.iteritems():
					nid = v[0]

					label = 0

					if v[1] == 1:
						label = 1 

					if v[1] == 2:
						label = 2 

					if v[1] == 3:
						label = 3 

					if v[1] == 4:
						label = 4 

					self.target[nid] = label


		return self.target 


	def GetTargetMask(self):
		if self.target_mask is None:
			self.target_mask = np.zeros((len(self.nodes), 1), dtype = np.float32)

			for k,v in self.nodes.iteritems():
				nid = v[0]

				if v[1] > 0 and len(self.node_degree[nid])<= 2:
					self.target_mask[nid][0] = 1.0


		return self.target_mask


	def Visualize(self, output = None, dump_file = "default.svg"):
		lat_min, lon_min, lat_max, lon_max = self.region[0], self.region[1], self.region[2], self.region[3]
		res = 0.5 


		sizex = int((lat_max-lat_min)*111111.0/res) 
		sizey = int((lon_max-lon_min)*111111.0*math.cos(math.radians(lat_min))/res )

		dwg = svgwrite.Drawing(dump_file, profile='tiny')
		dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='rgb(90,90,90)'))
		dwg.add(svgimage("file:///Users/songtaohe/Project/MapMaker/gcn_lane_detection/"+self.image_file, insert=(0,0)))

		for edge in self.edges:
			lat1 = self.nid2loc[edge[0]][0]/111111.0
			lon1 = self.nid2loc[edge[0]][1]/111111.0

			lat2 = self.nid2loc[edge[1]][0]/111111.0
			lon2 = self.nid2loc[edge[1]][1]/111111.0

			ilat2, ilon2 = Coord2Pixels(lat1, lon1, lat_min, lon_min, lat_max, lon_max, sizex, sizey)
			ilat, ilon = Coord2Pixels(lat2, lon2, lat_min, lon_min, lat_max, lon_max, sizex, sizey)

			dwg.add(dwg.line((ilon2, ilat2), (ilon, ilat),stroke='rgb(0,0,255)'))


		for k,v in self.nodes.iteritems():
			x,y = Coord2Pixels(k[0]/111111.0, k[1]/111111.0, lat_min, lon_min, lat_max, lon_max, sizex, sizey)

			nid = v[0]

			gt = self.target[nid]
			ov = 1.0 
			if output is None:
				pred = gt 
			else:

				if self.loss_func == "L2":
					pred = output[nid][0]
					ov = 1.0
				else:
					pred = np.argmax(output[nid])
					ov = output[nid][pred]


			if self.target_mask[nid][0]<0.5:
				color = "rgb(128,128,128)"
			else:
				error = np.abs(gt-pred)

				if error > 2.0:
					error = 2.0 

				error/= 2.0

				color = "rgb(%d,%d,0)" % (int(error*255), int((1-error)*255))


			dwg.add(dwg.circle(center = (y,x), r = 15, stroke=color, fill=color))
			if self.loss_func == "L2":
				dwg.add(dwg.text("%.2f(%d)" % (pred, int(gt)), insert = (y-15, x+3), fill = "rgb(255,255,0)",  )) #style = "font-size:6px; font-family:Arial"))
			else:
				dwg.add(dwg.text("%d %d(%.2f)" % (int(gt), pred, ov), insert = (y-15, x+3), fill = "rgb(255,255,0)",  )) #style = "font-size:6px; font-family:Arial"))



		dwg.save()


	def RandomSample(self, size = 512, name = "default"):
		lat_min, lon_min, lat_max, lon_max = self.region[0], self.region[1], self.region[2], self.region[3]
		res = 0.5 

		lat_size = size / 111111.0 
		lon_size = size / 111111.0 / math.cos(math.radians(lat_min))


		lat_start = random.uniform(lat_min+10.0/111111.0, lat_max - lat_size-10.0/111111.0)
		lon_start = random.uniform(lon_min+10.0/111111.0, lon_max - lon_size-10.0/111111.0)


		newRoadNetwork = RoadNetwork() 
		new_region = [lat_start, lon_start, lat_start+lat_size, lon_start+lon_size]

		counter = 0 

		for edge in self.edges:
			nid1 = edge[0]
			nid2 = edge[1]

			loc1 = self.nid2loc[nid1]
			loc2 = self.nid2loc[nid2]



			def inrange(lat,lon, region):
				lat_mergin = 70*0.5/111111.0
				lon_mergin = 70*0.5/111111.0 / math.cos(math.radians(region[0]))

				if lat-region[0] > lat_mergin and region[2] - lat > lat_mergin and lon-region[1] > lon_mergin and region[3] - lon > lon_mergin:
					return True
				else:
					return False


			if inrange(loc1[0]/111111.0, loc1[1]/111111.0, new_region) and inrange(loc2[0]/111111.0, loc2[1]/111111.0, new_region):
				newRoadNetwork.AddEdge(newRoadNetwork.AddNode(loc1[0], loc1[1], self.nodes[loc1][1]), newRoadNetwork.AddNode(loc2[0], loc2[1], self.nodes[loc2][1]))



		if len(newRoadNetwork.nodes) < 20:
			self.RandomSample(size,name)
			return 


		# load images 
		gps_img = scipy.ndimage.imread(self.image_file)
		road_img = scipy.ndimage.imread(self.road_image_file)

		# crop it 
		x = int((lat_max-lat_start-lat_size)/(lat_max-lat_min) * np.shape(gps_img)[0])
		y = int((lon_start - lon_min)/(lon_max-lon_min) * np.shape(gps_img)[1])


		# XXX
		Image.fromarray(gps_img[x:x+size*2, y:y+size*2]).save(name+".png")
		Image.fromarray(road_img[x:x+size*2, y:y+size*2]).save(name+"_road.png")



		newRoadNetwork.region = new_region
		newRoadNetwork.image_file = name+".png"
		newRoadNetwork.road_image_file = name+"_road.png"

		newRoadNetwork.loss_func = self.loss_func
		newRoadNetwork.DumpToFile(name+".p")
		newRoadNetwork.GetImages()
		newRoadNetwork.GetTarget()
		newRoadNetwork.GetTargetMask()
		newRoadNetwork.Visualize(dump_file = name+".svg")



	def generate_road_segment(self):
		visited = []

		self.seg_id = 0 
		self.segments = {}
		self.node_to_segment = {}


		for nid in self.nid2loc.keys():
			if nid in visited:
				continue 

			if len(self.node_degree[nid]) == 1:
				seg = [nid]
				local_queue = [self.node_degree[nid][0]]
				local_visited = [nid]
				visited.append(nid)

				while len(local_queue) != 0:
					head = local_queue.pop(0)

					if head in local_visited:
						continue 

					if len(self.node_degree[head]) != 2:
						seg.append(head)
						# don't add to visited
						break 

					local_visited.append(head)
					visited.append(head)
					seg.append(head)
					for next_node in self.node_degree[head]:
						local_queue.append(next_node)


				self.segments[self.seg_id] = list(seg)

				for node in seg :
					self.node_to_segment[node] = self.seg_id 

				self.seg_id += 1 


			if len(self.node_degree[nid]) > 2 :
				visited.append(nid)

				for next_node in self.node_degree[nid]:
					seg = [nid]
					local_queue = [next_node]

					while len(local_queue) != 0:
						head = local_queue.pop(0)

						if head in visited:
							continue 

						if len(self.node_degree[head]) != 2:
							seg.append(head)
							# don't add to visited
							break 

						visited.append(head)
						seg.append(head)
						for next_nid in self.node_degree[head]:
							local_queue.append(next_nid)

					if len(seg) != 1 :
						self.segments[self.seg_id] = list(seg)
						for node in seg :
							self.node_to_segment[node] = self.seg_id 

						self.seg_id += 1 


	def loadAnnotation(self):
		# define 
		# self.target_shape
		# self.targets
		# self.mask
		# self.preload_img
		# self.annotation = annotation?
		# self.config 

		# call self.generate_road_segment()
		


	def dumpConsistency(self, root_folder = ""):
		Popen("mkdir -p %s" % (root_folder + self.output_folder), shell=True).wait()
		pickle.dump(self.consistency, open(root_folder+self.output_folder+"/consistency.p", "w"))




def get_image_coordinate(lat, lon, size, region):
	x = int((region[2]-lat)/(region[2]-region[0])*size)
	y = int((lon-region[1])/(region[3]-region[1])*size)

	return x,y 

def directionScore(roadNetwork, n1,n2,n3):
		v1_lat = roadNetwork.nid2loc[n2][0] - roadNetwork.nid2loc[n1][0]
		v1_lon = (roadNetwork.nid2loc[n2][1] - roadNetwork.nid2loc[n1][1]) * math.cos(math.radians(roadNetwork.nid2loc[n2][0]/111111.0))

		v2_lat = roadNetwork.nid2loc[n3][0] - roadNetwork.nid2loc[n2][0]
		v2_lon = (roadNetwork.nid2loc[n3][1] - roadNetwork.nid2loc[n2][1]) * math.cos(math.radians(roadNetwork.nid2loc[n2][0]/111111.0))


		v1_l = np.sqrt(v1_lat * v1_lat + v1_lon * v1_lon)

		if v1_l == 0:
			print(n1,n2,n3)



		v1_lat /= v1_l 
		v1_lon /= v1_l 


		v2_l = np.sqrt(v2_lat * v2_lat + v2_lon * v2_lon)
		v2_lat /= v2_l 
		v2_lon /= v2_l 


		return v1_lat * v2_lat + v1_lon * v2_lon


class SubRoadNetwork():
	def __init__(self, parentRoadNetowrk, train_cnn_only = False, train_cnn_batch = 64,train_cnn_preload = 256, graph_size = 256, augmentation = True, search_mode = 0, partial = False, remove_adjacent_matrix = 0, output_folder = None, reseed = False, no_image = False, tiles_name = "tiles", image_size = 384, cnn_embedding_dim = 64 ):
		self.parentRoadNetowrk = parentRoadNetowrk
		self.annotation = parentRoadNetowrk.annotation
		self.config = parentRoadNetowrk.config 
		

		self.train_cnn_only = train_cnn_only
		self.train_cnn_batch = train_cnn_batch

		label_list = []
		for k in self.parentRoadNetowrk.nodes.keys():
			label_list.append(k) 

		seed_node = random.choice(label_list)


		# if partial == True:
		# 	for k,v in self.annotation.iteritems():
		# 		if 'labelled' in v and v['labelled'] == 1 and v['remove'] != 1:
		# 			label_list.append(k)



		if train_cnn_only == False:

			
			# bfs !
			visited = []
			newGraphNodes = []
			#graph_size

			queue = [seed_node]
			resample = 0 
			while True:
				while  len(queue) != 0:
					if search_mode == 0:
						current_node = queue.pop(0) # bfs
					elif search_mode == 1:
						current_node = queue.pop() # dfs
					else:
						current_node = queue.pop(random.randint(0,len(queue)-1))

					if current_node in visited:
						continue
					else:
						visited.append(current_node)

						if len(newGraphNodes) < graph_size:
							if partial == True:
								if current_node in label_list and self.annotation[current_node]['remove'] == 0:
									newGraphNodes.append(current_node)
							else:
								if self.annotation[current_node]['remove'] == 0:
									newGraphNodes.append(current_node)

						else:
							break

						for next_node in self.parentRoadNetowrk.node_degree[current_node]:
							queue.append(next_node)

				if reseed == False:
					break 

				if len(newGraphNodes) != len(label_list) and len(newGraphNodes) < graph_size:
					unsampled_node = []
					for x in label_list:
						if x not in newGraphNodes:
							unsampled_node.append(x)


					if len(unsampled_node)!=0:
						queue = [random.choice(unsampled_node)]

					else:
						break 
				else:
					break 

				resample += 1

				if resample > 5000:
					break 
		else:
			# just random sample nodes with labels 
			newGraphNodes = []

			for i in xrange(train_cnn_preload):
				newGraphNodes.append(random.choice(label_list))
				

		self.node_mapping = {}

		self.subGraphNoadList = [0] * len(newGraphNodes)

		st_ptr = 0
		ed_ptr = len(newGraphNodes)-1


		for nid in newGraphNodes:
			hv = self.annotation[nid]['heading_vector']

			if hv[0]*hv[0] + hv[1]*hv[1] < 0.1:
				self.subGraphNoadList[ed_ptr] = nid 
				self.node_mapping[nid] = ed_ptr
				ed_ptr -= 1 

			else:
				self.subGraphNoadList[st_ptr] = nid 
				self.node_mapping[nid] = st_ptr
				st_ptr += 1 

		self.nonIntersectionNodeNum = st_ptr 

		print(graph_size, search_mode, st_ptr)


		self.virtual_intersection_node_ptr = len(self.subGraphNoadList)

		self.generate_decomposited_graph()
		self.generate_auxiliary_graph()
		self.generate_fully_connected_graph()


		#todo load graph   D-1/2 . A . D-1/2
		adj = np.zeros((len(self.subGraphNoadList), len(self.subGraphNoadList)))

		# # road network
		# for edge in self.parentRoadNetowrk.edges:
		# 	if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
		# 		e0 = self.node_mapping[edge[0]]
		# 		e1 = self.node_mapping[edge[1]]


		# 		adj[e0,e1] = 1.0 
		# 		adj[e1,e0] = 1.0 


		# road link 
		for start_node in self.subGraphNoadList:
			if self.annotation[start_node]['degree'] > 2 :
				continue
			else:
				for next_node in self.parentRoadNetowrk.node_degree[start_node]:
					if next_node not in self.subGraphNoadList:
						continue
					if self.annotation[next_node]['degree'] > 2:

						best_next_node = next_node 
						best_d = -10 

						for next_next_node in self.parentRoadNetowrk.node_degree[next_node]:
							if next_next_node not in self.subGraphNoadList:
								continue
							if next_next_node == start_node:
								continue


							score = directionScore(self.parentRoadNetowrk, start_node, next_node, next_next_node)

							if score > best_d:
								best_d = score 
								best_next_node = next_next_node

						
						if best_d > 0.2:
							e0 = self.node_mapping[start_node]
							e1 = self.node_mapping[best_next_node]

							adj[e0,e1] = 1.0 
							adj[e1,e0] = 1.0 


					else:
						e0 = self.node_mapping[start_node]
						e1 = self.node_mapping[next_node]

						adj[e0,e1] = 1.0 
						adj[e1,e0] = 1.0 

		adj = sp.coo_matrix(adj)
		rowsum = np.array(adj.sum(1))

		self.homogeneous_loss_mask = np.zeros(len(self.subGraphNoadList))


		cc = len(np.where(rowsum<=2)[0]) + 1.0 

		factor = len(self.subGraphNoadList)/cc


		self.homogeneous_loss_mask[np.where(rowsum<=2)[0]] = factor


		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.

		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

		new_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 


		self.spares_graph_structure = {}



		self.remove_adjacent_matrix = remove_adjacent_matrix

		if remove_adjacent_matrix == 1:
			self.spares_graph_structure['indices'] = [(x,x) for x in xrange(len(self.subGraphNoadList))]
			self.spares_graph_structure['values'] = [1.0 for x in xrange(len(self.subGraphNoadList))]
			self.spares_graph_structure['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]


		else:

			self.spares_graph_structure['indices'] = np.vstack((new_adj.row, new_adj.col)).transpose()
			self.spares_graph_structure['values'] = new_adj.data 
			self.spares_graph_structure['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]


		# for edge in self.parentRoadNetowrk.edges:
		# 	if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
		# 		e0 = self.node_mapping[edge[0]]
		# 		e1 = self.node_mapping[edge[1]]

		# 		self.spares_graph_structure['indices'].append([e0, e1])
		# 		self.spares_graph_structure['indices'].append([e1, e0])

		# 		self.spares_graph_structure['values'].append(1.0)
		# 		self.spares_graph_structure['values'].append(1.0)


		self.tf_spares_graph_structure = tf.SparseTensorValue(self.spares_graph_structure['indices'], self.spares_graph_structure['values'], self.spares_graph_structure['shape']) 
		#load images

		self.images = np.zeros((self.nonIntersectionNodeNum, image_size, image_size, 3), dtype = np.float32)

		if output_folder is None:
			output_folder = self.config["folder"]
		else:
			output_folder = output_folder + self.config["folder"]
		
		if no_image == False:
			c = 0 

			has_titles = os.path.isdir(output_folder + "/"+tiles_name) or os.path.isdir("/data/songtao/DeepRoadMateinfo/"+ self.config["folder"] + "/"+tiles_name)

			if has_titles == False:
				large_image = scipy.ndimage.imread(output_folder+"/sat_16384.png").astype(np.uint8)


			for nid in self.subGraphNoadList[:self.nonIntersectionNodeNum]:
				if has_titles:
					if self.parentRoadNetowrk.preload_img is not None:
						img = self.parentRoadNetowrk.preload_img[nid]
					else:
						print("Error! No preloaded images")
						exit()
						try:
							img = scipy.ndimage.imread(output_folder + "/"+tiles_name+"/img_%d.png" % nid).astype(np.float32)/255.0 
						except:

							img = scipy.ndimage.imread("/data/songtao/DeepRoadMateinfo/"+ self.config["folder"] + "/"+tiles_name+"/img_%d.png" % nid).astype(np.float32)/255.0 
						
							print(nid)
							print(output_folder + "/"+tiles_name+"/img_%d.png" % nid)
				# else:
				# 	v = self.annotation[nid]
				# 	img_size = 16384
				# 	region = self.config["region"]

				# 	loc = self.parentRoadNetowrk.nid2loc[k]
				# 	x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, img_size,region)

				# 	heading_vector_lat, heading_vector_lon = v['heading_vector']
				# 	subimage = large_image[x-272:x+272, y-272:y+272]

				# 	if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
				# 		angle = 0.0
				# 	else:
				# 		angle = math.degrees(math.atan2(heading_vector_lon, heading_vector_lat))
					
				# 	#print(k, angle)

				# 	img_rot = scipy.ndimage.interpolation.rotate(subimage, angle)


				# 	center = np.shape(img_rot)[0]/2

				# 	r = 192 # 384
				# 	img = img_rot[center-r:center+r, center-r: center+ r,:]

				# 	img = img.astype(np.float32)/255.0 





				self.images[c,:,:,:] = self.image_augmentation(img, flag=augmentation) 
				c = c + 1 







		#load targets and masks

		self.targets = np.zeros((self.nonIntersectionNodeNum, 6), dtype = np.int32)
		self.mask = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)

		
		c = 0 
		for nid in self.subGraphNoadList[:self.nonIntersectionNodeNum]:
			self.targets[c,:] = self.parentRoadNetowrk.targets[nid,:]
			self.mask[c] = self.parentRoadNetowrk.mask[nid]


			c = c + 1
	
		#print(np.amin(self.targets[:,0]), np.amax(self.targets[:,0]))
		#load heading vector 
		c = 0 
		self.heading_vector = np.zeros((self.nonIntersectionNodeNum, 2), dtype = np.float32)
		for nid in self.subGraphNoadList[:self.nonIntersectionNodeNum]:
			self.heading_vector[c,0] = self.annotation[nid]['heading_vector'][0]
			self.heading_vector[c,1] = self.annotation[nid]['heading_vector'][1]
			c = c + 1


		#intersection features 

		self.intersectionFeatures = np.zeros((len(newGraphNodes)-self.nonIntersectionNodeNum, cnn_embedding_dim), dtype = np.float32)


	# todo, if node not in this graph, remove the target! 
	def generate_decomposited_graph(self, use_virtual_intersection = False):

		# tmp
		adj = np.zeros((len(self.subGraphNoadList),len(self.subGraphNoadList)))
		
		adj_dir1 = np.zeros((len(self.subGraphNoadList),len(self.subGraphNoadList)))
		adj_dir2 = np.zeros((len(self.subGraphNoadList),len(self.subGraphNoadList)))


		new_neighbor = {}

		for start_node in self.subGraphNoadList:
			if self.annotation[start_node]['degree'] > 2 :
				continue
			else:
				for next_node in self.parentRoadNetowrk.node_degree[start_node]:
					if next_node not in self.subGraphNoadList:
						continue
					if self.annotation[next_node]['degree'] > 2:

						best_next_node = next_node 
						best_d = -10 

						for next_next_node in self.parentRoadNetowrk.node_degree[next_node]:
							if next_next_node not in self.subGraphNoadList:
								continue
							if next_next_node == start_node:
								continue

							score = directionScore(self.parentRoadNetowrk, start_node, next_node, next_next_node)

							if score > best_d:
								best_d = score 
								best_next_node = next_next_node

						
						if best_d > 0.2:
							e0 = self.node_mapping[start_node]
							e1 = self.node_mapping[best_next_node]

							adj[e0,e1] = 1.0 
							adj[e1,e0] = 1.0 

							if e0 not in new_neighbor:
								new_neighbor[e0] = [e1]
							else:
								if e1 not in new_neighbor[e0]:
									new_neighbor[e0].append(e1)

							if e1 not in new_neighbor:
								new_neighbor[e1] = [e0]
							else:
								if e0 not in new_neighbor[e1]:
									new_neighbor[e1].append(e0)


					else:
						e0 = self.node_mapping[start_node]
						e1 = self.node_mapping[next_node]

						adj[e0,e1] = 1.0 
						adj[e1,e0] = 1.0		

						if e0 not in new_neighbor:
							new_neighbor[e0] = [e1]
						else:
							if e1 not in new_neighbor[e0]:
								new_neighbor[e0].append(e1)

						if e1 not in new_neighbor:
							new_neighbor[e1] = [e0]
						else:
							if e0 not in new_neighbor[e1]:
								new_neighbor[e1].append(e0)

		visited = []

		#print(new_neighbor)

		for node_id in xrange(len(self.subGraphNoadList)):
			if node_id not in new_neighbor:
				continue

			if node_id in visited:
				continue 


			# define base direction 

			neighbor1 = new_neighbor[node_id][0] 

			neighbor2 = None 

			if len(new_neighbor[node_id]) > 1:
				best_d = -10
				best_neighbor = None
				for next_node in new_neighbor[node_id][1:]:
					d_score = directionScore(self.parentRoadNetowrk, self.subGraphNoadList[neighbor1], self.subGraphNoadList[node_id], self.subGraphNoadList[next_node])

					if d_score > best_d:
						best_d = d_score
						best_neighbor = next_node 


				if best_d > 0.2 :
					neighbor2 = best_neighbor

			# now we have the direction 1, which is 
			# neighbor 2  --> node_id --> neighbor 1 
			# we will explore them one by on 


			nd1 = node_id 
			nd2 = neighbor1

			visited.append(nd1)

			while True:
				adj_dir1[nd1, nd2] = 1.0 
				adj_dir2[nd2, nd1] = 1.0

				if nd2 in visited:
					break

				visited.append(nd2)




				best_d = -10
				best_neighbor = None
				for next_node in new_neighbor[nd2]:
					if next_node == nd1 :
						continue
					d_score = directionScore(self.parentRoadNetowrk, self.subGraphNoadList[nd1], self.subGraphNoadList[nd2], self.subGraphNoadList[next_node])

					if d_score > best_d:
						best_d = d_score
						best_neighbor = next_node 

				if best_d > 0.2:
					nd1, nd2 = nd2, best_neighbor
				else:
					break



			if neighbor2!= None:
				nd1 = node_id 
				nd2 = neighbor2

				while True:
					adj_dir1[nd2, nd1] = 1.0 
					adj_dir2[nd1, nd2] = 1.0

					if nd2 in visited:
						break

					visited.append(nd2)

					best_d = -10
					best_neighbor = None
					for next_node in new_neighbor[nd2]:
						if next_node == nd1 :
							continue
						d_score = directionScore(self.parentRoadNetowrk, self.subGraphNoadList[nd1], self.subGraphNoadList[nd2], self.subGraphNoadList[next_node])

						if d_score > best_d:
							best_d = d_score
							best_neighbor = next_node 

					if best_d > 0.2:
						nd1, nd2 = nd2, best_neighbor
					else:
						break



		# adj to sparse matrix 

		adj = sp.coo_matrix(adj_dir1)
		rowsum = np.array(adj.sum(1))
		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.

		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

		new_adj_dir1 = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

		adj = sp.coo_matrix(adj_dir2)
		rowsum = np.array(adj.sum(1))
		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.

		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

		new_adj_dir2 = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 



		self.spares_graph_structure_direction1 = {}
		self.spares_graph_structure_direction2 = {}

		self.spares_graph_structure_direction1['indices'] = np.vstack((new_adj_dir1.row, new_adj_dir1.col)).transpose()
		self.spares_graph_structure_direction1['values'] = new_adj_dir1.data 
		self.spares_graph_structure_direction1['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]

		self.spares_graph_structure_direction2['indices'] = np.vstack((new_adj_dir2.row, new_adj_dir2.col)).transpose()
		self.spares_graph_structure_direction2['values'] = new_adj_dir2.data 
		self.spares_graph_structure_direction2['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]


		#print(self.spares_graph_structure_direction1, self.spares_graph_structure_direction2)

		self.tf_spares_graph_structure_direction1 = tf.SparseTensorValue(self.spares_graph_structure_direction1['indices'], self.spares_graph_structure_direction1['values'], self.spares_graph_structure_direction1['shape']) 
		self.tf_spares_graph_structure_direction2 = tf.SparseTensorValue(self.spares_graph_structure_direction2['indices'], self.spares_graph_structure_direction2['values'], self.spares_graph_structure_direction2['shape']) 
		
		#load images


	def generate_auxiliary_graph(self):
		
		# parallel roads

		idx = index.Index()

		adj = np.zeros((len(self.subGraphNoadList), len(self.subGraphNoadList)))

		candidates = []

		for nid in self.subGraphNoadList:
			if len(self.parentRoadNetowrk.node_degree[nid]) != 2:
				continue

			d_score = directionScore(self.parentRoadNetowrk, self.parentRoadNetowrk.node_degree[nid][0], nid, self.parentRoadNetowrk.node_degree[nid][1])

			if d_score > 0.9 : # straight
				lat = self.parentRoadNetowrk.nid2loc[nid][0]
				lon = self.parentRoadNetowrk.nid2loc[nid][1]

				idx.insert(nid, (lat-1,lon-1, lat+1, lon+1))

				candidates.append(nid)


		print("number of parallel roads candidates", len(candidates))


		for nid in candidates:
			lat = self.parentRoadNetowrk.nid2loc[nid][0]
			lon = self.parentRoadNetowrk.nid2loc[nid][1]

			neighbors =  list(idx.intersection((lat-20,lon-30,lat+20,lon+30)))

			#print(nid, neighbors)

			best_id = None 
			best_distance = 10000 # 30 meter 

			for nnid in neighbors:
				if nnid in self.parentRoadNetowrk.node_degree[nid] or nnid == nid:
					continue

				d_score1 = directionScore(self.parentRoadNetowrk, nnid, nid, self.parentRoadNetowrk.node_degree[nid][0])
				d_score2 = directionScore(self.parentRoadNetowrk, nnid, nid, self.parentRoadNetowrk.node_degree[nid][1])
				d_score3 = directionScore(self.parentRoadNetowrk, nid, nnid, self.parentRoadNetowrk.node_degree[nnid][0])
				d_score4 = directionScore(self.parentRoadNetowrk, nid, nnid, self.parentRoadNetowrk.node_degree[nnid][1])


				#print(d_score1, d_score2, lat,lon)

				if abs(d_score1) < 0.3 and abs(d_score2) < 0.3 and abs(d_score3) < 0.3 and abs(d_score4) < 0.3:
					lat2 = self.parentRoadNetowrk.nid2loc[nnid][0]
					lon2 = self.parentRoadNetowrk.nid2loc[nnid][1]

					a = lat2 - lat
					b = (lon2 - lon) * math.cos(math.radians(lat2/111111.0)) 


					d = a*a + b*b

					if d < best_distance:
						best_id = nnid 
						best_distance = d 


			if best_id is not None:
				n0 = self.node_mapping[best_id]
				n1 = self.node_mapping[nid]

				adj[n0,n1] = 1.0 
				adj[n1,n0] = 1.0 


		adj = sp.coo_matrix(adj)
		rowsum = np.array(adj.sum(1))
		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.

		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

		new_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 
		self.spares_graph_structure_auxiliary = {}

		self.spares_graph_structure_auxiliary['indices'] = np.vstack((new_adj.row, new_adj.col)).transpose()
		self.spares_graph_structure_auxiliary['values'] = new_adj.data 
		self.spares_graph_structure_auxiliary['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]

		self.tf_spares_graph_structure_auxiliary = tf.SparseTensorValue(self.spares_graph_structure_auxiliary['indices'], self.spares_graph_structure_auxiliary['values'], self.spares_graph_structure_auxiliary['shape']) 
		


		pass

	def generate_fully_connected_graph(self):
		adj = np.zeros((len(self.subGraphNoadList), len(self.subGraphNoadList)))

		# road network
		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				e0 = self.node_mapping[edge[0]]
				e1 = self.node_mapping[edge[1]]


				adj[e0,e1] = 1.0 
				adj[e1,e0] = 1.0 


		adj = sp.coo_matrix(adj)
		rowsum = np.array(adj.sum(1))
		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.

		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

		new_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 
		self.spares_graph_structure_fully_connected = {}

		self.spares_graph_structure_fully_connected['indices'] = np.vstack((new_adj.row, new_adj.col)).transpose()
		self.spares_graph_structure_fully_connected['values'] = new_adj.data 
		self.spares_graph_structure_fully_connected['shape'] = [len(self.subGraphNoadList), len(self.subGraphNoadList)]

		self.tf_spares_graph_structure_fully_connected = tf.SparseTensorValue(self.spares_graph_structure_fully_connected['indices'], self.spares_graph_structure_fully_connected['values'], self.spares_graph_structure_fully_connected['shape']) 
		

 		pass


	def GetGraphStructure(self):
		return self.tf_spares_graph_structure

	def RandomBatchST(self):
		self.st = random.randint(0, self.nonIntersectionNodeNum-self.train_cnn_batch-1)


	def GetImages(self, batch_size = None):
		if batch_size is None :
			return self.images 
		else:
			return self.images[self.st:self.st+batch_size]


	def GetTarget(self, batch_size = None):
		if batch_size is None:
			return self.targets 
		else:
			return self.targets[self.st:self.st+batch_size]


	def GetTargetMask(self, batch_size = None):
		if batch_size is None:
			return self.mask
		else:
			return self.mask[self.st:self.st+batch_size]


	def GetHeadingVector(self, use_random = True):

		# randomly rotate it 
		heading_vector_ = np.zeros((self.nonIntersectionNodeNum, 2), dtype = np.float32)


		if use_random == False:
			return self.heading_vector

		angle = random.random()*3.1415926*2 

		heading_vector_[:,0] = math.cos(angle) * self.heading_vector[:,0] - math.sin(angle) * self.heading_vector[:,1]
		heading_vector_[:,1] = math.sin(angle) * self.heading_vector[:,0] + math.cos(angle) * self.heading_vector[:,1]
		

		return heading_vector_

	def GetIntersectionFeatures(self):
		return self.intersectionFeatures 



	def GetNodeDropoutMask(self, use_random, batch_size = None, rate = 0.1, stop_gradient = False):
		if batch_size is None:
			size = self.nonIntersectionNodeNum
		else:
			size = batch_size

		if stop_gradient == True:
			ret = np.ones((size, 62))
			gradient_mask = np.ones((size, 62))

			return ret, gradient_mask

		ret = np.ones((size, 62))
		gradient_mask = np.zeros((size, 62))



		if use_random:
			for i in xrange(int(size*rate)):
				c = random.randint(0,size-1)
				ret[c,:] = np.random.uniform(-1.0, 1.0, size=62)
				gradient_mask[c,:] = 1.0

		return ret, gradient_mask

	def GetGlobalLossMask(self, batch_size = None):
		ret = np.ones((self.nonIntersectionNodeNum))
		if batch_size is None:
			return ret 
		else:
			batch_size = min(batch_size, self.nonIntersectionNodeNum)
			c = random.randint(0, self.nonIntersectionNodeNum-batch_size)

			ret[:] = 0.0

			factor = 1.0 / float(batch_size) * self.nonIntersectionNodeNum 
			if factor < 0.5:
				factor = 0.5 
			if factor > 2.0:
				factor = 2.0 

			ret[c:c+batch_size] = factor
			
			return ret 

	def GetHomogeneousLossMask(self):  # only for GNN 
		return self.homogeneous_loss_mask


	def image_augmentation(self, img, flag = True ):

		img = np.copy(img)

		if flag == False:
			return img 
		else:
			k = random.randint(0,7)

			if k % 2 == 1:
				img = img * (random.random()*0.5 + 0.5)

			if k % 4 > 2:
				img = np.clip(img + np.random.normal(0,0.05,np.shape(img)),0.0,1.0)

			if k > 4:	

				dim = np.shape(img)

				block_x = random.randint(0, dim[0]-65)
				block_y = random.randint(0, dim[1]-65)

				img[block_x:block_x+64, block_y:block_y+64,:] = 0.5

			return img 


	def Visualize(self, output = "default.png"):
		img = np.copy(self.parentRoadNetowrk.sat_image)

		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				loc0 = self.parentRoadNetowrk.nid2loc[edge[0]]
				loc1 = self.parentRoadNetowrk.nid2loc[edge[1]]

				d = np.shape(self.parentRoadNetowrk.sat_image)[0]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, d,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, d,self.parentRoadNetowrk.region)


				cv2.line(img, (y0,x0), (y1,x1), (0,0,255),2)

				cv2.circle(img, (y0,x0), 3, (0,255,0), -1)
				cv2.circle(img, (y1,x1), 3, (0,255,0), -1)

	
		Image.fromarray(img).save(output)

	
	






