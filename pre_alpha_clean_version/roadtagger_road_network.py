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



	def loadAnnotation(self, filename, osm_auto = False, preload_img = False, lane_numbers = [0]*6, root_folder="", force_osm = False ):
		self.config = json.load(open(filename,"r"))
		output_folder = self.config["folder"]
		self.output_folder = output_folder

		if force_osm:
			print("load annotation_osm.p")
			self.annotation = pickle.load(open(root_folder+"/"+output_folder+"/annotation_osm.p", "r"))
		else:

			for a_filename in  ["/annotation_fixed.p", "/annotation.p", "/annotation_osm.p"]:

				try:
					self.annotation = pickle.load(open(root_folder+"/"+output_folder+a_filename, "r"))
					print(a_filename)
					break
				except:
					if a_filename == "/annotation_osm.p":
						print("no annotation file")
						exit()

					continue
					#print("load annotation_osm.p")
					#self.annotation = pickle.load(open(root_folder+"/"+output_folder+"/annotation_osm.p", "r"))



		# todo target and mask 
		self.targets = np.zeros((len(self.nodes.keys()),6), dtype = np.int32)
		self.mask = np.zeros((len(self.nodes.keys())), dtype = np.float32)


		self.consistency = np.ones((len(self.nodes.keys()),6), dtype = np.float32) 


		lane_total_counter = 0 
		lane_has_labbel_counter = 0 


		self.node_with_lane = []

		for k,v in self.annotation.iteritems():
			vector = v['heading_vector']

			if vector[0]*vector[0] + vector[1]*vector[1] < 0.1:
				continue

			if v['remove'] == 1:
				continue 


			lane_total_counter += 1

			# if v['confidence'] == 1:
			# 	self.mask[k] = 1.0 
			# else:
			# 	self.mask[k] = 0.5 # 

			if v['roadtype'] == 1:
				self.mask[k] = 1.0 
			else:
				self.mask[k] = 0.0 # for residential roads 


			if osm_auto == True:
				#print(v['number_of_lane'])
				self.targets[k,0] = v['number_of_lane'] - 1
			else:
				if v['number_of_lane'] > 0:
					self.targets[k,0] = v['number_of_lane'] - 1 
				else:
					self.targets[k,0] = int(v['transition']-0.5)


			if self.targets[k,0] >= 0 :
				nl = self.targets[k,0]

				if nl > 5:
					nl = 5 

				lane_numbers[nl] += 1


			if v['roadtype'] == 1 and self.targets[k,0] >= 0:
				lane_has_labbel_counter += 1
				self.node_with_lane.append(k)


			self.targets[k,1] = v['left_park']
			self.targets[k,2] = v['left_bike']
			self.targets[k,3] = v['right_bike']
			self.targets[k,4] = v['right_park']
			self.targets[k,5] = v['roadtype']

		self.preload_img = None 
		if preload_img == True:
			self.preload_img = {}
			for k in self.annotation.keys():
				img = scipy.ndimage.imread(output_folder + "/tiles/img_%d.png" % k).astype(np.float32)/255.0 
				self.preload_img[k] = img


				if k % 100 == 0:
					print(k)





		self.generate_road_segment()


		self.node_with_lane_change = []

		for edge in self.edges:
			n1 = edge[0]
			n2 = edge[1]

			if len(self.node_degree[n1]) != 2 or len(self.node_degree[n2]) != 2:
				continue
			
			if self.mask[n1] == 0 or self.mask[n2] == 0:
				continue

			if self.targets[n1,0] < 0 or self.targets[n2,0] < 0 :
				continue


			if self.targets[n1,0] != self.targets[n2,0]:
				if n1 not in self.node_with_lane_change :
					self.node_with_lane_change.append(n1)


		print("node with lane change:", len(self.node_with_lane_change), self.node_with_lane_change)





		return lane_numbers, [lane_total_counter, lane_has_labbel_counter]

	def dumpConsistency(self, root_folder = ""):
		Popen("mkdir -p %s" % (root_folder + self.output_folder), shell=True).wait()
		pickle.dump(self.consistency, open(root_folder+self.output_folder+"/consistency.p", "w"))



def GenerateMicroBenchmark():
	# 4096 * 4096 8 roads
	base_lat = 40
	base_lon = -70


	region = [base_lat, base_lon, base_lat + 4096*0.125 * 1.0/111111.0, base_lon + 4096*0.125 * 1.0/111111.0 / math.cos(math.radians(base_lat))]


	roadnetwork = RoadNetwork()

	roadnetwork.region = region 


	for i in xrange(8):
		for j in xrange(21):
			lat1 = base_lat + (32 + 64*i) * 1.0/111111.0
			lon1 = base_lon + (32 + 20*j) * 1.0/111111.0 / math.cos(math.radians(base_lat))

			lat2 = base_lat + (32 + 64*i) * 1.0/111111.0
			lon2 = base_lon + (32 + 20*(j+1)) * 1.0/111111.0 / math.cos(math.radians(base_lat))

			n1 = roadnetwork.AddNode(int(lat1*111111.0), int(lon1*111111.0),[i,j])
			n2 = roadnetwork.AddNode(int(lat2*111111.0), int(lon2*111111.0),[i,j+1])

			roadnetwork.AddEdge(n1,n2)



	roadnetwork.targets = np.zeros((len(roadnetwork.nodes.keys()),6), dtype = np.int32)
	roadnetwork.mask = np.zeros((len(roadnetwork.nodes.keys())), dtype = np.int32)
	roadnetwork.targets[:,:] = 3

	roadnetwork.targets[159:167,:] = 2	

	roadnetwork.node_with_lane = [k for k in roadnetwork.nodes.keys()]

	roadnetwork.annotation = {}

	roadnetwork.config = {}
	roadnetwork.config['folder'] = "123"

	for k, va in roadnetwork.nodes.iteritems():
		v = {}

		v['labelled'] = 1
		v['remove'] = 0 

		hv = [0,1]

		v['heading_vector'] = hv 

		nid = va[0]

		v['degree'] = len(roadnetwork.node_degree[nid])

		roadnetwork.annotation[nid] = v


	return roadnetwork

def SubRoadNetworkAddImageForMicrobenchmark(subnet, img):

	for k,v in subnet.parentRoadNetowrk.nodes.iteritems():
		nid = v[0]
		loc_image = v[1]

		x = loc_image[1] * 160 + 256
		y = 4096 - (loc_image[0] * 512 + 256)

		ind = subnet.node_mapping[nid]
		
		subnet.images[ind, :,:,:] = np.rot90(img[y-192:y+192, x-192:x+192,0:3].astype(np.float32)/255.0,axes=(-3,-2))


		if loc_image[0] == 7:
			Image.fromarray(np.rot90(img[y-192:y+192, x-192:x+192,0:3],axes=(-3,-2))).save("debug%d.png" % loc_image[1])







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
	def __init__(self, parentRoadNetowrk, train_cnn_only = False, train_cnn_batch = 64,train_cnn_preload = 256, graph_size = 256, augmentation = True, search_mode = 0, partial = False, noLeftRight = False, remove_adjacent_matrix = 0, output_folder = None, lane_balance_factor=None, reseed = False, must_have_lane = False, must_have_lane_change = False, no_image = False, tiles_name = "tiles" ):
		self.parentRoadNetowrk = parentRoadNetowrk
		self.annotation = parentRoadNetowrk.annotation
		self.config = parentRoadNetowrk.config 
		self.noLeftRight = noLeftRight


		self.train_cnn_only = train_cnn_only
		self.train_cnn_batch = train_cnn_batch

		label_list = [] 

		if partial == True:
			for k,v in self.annotation.iteritems():
				if 'labelled' in v and v['labelled'] == 1 and v['remove'] != 1:
					label_list.append(k)


		if train_cnn_only == False:

			while  True:
				if partial == True:
					seed_node = random.choice(label_list)
					if must_have_lane == True:
						seed_node = random.choice(self.parentRoadNetowrk.node_with_lane)
					if must_have_lane_change == True:
						seed_node = random.choice(self.parentRoadNetowrk.node_with_lane_change)
				else:
					seed_node = random.choice(self.parentRoadNetowrk.nid2loc.keys())
					if must_have_lane == True:
						seed_node = random.choice(self.parentRoadNetowrk.node_with_lane)
					if must_have_lane_change == True:
						seed_node = random.choice(self.parentRoadNetowrk.node_with_lane_change)

				if self.annotation[seed_node]['remove'] == 1:
					continue
				else:
					break 

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
				if i % 2 == 0:
					newGraphNodes.append(random.choice(label_list))
				else:
					newGraphNodes.append(random.choice(self.parentRoadNetowrk.node_with_lane))
			

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

		self.images = np.zeros((self.nonIntersectionNodeNum, 384, 384, 3), dtype = np.float32)

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
						try:
							img = scipy.ndimage.imread(output_folder + "/"+tiles_name+"/img_%d.png" % nid).astype(np.float32)/255.0 
						except:

							img = scipy.ndimage.imread("/data/songtao/DeepRoadMateinfo/"+ self.config["folder"] + "/"+tiles_name+"/img_%d.png" % nid).astype(np.float32)/255.0 
						
							print(nid)
							print(output_folder + "/"+tiles_name+"/img_%d.png" % nid)
				else:
					v = self.annotation[nid]
					img_size = 16384
					region = self.config["region"]

					loc = self.parentRoadNetowrk.nid2loc[k]
					x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, img_size,region)

					heading_vector_lat, heading_vector_lon = v['heading_vector']
					subimage = large_image[x-272:x+272, y-272:y+272]

					if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
						angle = 0.0
					else:
						angle = math.degrees(math.atan2(heading_vector_lon, heading_vector_lat))
					
					#print(k, angle)

					img_rot = scipy.ndimage.interpolation.rotate(subimage, angle)


					center = np.shape(img_rot)[0]/2

					r = 192 # 384
					img = img_rot[center-r:center+r, center-r: center+ r,:]

					img = img.astype(np.float32)/255.0 





				self.images[c,:,:,:] = self.image_augmentation(img, flag=augmentation) 
				c = c + 1 







		#load targets and masks

		self.targets = np.zeros((self.nonIntersectionNodeNum, 6), dtype = np.int32)
		self.mask = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)

		self.lane_number_balance = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)
		self.parking_balance = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)
		self.biking_balance = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)
		self.roadtype_balance = np.zeros((self.nonIntersectionNodeNum), dtype = np.float32)



		c = 0 
		for nid in self.subGraphNoadList[:self.nonIntersectionNodeNum]:
			self.targets[c,:] = self.parentRoadNetowrk.targets[nid,:]

			if noLeftRight:
				if self.targets[c,1] == 1 or self.targets[c,4] == 1:
					self.targets[c,1] = 1
					self.targets[c,4] = 1

				if self.targets[c,2] == 1 or self.targets[c,3] == 1:
					self.targets[c,2] = 1
					self.targets[c,3] = 1


			self.mask[c] = self.parentRoadNetowrk.mask[nid]


			if self.targets[c,0] < 0:
				self.lane_number_balance[c] = 0.0
				self.targets[c,0] = 0
			else:
				if lane_balance_factor is not None:
					if self.targets[c,0] > 5:
						self.targets[c,0] = 5
					self.lane_number_balance[c] = self.mask[c] * lane_balance_factor[self.targets[c,0]]
				else:
					self.lane_number_balance[c] = self.mask[c] 

			#print(c, self.mask[c], self.targets[c,0], self.lane_number_balance[c] )


			if self.targets[c,0] > 5:
				self.targets[c,0] = 5


				

			self.parking_balance[c] = self.mask[c]
			self.biking_balance[c] = self.mask[c]
			self.roadtype_balance[c] = 1.0
			


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

		self.intersectionFeatures = np.zeros((len(newGraphNodes)-self.nonIntersectionNodeNum, 64), dtype = np.float32)


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


	def Get_lane_number_balance(self, batch_size = None):
		if batch_size is None:
			return self.lane_number_balance
		else:
			return self.lane_number_balance[self.st:self.st+batch_size]


	def Get_parking_balance(self, batch_size = None):
		if batch_size is None:
			return self.parking_balance
		else:
			return self.parking_balance[self.st:self.st+batch_size]


	def Get_biking_balance(self, batch_size = None):
		if batch_size is None:
			return self.biking_balance
		else:
			return self.biking_balance[self.st:self.st+batch_size]


	def Get_roadtype_balance(self, batch_size = None):
		if batch_size is None:
			return self.roadtype_balance
		else:
			return self.roadtype_balance[self.st:self.st+batch_size]


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

	def UpdateConsistency(self, output_lane_number, output_road_type, decay = 0.99):

		for i in xrange(self.nonIntersectionNodeNum):
			nid = self.subGraphNoadList[i]
			if self.targets[i][5] == 1: # primary road 
				if self.targets[i][0] >= 0:
					number_of_lane = np.argmax(output_lane_number[i,:].reshape((6)))
					if number_of_lane == self.targets[i][0]:
						self.parentRoadNetowrk.consistency[nid][0] = self.parentRoadNetowrk.consistency[nid][0] * decay + 1.0 * (1-decay)
						# todo increase score 
						pass
					else:

						self.parentRoadNetowrk.consistency[nid][0] *= decay
						# decrease the score 
						pass

			roadtype = np.argmax(output_road_type[i,:].reshape((2)))
			if roadtype == self.targets[i][5]:
				self.parentRoadNetowrk.consistency[nid][5] = self.parentRoadNetowrk.consistency[nid][5] * decay + 1.0 * (1-decay)
						
				pass
			else:

				self.parentRoadNetowrk.consistency[nid][5] *= decay
				pass


	def GetAccuracyStatistic(self, outputs, accumulate = None, dump=True, validation_set=[], mask = None ):
		def update_binary_classifier_counter(x, output, target):
			x['total'][target] += 1

			if output == target:
				x['correct'][target] += 1

		def get_binary_classifier_counter():
			x = {}
			x['total'] = [0.0001,0.0001]
			x['correct'] = [0,0]

			return x 

		def print_binary_classifier_counter(x, prefix = "", dump=True):
			if dump:
				print(prefix + "\t predict 0 %d/%d (%.3f)  \t predict 1 %d/%d (%.3f) \t overall %d/%d (%.3f)" 
				% (x['correct'][0], x['total'][0], float(x['correct'][0])/x['total'][0],
				 x['correct'][1], x['total'][1], float(x['correct'][1])/x['total'][1],
				 np.sum(x['correct']), np.sum(x['total']), float(np.sum(x['correct']))/np.sum(x['total'])))


			return float(np.sum(x['correct']))/np.sum(x['total'])

		if accumulate is None:

			per_lane_correct_counter = [0] * 6
			per_lane_counter = [0.000001] * 6
			lane_abs_error = 0

			roadtype_counter = get_binary_classifier_counter()

			counter_group = [get_binary_classifier_counter() for i in range(4)]

		else:
			per_lane_correct_counter = accumulate[0]
			per_lane_counter = accumulate[1]
			roadtype_counter = accumulate[2]
			counter_group = accumulate[3]
			lane_abs_error = accumulate[4]


		for i in xrange(self.nonIntersectionNodeNum):

			if self.subGraphNoadList[i] in validation_set:
				continue

			if mask is not None:
				if self.subGraphNoadList[i] not in mask:
					continue

			if self.targets[i][5] == 1: # primary road 
				if self.targets[i][0] >= 0 and self.lane_number_balance[i] != 0:
					number_of_lane = np.argmax(outputs[1][i,:].reshape((6)))

					per_lane_counter[self.targets[i][0]] += 1 

					if number_of_lane == self.targets[i][0]:
						per_lane_correct_counter[number_of_lane] += 1

					lane_abs_error += np.abs(number_of_lane-self.targets[i][0])


					for j in xrange(4):
						output = np.argmax(outputs[2+j][i,:].reshape((2)))
						update_binary_classifier_counter(counter_group[j], output, self.targets[i][j+1])

			output = np.argmax(outputs[6][i,:].reshape((2)))
			update_binary_classifier_counter(roadtype_counter, output, self.targets[i][5])

		acc_roadtype = print_binary_classifier_counter(roadtype_counter, "roadtype", dump=dump)
		acc_l_parking = print_binary_classifier_counter(counter_group[0], "left_parking ",dump=dump)
		if self.noLeftRight == False:
			acc_r_parking = print_binary_classifier_counter(counter_group[3], "right_parking",dump=dump)
		acc_l_biking = print_binary_classifier_counter(counter_group[1], "left_biking  ",dump=dump)
		if self.noLeftRight == False:
			acc_r_biking = print_binary_classifier_counter(counter_group[2], "right_biking ",dump=dump)

		if dump :
			for i in xrange(6):
				print("lane %d \t %d/%d (%.3f)" % (i+1, per_lane_correct_counter[i], per_lane_counter[i], float(per_lane_correct_counter[i])/per_lane_counter[i] ))
				

			print("lane overall \t %d/%d (%.3f)"  % (np.sum(per_lane_correct_counter), np.sum(per_lane_counter), float(np.sum(per_lane_correct_counter))/np.sum(per_lane_counter) ))
			print("lane error %.3f" % (float(lane_abs_error)/np.sum(per_lane_counter)))



		acc_lane = float(np.sum(per_lane_correct_counter))/np.sum(per_lane_counter)

		if self.noLeftRight == True:
			acc_overall = (acc_roadtype + 1.0*acc_l_parking + 1.0*acc_l_biking  + acc_lane)/4.0
		else:
			acc_overall = (acc_roadtype + 0.5*acc_l_parking + 0.5*acc_r_parking + 0.5*acc_l_biking + 0.5*acc_r_biking + acc_lane)/4.0
		if dump:
			print("overall score: %.4f" % acc_overall)


		return [acc_overall, acc_roadtype, acc_lane, acc_l_parking, acc_l_biking], [per_lane_correct_counter, per_lane_counter, roadtype_counter, counter_group, lane_abs_error]

	def SegmentSmoothSlidingWindow(self, outputs, window_size = 3):
		for seg_id, seg in self.parentRoadNetowrk.segments.iteritems():
			if len(seg)<=2+window_size-1:
				continue

			new_labels_lane = [[0] * 6 for _ in xrange(len(seg))]
			new_labels_roadtype = [[0] * 2 for _ in xrange(len(seg))]

			for start_pos in xrange(1,len(seg)-1):

				scores_lane = [0]*6
				scores_roadtype = [0]*2 
				flag = True 


				if start_pos == 1 :
					st = start_pos
					ed = start_pos + 2
					ll = 2

				elif start_pos == len(seg)-2:
					st = start_pos-1
					ed = start_pos + 1
					ll = 2
				else:
					st = start_pos-1
					ed = start_pos + 2
					ll = 3

				for nid in seg[st:ed]:
					if nid not in self.node_mapping:
						flag = False 
						break

					ind = self.node_mapping[nid]

					for i in xrange(6):
						scores_lane[i] += outputs[1][ind,:].reshape((6))[i]

					for i in xrange(2):
						scores_roadtype[i] += outputs[6][ind,:].reshape((2))[i]

				if flag == False:

					if seg[start_pos] in self.node_mapping:
						ind = self.node_mapping[seg[start_pos]]

						for i in xrange(6):
							new_labels_lane[start_pos][i] = outputs[1][ind,i]
						for i in xrange(2):
							new_labels_roadtype[start_pos][i] = outputs[6][ind,i]


					continue

				for i in xrange(6):
					new_labels_lane[start_pos][i] = scores_lane[i] / ll

				for i in xrange(2):
					new_labels_roadtype[start_pos][i] = scores_roadtype[i] / ll


			for start_pos in xrange(1,len(seg)-1):
				nid = seg[start_pos]

				if nid not in self.node_mapping:
					continue

				ind = self.node_mapping[nid]

				for i in xrange(6):
					outputs[1][ind,i] = new_labels_lane[start_pos][i] 


				for i in xrange(2):
					outputs[6][ind,i] = new_labels_roadtype[start_pos][i]



				

		return outputs 


	def PostProcessWithMRF(self, outputs, weight = 10, norm = 2.0, ind = 1, labels = 6, mask = None  ):
		# for lanes 

		nLanes = labels
		nNode = self.nonIntersectionNodeNum

		gm = opengm.gm([nLanes] * nNode, operator='adder')
		

		graph = self.spares_graph_structure_direction1 


		pairwise_dict = {}
		for edge in graph['indices']:
			n1 = edge[0]
			n2 = edge[1]

			if mask is not None:
				if n1 not in mask or n2 not in mask:
					continue

			if (n1,n2) in pairwise_dict or (n2,n1) in pairwise_dict:
				pass
			else:
				pairwise_dict[(n1,n2)] = True

		#gm.reserveFunctions(nNode + len(pairwise_dict),'explicit')
		#gm.reserveFactors(nNode + len(pairwise_dict))

		#unaries = outputs[1] # lanes

		unaries = np.zeros((nNode, nLanes))
		for i in range(nNode):
			for j in range(nLanes):
				unaries[i][j] = -np.log(max(outputs[ind][i,j],0.00000001))




		# - add unaries to the graphical model
		#fids=gm.addFunctions(unaries.astype(opengm.value_type))

		for i in range(nNode):
			fid =  gm.addFunction(unaries[i][:].reshape((nLanes)).astype(opengm.value_type))
			gm.addFactor(fid, np.array([i]).astype(opengm.index_type))

			


		# add pairwise energy functions 

		weights = weight
		norm = norm 

		pf = np.zeros((nLanes, nLanes))

		for i in range(nLanes):
			for j in range(nLanes):
				pf[i][j] =  weights * abs(i-j)**norm 

		#print(pf)

		for edge in pairwise_dict.keys():
			#n1 = self.node_mapping[edge[0]]
			#n2 = self.node_mapping[edge[1]]

			n1 = edge[0]
			n2 = edge[1]

			if n1 >= nNode or n2 >= nNode:
				print("This should not happen! ", n1, n2)
				continue
			#gm.addFactors(gm.addFunction(pf),[n1,n2])
			fid = gm.addFunction(pf.astype(opengm.value_type))

			if n1<=n2:
				gm.addFactor(fid, np.array([n1,n2]).astype(opengm.index_type))
			else:
				gm.addFactor(fid, np.array([n2,n1]).astype(opengm.index_type))
			#gm.addFactors(gm.addFunction(pf), np.array([n2,n1]).astype(opengm.index_type))

			#print(n1,n2)

			
		class PyCallback(object):
			"""
			callback functor which will be passed to an inference
			visitor.
			In that way, pure python code can be injected into the c++ inference.
			This functor visualizes the labeling as an image during inference.

			Args :
				shape : shape of the image 
				numLabels : number of labels
			"""
			def __init__(self):
				self.step = 0
				pass 
				
			def begin(self,inference):
				"""
				this function is called from c++ when inference is started

				Args : 
					inference : python wrapped c++ solver which is passed from c++
				"""
				print("begin")

			def end(self,inference):
				"""
				this function is called from c++ when inference ends

				Args : 
					inference : python wrapped c++ solver which is passed from c++
				"""
				arg = inference.arg()
				gm  = inference.gm()
				print(self.step, "energy ",gm.evaluate(arg))
				print("end")

			def visit(self,inference):
				"""
				this function is called from c++ each time the visitor is called

				Args : 
					inference : python wrapped c++ solver which is passed from c++
				"""
				
				# arg = inference.arg()
				# gm  = inference.gm()
				# print(self.step, "energy ",gm.evaluate(arg))
				self.step += 1
				
		#inf=opengm.inference.Icm(gm)
		#inf = opengm.inference.Icm(gm,parameter=opengm.InfParam())
		#inf.setStartingPoint(inf.arg())

		inf=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=100,damping=0.9,convergenceBound=0.00001))
		
		callback = PyCallback() 
		visitor=inf.pythonVisitor(callback,visitNth=1)
		inf.infer(visitor)
		#inf.infer()

		arg=inf.arg()

		print(np.shape(arg))
		print(arg)

		for i in range(len(outputs[1])):
			for j in range(nLanes):
				outputs[ind][i,j]=0
			outputs[ind][i,arg[i]]=1

		return outputs

	def SegmentAccuracy(self, outputs):

		segments_score_lane = []  # (id , score) 
		segments_score_roadtype = []  # (id , score) 


		for seg_id, seg in self.parentRoadNetowrk.segments.iteritems():
			
			if len(seg)<=2:
				continue

			flag = True

			scores_lane = []
			scores_roadtype = []


			for nid in seg[1:len(seg)-1]:

				if nid not in self.subGraphNoadList:
					flag=False 
					break 

				# todo local score?
				ind = self.node_mapping[nid]

				if self.lane_number_balance[ind] != 0:
					gt = self.targets[ind,0]

					p = max(outputs[1][ind,:].reshape((6))[gt], 0.00000000000000001)

					if gt != np.argmax(outputs[1][ind,:].reshape((6))):
						scores_lane.append(-math.log(p))
					else:
						scores_lane.append(0)

				else:
					scores_lane.append(-1.0)


				# roadtype 
				gt = self.targets[ind,5]
				p = max(outputs[6][ind,:].reshape((2))[gt], 0.00000000000000001) 

				if gt != np.argmax(outputs[6][ind,:].reshape((2))):
					scores_roadtype.append(-math.log(p))
				else:
					scores_roadtype.append(0.0)


			if flag == False :
				continue 


			if np.sum(scores_lane) > 0:  # 
				segments_score_lane.append((seg_id, np.mean(scores_lane), list(scores_lane)))

			if np.sum(scores_roadtype) > 0: #
				segments_score_roadtype.append((seg_id, np.mean(scores_roadtype), list(scores_roadtype)))


		segments_score_lane = sorted(segments_score_lane, key=lambda x: x[1])
		segments_score_roadtype = sorted(segments_score_roadtype, key=lambda x: x[1])

		print(segments_score_lane[0], segments_score_lane[len(segments_score_lane)-1])
		print(segments_score_roadtype[0], segments_score_roadtype[len(segments_score_roadtype)-1])


		segments_score_lane_dict = {}

		for i in xrange(len(segments_score_lane)):
			rank = float(i)/len(segments_score_lane)
			segments_score_lane_dict[segments_score_lane[i][0]] = (segments_score_lane[i][1], rank)

		segments_score_roadtype_dict = {}

		for i in xrange(len(segments_score_roadtype)):
			rank = float(i)/len(segments_score_roadtype)
			segments_score_roadtype_dict[segments_score_roadtype[i][0]] = (segments_score_roadtype[i][1], rank)
		


		return segments_score_lane_dict, segments_score_roadtype_dict



	def RecommendationAccNode(self, outputs, gt_roadnetwork, validation_set=[]):


		# find top errors 

		roadtype_score = []
		lane_score = []


		for nid in self.subGraphNoadList[:self.nonIntersectionNodeNum]:

			if nid in validation_set:
				continue

			ind = self.node_mapping[nid]

			if nid in gt_roadnetwork.node_mapping:
				ind_gt = gt_roadnetwork.node_mapping[nid]



			else:
				continue




			if self.lane_number_balance[ind] != 0 and gt_roadnetwork.lane_number_balance[ind_gt] != 0:
				gt_osm = self.targets[ind,0]

				gtgt = gt_roadnetwork.targets[ind_gt,0]



				if gt_osm !=  np.argmax(outputs[1][ind_gt,:].reshape((6))):

					score = np.amax(outputs[1][ind_gt,:])

					if np.argmax(outputs[1][ind_gt,:].reshape((6))) == gtgt:
						correct = 1
					else:
						correct = 0

					lane_score.append((nid,score,correct))



			# roadtype 
			gt_osm = self.targets[ind,5]
			gtgt = gt_roadnetwork.targets[ind_gt,5]
			

			if gt_osm != np.argmax(outputs[6][ind_gt,:].reshape((2))):
				score = np.amax(outputs[6][ind_gt,:])

				if gtgt == np.argmax(outputs[6][ind_gt,:].reshape((2))):
					correct = 1
				else:
					correct = 0 


				roadtype_score.append((nid,score,correct))


			
		print(len(roadtype_score), len(lane_score), self.nonIntersectionNodeNum)


		lane_score = sorted(lane_score, key = lambda x : -x[1])

		roadtype_score = sorted(roadtype_score, key = lambda x : -x[1])




		def topk(k, a, name):

			k = min(k, len(a))

			true_positive = np.sum([x[2] for x in a[:k]]) / float(k)


			print(name, "top-%d" % k, "True Positive %.3f" % true_positive)


			return true_positive 



		output = []

		output.append(topk(10, lane_score, "lane"))
		output.append(topk(20, lane_score, "lane"))
		output.append(topk(50, lane_score, "lane"))
		output.append(topk(100, lane_score, "lane"))
		output.append(topk(200, lane_score, "lane"))
		output.append(topk(500, lane_score, "lane"))
		

		output.append(topk(10, roadtype_score, "roadtype"))
		output.append(topk(20, roadtype_score, "roadtype"))
		output.append(topk(50, roadtype_score, "roadtype"))
		output.append(topk(100, roadtype_score, "roadtype"))
		output.append(topk(200, roadtype_score, "roadtype"))
		output.append(topk(500, roadtype_score, "roadtype"))

		return output




	def VisualizeOutput(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				loc0 = self.parentRoadNetowrk.nid2loc[edge[0]]
				loc1 = self.parentRoadNetowrk.nid2loc[edge[1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)


				cv2.line(img, (y0,x0), (y1,x1), (255,255,0),1)

				#cv2.circle(img, (y0,x0), 3, (0,255,0), -1)
				#cv2.circle(img, (y1,x1), 3, (0,255,0), -1)
				
				#print(outputs)

				def draw_output_info(i,x,y,dx,dy):
					num_of_lane = np.argmax(outputs[1][i,:].reshape((6)))
					num_of_lane_confidence = outputs[1][i,num_of_lane]
					num_of_lane = num_of_lane + 1

					step = 7

					if outputs[6][i,0] > 0.5:
						color = (255,255-int(outputs[6][i,0]*255),255)


						cv2.circle(img, (y,x), 3, color, -1)


					for ii in range(num_of_lane):
						dl = ii - float(num_of_lane-1)/2

						color = (255, 255-int(num_of_lane_confidence*255),255-int(num_of_lane_confidence*255))
						cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 1, color, -1)


					if outputs[2][i,1] > 0.5 and draw_parking == True:
						color = (255-int(outputs[2][i,1]*255),255-int(outputs[2][i,1]*255),255)

						dl = -2 - float(num_of_lane-1)/2

						cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 1, color, -1)

					if outputs[3][i,1] > 0.5 and draw_biking == True:
						color = (255-int(outputs[3][i,1]*255),255, 255-int(outputs[3][i,1]*255))

						dl = -1 - float(num_of_lane-1)/2

						cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 1, color, -1)

					# if outputs[4][i,1] > 0.5 and draw_biking == True:
					# 	color = (255-int(outputs[4][i,1]*255),255, 255-int(outputs[4][i,1]*255))

					# 	dl = num_of_lane - float(num_of_lane-1)/2

					# 	cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 1, color, -1)

					# if outputs[5][i,1] > 0.5 and draw_parking == True:
					# 	color = (255-int(outputs[5][i,1]*255),255-int(outputs[5][i,1]*255),255)

					# 	dl = num_of_lane+1 - float(num_of_lane-1)/2

					# 	cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 1, color, -1)

					




				ind0 = self.node_mapping[edge[0]]
				ind1 = self.node_mapping[edge[1]]

				# if (ind0,ind1) in self.spares_graph_structure['indices'] or (ind0,ind1) in self.spares_graph_structure['indices']
				# 	pass 
				# else:
				# 	if remove_adjacent_matrix != 1:
				# 		continue





				if ind0 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind0,0]
					dy = self.heading_vector[ind0,1]

					draw_output_info(ind0, x0, y0, dy, dx)

				if ind1 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind1,0]
					dy = self.heading_vector[ind1,1]

					draw_output_info(ind1, x1, y1, dy, dx)



		Image.fromarray(img).save(output_file)
		pass

	def VisualizeOutputLane(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				loc0 = self.parentRoadNetowrk.nid2loc[edge[0]]
				loc1 = self.parentRoadNetowrk.nid2loc[edge[1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)


				cv2.line(img, (y0,x0), (y1,x1), (255,255,0),1)

				#cv2.circle(img, (y0,x0), 3, (0,255,0), -1)
				#cv2.circle(img, (y1,x1), 3, (0,255,0), -1)
				
				#print(outputs)

				def draw_output_info(i,x,y,dx,dy, color_type = "blue"):
					num_of_lane = np.argmax(outputs[1][i,:].reshape((6)))
					num_of_lane_confidence = outputs[1][i,num_of_lane]
					num_of_lane = num_of_lane + 1


					if self.lane_number_balance[i] != 0:
						if num_of_lane-1 == self.targets[i,0]:
							color_type = "green"
						else:
							color_type = "red"


					step = 7

					# if outputs[6][i,0] > 0.5:
					# 	color = (255,255-int(outputs[6][i,0]*255),255)


					# 	cv2.circle(img, (y,x), 3, color, -1)

					for ii in range(num_of_lane):
						dl = ii - float(num_of_lane-1)/2
						#print(color_type)

						if color_type == "green" :
							color = (255-int(num_of_lane_confidence*255),255,255-int(num_of_lane_confidence*255))
						elif color_type == "red":
							color = (255, 255-int(num_of_lane_confidence*255),255-int(num_of_lane_confidence*255))
						else:
							color = (255-int(num_of_lane_confidence*255),255-int(num_of_lane_confidence*255), 255)




						cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 3, color, -1)


					
				ind0 = self.node_mapping[edge[0]]
				ind1 = self.node_mapping[edge[1]]

				# if (ind0,ind1) in self.spares_graph_structure['indices'] or (ind0,ind1) in self.spares_graph_structure['indices']
				# 	pass 
				# else:
				# 	if remove_adjacent_matrix != 1:
				# 		continue



				if ind0 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind0,0]
					dy = self.heading_vector[ind0,1]



					if edge[0] not in self.validation_set:

						draw_output_info(ind0, x0, y0, dy, dx)

				if ind1 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind1,0]
					dy = self.heading_vector[ind1,1]
					if edge[1] not in self.validation_set:
						draw_output_info(ind1, x1, y1, dy, dx)





		Image.fromarray(img).save(output_file)



		pass

	def VisualizeOutputLaneMicro(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				loc0 = self.parentRoadNetowrk.nid2loc[edge[0]]
				loc1 = self.parentRoadNetowrk.nid2loc[edge[1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)


				#cv2.line(img, (y0,x0), (y1,x1), (255,255,0),2)

				#cv2.circle(img, (y0,x0), 3, (0,255,0), -1)
				#cv2.circle(img, (y1,x1), 3, (0,255,0), -1)
				
				#print(outputs)

				def draw_output_info(i,x,y,dx,dy, color_type = "blue"):
					num_of_lane = np.argmax(outputs[1][i,:].reshape((6)))
					num_of_lane_confidence = outputs[1][i,num_of_lane]
					num_of_lane = num_of_lane + 1


					#if self.lane_number_balance[i] != 0:
					if num_of_lane-1 == self.targets[i,0]:
						color_type = "green"
					else:
						color_type = "red"


					step = 28

					# if outputs[6][i,0] > 0.5:
					# 	color = (255,255-int(outputs[6][i,0]*255),255)


					# 	cv2.circle(img, (y,x), 3, color, -1)
					#color_type = "green"
					#num_of_lane_confidence = 1.0 

					for ii in range(num_of_lane):
						dl = ii - float(num_of_lane-1)/2
						#print(color_type)

						if color_type == "green" :
							color = (255-int(num_of_lane_confidence*255),255,255-int(num_of_lane_confidence*255))
						elif color_type == "red":
							color = (255, 255-int(num_of_lane_confidence*255),255-int(num_of_lane_confidence*255))
						else:
							color = (255-int(num_of_lane_confidence*255),255-int(num_of_lane_confidence*255), 255)




						cv2.circle(img, (y+int(dy*step*dl),x+int(dx*step*dl)), 12, color, -1)


					
				ind0 = self.node_mapping[edge[0]]
				ind1 = self.node_mapping[edge[1]]

				# if (ind0,ind1) in self.spares_graph_structure['indices'] or (ind0,ind1) in self.spares_graph_structure['indices']
				# 	pass 
				# else:
				# 	if remove_adjacent_matrix != 1:
				# 		continue



				if ind0 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind0,0]
					dy = self.heading_vector[ind0,1]





					draw_output_info(ind0, x0, y0, dy, dx)

				if ind1 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind1,0]
					dy = self.heading_vector[ind1,1]

					draw_output_info(ind1, x1, y1, dy, dx)





		Image.fromarray(img).save(output_file)



		pass

	def VisualizeOutputRoadType(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		for edge in self.parentRoadNetowrk.edges:
			if edge[0] in self.subGraphNoadList and edge[1] in self.subGraphNoadList:
				loc0 = self.parentRoadNetowrk.nid2loc[edge[0]]
				loc1 = self.parentRoadNetowrk.nid2loc[edge[1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)


				

				#cv2.circle(img, (y0,x0), 3, (0,255,0), -1)
				#cv2.circle(img, (y1,x1), 3, (0,255,0), -1)
				
				#print(outputs)

				def draw_output_info(i,x,y,dx,dy, color_type = "blue"):
					num_of_lane = np.argmax(outputs[1][i,:].reshape((6)))
					num_of_lane_confidence = outputs[1][i,num_of_lane]
					num_of_lane = num_of_lane + 1

					if (outputs[6][i,0] > 0.5 and self.targets[i,5] == 0) or (outputs[6][i,0] < 0.5 and self.targets[i,5] == 1):
						color = (0,255,0)

						cv2.circle(img, (y,x), 5, color, -1)
					else:
						color = (255,0,0)

						cv2.circle(img, (y,x), 5, color, -1)


				ind0 = self.node_mapping[edge[0]]
				ind1 = self.node_mapping[edge[1]]

				# if (ind0,ind1) in self.spares_graph_structure['indices'] or (ind0,ind1) in self.spares_graph_structure['indices']
				# 	pass 
				# else:
				# 	if remove_adjacent_matrix != 1:
				# 		continue



				residential = False 

				if ind0 < self.nonIntersectionNodeNum:
					if self.targets[ind0,5] == 0 :
						residential = True 

				if ind1 < self.nonIntersectionNodeNum:
					if self.targets[ind1,5] == 0 :
						residential = True 



				if residential==True: 
					cv2.line(img, (y0,x0), (y1,x1), (0,255,255),2)
				else:
					cv2.line(img, (y0,x0), (y1,x1), (255,255,0),2)


				if ind0 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind0,0]
					dy = self.heading_vector[ind0,1]
					if edge[0] not in self.validation_set:
						draw_output_info(ind0, x0, y0, dy, dx)

				if ind1 < self.nonIntersectionNodeNum:
					dx = self.heading_vector[ind1,0]
					dy = self.heading_vector[ind1,1]
					if edge[1] not in self.validation_set:
						draw_output_info(ind1, x1, y1, dy, dx)



		Image.fromarray(img).save(output_file)



		pass

	def VisualizeAdjacent(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		for edge in self.spares_graph_structure['indices']:
			n1 = edge[0]
			n2 = edge[1]

			loc0 = self.parentRoadNetowrk.nid2loc[self.subGraphNoadList[n1]]
			loc1 = self.parentRoadNetowrk.nid2loc[self.subGraphNoadList[n2]]

			x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
			x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)

			cv2.line(img, (y0,x0), (y1,x1), (0,255,255),2)
			cv2.circle(img, (y0,x0), 5, (255,0,0), -1)
			cv2.circle(img, (y0,x0), 5, (255,0,0), -1)


		Image.fromarray(img).save(output_file)

		pass

	def VisualizeAdjacentDirection(self, structure, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)

		
		print(structure)

		for edge in structure['indices']:
			n1 = edge[0]
			n2 = edge[1]

			loc0 = self.parentRoadNetowrk.nid2loc[self.subGraphNoadList[n1]]
			loc1 = self.parentRoadNetowrk.nid2loc[self.subGraphNoadList[n2]]

			x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
			x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)

			cv2.line(img, (y0,x0), (y1,x1), (0,255,255),2)


			dx = (x1-x0)/5
			dy = (y1-y0)/5


			cv2.line(img, (y1-dy+dx,x1-dx-dy), (y1,x1), (0,255,255),2)
			cv2.line(img, (y1-dy-dx,x1-dx+dy), (y1,x1), (0,255,255),2)

			cv2.circle(img, (y0,x0), 5, (255,0,0), -1)
			cv2.circle(img, (y0,x0), 5, (255,0,0), -1)


		Image.fromarray(img).save(output_file)

		pass

	
	def VisualizeRoadSegment(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)


		colors = [(255,0,0), (0,255,0), (0,0,255),(255,255,0), (0,255,255), (255,0,255),(255,255,255)]

		for seg_id, seg in self.parentRoadNetowrk.segments.iteritems():
			color = colors[seg_id % (len(colors))]
			for i in xrange(len(seg)-1):
				loc0 = self.parentRoadNetowrk.nid2loc[seg[i]]
				loc1 = self.parentRoadNetowrk.nid2loc[seg[i+1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)

				cv2.line(img, (y0,x0), (y1,x1), color,2)
				cv2.circle(img, (y0,x0), 5, (128,128,128), -1)
				cv2.circle(img, (y1,x1), 5, (128,128,128), -1)



		Image.fromarray(img).save(output_file)

		pass


	def VisualizeSegmentAccuracy(self, outputs, output_file = "default.png", draw_biking = True, draw_parking = True):
		img = np.copy(self.parentRoadNetowrk.sat_image)

		lane_acc, roadtype_acc = self.SegmentAccuracy(outputs)

		for seg_id, seg in self.parentRoadNetowrk.segments.iteritems():
		
			lane_f = 1.0 
			roadtype_f = 1.0 

			if seg_id in lane_acc:
				lane_f = 1.0-lane_acc[seg_id][1]

			if seg_id in roadtype_acc:
				roadtype_f = 1.0-roadtype_acc[seg_id][1]



			color_link = (255 - int(roadtype_f*255.0),int(roadtype_f*255.0),0)
			color_node = (255 - int(lane_f*255.0),int(lane_f*255.0),0)

			for i in xrange(len(seg)-1):
				loc0 = self.parentRoadNetowrk.nid2loc[seg[i]]
				loc1 = self.parentRoadNetowrk.nid2loc[seg[i+1]]

				x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0, 4096,self.parentRoadNetowrk.region)
				if seg_id in roadtype_acc:
					cv2.line(img, (y0,x0), (y1,x1), color_link,2)
				
				if i != 0:
					if seg_id in lane_acc:
						cv2.circle(img, (y0,x0), 5, color_node, -1)



		Image.fromarray(img).save(output_file)

		pass



	def DumpInputImages(self, output_folder = "default"):
		for i in range(np.shape(self.images)[0]):
			Image.fromarray((self.images[i,:,:,:]*255).astype(np.uint8).reshape((384,384,3))).save(output_folder+"img%d.png" % i)















