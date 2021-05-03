try:
	from tkinter import *
	import tkinter
except:
	from Tkinter import *
	import Tkinter
import json
import cv2
from subprocess import Popen
import math 
from PIL import Image
from scipy.misc import imresize
import sys 
import numpy as np 
from roadtagger_road_network import * 
import os 
import pickle 
import scipy.ndimage
from PIL import ImageTk

Image.MAX_IMAGE_PIXELS = None

# # these three files are from RoadRunner
import helper_mapdriver as md 
import helper_road_structure as splfy
import helper_road_structure_topo as topo

# config should have 
# - region [lat/lon]
# - output folder
# - (res) default is 0.25
def generate_dataset(config):
	output_folder = config["folder"]
	region = config["region"]
	img_size = config["size"]*2


	if os.path.isfile(output_folder+"/sat_16384.png") == False:
		# generate sat image 
		subfolder_cache = output_folder + "/cache/"
		Popen("mkdir -p %s" % subfolder_cache, shell=True).wait()

		img, ok = md.GetMapInRect(region[0], region[1], region[2], region[3], folder = subfolder_cache, start_lat = region[0], start_lon = region[1], zoom = 19, scale = 1)

		#print("raw resolution", np.shape(img))
		#Image.fromarray(img).save(output_folder+"/sat_raw.png")

		img2 = imresize(img, (img_size, img_size))
		Image.fromarray(img2).save(output_folder+"/sat_16384.png")



	if os.path.isfile(output_folder+"/roadnetwork.p"):
		print("[Warning]road network pickle files already existed. ???")
	

	# load osm
	roadNetwork = RoadNetwork()

	OSMMap = md.OSMLoader(region, True)
	OSMRoadGraph = splfy.RoadGraph()

	for nodeid, node in OSMMap.nodedict.iteritems():
		for next_node in node['to']:
			OSMRoadGraph.addEdge(nodeid, node['lat'], node['lon'], next_node, OSMMap.nodedict[next_node]['lat'], OSMMap.nodedict[next_node]['lon'])

	OSMRoadGraph.ReverseDirectionLink()

	for node in OSMRoadGraph.nodes.keys():
		OSMRoadGraph.nodeScore[node] = 100

	for edge in OSMRoadGraph.edges.keys():
		OSMRoadGraph.edgeScore[edge] = 100


	topo.TOPOGenerateStartingPoints(OSMRoadGraph, density = 0.00020, region=region, image='NULL', check = False, direction = True, metaData = OSMMap, RoadNetworkCallback = roadNetwork)


	roadNetwork.region = region 
	roadNetwork.image_file = output_folder+"/sat_16384.png"

	# don't overwrite
	if os.path.isfile(output_folder+"/roadnetwork.p") == False:
		roadNetwork.DumpToFile(output_folder+"/roadnetwork.p")
	else:
		roadNetwork.DumpToFile(output_folder+"/roadnetwork_old_file_exist.p")

	pass



lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0 
max_lon = -71.0 


# only for 1 meter resolution
def graphDensify(input_pickle_graph, vis=True):

	def xy2latlon(x,y):
		lat = lat_top_left - x * 1.0 / 111111.0
		lon = lon_top_left + (y * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))

		return lat, lon 


	def create_graph(m):
		global min_lat 
		global max_lon 

		graph = splfy.RoadGraph() 

		nid = 0 
		idmap = {}

		def getid(k, idmap):
			if k in idmap :
				return idmap[k]
		
			idmap[k] = nid 
			nid += 1 

			return idmap[k]


		for k, v in m.items():
			n1 = k 

			lat1, lon1 = xy2latlon(n1[0],n1[1])

			if lat1 < min_lat:
				min_lat = lat1 

			if lon1 > max_lon :
				max_lon = lon1 

			for n2 in v:
				lat2, lon2 = xy2latlon(n2[0],n2[1])

				if n1 in idmap:
					id1 = idmap[n1]
				else:
					id1 = nid 
					idmap[n1] = nid 
					nid = nid + 1

				if n2 in idmap:
					id2 = idmap[n2]
				else:
					id2 = nid 
					idmap[n2] = nid 
					nid = nid + 1

				graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
		
		graph.ReverseDirectionLink() 

		for node in graph.nodes.keys():
			graph.nodeScore[node] = 100

		for edge in graph.edges.keys():
			graph.edgeScore[edge] = 100


		return graph 

	map1 = pickle.load(open(input_pickle_graph, "r"))
	graph = create_graph(map1)

	region = [min_lat-30 * 1.0/111111.0, lon_top_left-50 * 1.0/111111.0, lat_top_left+30 * 1.0/111111.0, max_lon+50 * 1.0/111111.0]

	roadNetwork = RoadNetwork()
	topo.TOPOGenerateStartingPoints(graph, density = 0.00020, region=region, image='NULL', check = False, direction = True, metaData = None, RoadNetworkCallback = roadNetwork)

	node_neighbors = {}
	def get_image_coordinate(lat, lon):
		x = int((lat_top_left - lat) / (1.0/111111.0))
		y = int((lon - lon_top_left) / (1.0/111111.0 / math.cos(math.radians(lat_top_left))))
		return x,y 

	roadnetwork = roadNetwork

	for edge in roadnetwork.edges:
		
		loc0 = roadnetwork.nid2loc[edge[0]]
		loc1 = roadnetwork.nid2loc[edge[1]]

		x0,y0 = get_image_coordinate(loc0[0]/111111.0, loc0[1]/111111.0)
		x1,y1 = get_image_coordinate(loc1[0]/111111.0, loc1[1]/111111.0)

		#print(edge, loc0/111111.0, loc1/111111.0, x0,y0,x1,y1)

		n1key = (x0,y0)
		n2key = (x1,y1)

		if n1key != n2key:

			if n1key in node_neighbors:
				if n2key in node_neighbors[n1key]:
					pass 
				else:
					node_neighbors[n1key].append(n2key)
			else:
				node_neighbors[n1key] = [n2key]

			if n2key in node_neighbors:
				if n1key in node_neighbors[n2key]:
					pass 
				else:
					node_neighbors[n2key].append(n1key)
			else:
				node_neighbors[n2key] = [n1key]

	pickle.dump(node_neighbors, open(input_pickle_graph.replace(".p", "_dense.p"), "w"))

	img = np.zeros((512,512), dtype=np.uint8)

	#node_neighbors = map1 


	if vis == True:
		print("size" , len(node_neighbors))
		for k,v in node_neighbors.iteritems():
			n1 = k 
			for n2 in v:
				cv2.line(img, (n1[1], n1[0]), (n2[1], n2[0]), (255),1)

		for k,_ in node_neighbors.iteritems():
			cv2.circle(img, (k[1],k[0]), 2, (255), -1)

	Image.fromarray(img).save(input_pickle_graph.replace(".p", "_dense.png"))

	#exit()
# 384
def generate_per_node_image(config, input_sat = "sat_16384.png", output_name = "tiles", scale = 1.0):
	output_folder = config["folder"]
	region = config["region"]
	img_size = config["size"]*2

	image = scipy.ndimage.imread(output_folder+"/"+input_sat).astype(np.uint8)
	try:
		annotation = pickle.load(open(output_folder+"/annotation.p", "r"))
	except:
		annotation = pickle.load(open(output_folder+"/annotation_osm.p", "r"))
	roadNetwork =  pickle.load(open(output_folder+"/roadnetwork.p", "r"))

	Popen("mkdir -p %s" % (output_folder + "/"+output_name+"/"), shell=True).wait()

	# pad 512
	image = np.pad(image, ((512,512), (512,512),(0,0)), 'constant')

	for k,v in annotation.iteritems():
		loc = roadNetwork.nid2loc[k]
		x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, img_size,region)

		x += 512
		y += 512

		heading_vector_lat, heading_vector_lon = v['heading_vector']
		rr = int(272*scale)
		subimage = image[x-rr:x+rr, y-rr:y+rr]

		subimage = scipy.misc.imresize(subimage, (272*2, 272*2)).astype(np.uint8)


		if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
			angle = 0.0
		else:
			angle = math.degrees(math.atan2(heading_vector_lon, heading_vector_lat))
		
		if math.isnan(angle):
			angle = 0.0
		

		print(k, angle)

		img = scipy.ndimage.interpolation.rotate(subimage, angle)


		center = np.shape(img)[0]/2

		r = 192 # 384
		result = img[center-r:center+r, center-r: center+ r,:]
    
		Image.fromarray(result).save(output_folder + "/"+output_name+"/"+"img_%d.png" % k)



		# if k > 100:
		# 	break 



def get_image_coordinate(lat, lon, size, region):
	x = int((region[2]-lat)/(region[2]-region[0])*size)
	y = int((lon-region[1])/(region[3]-region[1])*size)

	return x,y 


def color_encode(label):
	clabel = np.zeros((4,20,3), dtype = np.uint8)

	try:
		if label['number_of_lane']>0:
			if label['number_of_lane'] == 1:
				clabel[:,8:10,0:2] = 255
			if label['number_of_lane'] == 2:
				clabel[:,8:12,0:2] = 255
			if label['number_of_lane'] == 3:
				clabel[:,8:14,0:2] = 255
			if label['number_of_lane'] == 4:
				clabel[:,6:14,0:2] = 255
			if label['number_of_lane'] == 5:
				clabel[:,6:16,0:2] = 255
			if label['number_of_lane'] == 6:
				clabel[:,4:16,0:2] = 255

		else:
			clabel[:,4:16,0] = 255
			clabel[:,4:16,0] = 128


		if label['left_park'] == 1:
			clabel[:,0:2,2] = 255
		if label['right_park'] == 1:
			clabel[:,18:20,2] = 255
		if label['left_bike'] == 1:
			clabel[:,2:4,1] = 255
		if label['right_bike'] == 1:
			clabel[:,16:18,1] = 255

		if label['roadtype'] == 0:
			clabel[:,:,0] = 255
			clabel[:,:,1] = 0
			clabel[:,:,2] = 255


		if label['remove'] == 1:
			clabel[:,:,:] = 128 


	except:
		clabel[:,:,:] = 128 


	return clabel 





def GenerateRoadSegmentList(roadNetwork):
	visited = []
	mapping_list = []
	segment_position = []
	seg_num = 0


	segments = []

	def directionScore(n1,n2,n3):
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


	for nid in roadNetwork.nid2loc.keys():
		if nid in visited:
			continue
		else:
			if len(roadNetwork.node_degree[nid])> 2 : # never start from intersection 
				continue

			if len(roadNetwork.node_degree[nid]) == 2: # tracing two direction 
				seg_start = roadNetwork.node_degree[nid][0] 
				seg_end = roadNetwork.node_degree[nid][1]

				node_list = [seg_start, nid, seg_end]
				#print("nodelist", node_list)
				for tmp_nid in node_list:
					if tmp_nid not in visited:
						visited.append(tmp_nid)

				while True:
					neighbor = roadNetwork.node_degree[seg_end]
					#print('neightbor', neightbor)
					best_next_nid = -1 
					best_next_score = -1 
					for next_nid in neighbor:
						if next_nid in visited and len(roadNetwork.node_degree[next_nid]) <= 2:
							continue
						if next_nid in node_list:
							continue
						score = directionScore(node_list[len(node_list)-2], node_list[len(node_list)-1], next_nid)

						if score > best_next_score:
							best_next_score = score 
							best_next_nid = next_nid  

					if best_next_score > 0.5 or (best_next_nid!=-1 and len(neighbor)==2):
						node_list.append(best_next_nid)
						#print(node_list, best_next_score)
						if len(roadNetwork.node_degree[best_next_nid]) <= 2:
							if best_next_nid not in visited:
								visited.append(best_next_nid)

						seg_end = best_next_nid

					else:
						break

				while True:
					neighbor = roadNetwork.node_degree[seg_start]
					best_next_nid = -1 
					best_next_score = -1 
					for next_nid in neighbor:
						if next_nid in visited and len(roadNetwork.node_degree[next_nid]) <= 2:
							continue
						if next_nid in node_list:
							continue
						score = directionScore(node_list[1], node_list[0], next_nid)

						if score > best_next_score:
							best_next_score = score 
							best_next_nid = next_nid  

					if best_next_score > 0.5 or (best_next_nid!=-1 and len(neighbor)==2):
						node_list = [best_next_nid] + node_list
						#print(node_list)
						if len(roadNetwork.node_degree[best_next_nid]) <= 2:
							if best_next_nid not in visited:
								visited.append(best_next_nid)

						seg_start = best_next_nid

					else:
						break 

				segments.append([node_list, len(node_list)])
				#print(node_list)
				#exit()

	segments.sort(key=lambda x: x[1], reverse=True)

	print(segments[0:5],segments[-1])

	return segments



def annotate_dataset_osm(config):
	output_folder = config["folder"]
	region = config["region"]
	img_size = config["size"]

	roadNetwork = pickle.load(open(output_folder+"/roadnetwork.p", "rb"))

	annotation_file = output_folder+"/annotation_osm.p"

	if os.path.isfile(annotation_file):
		annotation = pickle.load(open(annotation_file, "r"))
	else:
		annotation = {}
		
		for anid in roadNetwork.nid2loc.keys():


			annotation[anid] = {}

 			annotation[anid]['number_of_lane'] = 1 # 1,2,3,4,5,6 or -1
 			annotation[anid]['transition'] = -1 # 1.5, 2.5, 3.5, 4.5, 5.5 

 			# 0 or 1 
 			annotation[anid]['left_bike'] = 0
 			annotation[anid]['right_bike'] = 0

 			annotation[anid]['left_park'] = 0
 			annotation[anid]['right_park'] = 0


 			# confidence 
 			annotation[anid]['confidence'] = 1 

 			# not consider
 			annotation[anid]['remove'] = 0

 			annotation[anid]['labelled'] = 1




 			heading_vector_lat = 0 
	 		heading_vector_lon = 0



	 		if len(roadNetwork.node_degree[anid]) > 2:
	 			heading_vector_lat = 0 
	 			heading_vector_lon = 0
	 		elif len(roadNetwork.node_degree[anid]) == 1:
	 			loc1 = roadNetwork.nid2loc[anid]
	 			loc2 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][0]]

	 			dlat = loc1[0] - loc2[0]
	 			dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]/111111.0))



	 			l = np.sqrt(dlat*dlat + dlon * dlon)

	 			dlat /= l
	 			dlon /= l 

	 			heading_vector_lat = dlat 
	 			heading_vector_lon = dlon 
	 		elif len(roadNetwork.node_degree[anid]) == 2:
	 			loc1 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][1]]
	 			loc2 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][0]]

	 			dlat = loc1[0] - loc2[0]
	 			dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]/111111.0))

	 			l = np.sqrt(dlat*dlat + dlon * dlon)

	 			dlat /= l
	 			dlon /= l 

	 			heading_vector_lat = dlat 
	 			heading_vector_lon = dlon 


	 		annotation[anid]['heading_vector'] = (heading_vector_lat, heading_vector_lon)
	 		annotation[anid]['degree'] = len(roadNetwork.node_degree[anid])
	 		



	 		road_info = roadNetwork.nodes[roadNetwork.nid2loc[anid]][1]

			if road_info['roadtype'] in ['residential', 'service']:
				annotation[anid]['roadtype'] = 0
			else:
				annotation[anid]['roadtype'] = 1


			annotation[anid]['number_of_lane'] = road_info['lane']


			if road_info['cycleway'] not in ['none', 'shared_lane', 'share_busway', 'shared', 'no']:
				annotation[anid]['left_bike'] = 1 
			else:
				annotation[anid]['left_bike'] = 0 



		

		raw_img = scipy.ndimage.imread(output_folder+"/sat_16384.png").astype(np.uint8)
		overview = scipy.misc.imresize(raw_img, (4096, 4096)).astype(np.uint8)


		for loc, node in roadNetwork.nodes.iteritems():
			# draw a circle

			x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, 4096,region)


			
			clabel = color_encode(annotation[node[0]])
			if annotation[node[0]]['remove'] == 1:
				color = (128,128,128)
			elif 'labelled' in annotation[node[0]] and annotation[node[0]]['labelled'] == 1:
				color = (0,255,0)
			else:
				color = (255,0,0)

			overview[x-9:x-5,y-10:y+10, :] = clabel

			cv2.circle(overview, (y,x), 3, color, -1)


		Image.fromarray(overview).save(output_folder+"/overview.png")
		pickle.dump(annotation, open(output_folder+"/annotation_osm.p", "w"))

		# 	self.active_node_x, self.active_node_y = get_image_coordinate(roadNetwork.nid2loc[self.active_node][0]/111111.0, roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,region)
		# 	self.InitNode()
		# 	self.updateHeading()

		# self.OSMLabelRoadType()
		# self.OSMLabelLanes()



# an interactive interface to label all the roads 
class annotate_dataset():
	def __init__(self, config, osm_auto = False):
		output_folder = config["folder"]
		region = config["region"]
		img_size = config["size"]

		self.w = None 

		roadNetwork = pickle.load(open(output_folder+"/roadnetwork.p", "rb"))
		self.roadNetwork = roadNetwork


		annotation_file = output_folder+"/annotation.p"

		if os.path.isfile(annotation_file):
			annotation = pickle.load(open(annotation_file, "r"))
			self.annotation = annotation
		else:
			annotation = {}
			self.annotation = annotation
			
			for self.active_node in roadNetwork.nid2loc.keys():
				self.active_node_x, self.active_node_y = get_image_coordinate(roadNetwork.nid2loc[self.active_node][0]/111111.0, roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,region)
				self.InitNode()
				self.updateHeading()

			self.OSMLabelRoadType()
			self.OSMLabelLanes()


		print(annotation.keys())

		for nid in self.annotation.keys():
			if 'labeled' in self.annotation[nid]:
				print(self.annotation[nid])
				self.annotation[nid]['labelled'] = self.annotation[nid]['labeled']
				del self.annotation[nid]['labeled']
			elif 'labelled' not in self.annotation[nid]:
				self.annotation[nid]['labelled'] = 0


		# label from osm (be carefull! this will override original annotation)

		#self.OSMLabelRoadType(only_not_labelled=True)
		#self.OSMLabelLanes()






		# overview

		self.raw_img = scipy.ndimage.imread(output_folder+"/sat_16384.png").astype(np.uint8)
		self.overview = scipy.misc.imresize(self.raw_img, (4096, 4096)).astype(np.uint8)
		self.region = region 
		
		self.clipboard = []
		
		
		for loc, node in roadNetwork.nodes.iteritems():
			# draw a circle

			x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, 4096,self.region)


			if node[0] not in annotation:
				color = (255,0,0)
			else:
				clabel = color_encode(annotation[node[0]])
				if annotation[node[0]]['remove'] == 1:
					color = (128,128,128)
				elif 'labelled' in annotation[node[0]] and annotation[node[0]]['labelled'] == 1:
					color = (0,255,0)
				else:
					color = (255,0,0)

				self.overview[x-9:x-5,y-10:y+10, :] = clabel

			
			cv2.circle(self.overview, (y,x), 3, color, -1)


		self.master = Tk()

		
		# annotation UI 
		
		self.sliding_window_x = 0
		self.sliding_window_y = 0 
		self.overview_size = 800


		if len(sys.argv)>3:
			self.active_node = int(sys.argv[3])
		else:
			self.active_node = 0  

		

		self.segments = GenerateRoadSegmentList(roadNetwork)

		self.seg_ptr = 0
		self.seg_ptr_inner = 0 





		self.active_node_x, self.active_node_y = get_image_coordinate(roadNetwork.nid2loc[0][0]/111111.0, roadNetwork.nid2loc[0][1]/111111.0,16384, self.region)


		self.overview_tk = ImageTk.PhotoImage(image = Image.fromarray(self.overview[self.sliding_window_x:self.sliding_window_x+self.overview_size,self.sliding_window_y:self.sliding_window_y+self.overview_size,:]))
		act_img =  Image.fromarray(self.raw_img[self.active_node_x-128:self.active_node_x+128,self.active_node_y-128:self.active_node_y+128,:])
		act_img = act_img.resize((512,512), Image.ANTIALIAS)
		self.active_image_tk = ImageTk.PhotoImage(image = act_img)
	 	#self.active_image_tk.zoom(2,2)


		self.canvas_width = 1400
		self.canvas_height = 800

		self.w = Canvas(self.master, width=self.canvas_width, height=self.canvas_height)

		self.overview_image_object = self.w.create_image(0,0,image=self.overview_tk, anchor=tkinter.NW)
		
		self.active_image_object = self.w.create_image(844,0,image=self.active_image_tk, anchor=tkinter.NW)

		self.roadNetwork = roadNetwork
		self.annotation = annotation

		self.heading_line = self.w.create_line(1100, 256, 1100, 256, fill="cyan", width=2)
		self.center_line = self.w.create_rectangle(1100-2, 256-2, 1100+2, 256+2, fill="cyan")
		self.center_line_left = self.w.create_rectangle(400-2, 400-2, 400+2, 400+2, fill="cyan")


		self.text_info_vis1 = self.w.create_text(1100, 550, text = "|park|bike|number of lane(s)|bike|park|conf.|remove|", font=("Purisa", 24))
		self.text_info_vis2 = self.w.create_text(1100, 580, text = "|   *  |   *  |      0 lane(s)    |  *   |  *   |  1  |  0   |", font=("Purisa", 24))
		self.text_info_vis3 = self.w.create_text(1100, 20, text = "|   *  |   *  |      0 lane(s)    |  *   |  *   |  1  |  0   |", font=("Purisa", 24), fill='yellow')
		self.text_info_vis4 = self.w.create_text(1100, 610, text = "RoadType:", font=("Purisa", 24))
		


		self.text_info1 = self.w.create_text(1100, 640, text = "Node ID 0\t Total 0\t Complete 0% ", font=("Purisa", 16))

		self.text_info2 = self.w.create_text(1100, 670, text = "1-6 Number of lanes   q,w,e,r,t --> transition", font=("Purisa", 16))
		self.text_info3 = self.w.create_text(1100, 700, text = "a,s,d,f --> left park/bike  right park/bike", font=("Purisa", 16))
		self.text_info4 = self.w.create_text(1100, 730, text = "c/v copy/paste setting, z unsure, x not consider", font=("Purisa", 16))
              

	 	self.w.bind("<1>", lambda event: self.w.focus_set())
	 	self.w.bind("<Key>", self.Key)

	 	self.annotation_file = annotation_file 

		self.w.pack()
		mainloop()


	def OSMLabelRoadType(self, only_not_labelled = True):

		for nid in self.annotation.keys():

			if self.annotation[nid]['labelled'] == 1 and only_not_labelled == True:
				continue

			road_info = self.roadNetwork.nodes[self.roadNetwork.nid2loc[nid]][1]

			if road_info['roadtype'] in ['residential', 'service'] and road_info['lane'] <= 1:
				self.annotation[nid]['roadtype'] = 0
			else:
				self.annotation[nid]['roadtype'] = 1


	def OSMLabelLanes(self,only_not_labelled = True):

		for nid in self.annotation.keys():
			if self.annotation[nid]['labelled'] == 1 and only_not_labelled == True:
				continue


			road_info = self.roadNetwork.nodes[self.roadNetwork.nid2loc[nid]][1]

			self.annotation[nid]['number_of_lane'] = max(1,road_info['lane'])
			







	def Key(self,event):

 		print("pressed", repr(event.char), event.char)

 		self.InitNode()
 		self.updateHeading()

 		loc = self.roadNetwork.nid2loc[self.active_node]
 		color = (0,255,0)
 		x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, 4096,self.region)

		cv2.circle(self.overview, (y,x), 3, color, -1)

		clabel = color_encode(self.annotation[self.active_node])
		self.overview[x-9:x-5,y-10:y+10, :] = clabel

 		if repr(event.char) == "','":
 			self.annotation[self.active_node]['labelled'] = 1
 			pickle.dump(self.annotation, open(self.annotation_file, "w"))
 			json.dump(self.annotation, open(self.annotation_file+".json", "w"), indent = 4)
 			self.PreviousNode()
 			
 			self.UpdateText()
 			return 

 		if repr(event.char) == "'.'":
 			self.annotation[self.active_node]['labelled'] = 1
 			pickle.dump(self.annotation, open(self.annotation_file, "w"))
 			json.dump(self.annotation, open(self.annotation_file+".json", "w"), indent = 4)
 			self.NextNode()
 			
 			self.UpdateText()
 			return 

 		ch = repr(event.char)
 		if ch == "'1'" or ch == "'2'" or ch == "'3'" or ch == "'4'" or ch == "'5'" or ch == "'6'":
 			self.annotation[self.active_node]['number_of_lane'] = int(ch.split("'")[1])
 			self.annotation[self.active_node]['transition'] = -1 
 			self.UpdateText()
 			return 

 		ch = ch.split("'")[1]


 		if ch == "[":

 			self.PreviousSegment()
 			self.UpdateText()


 		if ch == "]":

 			self.NextSegment()
 			self.UpdateText()


 		if ch == "c":
 			self.clipboard = [self.annotation[self.active_node]['number_of_lane'], 
 			self.annotation[self.active_node]['transition'],
 			self.annotation[self.active_node]['left_park'],
 			self.annotation[self.active_node]['right_park'],
 			self.annotation[self.active_node]['left_bike'],
 			self.annotation[self.active_node]['right_bike'],
 			self.annotation[self.active_node]['confidence'],
 			self.annotation[self.active_node]['remove']]

 			print("clipboard", self.clipboard)

 			return 

 		if ch == "v":
 			if len(self.clipboard) != 0:
 				self.annotation[self.active_node]['number_of_lane'] = self.clipboard[0]
	 			self.annotation[self.active_node]['transition']= self.clipboard[1]
	 			self.annotation[self.active_node]['left_park']= self.clipboard[2]
	 			self.annotation[self.active_node]['right_park']= self.clipboard[3]
	 			self.annotation[self.active_node]['left_bike']= self.clipboard[4]
	 			self.annotation[self.active_node]['right_bike']= self.clipboard[5]
	 			self.annotation[self.active_node]['confidence']= self.clipboard[6]
	 			self.annotation[self.active_node]['remove'] = self.clipboard[7]
	 			


 		if ch == "q" or ch == "w" or ch == "e" or ch == "r" or ch == "t":
 			self.annotation[self.active_node]['number_of_lane'] = -1 

 			if ch == "q":
 				self.annotation[self.active_node]['transition'] = 1.5
 			if ch == "w":
 				self.annotation[self.active_node]['transition'] = 2.5
 			if ch == "e":
 				self.annotation[self.active_node]['transition'] = 3.5
 			if ch == "r":
 				self.annotation[self.active_node]['transition'] = 4.5
 			if ch == "t":
 				self.annotation[self.active_node]['transition'] = 5.5

 			self.UpdateText()
 			return 


 		if ch == "a":
 			self.annotation[self.active_node]['left_park'] = 1 - self.annotation[self.active_node]['left_park']

 		if ch == "s":
 			self.annotation[self.active_node]['left_bike'] = 1 - self.annotation[self.active_node]['left_bike']

 		if ch == "f":
 			self.annotation[self.active_node]['right_park'] = 1 - self.annotation[self.active_node]['right_park']

 		if ch == "d":
 			self.annotation[self.active_node]['right_bike'] = 1 - self.annotation[self.active_node]['right_bike']


 		if ch == "b":
 			self.annotation[self.active_node]['roadtype'] = 1-self.annotation[self.active_node]['roadtype']


 		if ch == "x":
 			self.annotation[self.active_node]['remove'] = 1 - self.annotation[self.active_node]['remove']


 		if ch == "z":
 			self.annotation[self.active_node]['confidence'] = 1 - self.annotation[self.active_node]['confidence']

 		if ch == "m":
 			print("save image")
 			Image.fromarray(self.overview).save("overview.png")


 		self.UpdateText()
 		return 

 		

	def updateImage(self):
		self.sliding_window_x = max(self.active_node_x/4 - 400,0)
		self.sliding_window_y = max(self.active_node_y/4 - 400,0)


		print(self.sliding_window_x, self.sliding_window_y)

		self.overview_tk = ImageTk.PhotoImage(image = Image.fromarray(self.overview[self.sliding_window_x:self.sliding_window_x+self.overview_size,self.sliding_window_y:self.sliding_window_y+self.overview_size,:]))
		act_img =  Image.fromarray(self.raw_img[self.active_node_x-128:self.active_node_x+128,self.active_node_y-128:self.active_node_y+128,:])
		act_img = act_img.resize((512,512), Image.ANTIALIAS)
		self.active_image_tk = ImageTk.PhotoImage(image = act_img)
		#self.active_image_tk = ImageTk.PhotoImage(image = Image.fromarray(self.raw_img[self.active_node_x-128:self.active_node_x+128,self.active_node_y-128:self.active_node_y+128,:]))
 	
		#self.active_image_tk.zoom(2,2)

 		self.w.itemconfig(self.overview_image_object, image=self.overview_tk)
 		self.w.itemconfig(self.active_image_object, image=self.active_image_tk)





 	def updateHeading(self):

 		anid = self.active_node

 		heading_vector_lat = 0 
 		heading_vector_lon = 0



 		if len(self.roadNetwork.node_degree[anid]) > 2:
 			heading_vector_lat = 0 
 			heading_vector_lon = 0
 		elif len(self.roadNetwork.node_degree[anid]) == 1:
 			loc1 = self.roadNetwork.nid2loc[anid]
 			loc2 = self.roadNetwork.nid2loc[self.roadNetwork.node_degree[anid][0]]

 			dlat = loc1[0] - loc2[0]
 			dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]/111111.0))



 			l = np.sqrt(dlat*dlat + dlon * dlon)

 			dlat /= l
 			dlon /= l 

 			heading_vector_lat = dlat 
 			heading_vector_lon = dlon 
 		elif len(self.roadNetwork.node_degree[anid]) == 2:
 			loc1 = self.roadNetwork.nid2loc[self.roadNetwork.node_degree[anid][1]]
 			loc2 = self.roadNetwork.nid2loc[self.roadNetwork.node_degree[anid][0]]

 			dlat = loc1[0] - loc2[0]
 			dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]/111111.0))

 			l = np.sqrt(dlat*dlat + dlon * dlon)

 			dlat /= l
 			dlon /= l 

 			heading_vector_lat = dlat 
 			heading_vector_lon = dlon 


 		if anid in self.annotation:
 			self.annotation[anid]['heading_vector'] = (heading_vector_lat, heading_vector_lon)
 			self.annotation[anid]['degree'] = len(self.roadNetwork.node_degree[anid])
 		else:
 			self.annotation[anid] = {}
 			self.annotation[anid]['heading_vector'] = (heading_vector_lat, heading_vector_lon)
 			self.annotation[anid]['degree'] = len(self.roadNetwork.node_degree[anid])


 		print(anid, self.roadNetwork.node_degree[anid], heading_vector_lat, heading_vector_lon)
 		if self.w is not None:
 			self.w.coords(self.heading_line, 1100, 256, 1100+int(heading_vector_lon*50), 256-int(heading_vector_lat*50))


 	def InitNode(self):

 		anid = self.active_node
 		print("init node", anid)
 		if anid not in self.annotation:
 			self.annotation[anid] = {}

 			self.annotation[anid]['number_of_lane'] = 1 # 1,2,3,4,5,6 or -1
 			self.annotation[anid]['transition'] = -1 # 1.5, 2.5, 3.5, 4.5, 5.5 

 			# 0 or 1 
 			self.annotation[anid]['left_bike'] = 0
 			self.annotation[anid]['right_bike'] = 0

 			self.annotation[anid]['left_park'] = 0
 			self.annotation[anid]['right_park'] = 0


 			# confidence 
 			self.annotation[anid]['confidence'] = 1 

 			# not consider
 			self.annotation[anid]['remove'] = 0

 			self.annotation[anid]['labelled'] = 0 

 	def UpdateText(self):
 		anid = self.active_node

 		print(anid)

 		if self.annotation[anid]['number_of_lane'] != -1 :
 			number_of_lane_str = "      %d lane(s)   " % self.annotation[anid]['number_of_lane']
 		else:
 			number_of_lane_str = "  %d or %d lane(s) " % (int(self.annotation[anid]['transition']-0.5),int(self.annotation[anid]['transition']+0.5))

 		if self.annotation[anid]['left_bike'] == 0 :
 			left_bike_str = "      "
 		else:
 			left_bike_str = "   *  "

 		if self.annotation[anid]['left_park'] == 0 :
 			left_park_str = "      "
 		else:
 			left_park_str = "   *  "

 		if self.annotation[anid]['right_bike'] == 0 :
 			right_bike_str = "      "
 		else:
 			right_bike_str = "  *   "

 		if self.annotation[anid]['right_park'] == 0 :
 			right_park_str = "      "
 		else:
 			right_park_str = "  *   "


 		if self.annotation[anid]['confidence'] == 0 :
 			confidence_str = "  0  "
 		else:
 			confidence_str = "  1  "

 		if self.annotation[anid]['remove'] == 0 :
 			remove_str = "  0   "
 		else:
 			remove_str = "  1   "

 		if self.annotation[anid]['roadtype'] == 0:
 			roadtype = "residential/service"
 		else:
 			roadtype = "primary/highway"


 		self.w.itemconfig(self.text_info_vis4, text = "roadtype: "+roadtype)



 		info_str = "|%s|%s|%s|%s|%s|%s|%s|" % (left_park_str, left_bike_str, number_of_lane_str, right_bike_str, right_park_str, confidence_str, remove_str)


 		self.w.itemconfig(self.text_info_vis2, text=info_str)
 		self.w.itemconfig(self.text_info_vis3, text=info_str)


 		sub_ss = [x[1] for x in self.segments]

 		current_s = sum(sub_ss[:self.seg_ptr]) + self.seg_ptr_inner
 		total_s = sum(sub_ss)


 		self.w.itemconfig(self.text_info1, text = "Labelled %d  Total %d  Total2 %d  Complete %.1f %% " % (current_s, len(self.roadNetwork.nodes), total_s, float(current_s)/total_s*100.0))

 		


 		



	def NextNode(self, use_segments = True):
 		

 		if use_segments == True:
 			if self.seg_ptr_inner < len(self.segments[self.seg_ptr][0])-1:
 				self.seg_ptr_inner += 1
 			else:
 				if self.seg_ptr < len(self.segments)-1:
 					self.seg_ptr += 1
 					self.seg_ptr_inner = 0
 			print(self.seg_ptr, self.seg_ptr_inner)

 			self.active_node = self.segments[self.seg_ptr][0][self.seg_ptr_inner]
 		else:
	 		self.active_node += 1

			if self.active_node >= len(self.roadNetwork.nodes):
				self.active_node = len(self.roadNetwork.nodes)-1




		self.active_node_x, self.active_node_y = get_image_coordinate(self.roadNetwork.nid2loc[self.active_node][0]/111111.0, self.roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,self.region)
		self.InitNode()


		self.updateImage()
		self.updateHeading()



	def PreviousNode(self, use_segments = True):
		

		if use_segments == True:
 			if self.seg_ptr_inner > 0:
 				self.seg_ptr_inner -= 1
 			else:
 				if self.seg_ptr > 0:
 					self.seg_ptr -= 1
 					self.seg_ptr_inner = self.segments[self.seg_ptr][1]-1

 			print(self.seg_ptr, self.seg_ptr_inner)
 			
 			self.active_node = self.segments[self.seg_ptr][0][self.seg_ptr_inner]
 		else:
	 		self.active_node -= 1

			if self.active_node < 0:
				self.active_node = 0





		self.active_node_x, self.active_node_y = get_image_coordinate(self.roadNetwork.nid2loc[self.active_node][0]/111111.0, self.roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,self.region)
		self.InitNode()

		

		self.updateImage()
		self.updateHeading()

	def NextSegment(self):
 		
 		
		if self.seg_ptr < len(self.segments):
			self.seg_ptr += 1
			self.seg_ptr_inner = 0
 			print(self.seg_ptr, self.seg_ptr_inner)

 			self.active_node = self.segments[self.seg_ptr][0][self.seg_ptr_inner]
 		

		self.active_node_x, self.active_node_y = get_image_coordinate(self.roadNetwork.nid2loc[self.active_node][0]/111111.0, self.roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,self.region)
		self.InitNode()


		self.updateImage()
		self.updateHeading()



	def PreviousSegment(self, use_segments = True):
		

		
		if self.seg_ptr > 0:
			self.seg_ptr -= 1
			self.seg_ptr_inner = 0

 			print(self.seg_ptr, self.seg_ptr_inner)
 			
 			self.active_node = self.segments[self.seg_ptr][0][self.seg_ptr_inner]
 		




		self.active_node_x, self.active_node_y = get_image_coordinate(self.roadNetwork.nid2loc[self.active_node][0]/111111.0, self.roadNetwork.nid2loc[self.active_node][1]/111111.0,16384,self.region)
		self.InitNode()

		

		self.updateImage()
		self.updateHeading()





	pass


def generate_config_files(start_lat, start_lon, nlat, nlon, output_folder, size = 2048, res = 0.25, skip_lat = 0, skip_lon = 0):

	Popen("mkdir -p %s" % output_folder, shell=True).wait()

	lat_step = size/111111.0
	lon_step = size/111111.0 / math.cos(math.radians(start_lat))


	for i in range(skip_lat, nlat):
		for j in range(skip_lon, nlon):
			subfolder = output_folder + "/region_%d_%d" % (i,j)
			Popen("mkdir -p %s" % subfolder, shell=True).wait()

			config = {}

			config['region'] = [start_lat+i*lat_step, start_lon + j * lon_step, start_lat + (i+1)*lat_step, start_lon + (j+1)*lon_step]
			config['folder'] = subfolder
			config['size'] = int(size/res)

			json.dump(config, open(subfolder+"/config.json","w"))



if __name__ == "__main__":
	cmd = sys.argv[1]


	if cmd.startswith("config"):
		start_lat = float(sys.argv[2])
		start_lon = float(sys.argv[3])

		nlat = int(sys.argv[4])
		nlon = int(sys.argv[5])

		#start_lat = 42.336395
		#start_lon = -71.142805

		output_folder = sys.argv[6]

		generate_config_files(start_lat, start_lon, nlat, nlon, output_folder)


	elif cmd.startswith("annotate"):
		config = json.load(open(sys.argv[2],"r"))
		a = annotate_dataset(config)


		pass

	elif cmd.startswith("generate"):

		config = json.load(open(sys.argv[2],"r"))
		generate_dataset(config)


	elif cmd.startswith("tiles"):

		input_sat = "sat_16384.png"
		output_name = cmd


		items = cmd.split("_")
		if len(items) > 1:
			scale = float(items[1])
		else:
			scale = 1.0 

		print("Scale: ", scale)

		if len(sys.argv)>3:
			input_sat = sys.argv[3]
			output_name = sys.argv[4]

			generate_per_node_image(json.load(open(sys.argv[2],"r")), input_sat = input_sat, output_name = output_name, scale = scale)

		else:

			generate_per_node_image(json.load(open(sys.argv[2],"r")), scale = scale, output_name = cmd)


	elif cmd.startswith("osmauto"):

		annotate_dataset_osm(json.load(open(sys.argv[2],"r")))




	pass
