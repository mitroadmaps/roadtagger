from subprocess import Popen 
import numpy as np 
import scipy.ndimage 
import helper_road_structure as splfy
import helper_road_structure_topo as topo
from roadtagger_road_network import * 
import json
import math 
from PIL import Image 
import pickle
import sys 

	# load graph
	# convert it to a config file 
	# create tiles
	# 
	#  

	# cmd = "cd roadtagger; "
	# cmd += "python roadtagger_evaluate.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/boston -config dataset/boston_auto/region_0_1/config.json -r model/model_best"

	# Popen(cmd, shell=True).wait()

target_gsd = 0.125
start_lat = 70
start_lon = 10
imgfile = "../sample.png"
input_gsd = float(sys.argv[1]) 
graphfile = "../samplegraph.json"

region_folder = "region_tmp/"
output_image_name = region_folder + "sat.png"
output_roadnetwork = region_folder + "roadnetwork.p"
output_tile_folder = region_folder + "tiles/"
output_annotation = region_folder + "annotation.p"

Popen("mkdir -p " + region_folder, shell=True).wait()

# load image 
img = scipy.ndimage.imread(imgfile)
dim = np.shape(img)
if dim[2] == 4:
	img=img[:,:,0:3] # only RGB channel

original_dim = dim

region = [start_lat, start_lon, start_lat + dim[0]*input_gsd * 1.0 / 111111.0, start_lon + dim[1]*input_gsd * 1.0 / 111111.0 / math.cos(math.radians(start_lat)) ]
print(region)
newdim = (int(dim[1]/target_gsd*input_gsd), int(dim[0]/target_gsd*input_gsd))
img = cv2.resize(img, newdim)
#img[:,:,0], img[:,:,2] = img[:,:,2], img[:,:,0] 
Image.fromarray(img).save(output_image_name)

# load graph 
MyRoadGraph = splfy.RoadGraph()
RoadNetworkRoadTagger = RoadNetwork()

nodehash = {}
nid = 0
graphedge = json.load(open(graphfile))
for edge in graphedge:
	n1,n2 = edge 

	if tuple(n1) not in nodehash:
		nodehash[tuple(n1)] = nid 
		nid += 1

	if tuple(n2) not in nodehash:
		nodehash[tuple(n2)] = nid 
		nid += 1

	nid1 = nodehash[tuple(n1)]
	nid2 = nodehash[tuple(n2)]


	def xy2latlon(xy):
		lon = start_lon + xy[0] * input_gsd * 1.0/111111.0 / math.cos(math.radians(start_lat))
		lat = region[2] - xy[1] * input_gsd * 1.0/111111.0
		return lat, lon 
	
	lat1,lon1 = xy2latlon(n1)
	lat2,lon2 = xy2latlon(n2)

	MyRoadGraph.addEdge(nid1, lat1, lon1, nid2, lat2, lon2)

MyRoadGraph.ReverseDirectionLink()

for node in MyRoadGraph.nodes.keys():
	MyRoadGraph.nodeScore[node] = 100

for edge in MyRoadGraph.edges.keys():
	MyRoadGraph.edgeScore[edge] = 100


topo.TOPOGenerateStartingPoints(MyRoadGraph, density = 0.00020, region=region, image='NULL', check = False, direction = True, metaData = None, RoadNetworkCallback = RoadNetworkRoadTagger, margin = 0.0)

RoadNetworkRoadTagger.region = region 
RoadNetworkRoadTagger.image_file = output_image_name

RoadNetworkRoadTagger.DumpToFile(output_roadnetwork)


# create heading vectors (using a fake annotation structure)
annotation = {}
roadNetwork = RoadNetworkRoadTagger

for anid in roadNetwork.nid2loc.keys():
	annotation[anid] = {}
	
	# create some fake placeholder labels
	annotation[anid]['number_of_lane'] = 1
	annotation[anid]['roadtype'] = 1
	annotation[anid]['remove'] = 0
	annotation[anid]['labelled'] = 1
	annotation[anid]['left_bike'] = 0
	annotation[anid]['right_bike'] = 0
	annotation[anid]['left_park'] = 0
	annotation[anid]['right_park'] = 0
	annotation[anid]['confidence'] = 1 

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

pickle.dump(annotation, open(output_annotation, 'w'))

# create tiles
Popen("mkdir -p "+output_tile_folder, shell=True).wait()
dim = np.shape(img)

image = np.pad(img, ((512,512), (512,512),(0,0)), 'constant')

def get_image_coordinate(lat, lon, sizelat, sizelon, region):
	x = int((region[2]-lat)/(region[2]-region[0])*sizelat)
	y = int((lon-region[1])/(region[3]-region[1])*sizelon)

	return x,y 

scale = 1.0 

for k,v in annotation.iteritems():
	loc = roadNetwork.nid2loc[k]
	x,y = get_image_coordinate(loc[0]/111111.0, loc[1]/111111.0, dim[0], dim[1],region)

	print("generating tile-%d, at (%d,%d)" % (k,x,y))

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
	
	img = scipy.ndimage.interpolation.rotate(subimage, angle)

	center = np.shape(img)[0]/2

	r = 192 # 384
	result = img[center-r:center+r, center-r: center+ r,:]

	Image.fromarray(result).save(output_tile_folder+"/img_%d.png" % k)


# create config file
config = {}
config["region"] = region
config["folder"] = region_folder
config["size"] = -1

json.dump(config, open("config_tmp.json", "w"), indent=2)


# run inference
cmd = "mkdir output; python roadtagger_custom_inference.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/custom -config config_tmp.json -r model/model_best"
Popen(cmd, shell=True).wait()

# convert result 
result = json.load(open("output/custom.json"))

new_result = []
for item in result:
	lat,lon = item[0]
	lat /= 111111.0
	lon /= 111111.0

	x = int((lon - start_lon) / (region[3]-region[1]) * original_dim[1])
	y = int((region[2] - lat) / (region[2]-region[0]) * original_dim[0])

	newitem = {}
	newitem["coordinate"] = [x,y]
	newitem["lane count"] = item[1]
	newitem["road type"] = "primary" if item[2] == 1 else "residential"
	newitem["lane count probability"] = item[3]
	newitem["road type probability"] = item[4]
	newitem["node id"] = item[5]
	
	new_result.append(newitem)

json.dump(new_result, open("../sampleresult.json", "w"), indent=2)




