import argparse

from roadtagger_model import * 
from roadtagger_road_network import * 

import pickle
from subprocess import Popen
import sys
from time import time,sleep
import random
import json 

from PIL import Image 


parser = argparse.ArgumentParser()

parser.add_argument('-model_config', action='store', dest='model_config', type=str,
                    help='model config', required =True)

parser.add_argument('-save_folder', action='store', dest='save_folder', type=str,
                    help='save folder')

parser.add_argument('-r', action='store', dest='model_recover', type=str,
                    help='path to model for recovering', default=None)

parser.add_argument('-d', action='store', dest='data_folder', type=str,
                    help='dataset folder', default='/data/songtao/')


parser.add_argument('-run', action='store', dest='run_name', type=str,
                    help='run name', default='run1')


parser.add_argument('-lr', action='store', dest='learning_rate', type=float,
                    help='initial learning_rate', default=0.0001)


parser.add_argument('--noLeftRight', action='store', dest='noLeftRight', type=bool,
                    help='noLeftRight flag', default=True)

parser.add_argument('--converage', action='store', dest='converage', type=bool,
                    help='converage flag', default=False)

parser.add_argument('--step', action='store', dest='step', type=int,
                    help='initial step', default=0)

parser.add_argument('--step_max', action='store', dest='step_max', type=int,
                    help='max_step', default=300000)

parser.add_argument('--train_cnn_only', action='store', dest='train_cnn_only', type=bool,
                    help='train_cnn_only', default=False)

parser.add_argument('--cnn_model', action='store', dest='cnn_model', type=str,
                    help='cnn_model', default="simple")

parser.add_argument('--gnn_model', action='store', dest='gnn_model', type=str,
                    help='gnn_model', default="simple")

parser.add_argument('--use_batchnorm', action='store', dest='use_batchnorm', type=bool,
                    help='use_batchnorm', default=False)

parser.add_argument('--lr_drop_interval', action='store', dest='lr_drop_interval', type=int,
                    help='lr_drop_interval', default=60000)

parser.add_argument('-o', action='store', dest='output_prefix', type=str,
                    help='output_prefix', default="evaluate_result_")

parser.add_argument('-config', action='store', dest='config', type=str,
                    help='config', default="None", required=True)


parser.add_argument('-validation_file', action='store', dest='validation_file', type=str,
                    help='validation_file', default="validation.p", required=False)

parser.add_argument('-tiles_name', action='store', dest='tiles_name', type=str,
                    help='tiles_name', default="tiles", required=False)


args = parser.parse_args()

print(args)


Image.MAX_IMAGE_PIXELS = None

noLeftRight = True 


if __name__ == "__main__":
	

	dataset_folder = args.data_folder

	result_output = args.output_prefix 


	config = json.load(open(args.config,"r"))
	output_folder = dataset_folder + config["folder"]
	roadNetwork =  pickle.load(open(output_folder+"/roadnetwork.p", "r"))

	roadNetwork.loadAnnotation(args.config, osm_auto=True, root_folder=dataset_folder)
	
	if args.tiles_name == "tiles":
		roadNetwork.sat_image = scipy.misc.imresize(scipy.ndimage.imread(output_folder+"/sat_16384.png").astype(np.uint8), (4096, 4096))
	else:
		roadNetwork.sat_image = scipy.misc.imresize(scipy.ndimage.imread(output_folder+"/sat_16384.png").astype(np.uint8), (4096, 4096))

	osm_roadNetwork = pickle.load(open(output_folder+"/roadnetwork.p", "r"))
	osm_roadNetwork.loadAnnotation(args.config, osm_auto=True, root_folder=dataset_folder, force_osm = True)


	validation_set = []
	if args.validation_file is not None:

		try:
			validation_dict = pickle.load(open(args.validation_file,"r"))

			for k,v in validation_dict.iteritems():
				if k in args.config:
					validation_set = v 
					print("set validation set to ", k)
					print("size ", len(v))

		except:
			print("validation set not found")


	model_config = args.model_config

	cnn_type = "resnet18" # "simple"
	gnn_type = "simple" # "none"

	number_of_gnn_layer = 4
	remove_adjacent_matrix = 0

	parking_weight = 1
	biking_weight = 1 
	lane_weight = 1 
	type_weight = 1 


	parking_weight = 0
	biking_weight = 0 
	lane_weight = 1 
	type_weight = 1 
	GRU=False

	if model_config.startswith("simpleCNN+GNN"):
		gnn_type = args.gnn_model
		cnn_type = args.cnn_model
		sampleSize = 172
		sampleNum = 32
		stride = 32

		items = model_config.split("_")

		number_of_gnn_layer = int(items[1])
		remove_adjacent_matrix = int(items[2]) # 0 or 1 
	
		if len(items) > 3:
			lane_weight = int(items[3])
			type_weight = int(items[4])

		if len(items) > 5:
			biking_weight = int(items[5])


		if model_config.startswith("simpleCNN+GNNGRU"):
			print("Use GRU")
			GRU = True 

	if model_config.startswith("simpleCNNonly"):

		gnn_type = "none"
		cnn_type = "simple"

		cnn_type = args.cnn_model

		print(model_config, cnn_type)

		sampleSize = 128
		sampleNum = 32
		stride = 0

		items = model_config.split("_")

		if len(items) > 2:
			lane_weight = int(items[1])
			type_weight = int(items[2])

		if len(items) > 3:
			biking_weight = int(items[3])

	# if model_config == "resnet18+GNN":
	# 	gnn_type = "simple"
	# 	cnn_type = "resnet18"
	# 	sampleSize = 160
	# 	sampleNum = 32
	# 	stride = 32

	# if model_config == "resnet18only":
	# 	gnn_type = "none"
	# 	cnn_type = "resnet18"
	# 	sampleSize = 8
	# 	sampleNum = 256
	# 	stride = 0


	random.seed(123)

	config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )


	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	#with tf.Session(config=config) as sess:

	input_region = SubRoadNetwork(roadNetwork, tiles_name = args.tiles_name,  graph_size = 6000, search_mode = 0, augmentation=False, partial = True, remove_adjacent_matrix = remove_adjacent_matrix, reseed = True )
	
	random.seed(123)
	osm_input_region = SubRoadNetwork(osm_roadNetwork, no_image = True, graph_size = 6000, search_mode = 0, augmentation=False, partial = True, remove_adjacent_matrix = remove_adjacent_matrix, reseed = True )
	

	# check whether the two map match 
	print(input_region.subGraphNoadList[:16])
	print(osm_input_region.subGraphNoadList[:16])

	if args.train_cnn_only == True:
		node_feature = np.zeros((input_region.nonIntersectionNodeNum, 16))
	else:
		node_feature = np.zeros((input_region.nonIntersectionNodeNum, 62))



	print("Stage1")
	time_cnn = 0 

	# stage 1 
	with tf.Session() as sess:
		model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type,lane_number_weight = lane_weight, roadtype_weight = type_weight, parking_weight = 0.0, biking_weight = biking_weight, number_of_gnn_layer = number_of_gnn_layer, GRU=GRU, noLeftRight =noLeftRight, use_batchnorm = args.use_batchnorm)
		model.restoreModel(args.model_recover)
		if model.dumpWeights()==True:
			pass
		else:
			print("load model failed, nan encountered!!! try reloading the model!!!!! ")

		batch_size = 256 

		st = 0

		while st < input_region.nonIntersectionNodeNum:
			print(st)
			ed = min(st+batch_size, input_region.nonIntersectionNodeNum)

			t0 = time() 
			node_feature[st:ed,:] = model.GetIntermediateNodeFeature(input_region, st, ed)[0]
			time_cnn += time() - t0 

			st += batch_size

	print("Stage2")
	# stage 2
	tf.reset_default_graph()
	time_gnn = 0 

	with tf.Session() as sess:
		

		model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type,lane_number_weight = lane_weight, roadtype_weight = type_weight, parking_weight = 0.0, biking_weight = biking_weight, number_of_gnn_layer = number_of_gnn_layer, GRU=GRU, noLeftRight =noLeftRight, use_batchnorm = args.use_batchnorm, stage = 2)
		model.restoreModel(args.model_recover)
		if model.dumpWeights()==True:
			pass
		else:
			print("load model failed, nan encountered!!! try reloading the model!!!!! ")

		t0 = time() 
		outputs = model.EvaluateWithIntermediateNodeFeature(input_region, node_feature)
		time_gnn = time() - t0 

		rec_result = osm_input_region.RecommendationAccNode(outputs, input_region, validation_set=validation_set)


		input_region.validation_set = validation_set
		input_region.VisualizeOutput(outputs,result_output+"wholeregion_output.png")
		input_region.VisualizeOutputRoadType(outputs,result_output+"wholeregion_output_roadtype.png")
		input_region.VisualizeOutputLane(outputs,result_output+"wholeregion_output_lane.png")
		input_region.VisualizeAdjacent(outputs,result_output+"wholeregion_output_adj.png")
		input_region.VisualizeSegmentAccuracy(outputs,result_output+"wholeregion_output_segacc.png")
		input_region.VisualizeRoadSegment(outputs,result_output+"wholeregion_output_seg.png")


		pickle.dump(outputs, open(result_output+"output.p", "w"))

		result1 = input_region.GetAccuracyStatistic(outputs, dump=True, validation_set = validation_set)

		
		time_mrf = 0 
		t0 = time() 
		outputs = input_region.PostProcessWithMRF(outputs,ind=1, labels=6, weight=6, norm = 1.0)
		outputs = input_region.PostProcessWithMRF(outputs,ind=6, labels=2, weight=8, norm = 1.0)
		time_mrf = time() - t0 


		input_region.VisualizeOutputRoadType(outputs,result_output+"wholeregion_output_roadtype_mrf.png")
		input_region.VisualizeOutputLane(outputs,result_output+"wholeregion_output_lane_mrf.png")
		
		result2 = input_region.GetAccuracyStatistic(outputs, dump=True, validation_set = validation_set)


		outputs = pickle.load(open(result_output+"output.p", "r"))

		outputs_smooth = input_region.SegmentSmoothSlidingWindow(outputs)

		input_region.VisualizeOutputLane(outputs_smooth, result_output+"wholeregion_output_micro_lane_smooth.png")
		
		result3 = input_region.GetAccuracyStatistic(outputs_smooth, dump=True, validation_set = validation_set)

		json.dump( [result1, result2, result3, rec_result], open(result_output+"result.json", "w"), indent=4)


		print("Time CNN", time_cnn)
		print("Time GNN", time_gnn)
		print("Time MRF", time_mrf)


		


			


















			
