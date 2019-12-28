import argparse
from roadtagger_model import * 
from roadtagger_road_network import * 

import pickle
from subprocess import Popen
import sys
from time import time,sleep
import random
import json 
import os 
import threading

from PIL import Image 


# AAAI-20 reproduce version

# time python roadtagger_train.py \
# -model_config simpleCNN+GNNGRU_8_0_1_1 \
# -save_folder folder/prefix \
# -d folder/ \
# -run run1 \
# -lr 0.0001  \
# --step_max 300000  \
# --cnn_model simple2 \
# --gnn_model RBDplusRawplusAux \
# --use_batchnorm True \
# --lr_drop_interval 30000 \
# --homogeneous_loss_factor 3.0 \
# --dataset config/dataset_180tiles.json


parser = argparse.ArgumentParser()

parser.add_argument('-model_config', action='store', dest='model_config', type=str,
                    help='model config', required =True)

parser.add_argument('-save_folder', action='store', dest='save_folder', type=str,
                    help='save folder', required =True)

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

parser.add_argument('--train_gnn_only', action='store', dest='train_gnn_only', type=bool,
                    help='train_gnn_only', default=False)


parser.add_argument('--lr_drop_interval', action='store', dest='lr_drop_interval', type=int,
                    help='lr_drop_interval', default=25000)


parser.add_argument('--use_node_drop', action='store', dest='use_node_drop', type=str,
                    help='use_node_drop', default="True")

parser.add_argument('--use_homogeneous_loss', action='store', dest='use_homogeneous_loss', type=bool,
                    help='use_homogeneous_loss', default=True)

parser.add_argument('--homogeneous_loss_factor', action='store', dest='homogeneous_loss_factor', type=float,
                    help='homogeneous_loss_factor', default=0.0)

parser.add_argument('--hst', action='store', dest='hst', type=int,
                    help='hst 1 2 3', default=0)

parser.add_argument('--crosscity', action='store', dest='crosscity', type=int,
                    help='crosscity 1 2 3', default=0)

parser.add_argument('--dataset', action='store', dest = 'dataset', type=str, help = 'dataset json file', default = None)

parser.add_argument('-tiles_name', action='store', dest='tiles_name', type=str,
                    help='tiles_name', default="tiles", required=False)


args = parser.parse_args()

print(args)

Image.MAX_IMAGE_PIXELS = None

noLeftRight = args.noLeftRight
converage = args.converage

if __name__ == "__main__":
	random.seed(123)
	learning_rate = args.learning_rate
	model_config = args.model_config

	suffix  = ""

	if args.use_batchnorm == True:
		suffix += "_batchnorm_"

	if args.use_node_drop == "True":
		args.use_node_drop = True
		suffix += "_use_drop_node_"
	else:
		args.use_node_drop = False 

	if args.train_gnn_only == True:
		suffix += "_trainGNNOnly_"

	suffix += "_tiletype_" + args.tiles_name+"_"


	suffix += "_hlf_" + str(int(args.homogeneous_loss_factor*10)) + "_"


	suffix += "_gnntype_" + args.gnn_model +"_"
	suffix += "_cnntype_" + args.cnn_model +"_"


	print(args.dataset, "dataset", dataset_cfg)

	if args.train_cnn_only == True:
		model_folder = args.save_folder + model_config + suffix + "cnnonly_"+ args.cnn_model
	else:
		model_folder = args.save_folder + model_config + suffix

	run = model_folder+"_"+args.run_name+"_"+str(learning_rate)

	log_folder = "alllogs/log"

	#learning_rate = float(sys.argv[5])

	step = args.step 


	Popen("mkdir -p %s" % model_folder, shell=True).wait()
	Popen("mkdir -p %s" % (model_folder+"/validation"), shell=True).wait()
	Popen("mkdir -p %s" % log_folder, shell=True).wait()

	Popen("mkdir -p samples", shell=True).wait()
	Popen("mkdir -p test", shell=True).wait()
	Popen("mkdir -p validation", shell=True).wait()

	
	cnn_type = "resnet18" # "simple"
	gnn_type = "simple" # "none"

	# default
	number_of_gnn_layer = 4
	remove_adjacent_matrix = 0

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


	# setup the dataset 
	training_configs = []
	validation_configs = []

	dataset_folder = args.data_folder
	if args.dataset is not None:
		dataset_cfg = json.load(open(args.dataset, "r"))
	else:
		dataset_cfg = []

	for city in dataset_cfg:
		nlat = city["lat_n"]
		nlon = city["lon_n"]
		city_name = city["cityname"]
		print(dataset_folder + city_name)

		for i in xrange(nlat):
			for j in xrange(nlon):
				if i == 0 and j == 1 and city_name in ['dataset/boston_auto','dataset/chicago_auto','dataset/dc_auto','dataset/seattle_auto']:
					validation_configs.append(dataset_folder+"%s/region_%d_%d/config.json" % (city_name,i,j))
				else:
					training_configs.append(dataset_folder+"%s/region_%d_%d/config.json" % (city_name,i,j))

	factor = 0.99
	if model_config == "resnet18only":
		factor = 0.2

	#gpu_options = tf.GPUOptions()


	best_testing_acc = 0
	best_testing_acc_last_step = 0 

	best_testing_loss = 1000.0
	best_testing_loss_last_step = 0 

	best_training_loss = 1000.0 
	best_training_loss_last_step = 0

	best_testing_lane_acc = 0
	best_testing_lane_acc_last_step = 0 

	best_testing_roadtype_acc = 0
	best_testing_roadtype_acc_last_step = 0 



	with tf.Session(config=tf.ConfigProto()) as sess:
		model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type,lane_number_weight = lane_weight, roadtype_weight = type_weight, parking_weight = 0.0, biking_weight = biking_weight, number_of_gnn_layer = number_of_gnn_layer, GRU=GRU, noLeftRight =noLeftRight, use_batchnorm = args.use_batchnorm, homogeneous_loss_factor = args.homogeneous_loss_factor)

		if args.model_recover is not None:
			model.restoreModel(args.model_recover)

		# loading dataset 
		training_networks = []
		lane_numbers = [100]*6

		for config_file in training_configs:
			print(config_file)
			config = json.load(open(config_file,"r"))
			output_folder = config["folder"]
			roadNetwork =  pickle.load(open(dataset_folder+output_folder+"/roadnetwork.p", "r"))

			lane_numbers,_ = roadNetwork.loadAnnotation(config_file, osm_auto=True, lane_numbers=lane_numbers, root_folder = dataset_folder)
			
			# if os.path.isfile(dataset_folder+output_folder+"/sat_4096.png"):
			# 	roadNetwork.sat_image = scipy.ndimage.imread(dataset_folder+output_folder+"/sat_4096.png").astype(np.uint8)
			# else:
			# 	roadNetwork.sat_image = scipy.misc.imresize(scipy.ndimage.imread(dataset_folder+output_folder+"/sat_16384.png").astype(np.uint8), (4096, 4096))
			# 	Image.fromarray(roadNetwork.sat_image).save(dataset_folder+output_folder+"/sat_4096.png")

			training_networks.append(roadNetwork)

		for config_file in validation_configs:
			print(config_file)
			config = json.load(open(config_file,"r"))
			output_folder = config["folder"]
			roadNetwork =  pickle.load(open(dataset_folder+output_folder+"/roadnetwork.p", "r"))

			lane_numbers,_ = roadNetwork.loadAnnotation(config_file, osm_auto=True, lane_numbers=lane_numbers, root_folder = dataset_folder)
			
			# if os.path.isfile(dataset_folder+output_folder+"/sat_4096.png"):
			# 	roadNetwork.sat_image = scipy.ndimage.imread(dataset_folder+output_folder+"/sat_4096.png").astype(np.uint8)
			# else:
			# 	roadNetwork.sat_image = scipy.misc.imresize(scipy.ndimage.imread(dataset_folder+output_folder+"/sat_16384.png").astype(np.uint8), (4096, 4096))
			# 	Image.fromarray(roadNetwork.sat_image).save(dataset_folder+output_folder+"/sat_4096.png")

			# this is not a bug, use training_networks here and pop later.
			training_networks.append(roadNetwork)



		s = np.sum(lane_numbers)
		lane_balance = [float(x)/s + 0.03 for x in lane_numbers]

		print(lane_balance)

		pod = 1.0 
		for l in lane_balance:
			pod *= l 


		lane_balance_factor = [l/x  for x in lane_balance]

		s = np.sum(lane_balance_factor)

		lane_balance_factor = [x/s*6 for x in lane_balance_factor]
		# disable this
		lane_balance_factor = [1.0 for x in lane_balance_factor]


		testing_network = {}
		testing_data = {}
		testing_data_all = []

		validation_set = {}

		for city_name in ['seattle','dc','chicago','boston']:
			validation_node_list = []

		#for city_name in ['boston']:
			testing_network[city_name] = training_networks.pop()  # region_0_1  in Boston

			testing_data[city_name] = []

			test_sampleSize = 172

			test_sampleNum = 8

			for i in range(test_sampleNum):
				if i % 2 == 0:
					must_have_lane = True 
				else:
					must_have_lane = False 

				subnet = SubRoadNetwork(testing_network[city_name], tiles_name = args.tiles_name, graph_size = test_sampleSize+32*random.randint(0,2), search_mode = random.randint(0,3), partial = True, remove_adjacent_matrix =remove_adjacent_matrix, output_folder = dataset_folder, lane_balance_factor=lane_balance_factor, must_have_lane = must_have_lane)

				testing_data[city_name].append(subnet)

				for nid in subnet.node_mapping.keys():
					if nid not in validation_node_list:
						validation_node_list.append(nid)

			print(city_name, "validation size", len(validation_node_list))

			validation_set[city_name] = validation_node_list
			testing_data_all += testing_data[city_name]


		pickle.dump(validation_set, open("validation.p","w"))


		writer = tf.summary.FileWriter(log_folder+"/"+run, sess.graph)

		Popen("pkill tensorboard", shell=True).wait()
		sleep(1)
		print("tensorboard --logdir=%s"%log_folder)
		logserver = Popen("tensorboard --logdir=%s"%log_folder, shell=True)



		s_loss = 0
		s_loss_details = [0]*6
		s_train_homogeneous_loss = 0
		#learning_rate = 0.00003

		#learning_rate = 0.00001

		#learning_rate = float(sys.argv[5])

		training_data = []

		if biking_weight == 1:
			train_op = np.random.choice([model.train_lane_op, model.train_type_op, model.train_bike_op, model.train_op])
		else:

			op_list = [model.train_op]
			if lane_weight == 1:
				op_list.append(model.train_lane_op)

			if type_weight == 1:
				op_list.append(model.train_type_op)

			train_op = np.random.choice(op_list)


		def loadTrainingDataAsyncBlock(training_data, st,ed):
			for i in range(st,ed):
				must_have_lane = True

				if i % 10> 5:
					must_have_lane = False

				batch_size = 64

				if args.cnn_model == "resnet18":
					batch_size = 32

				must_have_lane_change = False 

				if random.random() < 0.5:
					must_have_lane_change = True 


				tnid = random.randint(0,len(training_networks)-1)

				if must_have_lane_change:
					while len(training_networks[tnid].node_with_lane_change) == 0:
						tnid = random.randint(0,len(training_networks)-1)


				# old 
				# if tnid >= 32
				# 	must_have_lane = True 

				# 

				t1 = time()
				training_data.append(SubRoadNetwork(training_networks[tnid], must_have_lane_change = must_have_lane_change, tiles_name = args.tiles_name, train_cnn_only=args.train_cnn_only, train_cnn_batch=batch_size, train_cnn_preload=256,  graph_size = sampleSize+stride*random.randint(0,2), search_mode = random.randint(0,3), partial = True, noLeftRight =noLeftRight, remove_adjacent_matrix =remove_adjacent_matrix, output_folder = dataset_folder, lane_balance_factor=lane_balance_factor, must_have_lane =must_have_lane))
				print("time spent loading subgraph", time()-t1)



		def loadTrainingDataAsync(training_data):
			p_num = 4
			#training_data = []
			t0 = time()
				

			td = [[] for x in range(p_num)]

			procs = [threading.Thread(target = loadTrainingDataAsyncBlock, args = [td[i], i*(sampleNum/p_num), (i+1)*(sampleNum/p_num)]) for i in range(p_num)]

			for i in range(p_num):
				procs[i].start()

			for i in range(p_num):
				procs[i].join()

			for i in range(p_num):
				training_data += td[i]


			print("Time spent loading data", time() - t0)

			#return training_data

		training_data_bk = []
		loadingThread = threading.Thread(target=loadTrainingDataAsync, args = [training_data_bk])
		loadingThread.start() 

		t500 = time()

		while True:
			if step == args.step_max + 1:
				break

			if step % args.lr_drop_interval == 0 and step != 0:
				learning_rate /= 3 

			if converage == True:
				if step % 10000 == 0:
					learning_rate/=5
					print(learning_rate)

			if step % 500 == 0:
				loadingThread.join()

				training_data = training_data_bk

				training_data_bk = list()
				loadingThread = threading.Thread(target=loadTrainingDataAsync, args = [training_data_bk])
				loadingThread.start() 

				print("wall time", time() - t500)
				t500 = time()

			if step < 20000:
				if step % 5000 == 0 and step != 0:
					model.saveModel(model_folder+"/model%d"%step)
					model.saveCNNModel(model_folder+"/cnn_only_model%d"%step)
			else:
				if step % 5000 == 0 and step != 0:
					model.saveModel(model_folder+"/model%d"%step)
					model.saveCNNModel(model_folder+"/cnn_only_model%d"%step)
				

			loss_details = [0] * 6
			training_subgraph = training_data[random.randint(0,len(training_data)-1)]

			if args.train_cnn_only == True:
				training_subgraph.RandomBatchST()

			if step % 10 == 0:
				if biking_weight == 1:
					train_op = np.random.choice([model.train_lane_op, model.train_type_op, model.train_bike_op, model.train_op])
				else:

					op_list = [model.train_op]
					if lane_weight == 1:
						op_list.append(model.train_lane_op)

					if type_weight == 1:
						op_list.append(model.train_type_op)

					train_op = np.random.choice(op_list)

			batch_size = 64

			if args.train_cnn_only == False:
				batch_size = None 

			loss, output_lane_number, output_road_type, loss_details[0], loss_details[1], loss_details[2], loss_details[3], loss_details[4], loss_details[5],_, train_homogeneous_loss= model.Train(training_subgraph, learning_rate = learning_rate, train_op = train_op, batch_size = batch_size, use_drop_node = args.use_node_drop, train_gnn_only = args.train_gnn_only)
			
			if step % 10 == 0 and args.train_cnn_only == False:
				training_subgraph.UpdateConsistency(output_lane_number, output_road_type)

			s_loss += loss
			s_train_homogeneous_loss += train_homogeneous_loss

			for x in range(6):
				s_loss_details[x] += loss_details[x]

			step += 1

			if step % 200 == 0:
				for x in range(6):
					s_loss_details[x] /= 200

				
				test_loss = 0
				test_acc = 0
				test_acc_road = 0
				test_acc_lane = 0 
				test_acc_biking = 0 
				test_homogeneous_loss = 0

				statistic_result = None 
				ic = 0
				for test_region in testing_data_all:
					outputs = model.Evaluate(test_region)
					if step % 5000 == 0:
						#test_region.VisualizeOutput(outputs,model_folder+"/validation/test_graph%d_output.png" % ic, draw_biking=False, draw_parking = False)
						test_region.VisualizeOutput(outputs,"validation/test_graph%d_output.png" % ic, draw_biking=False, draw_parking = False)

					test_loss += outputs[0]
					test_homogeneous_loss += outputs[8]

					if ic == len(testing_data_all)-1 and step % 1000 == 0:
						dump = True 
					else:
						dump = False 
					test_acc_, statistic_result = test_region.GetAccuracyStatistic(outputs, statistic_result, dump=dump)
					
					if biking_weight == 0:
						test_acc = (test_acc_[1] + test_acc_[2])/2
					else:
						test_acc = (test_acc_[1] + test_acc_[2] + test_acc_[3])/3


					test_acc_road = test_acc_[1]
					test_acc_lane = test_acc_[2]
					test_acc_biking = test_acc_[3]

					ic = ic + 1


				test_loss/= len(testing_data_all)
				test_homogeneous_loss/= len(testing_data_all)
				

				if test_acc > best_testing_acc and step - best_testing_acc_last_step > 1000:
					print("New Best Model (testing acc) ", step, "acc", test_acc)

					model.saveModelBest(model.saver_best1, model_folder+"/best_model%d_%d"%(step, int(test_acc*1000)) )
					model.saveCNNModel(model_folder+"/cnn_only_best_model%d_%d"%(step, int(test_acc*1000)) )

					best_testing_acc = test_acc 
					best_testing_acc_last_step = step 

				train_loss = s_loss/200 + s_train_homogeneous_loss/200

				if train_loss < best_training_loss and step - best_training_loss_last_step > 1000 and step > 20000:
					print("New Best Model (training)", step, train_loss)

					model.saveModelBest(model.saver_best2, model_folder+"/best_model_training_%d_%d"%(step, int(train_loss*1000)) )
					
					best_training_loss = train_loss 
					best_training_loss_last_step = step 



				if test_loss< best_testing_loss and step - best_testing_loss_last_step > 1000 and step > 10000:
					print("New Best Model (testing)", step, test_loss)

					model.saveModelBest(model.saver_best3, model_folder+"/best_model_testing_%d_%d"%(step, int(test_loss*1000)) )
					
					best_testing_loss = test_loss
					best_testing_loss_last_step = step 


				if test_acc_road > best_testing_roadtype_acc and step - best_testing_roadtype_acc_last_step > 1000:
					print("New Best Model (test roadtype acc)", step, test_acc_road)

					model.saveModelBest(model.saver_best4, model_folder+"/best_model_roadtype_test_acc_%d_%d"%(step, int(test_acc_road*1000)) )
					
					best_testing_roadtype_acc = test_acc_road
					best_testing_roadtype_acc_last_step = step 


				if test_acc_lane > best_testing_lane_acc and step - best_testing_lane_acc_last_step > 1000:
					print("New Best Model (test lane acc)", step, test_acc_lane)

					model.saveModelBest(model.saver_best5, model_folder+"/best_model_lane_test_acc_%d_%d"%(step, int(test_acc_lane*1000)) )
					
					best_testing_lane_acc = test_acc_lane
					best_testing_lane_acc_last_step = step 


				print("train", step, test_acc, test_acc_road, test_acc_lane, test_acc_biking, test_loss, s_loss/200, s_train_homogeneous_loss/200, train_loss, s_loss_details)



				summary = model.addLog(test_loss, s_loss/200, total_train_loss = train_loss, train_homogeneous_loss=s_train_homogeneous_loss/200, test_homogeneous_loss= test_homogeneous_loss, test_acc_lane = test_acc_lane, test_acc_road_type = test_acc_road, test_acc_overall = test_acc, test_acc_biking = test_acc_biking)	
				writer.add_summary(summary, step)
				for x in range(6):
					s_loss_details[x] = 0 

				s_loss = 0
				s_train_homogeneous_loss = 0
