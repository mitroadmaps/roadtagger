# RoadTagger

RoadTagger Paper: https://arxiv.org/abs/1912.12408

**Abstract:**

Inferring road attributes such as lane count and road type from satellite imagery is challenging. Often, due to the occlusion in satellite imagery and the spatial correlation of road attributes, a road attribute at one position on a road may only be apparent when considering far-away segments of the road. Thus, to robustly infer road attributes, the model must integrate scattered information and capture the spatial correlation of features along roads. Existing solutions that rely on image classifiers fail to capture this correlation, resulting in poor accuracy. We find this failure is caused by a fundamental limitation -- the limited effective receptive field of image classifiers. To overcome this limitation, we propose RoadTagger, an end-to-end architecture which combines both Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) to infer road attributes. The usage of graph neural networks allows information propagation on the road network graph and eliminates the receptive field limitation of image classifiers. We evaluate RoadTagger on both a large real-world dataset covering 688 km^2 area in 20 U.S. cities and a synthesized micro-dataset. In the evaluation, RoadTagger improves inference accuracy over the CNN image classifier based approaches. RoadTagger also demonstrates strong robustness against different disruptions in the satellite imagery and the ability to learn complicated inductive rules for aggregating scattered information along the road network.


# About this Repository 

This repository consists of the source code of the [paper](https://arxiv.org/abs/1912.12408) and a [generic version](https://github.com/mitroadmaps/roadtagger/tree/master/generic_version) of the RoadTagger model. The original source code of RoadTagger is in the [pre_alpha_clean_version](https://github.com/mitroadmaps/roadtagger/tree/master/pre_alpha_clean_version) folder.

**IMPORTANT** As this is not yet a stable version, the code in the [pre_alpha_clean_version](https://github.com/mitroadmaps/roadtagger/tree/master/pre_alpha_clean_version) folder may only be used as reference - setting up your own data processing/training/evaluation code followed by the description in the [paper](https://arxiv.org/abs/1912.12408) and borrowing some code snippets from this repo would potentially save your time. :-) 

If you just want to play with the pretrained model and reproduce the numbers in the paper, you can checkout the instruction [here](#model). The code for reproduction has been tested. 


## Change Log 

**2020-07-23**
--------------------
- Fix bugs in the instruction for the pre-alpha clean version. Should use roadtagger_create_dataset_script.py to download and preprocess the dataset rather than roadtagger_generate_dataset.py . 

**2020-03-31**
--------------------
- Add download [link](https://mapster.csail.mit.edu/data/roadtagger/model.zip) for the pretrained model.
- Add download [link](https://mapster.csail.mit.edu/data/roadtagger/dataset_testing.zip) for the testing dataset and [link](https://mapster.csail.mit.edu/data/roadtagger/validation.p) for dataset partition.
- Add reproduce [instruction](#model). Now, you can reproduce the numbers in the paper with the pretrained model and the testing dataset.

**2020-01-30**
--------------------
- The [generic version](https://github.com/mitroadmaps/roadtagger/tree/master/generic_version) is working now (as an example).
- Add example graph loader, [roadtagger_generic_graph_loader.py](https://github.com/mitroadmaps/roadtagger/blob/master/generic_version/roadtagger_generic_graph_loader.py)
- Add example training code, [roadtagger_generic_train.py](https://github.com/mitroadmaps/roadtagger/blob/master/generic_version/roadtagger_generic_train.py)
- Add example evaluation code, [roadtagger_generic_eval.py](https://github.com/mitroadmaps/roadtagger/blob/master/generic_version/roadtagger_generic_eval.py)
- Add example config file, [lightpoles_example.json](https://github.com/mitroadmaps/roadtagger/blob/master/generic_version/configs/lightpoles_example.json)
- Fixed bugs in the [pre-alpha clean version](https://github.com/mitroadmaps/roadtagger/tree/master/pre_alpha_clean_version). 



## Instruction for the [generic version](https://github.com/mitroadmaps/roadtagger/tree/master/generic_version) (Recommended for Quick Start)

You can check out the example code for graph loading, training and evaluation in folder [generic_version/](https://github.com/mitroadmaps/roadtagger/tree/master/generic_version).

But for now, you may have to preprocess your training data according to the [example graph loader](https://github.com/mitroadmaps/roadtagger/blob/master/generic_version/roadtagger_generic_graph_loader.py). More detailed instructions are coming soon.


## Instruction for the [pre-alpha clean version](https://github.com/mitroadmaps/roadtagger/tree/master/pre_alpha_clean_version)
### Download the Dataset

Step-1:

In 'pre_alpha_clean_version' folder, 

**Edit line 43 in helper_mapdriver.py. Enter your Google API Key there.** You have to use your [Google Maps Static API key](https://developers.google.com/maps/documentation/maps-static/overview) to download the dataset. 

```
python roadtagger_create_dataset_script.py config/dataset_180tiles.json 
```

This script will download the dataset with 20 cities into the 'dataset' folder. The dataset is very large. You may need at least **250GB** free space to download the dataset. 

Downloading the images from Google is not free. But there is a free $200 monthly credit which is enough to cover this dataset.


**BEFORE** running this script, it would be highly recommended to try the sample dataset first to make sure there is no runtime issues. (2GB space needed)

```
python roadtagger_create_dataset_script.py config/dataset_sample.json 
```

The code automatically download OpenStreetMap data into the 'tmp' folder. Because the OpenStreetMap is changing all the time, we put the snapshots of four regions in the 'tmp' folder. The manual annotation in Step-2 **ONLY** matches these snapshots. 


Step-2:

Add the manual annotation.

```
cp annotation/boston_region_0_1_annotation.p dataset/boston_auto/region_0_1/annotation.p
cp annotation/chicago_region_0_1_annotation.p dataset/chicago_auto/region_0_1/annotation.p
cp annotation/dc_region_0_1_annotation.p dataset/dc_auto/region_0_1/annotation.p
cp annotation/seattle_region_0_1_annotation.p dataset/seattle_auto/region_0_1/annotation.p
```

### Train RoadTagger Model

Use the following code to train the model. 

```
time python roadtagger_train.py \
-model_config simpleCNN+GNNGRU_8_0_1_1 \
-save_folder model/prefix \
-d `pwd` \
-run run1 \
-lr 0.0001  \
--step_max 300000  \
--cnn_model simple2 \
--gnn_model RBDplusRawplusAux \
--use_batchnorm True \
--lr_drop_interval 30000 \
--homogeneous_loss_factor 3.0 \
--dataset config/dataset_180tiles.json
```

Here, the model is a CNN + GNN model. The GNN uses Gated Recurrent Unit (GRU). The propagation step of the graph neural network is 8. The graph neural network uses the graph structures that consists of the Road Bidirectional Decomposition graph, the original road graph and the auxiliary graph for parallel roads.  


### Evaluate the Model 

E.g., Boston. 

```
time python roadtagger_evaluate.py \
-model_config simpleCNN+GNNGRU_8_0_1_1 \
-d `pwd` \
--cnn_model simple2 \
--gnn_model RBDplusRawplusAux \
--use_batchnorm true \
-o output_folder/prefix \
-config dataset/boston_auto/region_0_1/config.json  \
-r path_to_model
```

### <a name="model"></a> Use Pre-trained Model and Reproduce the Numbers in the Paper
Go to the pre_alpha_clean_version folder.
```
cd pre_alpha_clean_version
```

Download the pretrained model (268M (unzip) /183M (zip)) and the testing dataset (3.8G). 
```
./download.sh 
```

After downloading, there will be two folders, 'model' and 'dataset', and one file 'validation.p' under the 'pre_alpha_clean_version' folder.

Then, you can use the following commands to evaluate the pretrined model on the testing dataset. The following commands print the results to the stdout (without post-processing, with MRF post-processing, and with smoothing post-processing) and dump all detailed results into the 'output' folder. 

If you got run-out-of memory error, try to reduce the batch size of the CNN evaluation through adding the '-cnnBatchSize X' flag to the following commands (The default batch size is 256).


Boston (RoadType: 0.925, LaneCount: 0.815),  
```
time python roadtagger_evaluate.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/boston -config dataset/boston_auto/region_0_1/config.json -r model/model_best

```

Chicago (RoadType: 0.937, LaneCount: 0.807),
```
time python roadtagger_evaluate.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/chicago -config dataset/chicago_auto/region_0_1/config.json -r model/model_best

```

DC (RoadType: 0.900, LaneCount: 0.675),
```
time python roadtagger_evaluate.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/dc -config dataset/dc_auto/region_0_1/config.json -r model/model_best

```

Seattle (RoadType: 0.961, LaneCount: 0.791),
```
time python roadtagger_evaluate.py -model_config simpleCNN+GNNGRU_8_0_1_1  -d `pwd`/ --cnn_model simple2 --gnn_model RBDplusRawplusAux --use_batchnorm true -o output/seattle -config dataset/seattle_auto/region_0_1/config.json -r model/model_best

```






# RoadTagger Results

**Results on Real-world Imagery**

![fig1](https://github.com/mitroadmaps/roadtagger/blob/master/figure/real.png "Results on Real-world Imagery")

**Results on Synthesized Imagery**

![fig2](https://github.com/mitroadmaps/roadtagger/blob/master/figure/synthesised.png "Results on Synthesized Imagery")










