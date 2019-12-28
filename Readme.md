### Download the Dataset

Step-1:

In 'pre_alpha_clean_version' folder, 

**Edit line 47 in helper_mapdriver.py. Enter your Google API Key there.**

```
python roadtagger_generate_dataset.py config/dataset_180tiles.json 
```

This script will download the dataset with 20 cities into the 'dataset' folder. 


Step-2:

Add the manual annotation.

```
cp annotation boston_region_0_1_annotation.p dataset/boston_auto/region_0_1/annotation.p
cp annotation chicago_region_0_1_annotation.p dataset/chicago_auto/region_0_1/annotation.p
cp annotation dc_region_0_1_annotation.p dataset/dc_auto/region_0_1/annotation.p
cp annotation seattle_region_0_1_annotation.p dataset/seattle_auto/region_0_1/annotation.p
```



### Train RoadTagger Model

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










