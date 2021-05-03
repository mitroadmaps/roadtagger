# RoadTagger Docker Container
We containerize the pre_alpha_clean_version of RoadTagger into a docker container image. You can use this docker container to reproduce the numbers in the paper and run RoadTagger on custom images (please check out the usage below).  

## Usage
### Docker 
To install docker (with GPU support), you can check out the instruction [here](https://www.tensorflow.org/install/docker). 

### Inference on Custom Image
After the docker environment is set up, start the RoadTagger inference server.
```bash
docker run -p8010:8000 -p8011:8001 -it --rm songtaohe/roadtagger_inference_server_cpu:latest
```

<!-- You can change 'cpu' to 'gpu' in the docker image name if you want to run the container with GPU support.  -->

To run RoadTagger on custom images, all you need is to provide an image and a graph file (edges in image coordinate), and specify the ground sampling distance (spatial resolution) of your image. 

We show an example (image and graph file) in the *examples* folder. To run RoadTagger on this example, 

```bash
cd script
python infer.py -image ../examples/e1.png -graph ../examples/e1.json -gsd 0.125
```

By default, this script will output the result to out.json. If everything goes well, the result should appear in out.json. 
```json
[
    {
    "node id": 0, 
    "road type": "primary", 
    "lane count probability": [
      0.01214602217078209, 
      0.09006574004888535, 
      0.810308039188385, 
      0.07358891516923904, 
      0.004188787192106247, 
      0.009702381677925587
    ], 
    "coordinate": [
      194, 
      176
    ], 
    "lane count": 3, 
    "road type probability": [
      0.34992465376853943, 
      0.6500753164291382
    ]
  }
]
```

### Build Docker Container
If you want to build this docker container image by yourself, you have to first download the RoadTagger model so that it can work with custom images. 
```bash
cd app/roadtagger
./downloadModel.sh
```

We have scripts to build and run the docker container on CPUs. You can also change the TensorFlow version in the Dockerfile to enable GPU support (the code should work with TensorFlow 1.15). 



