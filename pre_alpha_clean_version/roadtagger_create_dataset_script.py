import sys
import json 
from subprocess import Popen
from time import time, sleep 

# usage
# python roadtagger_create_dataset_script.py config/dataset_180tiles.json

dataset_cfg = json.load(open(sys.argv[1], "r"))

total_regions = 0

for item in dataset_cfg:
    prefix = item["cityname"]
    ilat = item["lat_n"]
    ilon = item["lon_n"]
    lat = item["lat"]
    lon = item["lon"]

    total_regions += ilat * ilon 

print("total regions", total_regions)
sleep(5)


pool = []
max_processes = 8


for item in dataset_cfg:
    prefix = item["cityname"]
    ilat = item["lat_n"]
    ilon = item["lon_n"]
    lat = item["lat"]
    lon = item["lon"]

    # Step-1 Generate Config Data
    Popen("python gen_dataset.py config %f %f %d %d %s" % (lat, lon, ilat, ilon, prefix), shell=True).wait()

    # Step-2
    # Download dataset from google map and OpenStreetMap
    for i in range(ilat):
        for j in range(ilon):
            while len(pool) == max_processes:
                sleep(1.0)
                new_pool = []
                for p in pool:
                    if p.poll() is None:
                        new_pool.append(p)

                pool = new_pool

            print("start a new process ",prefix, i, j)
            pool.append(Popen("python roadtagger_generate_dataset.py generate %s/region_%d_%d/config.json" % (prefix, i,j), shell=True))

    for p in pool:
        p.wait() 

     
    # Step-3
    # Add annotations from OpenStreetMap
    for i in range(ilat):
        for j in range(ilon):
            Popen("python roadtagger_generate_dataset.py osmauto %s/region_%d_%d/config.json" % (prefix, i,j), shell=True).wait()


    # Step-4
    # Create Tiles
    cc = 0 
    for i in range(ilat):
        for j in range(ilon):
            cc += 1 
            while len(pool) == max_processes:
                sleep(1.0)
                new_pool = []
                for p in pool:
                    if p.poll() is None:
                        new_pool.append(p)

                pool = new_pool

            print("start a new process ",prefix, i, j)
            pool.append(Popen("python roadtagger_generate_dataset.py tiles %s/region_%d_%d/config.json" % (prefix, i,j), shell=True))
            #pool.append(Popen("python gen_dataset.py generate %s/region_%d_%d/config.json" % (prefix, i,j), shell=True))

     
for p in pool:
    p.wait()