import sys
import logging
logging.basicConfig(level=logging.INFO)
import pointcloudscripte_binary
from plugin_uncertainty import RadianceField as InstantNGPRadianceField
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import time
import pickle
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

'------------------------------------------------------------------------------'
# Output:   Pointcloud (ply)
'------------------------------------------------------------------------------'
dataset_list = ["..."] # e.g. dataset_list = ["scan40"]

for dataset in dataset_list:
    print('dataset')
    print(dataset)

    if dataset == "...":
        min_coordinate = np.array([..,...,..])  #set boundingbox
        max_coordinate = np.array([..,...,..]) 
       

    resolution = ...   #e.g. 0.0075 (differs from scaling-space)
    start_time = time.time()
    ensemble = []
    for i in range(1, 11):   # e.g. for 10 ensemble member
        num = f"{i:02d}"
        
        input_dir = f"C:/.../scan{dataset}_member_{num}.msgpack"    #load trained networks
        pointcloud_name = f"{dataset}_ensemble.ply"
        pointcloud_name_member = f"{dataset}_member.ply"
    
        print('get .msgppack from input dir:')
        print(input_dir)
        output_dir = f"C:.../"
       
    
        radiance_field = InstantNGPRadianceField(
            batch_size = 100_000,                       
            trained_network_weights_file = input_dir,
            background_color = [1.0, 1.0, 1.0, 1.0]
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

#'------------------------------------------------------------------------------'

    # Step 1:
        coordinate_list = pointcloudscripte_binary.create_coordinate_list(min_coordinate, max_coordinate, resolution)

    # Step 2: 
        with pointcloudscripte_binary.PlyPointWriter(os.path.join(output_dir, pointcloud_name) + '.ply', use_color=True) as ply_writer:
            grid = pointcloudscripte_binary.create_member_grid(
                coordinate_list = coordinate_list,
                filename = pointcloud_name,
                radiance_field = radiance_field,
                ply_writer = ply_writer,
                use_threshold=False,
            )
        ensemble.append(grid)
      

    ensemble_exp = copy.deepcopy(ensemble)
    for grid in ensemble_exp:
        grid[:,3] = np.exp(grid[:,3])


#single ensemble member
    with nerftools.pointcloudscripte_binary.PlyPointWriter(os.path.join(output_dir, pointcloud_name) + '.ply', use_color=True) as ply_writer:
            grid_2 = nerftools.pointcloudscripte_binary.write_PC_(
                data = ensemble_exp[0],
                filename = pointcloud_name_member,
            )

    num_coords = len(coordinate_list)
    dimensions = (num_coords,9)
    ensemble_grid = np.zeros(dimensions,dtype=np.float32)


    ensemble_grid[:,0] = ensemble[0][:,0]    #coordinates                     -> x
    ensemble_grid[:,1] = ensemble[0][:,1]    #                                -> y
    ensemble_grid[:,2] = ensemble[0][:,2]    #                                -> z


    mean = np.mean(ensemble, axis=0)
    density_mean = mean[:,3]
    r_mean =  mean[:,4]
    g_mean =  mean[:,5]
    b_mean = mean[:,6]
    
    mean_exp = np.mean(ensemble_exp, axis=0)
    density_mean_exp = mean_exp[:,3]

    ensemble_grid[:,3] = density_mean      
    ensemble_grid[:,4] = density_std
    ensemble_grid[:,5] = r_mean
    ensemble_grid[:,6] = g_mean
    ensemble_grid[:,7] = b_mean

    std = np.std(ensemble, axis=0, ddof=1)
    density_std = std[:,3]
    condition = ensemble_grid[:,4] < ... #set value to filter low density values
    ensemble_grid = ensemble_grid[~condition]

### Speichere Ensemble_Punktwolke als .ply:
with nerftools.pointcloudscripte_binary.PlyPointWriter(os.path.join(output_dir, pointcloud_name) + '.ply', use_color=True) as ply_writer:
        grid = nerftools.pointcloudscripte_binary.write_PC_ensemble(
           data = ensemble_grid,
           filename = pointcloud_name,
        )         