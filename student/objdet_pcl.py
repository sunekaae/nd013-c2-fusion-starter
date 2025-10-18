# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from student.objdet_detect import load_configs
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

import open3d as o3d

point_cloud = o3d.geometry.PointCloud()

# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="pointcloud in open3d w callback", width=800, height=600)

    imgpcl = pcl[:, 0:3]

    # Convert the NumPy array to an Open3D PointCloud object
    point_cloud.points = o3d.utility.Vector3dVector(imgpcl)

    # Step 2: Add geometry (example: a simple point cloud)
    # Add the point cloud to the visualizer
    visualizer.add_geometry(point_cloud)

    # Step 3: Register a key callback (example: press 'Q' to quit)
    def quit_callback(vis):
        print("Quitting visualization...")
 #       vis.close()
        return False

    # Step 4: Run the visualizer
    visualizer.run()
    visualizer.destroy_window()

    # FIXME: maybe delete this change color stuff?
    # Define a callback function for a key press
    def change_color(vis):
        # Change the color of the point cloud
        point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        return False  # Return False to indicate no further updates are needed
    
    # Register the callback for the 'C' key (ASCII code 67)
    visualizer.register_key_callback(67, change_color)
    
    # step 2 : create instance of open3d point-cloud class

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    # FIXME: perhaps change current approach to align with above, downside is that each frame does not destroy the window and does not clean up after itself
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)

    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    pcl = tools.pcl_from_range_image(frame, lidar_name)
    
    #stats: ~148.457 rows/points -> x4 -> 593.828
    #x: observed: -73.24419427991357 to 75.59061433499369
    #y obserdver: -15.130397498534995 to 73.5403907234094
    #z observed: -1.483576463713748 to 5.325929859018363
    # intensity observed: 0.000362396240234375 to 22016.0

    # step 2 : extract the range and the intensity channel from the range image
    configs = load_configs() # cannot find another way to get access to configs

    # I observed many negative values outside of the lower limit (behind us, downwards). we clip data based on min/max in config
    pcl[:,0] = np.clip(pcl[:,0], configs.lim_x[0], configs.lim_x[1]) # 0 to 50
    pcl[:,1] = np.clip(pcl[:,1], configs.lim_y[0], configs.lim_y[1]) # -25 to 25
    pcl[:,2] = np.clip(pcl[:,2], configs.lim_z[0], configs.lim_z[1]) # -1 to 3
    pcl[:,3] = np.clip(pcl[:,3], configs.lim_r[0], configs.lim_r[1]) # 0 to 1
    # Note: The image in the homework shows a 360 image and text states:
    #       "Make sure that the entire range of the data is mapped appropriately onto the 8-bit channels of the OpenCV image so that no data is lost."
    #       but seems it makes more sense to just look ahead based on the config

    # x  # offset + discritization + floor
    x_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    pcl[:,0] = np.floor( (pcl[:,0]-configs.lim_x[0]) / x_discretization )

    # same for y
    y_discretization = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width
    pcl[:,1] = np.floor( (pcl[:,1]-configs.lim_y[0]) / y_discretization )

    # create the two image arrays
    range_img = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=int) # first variable x in lidar is forward, and first variable in image is height. It's 608 from config
    intensity_img = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=int)
    # Note: the +1 on the size initialisation enabled for rounding, but perhaps is not required.

    # step 3 : set values <0 to zero
    # NOTE: already done, before discretization
    
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    # normalise: (value - min) / (max - min) * 255
    # offset + normalize + floor
    pcl[:,2] = np.floor( (pcl[:,2]-configs.lim_z[0]) / (configs.lim_z[1]-configs.lim_z[0]) * 255 )
    # now, get index for sort to get the higest z for each x/y
    range_idx = np.lexsort(( -pcl[:,2], pcl[:,1], pcl[:,0] )) # sort by x, then y, then -z (highest z first)
    range_pcl = pcl[range_idx]
    _, range_unique_indices = np.unique(range_pcl[:,0:2], axis=0, return_index=True)
    range_pcl = range_pcl[range_unique_indices]
       
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    #first identify outliers and normalize:
    percentile_lo = np.percentile(pcl[:,3], 1)   # 1st percentile
    percentile_hi = np.percentile(pcl[:,3], 90) # 90th percentile
    # values below or above percentile gets set to the percentile value
    pcl[pcl[:, 3] < percentile_lo, 3] = percentile_lo
    pcl[pcl[:, 3] > percentile_hi, 3] = percentile_hi

    pcl[:,3] = np.floor( (pcl[:,3]-percentile_lo) / (percentile_hi-percentile_lo) * 255 )

    # sort and get unique as before
    intensity_pcl = pcl[:, [0,1,3]] # only x,y,intensity # doing this because picking 0, 1, 3 in the unique() method call wasn't working proper
    
    intensity_idx = np.lexsort(( -intensity_pcl[:,2], intensity_pcl[:,1], intensity_pcl[:,0] )) 
    intensity_pcl = intensity_pcl[intensity_idx]
    _, intensity_unique_indices = np.unique(intensity_pcl[:, 0:2], axis=0, return_index=True)
    intensity_pcl = intensity_pcl[intensity_unique_indices]

    # step 6 : stack the range and intensity image vertically using np.vstack
    range_img[ range_pcl[:,0].astype(int), range_pcl[:,1].astype(int) ] = range_pcl[:,2].astype(int)
    intensity_img[ intensity_pcl[:,0].astype(int), intensity_pcl[:,1].astype(int) ] = intensity_pcl[:,2].astype(int)

    stacked = np.vstack([range_img, intensity_img])

    img_range_intensity = stacked
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    x_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_copy = np.copy(lidar_pcl)
    lidar_pcl_copy[:,0] = np.floor(lidar_pcl_copy[:,0] / x_discretization)

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_copy[:,1] = lidar_pcl_copy[:,1] - configs.lim_y[0]
    lidar_pcl_copy[:,1] = np.floor(lidar_pcl_copy[:,1] / x_discretization)

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # show_pcl(lidar_pcl_copy)
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_img = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=float) # first variable x in lidar is forward, and first variable in image is height. It's 608 from config. x and y is are ints but value is float.

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -intensity (use numpy.lexsort)
    # now, get index for sort to get the higest intensity for each x/y
    # sort and get unique
    intensity_pcl = lidar_pcl_copy[:, [0,1,3]] # only x,y,intensity # doing this because picking 0, 1, 3 in the unique() method call wasn't working proper
    intensity_idx = np.lexsort(( -intensity_pcl[:,2], intensity_pcl[:,1], intensity_pcl[:,0] )) # sort by x, then y, then -z (highest z first)
    intensity_pcl = intensity_pcl[intensity_idx]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, range_unique_indices = np.unique(intensity_pcl[:,0:2], axis=0, return_index=True)
    intensity_pcl = intensity_pcl[range_unique_indices]
    # TODO: add the variable for counts?

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    percentile_lo = np.percentile(intensity_pcl[:,2], 1)   # 1st percentile
    percentile_hi = np.percentile(intensity_pcl[:,2], 90) # 90th percentile
    # values below or above percentile gets set to the percentile value
    intensity_pcl[intensity_pcl[:, 2] < percentile_lo, 2] = percentile_lo
    intensity_pcl[intensity_pcl[:, 2] > percentile_hi, 2] = percentile_hi

    intensity_pcl[:,2] = np.clip((intensity_pcl[:,2] - percentile_lo) / (percentile_hi - percentile_lo), 0, 1).astype(np.float32) # keep as float

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    intensity_img[ intensity_pcl[:,0].astype(int), intensity_pcl[:,1].astype(int) ] = (255*intensity_pcl[:,2]).astype(int)

    intensity_map = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=float)
    intensity_map[ intensity_pcl[:,0].astype(int), intensity_pcl[:,1].astype(int) ] = intensity_pcl[:,2]

    if False:
        img_intensity = intensity_img.astype(np.uint8)
        cv2.imshow('intensity_image', img_intensity)
        cv2.waitKey(0)


    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_img = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=float) # first variable x in lidar is forward, and first variable in image is height. It's 608 from config

    height_idx = np.lexsort(( -lidar_pcl_copy[:,2], lidar_pcl_copy[:,1], lidar_pcl_copy[:,0] )) # sort by x, then y, then -z (highest z first)
    height_pcl = lidar_pcl_copy[height_idx]
    _, height_unique_indices = np.unique(height_pcl[:,0:2], axis=0, return_index=True)

    height_pcl = height_pcl[height_unique_indices]
    low = np.amin(height_pcl[:,2])
    hi = np.amax(height_pcl[:,2])
    # height_pcl[:,2] = np.floor( (height_pcl[:,2]-low) / (hi-low) * 255 )
    height_pcl[:,2] = np.clip((height_pcl[:,2] - low) / (hi - low), 0, 1).astype(np.float32) # keep as float


    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_img[ height_pcl[:,0].astype(int), height_pcl[:,1].astype(int) ] = (255*height_pcl[:,2]).astype(int)

    height_map = np.zeros((configs.bev_height+1, configs.bev_width+1), dtype=float)
    height_map[ height_pcl[:,0].astype(int), height_pcl[:,1].astype(int) ] = height_pcl[:,2]

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    if False:
        img_range = height_img.astype(np.uint8)
        cv2.imshow('range_image', img_range)
        cv2.waitKey(0)

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    lidar_pcl_cpy = lidar_pcl_copy
    lidar_pcl_top = height_pcl
    height_map = height_map
    intensity_map = intensity_map

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(height_pcl[:, 0]), np.int_(height_pcl[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    bev_map = bev_map.astype(np.float32) # ensure BEV map is float32 before being passed to model.

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


