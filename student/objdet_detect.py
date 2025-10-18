# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
import os
import sys

from tools.objdet_models.resnet.utils.torch_utils import _sigmoid

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    configs.min_iou=0.5

    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.05
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.5
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")
        # from github repo: https://github.com/maudzung/SFA3D/blob/0e2f0b63dc4090bd6c08e15505f11d764390087c/sfa/test.py#L89
        # configs.head_conv = 64
        # hm_cen = 3
        # cen_offset = 2
        # z_coor = 1
        # dim = 3
        # direction = 2  # sin, cos

      
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')

        #configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.backbone = 'resnet18' 
        configs.batch_size = 4
        configs.cfgfile = None
        configs.conf_thresh = 0.5
        configs.distributed = False
#        configs.img_size = 608
#        configs.input_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
#        configs.use_giou_loss = False

        # from test.py
        configs.pin_memory = True
        configs.distributed = False  # For testing on 1 GPU only

        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50

        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4

        configs.K = 50
        configs.peak_thresh = 0.2


        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        #model = fpn_resnet.get_pose_net(num_layers=18, heads=configs.heads, head_conv=64, imagenet_pretrained=configs.pretrained_filename)
        #arch_parts = configs.arch.split('_')
        #num_layers = int(arch_parts[-1])
        num_layers = 18
        # above maybe not needed
        print('using ResNet architecture with feature pyramid')
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                            imagenet_pretrained=configs.imagenet_pretrained)
        # def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):

        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # FIXME: maybe needed for darknet troubleshooting:
        #input_bev_maps = input_bev_maps.to(torch.float32) / 255.0

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    
            print("done darknet detect")

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing

            # copied from darknet approach:

            #output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing_sfa3d(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
                


            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")

            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 

    ## step 1 : check whether there are any detections
    ## step 2 : loop over all detections
    for detectionTensor in detections:
        ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure

        # opposite of what we did previously:
        x_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height # 0.08223684210526316
        z_discretization = (configs.lim_z[1] - configs.lim_z[0]) / configs.bev_height # 0.08223684210526316

        # intended format: [1, x, y, z, h, w, l, yaw], where 1 denotes the class id for the object type vehicle.
        detection = [
            1, # 0: class id for the object type vehicle
            (float(detectionTensor[2]) * x_discretization) + configs.lim_x[0], # 1: x # ex: tensor(351.0266)
            (float(detectionTensor[1]) * x_discretization) + configs.lim_y[0],  # 2: y # ex tensor(217.6962)
            (float(detectionTensor[3]) * x_discretization) + configs.lim_z[0], # 3: z # ex 0.0
            (float(detectionTensor[4]) * x_discretization) + configs.lim_z[0], # 4: h # ex 1.5
            (float(detectionTensor[5]) * x_discretization), # 5: w # ex tensor(23.1756)
            (float(detectionTensor[6]) * x_discretization), # 6: l # ex 51.2127
            float(detectionTensor[7]) # 7: yaw # ex -0.0210
        ]
        ## step 4 : append the current object to the 'objects' array
        objects.append(detection)
        
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    


import torch.nn.functional as F

def prep_bev_for_darknet(bev, device, img_size=608):
    # bev can be np.ndarray HxWxC or torch.Tensor
#    if isinstance(bev, np.ndarray):
#        bev = torch.from_numpy(bev)

    # ensure float32 and normalize 0..1
    bev = bev.to(torch.float32) / 255.0

    # HWC -> CHW if needed
#    if bev.ndim == 3 and bev.shape[-1] in (1, 3):
#        bev = bev.permute(2, 0, 1)
#    elif bev.ndim == 2:
#        bev = bev.unsqueeze(0)  # 1xH×W

    # ensure 3 channels (Darknet expects 3)
#    if bev.shape[0] == 1:
#        bev = bev.repeat(3, 1, 1)
#    elif bev.shape[0] > 3:
#        bev = bev[:3, ...]  # take first 3 if you had more

    # resize to 608×608 (or configs.img_size)
#    bev = bev.unsqueeze(0)  # B=1
#    bev = F.interpolate(bev, size=(img_size, img_size), mode="bilinear", align_corners=False)

    return bev.to(device)

# usage:
# input_bev_maps = prep_bev_for_darknet(input_bev_maps, configs.device, img_size=configs.img_size)



def post_processing_sfa3d(detections, num_classes=3, down_ratio=4, peak_thresh=0.2):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y

    bound_size_x = 50 #TODO: hardcoded for now
    bound_size_y = 50 #TODO: hardcoded for now
    BEV_HEIGHT=608 #TODO: hardcoded for now
    BEV_WIDTH=608 #TODO: hardcoded for now
    # bound_size_z = boundary['maxZ'] - boundary['minZ']

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / bound_size_y * BEV_WIDTH,
                detections[i, inds, 6:7] / bound_size_x * BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]
        ret.append(top_preds)

    return ret



def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])
