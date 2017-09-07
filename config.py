#!/usr/bin/env python

"""
Configuration script, require easydict (`pip install easydict`).
Can be imported using:
`from config import cfg`
"""

import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C


##############################################
# General options                            #
##############################################

# Main directory
__C.ROOT_DIR = os.path.dirname(__file__)

# The directory where to find the models
__C.MODELS_DIR = os.path.join(__C.ROOT_DIR, 'models')

# The directory where to find the models
__C.CONFIGS_DIR = os.path.join(__C.ROOT_DIR, 'configs')

# The directory where to find the models
__C.DATASETS_DIR = os.path.join(__C.ROOT_DIR, 'datasets')

# The directory where to find the resources
__C.RESOURCES_DIR = os.path.join(__C.ROOT_DIR, 'res')

# Extension of the models
__C.MODELS_EXTENSION = 'pb'

# Test mode (for debug)
__C.TEST_MODE = False



##############################################
# Dataset options                            #
##############################################
__C.DATASET = edict()

# Number of categories
__C.DATASET.NB_CATEGS = { 'foodinc': 67, 'ms_coco': 80, 'uecfood_256': 256 }



##############################################
# Models options                             #
##############################################
__C.MODEL = edict()

# Number of categories
__C.MODEL.RPN_SHAPE = { 'frcnn': [1, None, None, 1024], 
                        'inception': [1, None, None, 1088], 
                        'rfcn': [1, None, None, 1024] }



##############################################
# API options                                #
##############################################
__C.API = edict()

# The port to use for the API
__C.API.PORT = 8000

# The route where to send the GET request
__C.API.ROUTE = 'detection'

# The complete path
__C.API.ADDRESS = 'http://' + 'localhost' + \
                  ':' + str(__C.API.PORT)

# The path to the GET host
__C.API.ADDRESS_GET = __C.API.ADDRESS + \
                      '/' + __C.API.ROUTE



##############################################
# Test settings (if test_mode = True)        #
##############################################
__C.TEST = edict()

# If True, the detection is done on a local image
__C.TEST.LOCAL_IMAGE = False

# If we use a local image, this is the image to use
__C.TEST.LOCAL_IMAGE_PATH = os.path.join(__C.RESOURCES_DIR, 'food1.jpg')

# If True, the detections are randomly computed (faster)
__C.TEST.FAKE_DETECTIONS = True

