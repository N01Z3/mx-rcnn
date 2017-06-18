import numpy as np
from easydict import EasyDict as edict


config = edict()

config.AUGMENTATION = edict()
# all aug until ~30
config.AUGMENTATION.PARAMS = {"geom_prob": 0.8,
                              "angle": (0, 360),
                              "shear": (0, 0),
                              "scale": (0.7, 1, 1.4),
                              "flip_h": 1,
                              "flip_v": 1,
                              "color_prob": 0,
                              "blur": (0, 1.1),
                              "noise": (0, 0),
                              "add_s": (0, 0),
                              "add_v": (0, 0),
                              "multiply_s": (1, 1),
                              "multiply_v": (1, 1),
                              "contrast": (1, 1),
                              "crop": True,
                              "crop_size": (1000, 1000),
                              "iou_threshold": 0.6}

config.AUGMENTATION.N_ATTEMPTS = 10

# network related params
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.IMAGE_STRIDE = 0
config.RPN_FEAT_STRIDE = 16
config.RCNN_FEAT_STRIDE = 16
config.FIXED_PARAMS = ['conv1', 'conv2']
config.FIXED_PARAMS_SHARED = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# dataset related params
config.NUM_CLASSES = 5
config.SCALES = [(2000, 2000)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = (8, 16, 32)
config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

config.TRAIN = edict()

# R-CNN and RPN
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 1
config.TRAIN.END2END = True
config.TRAIN.ASPECT_GROUPING = False
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 4

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

# default settings
default = edict()

# default network
default.network = 'vgg'
default.pretrained = 'model/vgg16'
default.pretrained_epoch = 0
default.base_lr = 0.001
# default dataset
default.dataset = 'noaa_lions'
default.image_set = 'Train'
default.test_image_set = 'Val'
default.root_path = '/home/aakuzin/dataset/'
default.dataset_path = '/home/aakuzin/dataset/noaa_sealines'
# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.e2e_prefix = 'model/e2e'
default.e2e_epoch = 10
default.e2e_lr = default.base_lr
default.e2e_lr_step = '7'
# default rpn
default.rpn_prefix = 'model/rpn'
default.rpn_epoch = 8
default.rpn_lr = default.base_lr
default.rpn_lr_step = '6'
# default rcnn
default.rcnn_prefix = 'model/rcnn'
default.rcnn_epoch = 8
default.rcnn_lr = default.base_lr
default.rcnn_lr_step = '6'

# network settings
network = edict()

network.vgg = edict()

network.resnet = edict()
network.resnet.pretrained = 'model/resnet-101'
network.resnet.pretrained_epoch = 0
network.resnet.PIXEL_MEANS = np.array([0, 0, 0])
network.resnet.IMAGE_STRIDE = 0
network.resnet.RPN_FEAT_STRIDE = 16
network.resnet.RCNN_FEAT_STRIDE = 16
network.resnet.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
network.resnet.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'gamma', 'beta']

# dataset settings
dataset = edict()

dataset.PascalVOC = edict()

dataset.coco = edict()
dataset.coco.dataset = 'coco'
dataset.coco.image_set = 'train2014'
dataset.coco.test_image_set = 'val2014'
dataset.coco.root_path = 'data'
dataset.coco.dataset_path = 'data/coco'
dataset.coco.NUM_CLASSES = 81


# dataset.noaa_lions = edict()
# dataset.noaa_lions.dataset = 'noaa_lions'
# dataset.noaa_lions.image_set = 'noaa_train'
# dataset.noaa_lions.test_image_set = 'noaa_test'
# dataset.noaa_lions.root_path = 'data'
# dataset.noaa_lions.dataset_path = 'd:/patches'
# dataset.noaa_lions.NUM_CLASSES = 5


dataset = edict()
dataset.noaa_lions = edict()
dataset.noaa_lions.dataset = 'General'
dataset.noaa_lions.image_set = 'Train'
dataset.noaa_lions.test_image_set = 'Val'
dataset.noaa_lions.root_path = "/home/aakuzin/dataset/noaa_sealines"
dataset.noaa_lions.dataset_path = "/home/aakuzin/dataset/noaa_sealines"
dataset.noaa_lions.CLASSES = ['__background__', 'adult_males','subadult_males','adult_females','juveniles','pups']
dataset.noaa_lions.NUM_CLASSES = len(dataset.noaa_lions.CLASSES)
dataset.noaa_lions.SCALES = [(1000, 1000)] # [(400, 400)]
#dataset.noaa_lions.ANCHOR_SCALES = (4, 8, 16, 32)
#dataset.noaa_lions.ANCHOR_RATIOS = (0.33, 0.5, 1, 2, 3)
dataset.noaa_lions.ANCHOR_SCALES = (2, 4, 8, 16, 32)
dataset.noaa_lions.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.noaa_lions.NUM_ANCHORS = len(dataset.noaa_lions.ANCHOR_SCALES) * len(dataset.noaa_lions.ANCHOR_RATIOS)


def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v

