import mxnet as mx
from rcnn.tools.wrappers import test_predictor, test_predict
from rcnn.config import config, generate_config, default
from rcnn.core.tester import im_detect

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os

from rcnn.symbol import *
from rcnn.dataset import *

from rcnn.io.image import transform
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from tqdm import tqdm

import cPickle
import copy


epoch = 50
thresh = 0.05

symbol = "/data/dataset/train/resnet-101"
network = "resnet"
dataset = "car"
generate_config(network, dataset)
with open("/data/dataset/val.txt") as f:
    val = map(lambda x: os.path.join(default.dataset_path, "images", x.strip()), f.readlines())
ims = [cv2.imread(i) for i in val[:]]


def predict():
    name = "{}/cache/{}_general_val_detections_val_{}.pkl".format(default.dataset_path, network, epoch)
    if os.path.exists(name):
        with open(name, 'rb') as fid:
            all_boxes = cPickle.load(fid)
    else:
        all_boxes = test_predict(network, symbol, epoch, dataset, ims, stride=630, threshold=thresh)

    return all_boxes


all_boxes = predict()
sym = eval('get_' + network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
imdb = eval(default.dataset)(default.test_image_set, default.root_path, default.dataset_path, config.CLASSES)
roidb = imdb.gt_roidb()
imdb.evaluate_detections(all_boxes)