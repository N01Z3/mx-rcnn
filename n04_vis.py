from collections import defaultdict
import cPickle
from rcnn.config import config, generate_config, default
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from rcnn.symbol import *
from rcnn.dataset import *
from rcnn.tools.wrappers import test_predictor, test_predict
from rcnn.core.tester import vis_all_detection
from rcnn.io.image import transform

radius = defaultdict(lambda: 30)
radius["adult_males"] = 12
radius["subadult_males"] = 40
radius["adult_females"] = 45

keys = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

color = {
    "adult_males": "blue",
    "subadult_males": "magenta",
    "adult_females": "black",
    "juveniles": "red",
    "pups": "cyan",
}


os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

epoch = 60
thresh = 0.05

symbol = "model/1800/vgg"
network = "vgg"
dataset = "noaa_lions"
generate_config(network, dataset)

with open("/home/aakuzin/dataset/noaa_sealines/Val.txt") as f:
    val = map(lambda x: os.path.join(default.dataset_path, "images", x.strip()), f.readlines())
ims = [cv2.imread(i) for i in val[:]]

print ims

def filter_box(box, r, kmax=4, kmin=0.7):
    w = box[2] - box[0]
    h = box[3] - box[1]
    s = w * h
    s_min = (kmin * r) * (kmin * r) * 0
    s_max = (kmax * r) * (kmax * r)
    # return 2*r*kmax >= h >= 2*r*kmin and 2*r*kmax >= w >= 2*r*kmin and w * h > (2*r*kmin)**2
    return s_max > s > s_min


def predict():
    name = "{}/cache/{}_general_val_detections_val_{}.pkl".format(default.dataset_path, network, epoch)
    if os.path.exists(name):
        with open(name, 'rb') as fid:
            all_boxes = cPickle.load(fid)
    else:
        all_boxes = test_predict(network, symbol, epoch, dataset, ims, stride=1200, threshold=thresh)

    return all_boxes


all_boxes = predict()
# if False:
#     for i, key in enumerate(keys):
#         for j, boxes in enumerate(all_boxes[i + 1]):
#             filtered = []
#             for box in boxes:
#                 if filter_box(box, radius[key], kmax=2.2, kmin=0.8):
#                     filtered.append(box)
#             all_boxes[i + 1][j] = np.array(filtered[:])

with open("/home/aakuzin/dataset/noaa_sealines/cache/vgg_general_Val_detections_val_60.pkl", 'rb') as fid:
    all_boxes = cPickle.load(fid)

# print all_boxes

# sym = eval('get_' + network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
# imdb = eval(default.dataset)(default.test_image_set, default.root_path, default.dataset_path, config.CLASSES)
# roidb = imdb.gt_roidb()
# imdb.evaluate_detections(all_boxes)

with open("/home/aakuzin/dataset/noaa_sealines/Val.txt") as f:
    val = map(lambda x: os.path.join(default.dataset_path, "images", x.strip()), f.readlines())
# val1 = map(lambda x: x.replace("/data/dataset/images", "/data/base/training_circle"), val)
ims = [cv2.imread(i) for i in val[:]]
i = 3
boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, config.NUM_CLASSES)]
vis_all_detection(transform(ims[i], config.PIXEL_MEANS), boxes_this_image, config.CLASSES[:], 1)
