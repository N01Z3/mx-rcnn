import mxnet as mx
from rcnn.tools.wrappers import test_predictor, test_predict, train_net
from rcnn.config import config, generate_config, default
from rcnn.core.tester import im_detect

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os

from rcnn.io.image import transform
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

network = "vgg"
dataset = "noaa_lions"
generate_config(network, dataset)

from rcnn.symbol import *
from rcnn.dataset import *

train_net(ctx=[mx.gpu(0)],
          pretrained=default.pretrained,
          epoch=default.pretrained_epoch,
          prefix="model/vgg",
          begin_epoch=0,
          end_epoch=100,
          shuffle=True,
          resume=False,
          frequent=500,
          lr=0.001,
          lr_step="10,15,20,25,30,35,40,45",
          lr_factor=0.5)
