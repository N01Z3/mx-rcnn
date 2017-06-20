import glob
import xml.etree.ElementTree
import xmltodict
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
from joblib import Parallel, delayed
import numpy as np

ISZ = 1400
PATH = 'images_'

def dump_images(file_name):
    img = cv2.imread(os.path.join(PATH, file_name))
    hgt, wdt, _ = img.shape

    xs, ys = hgt//1400, wdt//1400
    print xs,ys

if __name__ == '__main__':
    size = shift = 1000

    test_path = '/home/vladimir/workspace/data/kaggle_seals/Test'
    new_train_path = '/home/vladimir/workspace/data/kaggle_seals/test_patches'

    result = Parallel(n_jobs=8)(delayed(dump_images)(r) for r in os.listdir(test_path))