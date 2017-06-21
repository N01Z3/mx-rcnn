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

    xs, ys = hgt // 1400, wdt // 1400
    print xs, ys


if __name__ == '__main__':
    size = shift = 1000

    test_path = '/home/vladimir/workspace/data/kaggle_seals/Test'
    new_train_path = '/home/vladimir/workspace/data/kaggle_seals/test_patches'

    result = Parallel(n_jobs=8)(delayed(dump_images)(r) for r in os.listdir(test_path))

import xml.etree.ElementTree as ET
from collections import defaultdict

TSZ = 1400
PATH = 'images_'

RAW = '/home/n01z3/dataset/noaa_sealions/original'
OUT = '/home/n01z3/dataset/noaa_sealions/images_'
DOT = '/home/n01z3/dataset/noaa_sealions/TrainDotted'
TIL = '/home/n01z3/dataset/noaa_sealions/tiles'


def create_anno(fname):
    tree = ET.Element("annotation")
    fn = ET.SubElement(tree, "filename")
    fn.text = str(fname)
    size = ET.SubElement(tree, "size")
    w = ET.SubElement(size, "width")
    w.text = str(TSZ)
    h = ET.SubElement(size, "height")
    h.text = str(TSZ)
    d = ET.SubElement(size, "depth")
    d.text = "3"

    return tree


def add_rect(tree, rect, name):
    base = ET.SubElement(tree, "object")
    el = ET.SubElement(base, "name")
    el.text = name

    el = ET.SubElement(base, "difficult")
    el.text = "0"
    el = ET.SubElement(base, "truncated")
    el.text = "0"

    bb = ET.SubElement(base, "bndbox")
    el = ET.SubElement(bb, "xmin")
    el.text = str(rect[0])
    el = ET.SubElement(bb, "ymin")
    el.text = str(rect[1])
    el = ET.SubElement(bb, "xmax")
    el.text = str(rect[2])
    el = ET.SubElement(bb, "ymax")
    el.text = str(rect[3])

    return tree


def bbox(point, d):
    return int(point[0] - d / 2.0), int(point[1] - d / 2.0), int(point[0] + d / 2.0), int(point[1] + d / 2.0)


def transer_black(fn='1.jpg'):
    msk = cv2.imread(os.path.join(DOT, fn))
    img = cv2.imread(os.path.join(RAW, fn))
    try:
        img[msk == 0] = 0
    except:
        print fn
    cv2.imwrite(os.path.join(OUT, fn), img)


def blacked():
    result = Parallel(n_jobs=8)(delayed(transer_black)(r) for r in os.listdir(DOT))


def make_xml(fn, xt, yt):
    fn = 'data/annotations/%s.xml' % fn[:-4]
    with open(fn) as fd:
        doc = xmltodict.parse(fd.read())

    tree = create_anno(os.path.basename(fn).replace('xml', 'jpg'))
    cnt = 0
    for v in doc['annotation']['object']:
        t = v['bndbox']
        label = v['name']
        xmin, xmax = int(t['xmin']), int(t['xmax'])
        ymin, ymax = int(t['ymin']), int(t['ymax'])

        if xt < xmin < xt + TSZ and xt < xmax < xt + TSZ and yt < ymin < yt + TSZ and yt < ymax < yt + TSZ:
            tree = add_rect(tree, [xmin - xt, ymin - yt, xmax - xt, ymax - yt], label)
            cnt += 1

    tree = ET.ElementTree(tree)
    if cnt > 0:
        return tree

    return None


def dump_tiles(fn):
    img = cv2.imread(os.path.join(OUT, fn))
    if img is None:
        print os.path.join(OUT, fn)
    hgt, wdt, _ = img.shape
    ws = int((wdt - TSZ) / 3)
    hs = int((hgt - TSZ) / 2)

    for i in range(3):
        for j in range(4):
            y1 = i * hs
            x1 = j * ws
            fn_out = '%s_%d_%d' % (fn[:-4], i, j)
            crp = img[y1:y1 + TSZ, x1:x1 + TSZ]
            if crp.shape != (TSZ, TSZ, 3):
                print crp.shape, fn

            tree = make_xml(fn, x1, y1)

            if tree is not None:
                cv2.imwrite(os.path.join(TIL, fn_out + '.jpg'), crp)
                tree.write('data/anno_tiles/%s.xml' % fn_out)


def slice_to_tiles():
    with open('data/Train_ternaus.txt') as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    print content
    result = Parallel(n_jobs=8)(delayed(dump_tiles)(r) for r in content)


if __name__ == '__main__':
    # blacked()
    slice_to_tiles()

    # size = shift = 1000
    #
    # test_path = '/home/vladimir/workspace/data/kaggle_seals/Test'
    # new_train_path = '/home/vladimir/workspace/data/kaggle_seals/test_patches'
    #
    # result = Parallel(n_jobs=8)(delayed(dump_images)(r) for r in os.listdir(test_path))
