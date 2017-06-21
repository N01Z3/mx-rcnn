import numpy as np
import glob
import xml.etree.ElementTree
import xmltodict
import os
import pandas as pd
import glob
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FOLDER = '/home/n01z3/dataset/noaa_sealions/images/'
COLOR = ['r', 'g', 'b', 'c', 'k', 'y', 'navy', 'peru']

d = {'r': 'adult_males',
     'p': 'subadult_males',
     'b': 'adult_females',
     'bl': 'juveniles',
     'g': 'pups'}

anim = {'adult_males': 'r',
        'subadult_males': 'b',
        'adult_females': 'c',
        'juveniles': 'y',
        'pups': 'g'}


def conv_ternaus_xml(fn):
    with open(fn) as f:
        content = f.readlines()

    f.close()

    out = open(fn.replace('_ternaus', ''), 'w')

    for el in content:
        el = el.replace('</filename', '.jpg</filename')
        el = el.replace('annotations', 'annotation')

        if 'name' in el:
            for k in d.keys():
                el = el.replace('name>%s' % k, 'name>%s' % d.get(k))

        print el
        if 'path' not in el or 'path' not in el:
            # out.append(el)
            out.write(el)

    out.close()


def convert_files():
    fns = glob.glob('data/annotations_ternaus/*xml')
    for fn in fns:
        conv_ternaus_xml(fn)


def make_txt():
    out = open('data/tiles.txt', 'w')
    fns = glob.glob('data/anno_tiles/*xml')

    for fn in fns:
        out.write(fn.split('/')[-1].replace('xml', 'jpg\n'))

    out.close()


def show_bb():
    fn = glob.glob('data/anno_tiles/0_0_1.xml')[-1]
    n = os.path.basename(fn)[:-4]

    with open(fn) as fd:
        doc = xmltodict.parse(fd.read())

    img = plt.imread(os.path.join(FOLDER, n + '.jpg'))
    fig, ax = plt.subplots(1)
    print img.shape
    ax.imshow(img)

    print doc

    for v in doc['annotation']['object']:
        t = v['bndbox']
        label = v['name']
        x1, x2 = int(float(t['xmin'])), int(float(t['xmax']))
        y1, y2 = int(float(t['ymin'])), int(float(t['ymax']))

        w = (x2 - x1)  # //2
        h = (y2 - y1)  # //2
        xc = x1
        yc = y1

        print xc, yc, w, h, label

        rect = patches.Rectangle((xc, yc), w, h,
                                 linewidth=1, edgecolor=anim.get(label), facecolor='none')
        ax.add_patch(rect)

    plt.show()


def calc_sz():
    for fn in glob.glob('data/annotations/*.xml'):
        n = os.path.basename(fn)[:-4]
        img = plt.imread(os.path.join(FOLDER, n + '.jpg'))
        print img.shape


if __name__ == '__main__':
    show_bb()
