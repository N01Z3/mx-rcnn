import numpy as np
import glob

d = {'r': 'adult_males',
     'p': 'subadult_males',
     'b': 'adult_females',
     'bl': 'juveniles',
     'g': 'pups'}


def conv_ternaus_xml(fn):
    with open(fn) as f:
        content = f.readlines()

    f.close()

    out = open(fn.replace('_ternaus', ''), 'w')

    for el in content:
        el = el.replace('</filename', '.jpg</filename')

        if 'name' in el:
            for k in d.keys():
                el = el.replace('name>%s' % k, 'name>%s' % d.get(k))

        print el
        if 'path' not in el or 'path' not in el:
            # out.append(el)
            out.write(el)

    out.close()


def main():
    fns = glob.glob('data/annotations_ternaus/*xml')
    for fn in fns:
        conv_ternaus_xml(fn)


def make_txt():
    out = open('data/Train_ternaus.txt', 'w')
    fns = glob.glob('data/annotations/*xml')

    for fn in fns:
        out.write(fn.split('/')[-1].replace('xml', 'jpg\n'))

    out.close()


if __name__ == '__main__':
    make_txt()
