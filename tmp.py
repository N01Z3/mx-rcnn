import pandas as pd
import glob
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# df = pd.read_csv('/home/aakuzin/dataset/noaa_sealines/correct_coordinates.csv')
# # print df.head()
# for fn in sorted(set(df['filename'].tolist())):
#     print fn

# fns = glob.glob('/home/aakuzin/dataset/noaa_sealines/annotations_dirt/*xml')
# for fn in fns:
#     os.rename(fn, fn.replace('jpg.', ''))


# fns = glob.glob('/home/aakuzin/dataset/noaa_sealines/annotations/*lif')
# for fn in fns:
#     os.rename(fn, fn.replace('lif', 'xml'))
FOLDER = '/home/aakuzin/dataset/noaa_sealines/images/'
COLOR = ['r', 'g', 'b', 'c', 'k', 'y', 'navy', 'peru']

im = '20.jpg'
img = plt.imread(os.path.join(FOLDER, im))

fig, ax = plt.subplots(1)
print img.shape
ax.imshow(img)

for i, fn in enumerate(glob.glob('/home/aakuzin/dataset/noaa_sealines/results/*')):

    df = pd.read_csv(fn, sep=' ', names = ['fn', 'p', 'x1', 'y1', 'x2', 'y2'])

    df = df[df['fn'] == im]
    print df.shape
    if df.shape[0] > 1:
        df.reset_index(inplace=True)

        print df.head(10)
        print df.shape

        for n in range(df.shape[0]):
            rect = patches.Rectangle((df['x1'][n], df['y1'][n]), df['x2'][n]-df['x1'][n], df['y2'][n]-df['y1'][n],
                                     linewidth=1, edgecolor=COLOR[i], facecolor='none')
            ax.add_patch(rect)

plt.show()