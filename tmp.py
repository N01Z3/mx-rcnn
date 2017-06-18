import pandas as pd
import glob
import os
# df = pd.read_csv('/home/aakuzin/dataset/noaa_sealines/correct_coordinates.csv')
# # print df.head()
# for fn in sorted(set(df['filename'].tolist())):
#     print fn

# fns = glob.glob('/home/aakuzin/dataset/noaa_sealines/annotations_dirt/*xml')
# for fn in fns:
#     os.rename(fn, fn.replace('jpg.', ''))


fns = glob.glob('/home/aakuzin/dataset/noaa_sealines/annotations/*lif')
for fn in fns:
    os.rename(fn, fn.replace('lif', 'xml'))