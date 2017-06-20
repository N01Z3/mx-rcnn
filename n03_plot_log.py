import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.legend_handler import HandlerLine2D

COLOR = ['r', 'g', 'b', 'c', 'k', 'y', 'navy', 'peru']

acs = ['RPNAcc', 'RCNNAcc']
lss = ['RPNLogLoss', 'RCNNLogLoss']

fns  = ['model/1800_1gpu_t/log', 'model/1800_2gpu_t/log', 'model/1800_3gpu_t/log', 'model/1800x_log', 'model/1800_gpu_t_ti/log']
lbs = ['1gpu', '2gpu', '3gpu', 'old ann',  '2gputi']

plt.figure()
for z, fn in enumerate(fns):

    with open(fn) as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    for n, lst in enumerate([acs, lss]):
        plt.subplot(2, len(fns), len(fns) * n + 1 + z)
        for i, m in enumerate(lst):
            out = []
            for line in content:
                if 'Train-' + m in line and 'samples/sec' not in line:
                    out.append(float(line.split(m + '=')[-1]))

            line1, = plt.plot(out, color=COLOR[i], label='%0.4f: %s; %s' % (out[-1], m, lbs[z]))
            if 'Loss' in m:
                plt.ylim(0, 4)
            else:
                plt.ylim(0.6, 1.0)
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=3)})
plt.show()

# for fn, c in zip(fls,['r', 'g', 'b', 'c', 'k',  'y', 'navy', 'peru']):
#     out = []
#     with open(fn) as f:
#
#
#
#         for line in content:
#             if split in line:
#                 val = line.split(split)[-1]
#                 out.append(float(val))
#                 # print(val.replace('.',','))
#
#     # print(out)
#     # print(np.amax(out))
#     # print(np.amax(out))
#     print(c, fn)
#
#     print(np.array(out)[np.argsort(out)[-3:]], np.argsort(out)[-3:])
#
#     line1, = plt.plot(out, color = c, label= '%0.4f: %s' % (np.amax(out), fn.split('_')[-2]))
#     plt.ylim(0.60, 0.68)
#
