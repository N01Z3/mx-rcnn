from __future__ import print_function
import pprint
import os
import cPickle
import mxnet as mx
import numpy as np

from ..symbol import *
from ..dataset import *
from ..core.loader import TestLoader
from ..core.tester import Predictor
from ..utils.load_model import load_param
from ..config import config, generate_config, default
from ..core.tester import im_detect
from ..utils import sliding_window
from ..io.image import transform
from ..processing.nms import py_nms_wrapper
from tqdm import tqdm


def test_predictor(network, dataset, image_set, root_path, dataset_path,
                   ctx, prefix, epoch,
                   shuffle, has_rpn, proposal):
    # set config
    if has_rpn:
        config.TEST.HAS_RPN = True

    # print config
    # pprint.pprint(config)

    # load symbol and testing data
    if has_rpn:
        sym = eval('get_' + network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
        imdb = eval(dataset)(image_set, root_path, dataset_path, config.CLASSES)
        roidb = imdb.gt_roidb()
    else:
        sym = eval('get_' + network + '_rcnn_test')(num_classes=config.NUM_CLASSES)
        imdb = eval(dataset)(image_set, root_path, dataset_path, config.CLASSES)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape

    data_shape_dict = dict(test_data.provide_data)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
                arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
                aux_params[k].shape)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = None
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    if not has_rpn:
        max_data_shape.append(('rois', (1, config.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    return predictor, data_shape_dict


def test_predict(network, symbol, epoch, dataset, ims, stride, threshold=0.7, test=False):
    generate_config(network, dataset)
    predictor, data_shape_dict = test_predictor(default.network, default.dataset, default.test_image_set,
                                                default.root_path, default.dataset_path,
                                                mx.gpu(0), symbol, epoch,
                                                False, config.TEST.HAS_RPN, "rpn")

    nms = py_nms_wrapper(config.TEST.NMS)
    all_boxes = [[[] for _ in xrange(len(ims))]
                 for _ in xrange(config.NUM_CLASSES)]
    for i, im in tqdm(enumerate(ims)):
        positions = sliding_window.sliding_window(im, data_shape_dict["data"][2], int(stride), 0)[1]
        for position in positions:
            im_array, im_scale = im[position], 1
            im_array = transform(im_array, config.PIXEL_MEANS)
            im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
            data = [mx.nd.array(im_array), mx.nd.array(im_info)]
            data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
            data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
            scores, boxes, data_dict = im_detect(predictor, data_batch, ["data", "im_info"], 1)

            for j in range(1, config.NUM_CLASSES):
                indexes = np.where(scores[:, j] > threshold)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]
                for n in range(len(cls_boxes)):
                    cls_boxes[n][[0, 2]] += position[1].start
                    cls_boxes[n][[1, 3]] += position[0].start
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][i].append(cls_dets[keep, :])

    for i in range(len(all_boxes)):
        for j in range(len(all_boxes[i])):
            try:
                all_boxes[i][j] = np.vstack(all_boxes[i][j])
                keep = nms(all_boxes[i][j])
                all_boxes[i][j] = all_boxes[i][j][keep, :]
            except Exception as e:
                pass
    det_file = os.path.join(default.root_path, "cache",
                            "{}_general_{}_detections_{}_{}.pkl".format(network, default.test_image_set,
                                                                        "test" if test else "val", epoch))
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # boxes_this_image = [[]] + [all_boxes[j][0] for j in range(1, config.NUM_CLASSES)]

    return all_boxes


from ..config import config, default, generate_config
from ..symbol import *
from ..core import callback, metric
from ..core.loader import AnchorLoader
from ..core.module import MutableModule
from ..utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from ..utils.load_model import load_param
import logging


def train_net(ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              shuffle, resume, frequent,
              lr=0.001, lr_step='5', lr_factor=0.5):
    # set up logger
    log_file = "log"
    log_dir = prefix.rsplit("/", 1)[0]
    print(log_dir)
    log_file_full_name = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s Node[' + str(mx.kvstore.create("local").rank) + '] %(message)s'

    logger = logging.getLogger()
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # load symbol
    sym = eval('get_' + default.network + '_train')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)

    # load dataset and prepare imdb for training
    imdb = eval(default.dataset)(default.image_set, default.root_path, default.dataset_path,
                                 ['__background__', 'adult_males', 'subadult_males', 'adult_females', 'juveniles',
                                  'pups'])
    roidb = imdb.gt_roidb()
    roidb = filter_roidb(roidb)

    # load training data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=shuffle,
                              ctx=ctx, work_load_list=None,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # infer max shape
    max_data_shape = [
        ('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
    print('providing maximum shape', max_data_shape, max_label_shape)

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print('output shape')
    pprint.pprint(out_shape_dict)

    # load and initialize params
    if resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
                arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
                aux_params[k].shape)

    # create solver
    fixed_param_prefix = config.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=None,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNL1LossMetric()
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # decide learning rate
    base_lr = lr
    lr_factor = lr_factor
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)
