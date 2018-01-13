# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------														  
import _init_paths
import cv2
import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Retinanet')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import AnchorLoader
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)
    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()   

    sym = sym_instance.get_retina_symbol(config, is_train=True)
    feat_sym = []
     
    feat_sym_p4 = sym.get_internals()['box_pred/p4_output']
    feat_sym_p5 = sym.get_internals()['box_pred/p5_output']
    feat_sym_p6 = sym.get_internals()['box_pred/p6_output'] 
    feat_sym_p7 = sym.get_internals()['box_pred/p7_output']   

    feat_sym.append(feat_sym_p4)
    feat_sym.append(feat_sym_p5)
    feat_sym.append(feat_sym_p6)
    feat_sym.append(feat_sym_p7)
    #######
    feat_stride = []
    feat_stride.append(config.network.p4_RPN_FEAT_STRIDE)
    feat_stride.append(config.network.p5_RPN_FEAT_STRIDE)
    feat_stride.append(config.network.p6_RPN_FEAT_STRIDE)
    feat_stride.append(config.network.p7_RPN_FEAT_STRIDE)
    anchor_scales = []
   
    anchor_scales.append(config.network.p4_ANCHOR_SCALES)
    anchor_scales.append(config.network.p5_ANCHOR_SCALES)
    anchor_scales.append(config.network.p6_ANCHOR_SCALES)
    anchor_scales.append(config.network.p7_ANCHOR_SCALES)
    anchor_ratios = []
   
    anchor_ratios.append(config.network.p4_ANCHOR_RATIOS)
    anchor_ratios.append(config.network.p5_ANCHOR_RATIOS)
    anchor_ratios.append(config.network.p6_ANCHOR_RATIOS)
    anchor_ratios.append(config.network.p7_ANCHOR_RATIOS)
    #############


    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            flip=config.TRAIN.FLIP)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)

  

    roidb = filter_roidb(roidb, config)


    # load training data
    train_data = AnchorLoader(feat_sym,feat_stride,anchor_scales,anchor_ratios, 
                              roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx,                 
                              aspect_grouping=config.TRAIN.ASPECT_GROUPING)
    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    print 'providing maximum shape', max_data_shape, max_label_shape
        # infer max shape

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)
    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix)

    if config.TRAIN.RESUME:
        mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

    # decide training params
    # metric
    Retina_toal_eval_metric = metric.RetinaToalAccMetric()
    Retina_cls_metric = metric.RetinaFocalLossMetric()
    Retina_bbox_metric = metric.RetinaL1LossMetric()

    eval_metrics = mx.metric.CompositeEvalMetric()
    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [Retina_toal_eval_metric,Retina_cls_metric,Retina_bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print lr_step.split(',')
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {
                        'learning_rate': lr,
                        'wd':0.0001,
                        }

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)
    # train
    initializer = mx.init.MSRAPrelu(factor_type='out',slope = 0)
   # adam = mx.optimizer.AdaDelta(rho=0.09,  epsilon=1e-14)
    #optimizer_params=optimizer_params, 

    print "-----------------------train--------------------------------"
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params, initializer = initializer,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
