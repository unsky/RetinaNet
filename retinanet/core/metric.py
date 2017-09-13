# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx
import numpy as np


def get_rpn_names():
    pred = ['rpn_cls_prob/p3','rpn_bbox_loss/p3', 'rpn_cls_prob/p4', 'rpn_bbox_loss/p4','rpn_cls_prob/p5', 'rpn_bbox_loss/p5']

    label = ['rpn_label/p3','rpn_label/p4','rpn_label/p5','rpn_bbox_target/p3', 'rpn_bbox_target/p4', 'rpn_bbox_target/p5','rpn_bbox_weight/p3','rpn_bbox_weight/p4','rpn_bbox_weight/p5']
    return pred, label


class RetinaAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaAccMetric, self).__init__('RetinaAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('rpn_cls_prob/p3')]
        label = labels[self.label.index('rpn_label/p3')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')[0]
       # label = label[0]
        # filter with keep_inds
#        print len(label)
        keep_inds = np.where((label != -1)&(label!=0))


 #       print len(keep_inds[0])

        pred_label = pred_label[keep_inds]

        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

        pred = preds[self.pred.index('rpn_cls_prob/p4')]
        label = labels[self.label.index('rpn_label/p4')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')[0]
        #label = label[0]
        # filter with keep_inds
        keep_inds = np.where((label != -1)&(label!=0))
  #      keep_inds = np.where(label != -1)
     #   keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

        pred = preds[self.pred.index('rpn_cls_prob/p5')]
        label = labels[self.label.index('rpn_label/p5')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')[0]
       # label = label[0]
        keep_inds = np.where((label != -1)&(label!=0))
    #    keep_inds = np.where(label != -1)
        # filter with keep_inds
       # keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)




class RetinaFocalLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaFocalLossMetric, self).__init__('RetinaFocalLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob/p3')]
        label = labels[self.label.index('rpn_label/p3')]
        gamma = 2
        alpha = 0.25
        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss =  -1 *alpha* np.power(1 - cls, gamma) * np.log(cls)

        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

        pred = preds[self.pred.index('rpn_cls_prob/p4')]
        label = labels[self.label.index('rpn_label/p4')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss =  -1 *alpha* np.power(1 - cls, gamma) * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

        pred = preds[self.pred.index('rpn_cls_prob/p5')]
        label = labels[self.label.index('rpn_label/p5')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
       # cls_loss = -1 * np.log(cls)
        cls_loss =  -1 *alpha* np.power(1 - cls, gamma) * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RetinaL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaL1LossMetric, self).__init__('RetinaL1Loss/p3')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss/p3')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p3')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p4')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

        bbox_loss = preds[self.pred.index('rpn_bbox_loss/p5')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p5')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


