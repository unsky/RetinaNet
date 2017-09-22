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
    pred = ['cls_prob','bbox_loss']

    label = ['rpn_label/p2','rpn_label/p3','rpn_label/p4','rpn_label/p5','rpn_bbox_target/p2','rpn_bbox_target/p3', 'rpn_bbox_target/p4', 'rpn_bbox_weight/p2','rpn_bbox_target/p5','rpn_bbox_weight/p2','rpn_bbox_weight/p3','rpn_bbox_weight/p4','rpn_bbox_weight/p5']
    return pred, label


class RetinaAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaAccMetric, self).__init__('RetinaAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('cls_prob')]
        label1 = labels[self.label.index('rpn_label/p2')].asnumpy().astype('int32')[0]
        label2 = labels[self.label.index('rpn_label/p3')].asnumpy().astype('int32')[0]
        label3 = labels[self.label.index('rpn_label/p4')].asnumpy().astype('int32')[0]
        label4 = labels[self.label.index('rpn_label/p5')].asnumpy().astype('int32')[0]
        label = np.hstack((label1,label2,label3,label4))

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
    #    label = label.asnumpy().astype('int32')[0]
        
       # label = label[0]
        # filter with keep_inds
#        print len(label)
        keep_inds = np.where((label != -1)&(label !=0))
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
    
        # keep_inds_1 = np.where((label != -1)&(label!=0))
        # pred_label_1 = pred_label[keep_inds_1]
        # label_1 = label[keep_inds_1]
        # print "The imblance rate(!0 vs 0 vs -1):", len(label_1),' vs ',len(label),'vs',len(label[label==-1])
        # print 'The total acc (per/img):', np.sum(pred_label.flat == label.flat)*1.0/len(label)
        # print 'The acc without background class:',np.sum(pred_label_1.flat == label_1.flat)*1.0/len(label_1)*1.0
        


        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class RetinaToalAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaToalAccMetric, self).__init__('RetinaToalAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('cls_prob')]
        label1 = labels[self.label.index('rpn_label/p2')].asnumpy().astype('int32')[0]
        label2 = labels[self.label.index('rpn_label/p3')].asnumpy().astype('int32')[0]
        label3 = labels[self.label.index('rpn_label/p4')].asnumpy().astype('int32')[0]
        label4 = labels[self.label.index('rpn_label/p5')].asnumpy().astype('int32')[0]
        label = np.hstack((label1,label2,label3,label4))

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
    #    label = label.asnumpy().astype('int32')[0]
        
       # label = label[0]
        # filter with keep_inds
#        print len(label)
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
    
        # keep_inds_1 = np.where((label != -1)&(label!=0))
        # pred_label_1 = pred_label[keep_inds_1]
        # label_1 = label[keep_inds_1]
        # print "The imblance rate(!0 vs 0 vs -1):", len(label_1),' vs ',len(label),'vs',len(label[label==-1])
        # print 'The total acc (per/img):', np.sum(pred_label.flat == label.flat)*1.0/len(label)
        # print 'The acc without background class:',np.sum(pred_label_1.flat == label_1.flat)*1.0/len(label_1)*1.0
        


        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)



class RetinaFocalLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaFocalLossMetric, self).__init__('RetinaFocalLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('cls_prob')]
        label1 = labels[self.label.index('rpn_label/p2')].asnumpy().astype('int32')[0]
        label2 = labels[self.label.index('rpn_label/p3')].asnumpy().astype('int32')[0]
        label3 = labels[self.label.index('rpn_label/p4')].asnumpy().astype('int32')[0]
        label4 = labels[self.label.index('rpn_label/p5')].asnumpy().astype('int32')[0]
        label = np.hstack((label1,label2,label3,label4)).reshape((-1))

        gamma = 2
        alpha = 0.25
        # label (b, p)
     #   label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        #cls_loss =  -1 *np.log(cls)
        cls_loss = -1 *alpha* np.power(1 - cls, gamma) * np.log(cls)


        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RetinaL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaL1LossMetric, self).__init__('RetinaL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label1 = labels[self.label.index('rpn_label/p2')].asnumpy().astype('int32')[0]
        label2 = labels[self.label.index('rpn_label/p3')].asnumpy().astype('int32')[0]
        label3 = labels[self.label.index('rpn_label/p4')].asnumpy().astype('int32')[0]
        label4 = labels[self.label.index('rpn_label/p5')].asnumpy().astype('int32')[0]
        label = np.hstack((label1,label2,label3,label4))
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

  


