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

    label = ['rpn_label','rpn_bbox_target','rpn_bbox_weight']
    return pred, label




class RetinaToalAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaToalAccMetric, self).__init__('RetinaToalAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('cls_prob')].asnumpy()
        label = labels[self.label.index('rpn_label')].asnumpy().astype('int32').flatten()-1


        # pred (b, c, p) or (b, c, h, w)
        pred_label = np.argmax(pred,axis =1).astype('int').flatten()
        keep_inds = np.where(label > 0)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
        
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)



class RetinaFocalLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaFocalLossMetric, self).__init__('RetinaFocalLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('cls_prob')]
        label = labels[self.label.index('rpn_label')].asnumpy().astype('int32').flatten()[:] - 1
        gamma = 2
        alpha = 0.25
        pred = pred.asnumpy()
        num_classes =20

    	label[label<0] = 0
    	labels_ = np.zeros((1, num_classes, label.shape[0]))
    
    	labels_[0, label, np.arange(label.shape[0],dtype = 'int') ] = 1
        label_ig = labels[self.label.index('rpn_label')].asnumpy().astype('int32').flatten()[:] - 1
        ind_0 = np.where(label_ig<0)[0]
        labels_[0, :, ind_0] = 0

        ind_na = np.where(label_ig>-1)[0]
  

        cls_ = pred[:,:,ind_na]
        labels = labels_[:,:,ind_na]
        nom = cls_.shape[2]
 
        eps = 1e-14
        cls_loss =   np.sum(-1 * alpha * labels * np.power(1 -  cls_+eps, gamma) * np.log(cls_+eps) - (1-labels)*(1-alpha) * np.power(1 - ( 1-cls_)+eps, gamma) * np.log( 1-cls_+eps))
      
        self.sum_metric += cls_loss
        self.num_inst += (nom)

class RetinaL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RetinaL1LossMetric, self).__init__('RetinaL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy().astype('int32').flatten() - 1
 
        num_inst = np.sum(bbox_loss>0)+1

    
        
    
        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

  


