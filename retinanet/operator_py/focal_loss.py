# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss 
"""

import mxnet as mx
import numpy as np
class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self,  gamma, alpha,use_ignore,ignore_label):
        super(FocalLossOperator, self).__init__()
        self._gamma = gamma
        self._alpha = alpha 
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
    def forward(self, is_train, req, in_data, out_data, aux):
      
        cls_score = in_data[0]
        labels = in_data[1].asnumpy()[0]
    
        self._labels = labels
        pro_ = (mx.nd.sigmoid(cls_score) + 1e-14).asnumpy()
        print "pro:", pro_[0]

        # pro_ = np.exp(1e-14+cls_score - cls_score.max(axis=1).reshape((cls_score.shape[0], 1)))
        # pro_ /= pro_.sum(axis=1).reshape((cls_score.shape[0], 1))

        self.pro_ = pro_

       
        self._pt = pro_[np.arange(pro_.shape[0],dtype = 'int'), labels[0].astype('int')]
 
        ### note!!!!!!!!!!!!!!!!
        # focal loss value is not used in this place we should forward the cls_pro in this layer, 
        # the focal vale should be calculated in metric.py
        # the method is in readme
        #  focal loss (batch_size,num_class)
        #loss_ = -1 * np.power(1 - pro_, self._gamma) * np.log(pro_)
        self.assign(out_data[0],req[0],mx.nd.array(pro_))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        labels = self._labels.astype('int')
        pro_ = self.pro_
        #i!=j
        pt = self._pt + 1e-14
    
        pt = pt.reshape(len(pt),1)
        dx = (1-self._alpha)*np.power(1 - pt, self._gamma - 1) * (self._gamma * (-1 * pt * pro_) * np.log(pt) + pro_ * (1 - pt)) * 1.0 

        ####i==j 
        #reload pt
        pt = self._pt + 1e-14
        ig_inds = []
        negative_inds = np.where(labels>-100)

        if self.use_ignore:
            ig_inds= np.where(labels==self.ignore_label)
            dx[ig_inds,:] =0
            negative_inds = np.where(labels!=self.ignore_label)
            pt = pt[negative_inds]
        dx[negative_inds, labels[negative_inds].astype('int')]  = (self._alpha)*np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt -1) * (1.0)
        #0.25*(dx[negative_inds, labels[negative_inds].astype('int')]-1)
        # (self._alpha)*(-1)*dx[negative_inds, labels[negative_inds].astype('int')]
        #(self._alpha)*np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt -1) * (1.0)

        # print 'label:',labels[negative_inds][0]

        # print 'dx',dx[negative_inds][0]

        dx /= labels.shape[0] ##batch 
        # ig_labels = np.where(labels==-1)
        # dx[ig_labels,:]=0
        self.assign(in_grad[0], req[0], mx.nd.array(dx))
        self.assign(in_grad[1],req[1],0)
 
         

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma,alpha,use_ignore,ignore_label):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self.use_ignore =bool(use_ignore)
        print self.use_ignore
        self.ignore_label = int(ignore_label)

        self._gamma = float(gamma)
        self._alpha = float(alpha)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return  [data_shape, labels_shape],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._gamma,self._alpha,self.use_ignore,self.ignore_label)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []