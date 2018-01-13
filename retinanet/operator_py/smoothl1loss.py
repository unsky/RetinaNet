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
from mxnet import autograd
class SmoothL1LossOperator(mx.operator.CustomOp):
    def __init__(self):
        super(SmoothL1LossOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
      
        weight = in_data[0][:]
        pred = in_data[1][:]
        target = in_data[2][:]
        loss = weight * mx.nd.smooth_l1(data=(pred - target),scalar=3.0)
        

        self.assign(out_data[0],req[0],loss)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
       
        weight = in_data[0][:].asnumpy()
        pred = in_data[1][:]
        target = in_data[2][:]
        index = np.sum(weight>0)/4
        length =  np.max((1,index))
      #  print length
        weight = mx.nd.array(weight)
      
    	pred.attach_grad()
        with autograd.record():
            loss = mx.nd.sum(weight * mx.nd.smooth_l1(data=(pred - target),scalar=3.0))/length
        loss.backward()
        grad = pred.grad.asnumpy()
        # g = np.sum(grad.flatten()[grad.flatten()>0])
        # n = np.sum((1,np.sum(grad>0)))
        # print n,'bbox: ',g/n
      

    
        self.assign(in_grad[0], req[0], mx.nd.array(grad))
        self.assign(in_grad[1],req[1],0)
        self.assign(in_grad[2],req[2],0)
 
         

@mx.operator.register('SmoothL1Loss')
class SmoothL1LossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SmoothL1LossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['weight', 'bbox_pred','bbox_target']

    def list_outputs(self):
        return ['smooth_l1_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        out_shape = data_shape

        return  [in_shape[0], in_shape[1],in_shape[2]],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothL1LossOperator()

   
