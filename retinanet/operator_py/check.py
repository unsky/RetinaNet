# --------------------------------------------------------
# check value
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
check
"""

import mxnet as mx
import numpy as np
class CheckOperator(mx.operator.CustomOp):
    def __init__(self):
        super(CheckOperator, self).__init__()

   
    def forward(self, is_train, req, in_data, out_data, aux):
        # print "--------------------------"

        # value = in_data[0].asnumpy()
        # # print "forward:",np.sum(value)
        self.assign(out_data[0],req[0],in_data[0])
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        value = out_grad[0].asnumpy()
        # print 'backward:',np.sum(value)
        # print "--------------------------"    
        self.assign(in_grad[0], req[0], 0)

 
         

@mx.operator.register('Check')
class CheckProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CheckProp, self).__init__(True)

       
    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['Check']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
    #    print "date_shape:",data_shape
        return  [data_shape],[data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CheckOperator()

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     print "bbb"
    #     return []