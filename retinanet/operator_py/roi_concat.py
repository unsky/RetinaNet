"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import os,sys
import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool
import copy


from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

import time
# from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
# from rcnn.processing.generate_anchor import generate_anchors
# from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

DEBUG = False

class RoIConcatOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios):
        super(RoIConcatOperator, self).__init__()
        self._feat_stride = feat_stride
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')


        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

    def forward(self, is_train, req, in_data, out_data, aux):


        # the first set of anchors are background probabilities
        # keep the second part
        #print 'score_list shape:',scores_list.shape
        bbox_deltas = in_data[0].asnumpy()#[1,n*2]
        im_info = in_data[1].asnumpy()[0, :]
        p2_shape = in_data[2].asnumpy().shape
        p3_shape = in_data[3].asnumpy().shape
        p4_shape = in_data[4].asnumpy().shape
        p5_shape = in_data[5].asnumpy().shape
        feat_shape = []
        feat_shape.append(p2_shape)
        feat_shape.append(p3_shape)
        feat_shape.append(p4_shape)
        feat_shape.append(p5_shape)      
        #t = time.time()
        #print 'feat_shape:', feat_shape
        num_feat = len(feat_shape)#[1,5,4]
        all_anchors_list=[]
        #t_1 = time.time()
        for i in range(num_feat):
            feat_stride = int(self._feat_stride[i])#,8,16,32,64
            #print 'feat_stride:', feat_stride
            anchor = generate_anchors(feat_stride, scales=self._scales, ratios=self._ratios)
            num_anchors = anchor.shape[0]#3
            height = feat_shape[i][2]
            width = feat_shape[i][3]

            shift_x = np.arange(0, width) * feat_stride
            shift_y = np.arange(0, height) * feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
            A = num_anchors#3
            K = shifts.shape[0]#height*width
            anchors = anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))#3*height*widht,4
            all_anchors_list.append(anchors)


        all_anchors = np.concatenate(all_anchors_list, axis=0)
        proposals = bbox_pred(all_anchors, bbox_deltas)#debug here, corresponding?
        proposals = clip_boxes(proposals, im_info[:2])
  
        self.assign(out_data[0], req[0], proposals)
        #print 'roi concate spends :{:.4f}s'.format(time.time()-t)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[0], 0)
        self.assign(in_grad[4], req[0], 0)
        self.assign(in_grad[5], req[0], 0)


    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("roi_concat")
class RoIConcatProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='8,16,32,64', scales='(8)', ratios='(0.5, 1, 2)'):
        super(RoIConcatProp, self).__init__(need_top_grad=False)
        self._feat_stride = [int(i) for i in feat_stride.split(',')]
        self._scales = scales
        self._ratios = ratios

    def list_arguments(self):
        return ['bbox_pred','cls_pred', 'im_info', 'p2','p3','p4','p5']
    def list_outputs(self):

        return ['output']

    def infer_shape(self, in_shape):
        flatten_bbox_pred_shape = in_shape[0]
        im_info_shape = in_shape[2]


        output_shape = in_shape[0]



        return [flatten_bbox_pred_shape, in_shape[1],im_info_shape, in_shape[3],in_shape[4],in_shape[5],in_shape[6]], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RoIConcatOperator(self._feat_stride, self._scales, self._ratios)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
