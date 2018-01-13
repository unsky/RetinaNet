"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

DEBUG = False
def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def vis_all_detection(im_array, detections):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
   # print im_array.shape
    import matplotlib  
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import savefig  
    import random
    a =  [103.06 ,115.9 ,123.15]
    a = np.array(a)
    im = transform_inverse(im_array,a)
    plt.imshow(im)
    for j in range(detections.shape[0]):
        # if class_names[j] == 0:
        #     continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        det =dets
        bbox = det[0:] 
        score = det[0]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        # plt.gca().text(bbox[0], bbox[1] - 2,
        #                    '{:s} {:.3f}'.format(str(class_names[j]), score),
        #                    bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
    name = np.mean(im)
    savefig ('vis_restore/'+str(name)+'.png')
    plt.clf()
    plt.cla()

    plt. close(0)




class RestoreRoisOperator(mx.operator.CustomOp):
    def __init__(self,feat_stride, scales, ratios,num_classes, keep_num):
        super(RestoreRoisOperator, self).__init__()
        self._feat_stride = feat_stride
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._num_anchors = len(self._scales)*len(self._ratios)
        self._keep_num = keep_num
        self._num_classes = num_classes

    def forward(self, is_train, req, in_data, out_data, aux):


        cls_pro = in_data[4]
    
        bbox_pred_dict = {
            'stride128': in_data[3],
            'stride64': in_data[2],
            'stride32': in_data[1],
            'stride16': in_data[0],
        }
        cls_prob_dict = {
            'stride128': in_data[7],
            'stride64': in_data[6],
            'stride32': in_data[5],
            'stride16': in_data[4],
        }
        im_info = in_data[8].asnumpy()[0, :]
        im = in_data[9].asnumpy()
        

        proposal_list = []
        score_list = []

        destore_rois_list =[]
        destore_cls_list =[]

        for s in self._feat_stride:
            stride = int(s)
            sub_anchors = generate_anchors(base_size=stride, scales=self._scales, ratios=self._ratios)
            bbox_deltas = bbox_pred_dict['stride' + str(s)].asnumpy()
            # im_info = in_data[-1].asnumpy()[0, :]
            # 1. Generate proposals from bbox_deltas and shifted anchors
            # use real image size instead of padded feature map sizes
      
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            # Enumerate all shifts
            shift_x = np.arange(0, width) * stride
            shift_y = np.arange(0, height) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

            # Enumerate all shifted anchors:
            #
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = self._num_anchors
            K = shifts.shape[0]
            anchors = sub_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))
            # Transpose and reshape predicted bbox transformations to get them
            # into the same order as the anchors:
            #
            # bbox deltas will be (1, 4 * A, H, W) format
            # transpose to (1, H, W, 4 * A)
            # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
            # in slowest to fastest order
            bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
            # Same story for the scores:
            # scores are (1, A, H, W) format
            # transpose to (1, H, W, A)
            # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
            # Convert anchors into proposals via bbox transformations
            proposals = bbox_pred(anchors, bbox_deltas)

            proposals = clip_boxes(proposals, im_info[:2])
            
            scores = cls_prob_dict['stride' + str(s)].asnumpy()
            s_list = []
            start = 0
 
            for i in range(self._num_classes):
                s = scores[:, start : start + self._num_anchors, :, :]
                start = start + self._num_anchors
                s = self._clip_pad(s, (height, width))   
                s = s.transpose((0, 2, 3, 1)).reshape((-1, 1))
                s_list.append(s)
            scores = np.concatenate(s_list, axis=1)
            
            destore_rois_list.append(proposals)
            destore_cls_list.append(scores)

        destore_rois = np.concatenate(destore_rois_list, axis=0)
        destore_cls = np.concatenate(destore_cls_list, axis=0)
    
    #    print destore_cls
        s = np.max(destore_cls,axis = 1)
  #      print s
    
        order = s.ravel().argsort()[::-1]
        order = order[:self._keep_num]
        destore_cls = destore_cls[order, :]
        destore_rois = destore_rois[order,:]

        vis = False
        if vis:
            vis_all_detection(im, destore_rois[:,:])
    
        self.assign(out_data[0], req[0], mx.nd.array(destore_rois))
       
        self.assign(out_data[1], req[1], mx.nd.array(destore_cls))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

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


@mx.operator.register("restore_rois")
class RestoreRoisProp(mx.operator.CustomOpProp):
    def __init__(self,feat_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)',num_classes = 20,keep_num='200'):
        super(RestoreRoisProp, self).__init__(need_top_grad=False)
        self._feat_stride = np.fromstring(feat_stride[1:-1], dtype=int, sep=',')
        self._scales = scales
        self._ratios = ratios
        self._keep_num =int( keep_num)
        self._num_classes = int(num_classes)

    def list_arguments(self):
        return [ 'bbox_p4','bbox_p5','bbox_p6','bbox_p7','cls_p4','cls_p5','cls_p6','cls_p7','im_info','im']

    def list_outputs(self):
        
        return ['rois', 'cls_score']
    def infer_shape(self, in_shape):
    
        out_shape1 = [self._keep_num,4]
        out_shape2 = [self._keep_num,self._num_classes]
    


        return [in_shape[0],in_shape[1],in_shape[2],in_shape[3],in_shape[4],in_shape[5],in_shape[6],in_shape[7],in_shape[8],in_shape[9]], [out_shape1,out_shape2]

    def create_operator(self, ctx, shapes, dtypes):
        return RestoreRoisOperator(self._feat_stride, self._scales, self._ratios,self._num_classes,self._keep_num)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
