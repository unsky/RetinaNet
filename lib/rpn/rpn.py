"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from generate_anchor import generate_anchors
from bbox.bbox_transform import bbox_overlaps, bbox_transform

import time
def get_rpn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    data = [{'data': im_array[i],
            'im_info': im_info[i]} for i in range(len(roidb))]
    label = {}

    return data, label, im_info


def get_rpn_batch(roidb, cfg):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)
#    print roidb[0]
    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label


def assign_anchor(feat_shape_p3,feat_shape_p4,feat_shape_p5,
                  gt_boxes, im_info, cfg,
                  feat_stride_p3=4,scales_p3=(8,), ratios_p3=(0.75, 1, 1.5),
                  feat_stride_p4=8,scales_p4=(8,), ratios_p4=(0.75, 1, 1.5),
                  feat_stride_p5=16,scales_p5=(8,), ratios_p5=(0.75, 1, 1.5),allowed_border=1):
    
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: list of infer output shape
    :param gt_boxes: assign ground truth:[n, 5]
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
 
    feat_shape = [feat_shape_p3,feat_shape_p4,feat_shape_p5]
    feat_stride=[4,8,16]
    scales=(8,)
    ratios=(0.25,0.5, 1, 2,3)

    
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    debug = True
    im_info = im_info[0]
    #print 'im_info: ', im_info
    scales = np.array(scales, dtype=np.float32)
    if len(feat_stride) != len(feat_shape):
        assert('length of feat_stride is not equal to length of feat_shape')
    
    labels_list =[]
    bbox_targets_list =[]
    bbox_weights_list = []
    #print 'length of feat_shape: ',len(feat_shape)
    for i in range(len(feat_shape)):
        total_anchors = 0
        base_anchors = generate_anchors(base_size=feat_stride[i], ratios=list(ratios), scales=scales)
        num_anchors = base_anchors.shape[0]#3
        #print feat_shape[i]
        feat_height, feat_width = (feat_shape[i])[-2:]
        # 1. generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_stride[i]
        shift_y = np.arange(0, feat_height) * feat_stride[i]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors#3
        K = shifts.shape[0]#h*w
        all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))#(k*A,4) in the original image
        

         # keep only inside anchors
        anchors = all_anchors
        # inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
        #                    (all_anchors[:, 1] >= -allowed_border) &
        #                    (all_anchors[:, 2] < im_info[1] + allowed_border) &
        #                    (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        # label: 1 is positive, 0 is negative, -1 is dont care
        total_anchors = len(anchors)#3*w*h
     #   anchors = all_anchors[inds_inside, :]
        labels = np.empty((total_anchors,), dtype=np.float32)
        labels.fill(-1)

        if gt_boxes.size > 0:
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
      
            argmax_overlaps = overlaps.argmax(axis=1)
            gt_labels = gt_boxes[:,-1]
            gt_labels_ =  np.zeros((total_anchors, len(gt_labels)), dtype=np.int)
            gt_labels_[:,:] = gt_labels
    
            labels = gt_labels_[np.arange(total_anchors),argmax_overlaps]

            max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]
      
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
               labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    
            labels[gt_argmax_overlaps] = 1
            labels[(max_overlaps >= cfg.TRAIN.RPN_NEGATIVE_OVERLAP) & (max_overlaps < cfg.TRAIN.RPN_POSITIVE_OVERLAP)] = -1
            # bg_inds = np.where(labels == 0)[0]
            # if len(bg_inds) > 256:
            #     disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - 256), replace=False)
            # labels[disable_inds] = -1
        else:
            labels[:] = 0

        bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
        if gt_boxes.size > 0:
            bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])
        bbox_weights = np.zeros((total_anchors, 4), dtype=np.float32)
        bbox_weights[labels >0, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    
        # map up to original set of anchors
        labels = _unmap(labels, int(K * A), range(total_anchors), fill=-1)
        bbox_targets = _unmap(bbox_targets, int(K * A), range(total_anchors), fill=0)
        bbox_weights = _unmap(bbox_weights, int(K * A), range(total_anchors), fill=0)
 
        

        labels = labels.reshape((1, A * feat_height * feat_width))
    
        bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
        bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
        labels_list.append(labels)
        bbox_targets_list.append(bbox_targets)
        bbox_weights_list.append(bbox_weights)



    if len(feat_shape) == 3:
          label = {'label/p3': labels_list[0], 'label/p4': labels_list[1], 'label/p5': labels_list[2],
           'bbox_target/p3': bbox_targets_list[0], 'bbox_target/p4': bbox_targets_list[1], 'bbox_target/p5': bbox_targets_list[2],
           'bbox_weight/p3': bbox_weights_list[0], 'bbox_weight/p4': bbox_weights_list[1], 'bbox_weight/p5': bbox_weights_list[2]}
 
    return label
