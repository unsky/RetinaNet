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
   # print roidb
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

  #  gt boxes: (x1, y1, x2, y2, cls)
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


def assign_anchor(feat_shape_p4,feat_shape_p5,feat_shape_p6,feat_shape_p7,
                  gt_boxes, im_info, cfg,
                  feat_stride_p4=16,scales_p4=(8,), ratios_p4=(0.75, 1, 1.5),
                  feat_stride_p5=32,scales_p5=(8,), ratios_p5=(0.75, 1, 1.5),
                  feat_stride_p6=64,scales_p6=(8,), ratios_p6=(0.75, 1, 1.5),
                    feat_stride_p7=128,scales_p7=(8,), ratios_p7=(0.75, 1, 1.5),
                  allowed_border=0):
    
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
  
    feat_shapes = [feat_shape_p4,feat_shape_p5,feat_shape_p6,feat_shape_p7]
    feat_strides=[16,32,64,128]

    scales= np.array(scales_p5)

    ratios=np.array(ratios_p5)


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
    im_info = im_info[0]

    fpn_args = []
    fpn_anchors_fid = np.zeros(0).astype(int)
    fpn_anchors = np.zeros([0, 4])
    fpn_labels = np.zeros(0)
    fpn_inds_inside = []
    for feat_id in range(len(feat_strides)):
        # len(scales.shape) == 1 just for backward compatibility, will remove in the future
        if len(scales.shape) == 1:
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios, scales=scales)
        else:
            assert len(scales.shape) == len(ratios.shape) == 2
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios[feat_id], scales=scales[feat_id])
        num_anchors = base_anchors.shape[0]

        feat_height,feat_width= feat_shapes[feat_id][-2:]
    
        

        # 1. generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
        shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = [ind for ind in xrange(total_anchors)]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # for sigmoid classifier, ignore the 'background' class
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        fpn_anchors_fid = np.hstack((fpn_anchors_fid, len(inds_inside)))
        fpn_anchors = np.vstack((fpn_anchors, anchors))
        fpn_labels = np.hstack((fpn_labels, labels))
        fpn_inds_inside.append(inds_inside)
        fpn_args.append([feat_height, feat_width, A, total_anchors])

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(fpn_anchors.astype(np.float), gt_boxes.astype(np.float)) 
        argmax_overlaps = overlaps.argmax(axis=1) # (A)
        max_overlaps = overlaps[np.arange(len(fpn_anchors)), argmax_overlaps]


        labels = gt_boxes[argmax_overlaps, 4]
        labels[max_overlaps < cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        fpn_labels = labels
    else:
        fpn_labels[:] = 0

  #  subsample positive labels if we have too many
#     num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
#     fg_inds = np.where(fpn_labels >= 1)[0]
#     if len(fg_inds) > num_fg:
#         disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
#         fpn_labels[disable_inds] = -1
#   #  subsample negative labels if we have too many
#     num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else cfg.TRAIN.RPN_BATCH_SIZE - np.sum(fpn_labels >= 1)
#     bg_inds = np.where(fpn_labels == 0)[0]
    fpn_anchors_fid = np.hstack((0, fpn_anchors_fid.cumsum()))
    # if len(bg_inds) > num_bg:
    #     disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    #     fpn_labels[disable_inds] = -1

    fpn_bbox_targets = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
    if gt_boxes.size > 0:
         #fpn_bbox_targets[fpn_labels >= 1, :] = bbox_transform(fpn_anchors[fpn_labels >= 1, :], gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
         fpn_bbox_targets[:] = bbox_transform(fpn_anchors, gt_boxes[argmax_overlaps, :4])
    # fpn_bbox_targets = (fpn_bbox_targets - np.array(cfg.TRAIN.BBOX_MEANS)) / np.array(cfg.TRAIN.BBOX_STDS)
    fpn_bbox_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

    fpn_bbox_weights[fpn_labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    for feat_id in range(0, len(feat_strides)):
        feat_height, feat_width, A, total_anchors = fpn_args[feat_id]
        # map up to original set of anchors
        labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=-1)
        bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)

        labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, A * feat_height * feat_width))

        bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
        bbox_targets = bbox_targets.reshape((1, A * 4, -1))
        bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
        bbox_weights = bbox_weights.reshape((1, A * 4, -1))

        label_list.append(labels)
        bbox_target_list.append(bbox_targets)
        bbox_weight_list.append(bbox_weights)

    debug_label = np.concatenate(label_list, axis=1)
    # print debug_label
    # print"-----------total:",len(debug_label[0])
    # print "--------ig-",len(debug_label[debug_label==-1])
    # print "--------bg--",len(debug_label[debug_label==0])
    # print "--------gg--",len(debug_label[debug_label>=1])
    # print np.concatenate(label_list, axis=1)[np.concatenate(label_list, axis=1)>=1].shape
    #print np.concatenate(bbox_target_list, axis=2)


    label = {
        'label': np.concatenate(label_list, axis=1),
        'bbox_target': np.concatenate(bbox_target_list, axis=2),
        'bbox_weight': np.concatenate(bbox_weight_list, axis=2)
    }

    return label

