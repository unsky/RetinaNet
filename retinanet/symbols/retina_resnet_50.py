
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import mxnet as mx
import numpy as np
from utils.symbol import Symbol

from operator_py.focal_loss import *
from operator_py.restore_rois import *
from operator_py.smoothl1loss import *


class retina_resnet_50(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 2e-5
        self.USE_GLOBAL_STATS = True
        self.workspace = 512
        self.res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
    def residual_unit(self,data, num_filter, stride, dim_match, name,use_global_stats=True, bn_mom=0.9, bottle_neck=True, dilate=(1, 1)):
        workspace =self.workspace
        eps = self.eps
        if bottle_neck:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=dilate, 
                                    no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1), dilate=dilate, 
                                    no_bias=True, workspace=workspace, name=name + '_conv2')
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, dilate=dilate, 
                                    workspace=workspace, name=name + '_conv3')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, dilate=dilate, 
                                            workspace=workspace, name=name + '_sc')
            sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
            return sum
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), dilate=dilate, 
                                        no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), dilate=dilate, 
                                        no_bias=True, workspace=workspace, name=name + '_conv2')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True, dilate=dilate, 
                                                workspace=workspace, name=name+'_sc')
                
            sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
            return sum

    def get_fpn_resnet_conv(self,data, depth): #add bn to fpn layer:2017-08-01
        units = self.res_deps[str(depth)]
        filter_list = [256, 512, 1024, 2048, 256] if depth >= 50 else [64, 128, 256, 512, 256]

        bottle_neck = True if depth >= 50 else False
        USE_GLOBAL_STATS =self.USE_GLOBAL_STATS
        workspace = self.workspace
        eps = self.eps
        # res1
        data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
        conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
        bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn0')
        relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
        pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

        # res2
        conv1 = self.residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[0] + 1):
            conv1 = self.residual_unit(data=conv1, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                                bottle_neck=bottle_neck)
        #stride = 4

        # res3
        conv2 = self.residual_unit(data=conv1, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[1] + 1):
            conv2 = self.residual_unit(data=conv2, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                                bottle_neck=bottle_neck)
        # stride = 8
        # res4
        conv3 = self.residual_unit(data=conv2, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[2] + 1):
            conv3 = self.residual_unit(data=conv3, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                                bottle_neck=bottle_neck)
        #stride = 16
        # res5
        conv4 = self.residual_unit(data=conv3, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[3] + 1):
            conv4 = self.residual_unit(data=conv4, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i,
                                bottle_neck=bottle_neck)
        # bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='stage5_bn1')
        # act4 = mx.sym.Activation(data=bn4, act_type='relu', name='stage5_relu1')
        #stride = 32
      #  conv4 =  mx.symbol.Activation( data=conv4, act_type='relu') 
        up_conv6_out = mx.symbol.Convolution(data=conv4, kernel=(3, 3), pad=(1,1), stride=(2,2), num_filter=filter_list[4], name='stage6_conv_3*3')
        #stride = 64
        # de-res5
       # up_conv6_out_relu = mx.symbol.Activation( data=up_conv6_out, act_type='relu')
        up_conv7_out = mx.symbol.Convolution(data=up_conv6_out, kernel=(3, 3), pad=(1,1), stride=(2,2), num_filter=filter_list[4], name='stage7_conv_3*3')

        up_conv5_out = mx.symbol.Convolution(data=conv4, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='stage5_conv_1x1')

        up_conv4 = mx.symbol.UpSampling(up_conv5_out, scale=2, sample_type="nearest")
        #bn_up_conv4 = mx.sym.BatchNorm(data=up_conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv4_bn1')
        conv3_1 = mx.symbol.Convolution(
            data=conv3, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage4_conv_1x1')
        #bn_conv3_1 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv3_1_bn1')
        up_conv4_ = up_conv4 + conv3_1
        up_conv4_out = mx.symbol.Convolution(
            data=up_conv4_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage4_conv_3x3')
        

        
        output = []
     
        output.append(up_conv4_out)#stride:16
        output.append(up_conv5_out)#stride:32
        output.append(up_conv6_out)#stride:64
        output.append(up_conv7_out)#stride:128

        return output

    def get_retina_symbol(self, cfg, is_train=False):
        """ resnet symbol for scf train and test """
            # input init
        num_classes = cfg.dataset.NUM_CLASSES-1
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors_list = []
        num_anchors_list.append(cfg.network.p4_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p5_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p6_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p7_NUM_ANCHORS)

        if is_train:
            data = mx.sym.Variable(name="data")
            # gt_boxes = mx.sym.Variable(name="gt_boxes")
            retina_bbox_target= mx.sym.Variable(name='bbox_target')
            retina_bbox_weight = mx.sym.Variable(name='bbox_weight')
            retina_label = mx.sym.Variable(name='label')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

    #############share weight
        cls_conv1_3x3_weight = mx.symbol.Variable(name = 'cls_conv1_3x3_weight')
        cls_conv1_3x3_bias = mx.symbol.Variable(name = 'cls_conv1_3x3_bias')

        cls_conv2_3x3_weight = mx.symbol.Variable(name = 'cls_conv2_3x3_weight')
        cls_conv2_3x3_bias = mx.symbol.Variable(name = 'cls_conv2_3x3_bias')

        cls_conv3_3x3_weight = mx.symbol.Variable(name = 'cls_conv3_3x3_weight')
        cls_conv3_3x3_bias = mx.symbol.Variable(name = 'cls_conv3_3x3_bias')

        cls_conv4_3x3_weight = mx.symbol.Variable(name = 'cls_conv4_3x3_weight')
        cls_conv4_3x3_bias = mx.symbol.Variable(name = 'cls_conv4_3x3_bias')

        cls_score_weight =  mx.symbol.Variable(name = 'cls_score_weight')
        cls_score_bias = mx.symbol.Variable(name = 'cls_score_bias')

        box_conv1_weight = mx.symbol.Variable(name = 'box_conv1_weight')
        box_conv1_bias = mx.symbol.Variable(name = 'box_conv1_bias')

        box_conv2_weight = mx.symbol.Variable(name = 'box_conv2_weight')
        box_conv2_bias = mx.symbol.Variable(name = 'box_conv2_bias')

        box_conv3_weight = mx.symbol.Variable(name = 'box_conv3_weight')
        box_conv3_bias = mx.symbol.Variable(name = 'box_conv3_bias')

        box_conv4_weight = mx.symbol.Variable(name = 'box_conv4_weight')
        box_conv4_bias = mx.symbol.Variable(name = 'box_conv4_bias')

        box_pred_weight = mx.symbol.Variable(name = 'box_pred_weight')
        box_pred_bias = mx.symbol.Variable(name = 'box_pred_bias')
        ############################################
        depth = 50 ##resnet50
        conv_feat = self.get_fpn_resnet_conv(data, depth)
        sublayer_depth = 4
        # subnet
        #####################share params ##########################
        cls_score_list = []
        bbox_pred_list = []
        cls_score_test = []
        bbox_test = []
        for i in range(sublayer_depth):
            cls_conv1 = mx.sym.Convolution(
                    data=conv_feat[i], kernel=(3, 3), pad=(1, 1),weight = cls_conv1_3x3_weight,bias = cls_conv1_3x3_bias, num_filter=256, name="cls_conv1_3x3/p"+str(i+4))        
            cls_conv1 = mx.symbol.Activation( data=cls_conv1, act_type='relu')  
        #    cls_conv1 = mx.sym.Custom(op_type='Check',  data=cls_conv1)   
            cls_conv2 = mx.sym.Convolution(
                    data=cls_conv1, kernel=(3, 3), pad=(1, 1),weight = cls_conv2_3x3_weight, bias = cls_conv2_3x3_bias, num_filter=256, name="cls_conv2_3x3/p"+str(i+4))
            cls_conv2 = mx.symbol.Activation( data=cls_conv2, act_type='relu')
            cls_conv3 = mx.sym.Convolution(
                    data=cls_conv2, kernel=(3, 3),  pad=(1, 1), weight = cls_conv3_3x3_weight,bias = cls_conv3_3x3_bias,num_filter=256, name="cls_conv3_3x3/p"+str(i+4))      
            cls_conv3 = mx.symbol.Activation( data=cls_conv3, act_type='relu')    
            cls_conv4 = mx.sym.Convolution(
                    data=cls_conv3, kernel=(3, 3),  pad=(1, 1),weight = cls_conv4_3x3_weight,bias = cls_conv4_3x3_bias, num_filter=256, name="cls_conv4_3x3/p"+str(i+4)) 
            cls_conv4 = mx.symbol.Activation( data=cls_conv4, act_type='relu')
            cls_score = mx.sym.Convolution(
                    data=cls_conv4, kernel=(3, 3),pad=(1, 1), weight = cls_score_weight,bias= cls_score_bias, num_filter=num_classes * num_anchors_list[i], name="cls_score/p"+str(i+4))
            cls_score_reshape_ = mx.sym.Reshape(
                    data=cls_score, shape = (0, num_classes* num_anchors_list[i], -1, 0), name="cls_score_reshape/p"+str(i+4))         
            if is_train:
                cls_score_reshape = mx.sym.Reshape(
                    data=cls_score_reshape_, shape = (0, num_classes, -1)) 
            else:
                cls_score_reshape = mx.sym.sigmoid(cls_score_reshape_)
            cls_score_list.append(cls_score_reshape) 

            
            
            box_conv1 = mx.sym.Convolution(
                    data=conv_feat[i], kernel=(3, 3), pad=(1, 1),weight = box_conv1_weight,bias = box_conv1_bias, num_filter=256, name="box_conv1_3x3/p"+str(i+4))
            box_conv1 = mx.symbol.Activation( data=box_conv1, act_type='relu')
            box_conv2 = mx.sym.Convolution(
                    data = box_conv1, kernel=(3, 3), pad=(1, 1),weight = box_conv2_weight,bias = box_conv2_bias, num_filter=256, name="box_conv2_3x3/p"+str(i+4))
            box_conv2 = mx.symbol.Activation( data=box_conv2, act_type='relu')
            box_conv3 = mx.sym.Convolution(
                    data = box_conv2, kernel=(3, 3),pad=(1, 1),weight= box_conv3_weight,bias = box_conv3_bias, num_filter=256, name="box_conv3_3x3/p"+str(i+4))  
            box_conv3 = mx.symbol.Activation( data=box_conv3, act_type='relu')   
            box_conv4 = mx.sym.Convolution(
                    data = box_conv3, kernel=(3, 3), pad=(1, 1),weight = box_conv4_weight,bias = box_conv4_bias, num_filter=256, name="box_conv4_3x3/p"+str(i+4)) 
            box_conv4 = mx.symbol.Activation( data=box_conv4, act_type='relu')
            bbox_pred = mx.sym.Convolution(
                    data=box_conv4, kernel=(3, 3), pad=(1, 1),weight = box_pred_weight, bias = box_pred_bias,num_filter=4 * num_anchors_list[i], name="box_pred/p"+str(i+4)) 
            if is_train:
                bbox_pred = mx.sym.Reshape(data= bbox_pred,shape=(0, 0,-1))

            bbox_pred_list.append(bbox_pred)
        
        
         
        if is_train:
            cls_concat = mx.sym.Concat(cls_score_list[0],cls_score_list[1],cls_score_list[2],cls_score_list[3],dim=2,name='cls_concat')
            bbox_pred_concat = mx.sym.Concat(bbox_pred_list[0],bbox_pred_list[1],bbox_pred_list[2],bbox_pred_list[3],dim=2,name='bbox_pred_conncat') 
            retina_cls_prob = mx.sym.Custom(op_type='FocalLoss', name = 'cls_prob', data=cls_concat, num_classes = num_classes,labels=retina_label,alpha =0.25, gamma= 2,use_ignore=True,ignore_label=-1)                                     
            retina_bbox_loss_ = retina_bbox_weight * mx.sym.smooth_l1(name='retina_bbox_loss', scalar=3.0,
                                                                    data=(bbox_pred_concat - retina_bbox_target))
            retina_bbox_loss = mx.sym.Custom(op_type='SmoothL1Loss', name = 'retina_bbox_loss', 
                                     weight=retina_bbox_weight, bbox_pred = bbox_pred_concat,bbox_target = retina_bbox_target)
            group = mx.sym.Group([retina_cls_prob,retina_bbox_loss])
        else:
            rois, cls_score = mx.sym.Custom( bbox_p4 = bbox_pred_list[0],bbox_p5 = bbox_pred_list[1],bbox_p6 = bbox_pred_list[2],bbox_p7 = bbox_pred_list[3],
            cls_p4=cls_score_list[0],cls_p5=cls_score_list[1],cls_p6=cls_score_list[2],cls_p7=cls_score_list[3],im_info=im_info, im = data,
            name='restored_rois',op_type='restore_rois',scales=cfg.network.p4_ANCHOR_SCALES,ratios= cfg.network.p4_ANCHOR_RATIOS,feat_stride =[16,32,64,128],num_classes=num_classes,keep_num = 1000 )
            group = mx.sym.Group([cls_score,rois,mx.sym.BlockGrad(im_info)])
        self.sym =group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
            pi = 0.01
            arg_params['cls_score_bias'] = mx.nd.ones(shape=self.arg_shape_dict['cls_score_bias'])*(-np.log((1-pi)/pi))

