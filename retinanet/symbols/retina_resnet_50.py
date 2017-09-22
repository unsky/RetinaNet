
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import mxnet as mx
import numpy as np
from utils.symbol import Symbol
from operator_py.check import *

eps = 2e-5
USE_GLOBAL_STATS = True
workspace = 512
res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
class retina_resnet_50(Symbol):
    def __init__(self):
        self.eps = 2e-5
        self.USE_GLOBAL_STATS = True
        self.workspace = 512
        self.res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}

    def residual_unit(self,data, num_filter, stride, dim_match, name, use_global_stats=False, bn_mom=0.9, bottle_neck=True, dilate=(1, 1)):
        use_global_stats =self.USE_GLOBAL_STATS
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
        eps=self.eps
        USE_GLOBAL_STATS = self.USE_GLOBAL_STATS
        workspace = self.workspace
        filter_list = [256, 512, 1024, 2048, 256] if depth >= 50 else [64, 128, 256, 512, 256]

        bottle_neck = True if depth >= 50 else False
        
        # res1
        data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
        conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)

  #      conv0 = mx.sym.Custom(op_type='Check', name = 'check111', data=conv0) 

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
 #       conv2 = mx.sym.Custom(op_type='Check', name = 'che221', data=conv2) 
        for i in range(2, units[1] + 1):
            conv2 = self.residual_unit(data=conv2, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                                bottle_neck=bottle_neck)
      #  conv2 = mx.sym.Custom(op_type='Check', name = 'check11122', data=conv2) 
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

        #stride = 64
        # de-res5
                                  
        up_conv5_out = mx.symbol.Convolution(data=conv4, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='stage5_conv_1x1')

        c6 = up_conv_out = mx.symbol.Convolution(data = up_conv5_out,kernel=(3,3),num_filter=filter_list[4],stride=(2, 2), name='c6')


        up_conv4 = mx.symbol.UpSampling(up_conv5_out, scale=2, sample_type="nearest")
        #bn_up_conv4 = mx.sym.BatchNorm(data=up_conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv4_bn1')

      

        conv3_1 = mx.symbol.Convolution(
            data=conv3, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage4_conv_1x1')
        #bn_conv3_1 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv3_1_bn1')
        up_conv4_Crop = mx.sym.Crop(up_conv4,conv3_1)  
        up_conv4_ = up_conv4_Crop + conv3_1
        up_conv4_out = mx.symbol.Convolution(
            data=up_conv4_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage4_conv_3x3')
        
        # de-res4
        up_conv3 = mx.symbol.UpSampling(up_conv4_, scale=2, sample_type="nearest")
        #bn_up_conv3 = mx.sym.BatchNorm(data=up_conv3, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv3_bn1')

      

        conv2_1 = mx.symbol.Convolution(
            data=conv2, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage3_conv_1x1')
        #bn_conv2_1 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv2_1_bn1')
        up_conv3_Crop = mx.sym.Crop(up_conv3,conv2_1)  
        up_conv3_ = up_conv3_Crop + conv2_1
        up_conv3_out = mx.symbol.Convolution(
            data=up_conv3_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage3_conv_3x3')
        

        
        output = []
        output.append(up_conv3_out)#stride:8
        output.append(up_conv4_out)#stride:16
        output.append(up_conv5_out)#stride:32
        output.append(c6)#stride:64
        return output



    # eps = 2e-5
    # use_global_stats = False
    # res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}

    def get_retina_symbol(self, cfg, is_train=False):
        """ resnet symbol for scf train and test """
            # input init
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors_list = []
        num_anchors_list.append(cfg.network.p3_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p4_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p5_NUM_ANCHORS)
        num_anchors_list.append(cfg.network.p6_NUM_ANCHORS)

        if is_train:
            data = mx.sym.Variable(name="data")
            # gt_boxes = mx.sym.Variable(name="gt_boxes")
            
            bbox_target_p3 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_target/p3'),shape = (1,4*num_anchors_list[0],-1))
            bbox_target_p4 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_target/p4'),shape = (1,4*num_anchors_list[1],-1))
            bbox_target_p5 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_target/p5'),shape = (1,4*num_anchors_list[2],-1))
            bbox_target_p6 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_target/p6'),shape = (1,4*num_anchors_list[3],-1))

            
            bbox_weight_p3 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_weight/p3'),shape =(1,4*num_anchors_list[0],-1))
            bbox_weight_p4 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_weight/p4'),shape =(1,4*num_anchors_list[1],-1))
            bbox_weight_p5 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_weight/p5'),shape =(1,4*num_anchors_list[2],-1))
            bbox_weight_p6 = mx.sym.Reshape(data=mx.sym.Variable(name='bbox_weight/p6'),shape = (1,4*num_anchors_list[3],-1))

            retina_label = mx.sym.Concat(mx.sym.Variable(name='label/p3'),mx.sym.Variable(name='label/p4'),mx.sym.Variable(name='label/p5'),mx.sym.Variable(name='label/p6'),dim =1)
      
            retina_bbox_target = mx.sym.Concat(bbox_target_p3,bbox_target_p4,bbox_target_p5,bbox_target_p6,dim =2)
            retina_bbox_weight = mx.sym.Concat(bbox_weight_p3,bbox_weight_p4,bbox_weight_p5,bbox_weight_p6,dim=2)
   
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


        ##########################
        bn1_cls_gamma = mx.symbol.Variable(name = 'bn1_cls_gamma')
        bn1_cls_beta = mx.symbol.Variable(name = 'bn1_cls_beta')
        bn1_cls_moving_mean = mx.symbol.Variable(name = 'bn1_cls_moving_mean')
        bn1_cls_moving_var = mx.symbol.Variable(name = 'bn1_cls_moving_var')

        bn2_cls_gamma = mx.symbol.Variable(name = 'bn2_cls_gamma')
        bn2_cls_beta = mx.symbol.Variable(name = 'bn2_cls_beta')
        bn2_cls_moving_mean = mx.symbol.Variable(name = 'bn2_cls_moving_mean')
        bn2_cls_moving_var = mx.symbol.Variable(name = 'bn2_cls_moving_var')

        bn3_cls_gamma = mx.symbol.Variable(name = 'bn3_cls_gamma')
        bn3_cls_beta = mx.symbol.Variable(name = 'bn3_cls_beta')
        bn3_cls_moving_mean = mx.symbol.Variable(name = 'bn3_cls_moving_mean')
        bn3_cls_moving_var = mx.symbol.Variable(name = 'bn3_cls_moving_var')

        
        bn4_cls_gamma = mx.symbol.Variable(name = 'bn4_cls_gamma')
        bn4_cls_beta = mx.symbol.Variable(name = 'bn4_cls_beta')
        bn4_cls_moving_mean = mx.symbol.Variable(name = 'bn4_cls_moving_mean')
        bn4_cls_moving_var = mx.symbol.Variable(name = 'bn4_cls_moving_var')

        ##############################

        bn1_box_gamma = mx.symbol.Variable(name = 'bn1_box_gamma')
        bn1_box_beta = mx.symbol.Variable(name = 'bn1_box_beta')
        bn1_box_moving_mean = mx.symbol.Variable(name = 'bn1_box_moving_mean')
        bn1_box_moving_var = mx.symbol.Variable(name = 'bn1_box_moving_var')

        bn2_box_gamma = mx.symbol.Variable(name = 'bn2_box_gamma')
        bn2_box_beta = mx.symbol.Variable(name = 'bn2_box_beta')
        bn2_box_moving_mean = mx.symbol.Variable(name = 'bn2_box_moving_mean')
        bn2_box_moving_var = mx.symbol.Variable(name = 'bn2_box_moving_var')


        bn3_box_gamma = mx.symbol.Variable(name = 'bn3_box_gamma')
        bn3_box_beta = mx.symbol.Variable(name = 'bn3_box_beta')
        bn3_box_moving_mean = mx.symbol.Variable(name = 'bn3_box_moving_mean')
        bn3_box_moving_var = mx.symbol.Variable(name = 'bn3_box_moving_var')


        bn4_box_gamma = mx.symbol.Variable(name = 'bn4_box_gamma')
        bn4_box_beta = mx.symbol.Variable(name = 'bn4_box_beta')
        bn4_box_moving_mean = mx.symbol.Variable(name = 'bn4_box_moving_mean')
        bn4_box_moving_var = mx.symbol.Variable(name = 'bn4_box_moving_var')
        #############################


        depth = 50 ##resnet50
        #int(network.split('-')[1])
        conv_feat = self.get_fpn_resnet_conv(data, depth)
        sublayer_depth = 4

        # subnet
        #####################need to share params here!!!!!!!!!!!!!!##########################
        use_global_stats = True
        eps = 2e-5
        bn_mom  = 0.9
        cls_score_list = []
        bbox_pred_list = []
        for i in range(sublayer_depth):
                    ##########cls
            conv_relu = mx.symbol.Activation( data=conv_feat[i], act_type='relu')

            cls_conv1 = mx.sym.Convolution(
                    data=conv_relu, kernel=(3, 3), weight = cls_conv1_3x3_weight, bias = cls_conv1_3x3_bias,pad=(1, 1), num_filter=256, name="cls_conv1_3x3/p"+str(i+3))
            cls_conv1 = mx.sym.BatchNorm(data=cls_conv1, use_global_stats=False, eps=eps,gamma=bn1_cls_gamma, beta =bn1_cls_beta, moving_mean =bn1_cls_moving_mean,
            moving_var= bn1_cls_moving_var,momentum=bn_mom, name = 'bn1_cls')
            cls_conv1 = mx.symbol.Activation( data=cls_conv1, act_type='relu')
            

        #    cls_conv1 = mx.symbol.Dropout(data=cls_conv1, p=0.5)
          
            cls_conv2 = mx.sym.Convolution(
                    data=cls_conv1, kernel=(3, 3), weight=cls_conv2_3x3_weight, bias = cls_conv2_3x3_bias,pad=(1, 1), num_filter=256, name="cls_conv2_3x3/p"+str(i+3))
            cls_conv2 = mx.sym.BatchNorm(data=cls_conv2, use_global_stats=False, eps=eps,gamma=bn2_cls_gamma, beta =bn2_cls_beta, moving_mean =bn2_cls_moving_mean,
            moving_var= bn2_cls_moving_var,momentum=bn_mom, name = 'bn2_cls')
            cls_conv2 = mx.symbol.Activation( data=cls_conv2, act_type='relu')

         #   cls_conv2 = mx.symbol.Dropout(data=cls_conv2, p=0.5)
            cls_conv3 = mx.sym.Convolution(
                    data=cls_conv2, kernel=(3, 3), weight=cls_conv3_3x3_weight, bias = cls_conv3_3x3_bias, pad=(1, 1), num_filter=256, name="cls_conv3_3x3/p"+str(i+3))      
            cls_conv3 = mx.sym.BatchNorm(data=cls_conv3, use_global_stats=False, eps=eps,gamma=bn3_cls_gamma, beta =bn3_cls_beta, moving_mean =bn3_cls_moving_mean,
            moving_var= bn3_cls_moving_var,momentum=bn_mom, name = 'bn3_cls')
            cls_conv3 = mx.symbol.Activation( data=cls_conv3, act_type='relu')    

            cls_conv4 = mx.sym.Convolution(
                    data=cls_conv3, kernel=(3, 3), weight=cls_conv4_3x3_weight, bias = cls_conv4_3x3_bias, pad=(1, 1), num_filter=256, name="cls_conv4_3x3/p"+str(i+3)) 
            cls_conv4 = mx.sym.BatchNorm(data=cls_conv4, use_global_stats=False, eps=eps,gamma=bn4_cls_gamma, beta =bn4_cls_beta, moving_mean =bn4_cls_moving_mean,
            moving_var= bn4_cls_moving_var,momentum=bn_mom, name = 'bn4_cls')

            cls_conv4 = mx.symbol.Activation( data=cls_conv4, act_type='relu')
   

            cls_score = mx.sym.Convolution(
                    data=cls_conv4, kernel=(1, 1),weight=cls_score_weight,bias=cls_score_bias ,pad=(0, 0), num_filter=num_classes * num_anchors_list[i], name="cls_score/p"+str(i+3))
            cls_score_reshape = mx.sym.Reshape(
                    data=cls_score, shape = (-1, num_classes), name="cls_score_reshape/p"+str(i+3)) 
            cls_score_list.append(cls_score_reshape) 
                    ####bbox
          
            box_conv1 = mx.sym.Convolution(
                    data=conv_relu, kernel=(3, 3),weight = box_conv1_weight ,bias = box_conv1_bias, pad=(1, 1), num_filter=256, name="box_conv1_3x3/p"+str(i+3))
            box_conv1 = mx.sym.BatchNorm(data=box_conv1, use_global_stats=False, eps=eps,gamma=bn1_box_gamma, beta =bn1_box_beta, moving_mean =bn1_box_moving_mean,
            moving_var= bn1_box_moving_var,momentum=bn_mom, name = 'bn1_box')
            box_conv1 = mx.symbol.Activation( data=box_conv1, act_type='relu')
           # box_conv1  = mx.symbol.Dropout(data=box_conv1 , p=0.5)  
            box_conv2 = mx.sym.Convolution(
                    data = box_conv1, kernel=(3, 3), weight = box_conv2_weight, bias = box_conv2_bias,pad=(1, 1), num_filter=256, name="box_conv2_3x3/p"+str(i+3))
            box_conv2 = mx.sym.BatchNorm(data=box_conv2, use_global_stats=False, eps=eps,gamma=bn2_box_gamma, beta =bn2_box_beta, moving_mean =bn2_box_moving_mean,
            moving_var= bn2_box_moving_var,momentum=bn_mom, name = 'bn2_box')
            box_conv2 = mx.symbol.Activation( data=box_conv2, act_type='relu')

        #    box_conv2  = mx.symbol.Dropout(data=box_conv2 , p=0.5) 
            box_conv3 = mx.sym.Convolution(
                    data = box_conv2, kernel=(3, 3),weight = box_conv3_weight, bias = box_conv3_bias, pad=(1, 1), num_filter=256, name="box_conv3_3x3/p"+str(i+3))  
            box_conv3 = mx.sym.BatchNorm(data=box_conv3, use_global_stats=False, eps=eps,gamma=bn3_box_gamma, beta =bn3_box_beta, moving_mean =bn3_box_moving_mean,
            moving_var= bn3_box_moving_var,momentum=bn_mom, name = 'bn3_box')  

            box_conv3 = mx.symbol.Activation( data=box_conv3, act_type='relu')   
  
            box_conv4 = mx.sym.Convolution(
                    data = box_conv3, kernel=(3, 3),weight = box_conv4_weight, bias = box_conv4_bias, pad=(1, 1), num_filter=256, name="box_conv4_3x3/p"+str(i+3)) 
            box_conv4 = mx.sym.BatchNorm(data=box_conv4, use_global_stats=False, eps=eps,gamma=bn4_box_gamma, beta =bn4_box_beta, moving_mean =bn4_box_moving_mean,
            moving_var= bn4_box_moving_var,momentum=bn_mom, name = 'bn4_box')

            box_conv4 = mx.symbol.Activation( data=box_conv4, act_type='relu')
   
            bbox_pred = mx.sym.Convolution(
                    data=box_conv4, kernel=(1, 1), pad=(0, 0),weight=box_pred_weight,bias=box_pred_bias, num_filter=4 * num_anchors_list[i], name="box_pred/p"+str(i+3)) 
            bbox_pred = mx.sym.Reshape(data= bbox_pred,shape=(1,4 * num_anchors_list[i],-1))
            bbox_pred_list.append(bbox_pred)

        cls_concat = mx.sym.Concat(cls_score_list[0],cls_score_list[1],cls_score_list[2],cls_score_list[3],dim=0,name='cls_concat')
        bbox_pred_concat = mx.sym.Concat(bbox_pred_list[0],bbox_pred_list[1],bbox_pred_list[2],bbox_pred_list[3],dim=2,name='bbox_pred_conncat')


        if is_train:

            retina_cls_prob = mx.sym.Custom(op_type='FocalLoss', name = 'cls_prob', data=cls_concat, labels=retina_label,alpha =0.25, gamma= 2,use_ignore=True,ignore_label=-1)                                     
            retina_bbox_loss_ = retina_bbox_weight * mx.sym.smooth_l1(name='retina_bbox_loss', scalar=3.0,
                                                                    data=(bbox_pred_concat - retina_bbox_target))
 
            retina_bbox_loss = mx.sym.MakeLoss(name='retina_bbox_loss', data=retina_bbox_loss_,
                                                grad_scale=1.0 / 500.0)

            group = mx.sym.Group([retina_cls_prob,retina_bbox_loss])
            self.sym =group
            return group



    def init_weight(self, cfg, arg_params, aux_params):
            pi = 0.01
            arg_params['bn1_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_cls_gamma'])
            arg_params['bn1_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_cls_beta'])
            aux_params['bn1_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_cls_moving_mean'])
            aux_params['bn1_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_cls_moving_var'])

            arg_params['bn2_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_cls_gamma'])
            arg_params['bn2_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_cls_beta'])
            aux_params['bn2_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_cls_moving_mean'])
            aux_params['bn2_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_cls_moving_var'])

            arg_params['bn3_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn3_cls_gamma'])
            arg_params['bn3_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn3_cls_beta'])
            aux_params['bn3_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn3_cls_moving_mean'])
            aux_params['bn3_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn3_cls_moving_var'])
     
            arg_params['bn4_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn4_cls_gamma'])
            arg_params['bn4_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn4_cls_beta'])
            aux_params['bn4_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn4_cls_moving_mean'])
            aux_params['bn4_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn4_cls_moving_var'])




            arg_params['bn1_box_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_box_gamma'])
            arg_params['bn1_box_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_box_beta'])
            aux_params['bn1_box_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_box_moving_mean'])
            aux_params['bn1_box_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_box_moving_var'])

            arg_params['bn2_box_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_box_gamma'])
            arg_params['bn2_box_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_box_beta'])
            aux_params['bn2_box_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_box_moving_mean'])
            aux_params['bn2_box_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_box_moving_var'])

            arg_params['bn3_box_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn3_box_gamma'])
            arg_params['bn3_box_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn3_box_beta'])
            aux_params['bn3_box_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn3_box_moving_mean'])
            aux_params['bn3_box_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn3_box_moving_var'])

            arg_params['bn4_box_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn4_box_gamma'])
            arg_params['bn4_box_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn4_box_beta'])
            aux_params['bn4_box_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn4_box_moving_mean'])
            aux_params['bn4_box_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn4_box_moving_var'])

            arg_params['cls_conv1_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_conv1_3x3_weight'])
            arg_params['cls_conv1_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_conv1_3x3_bias'])
            arg_params['cls_conv2_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_conv2_3x3_weight'])
            arg_params['cls_conv2_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_conv2_3x3_bias']) 
            arg_params['cls_conv3_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_conv3_3x3_weight'])
            arg_params['cls_conv3_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_conv3_3x3_bias'])
            arg_params['cls_conv4_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_conv4_3x3_weight'])
            arg_params['cls_conv4_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_conv4_3x3_bias'])
            arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
            arg_params['cls_score_bias'] = mx.nd.ones(shape=self.arg_shape_dict['cls_score_bias'])*(-np.log((1-pi)/pi))

            arg_params['box_conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['box_conv1_weight'])
            arg_params['box_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['box_conv1_bias'])
            arg_params['box_conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['box_conv2_weight'])
            arg_params['box_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['box_conv2_bias']) 
            arg_params['box_conv3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['box_conv3_weight'])
            arg_params['box_conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['box_conv3_bias'])
            arg_params['box_conv4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['box_conv4_weight'])
            arg_params['box_conv4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['box_conv4_bias'])
            arg_params['box_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['box_pred_weight'])
            arg_params['box_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['box_pred_bias'])


            arg_params['c6_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['c6_weight'])
            arg_params['c6_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['c6_bias'])
                
            arg_params['stage5_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['stage5_conv_1x1_weight'])
            arg_params['stage5_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['stage5_conv_1x1_bias'])
            arg_params['up_stage4_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_1x1_weight'])
            arg_params['up_stage4_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_1x1_bias'])
            arg_params['up_stage4_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_3x3_weight'])
            arg_params['up_stage4_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_3x3_bias'])
            arg_params['up_stage3_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_1x1_weight'])
            arg_params['up_stage3_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_1x1_bias'])    
            arg_params['up_stage3_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_3x3_weight'])
            arg_params['up_stage3_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_3x3_bias'])          
            # arg_params['up_stage2_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_1x1_weight'])
            # arg_params['up_stage2_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_1x1_bias']) 
            # arg_params['up_stage2_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_3x3_weight'])
            # arg_params['up_stage2_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_3x3_bias'])     





