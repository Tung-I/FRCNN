import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.faster_rcnn.resnet import resnet50


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic=True):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_imgs, all_cls_gt_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = all_cls_gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


# class _fasterRCNN(nn.Module):
#     """ faster RCNN """
#     def __init__(self, classes, class_agnostic=True):
#         super(_fasterRCNN, self).__init__()
#         self.classes = classes
#         self.n_classes = len(classes)
#         self.class_agnostic = class_agnostic
#         # loss
#         self.RCNN_loss_cls = 0
#         self.RCNN_loss_bbox = 0

#         # define rpn
#         self.RCNN_rpn = _RPN(self.dout_base_model)
#         self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

#         # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
#         # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

#         self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
#         self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

#     def forward(self, im_data, im_info, gt_boxes, num_boxes, support_imgs, all_cls_gt_boxes):
#         batch_size = im_data.size(0)

#         im_info = im_info.data
#         gt_boxes = all_cls_gt_boxes.data
#         num_boxes = num_boxes.data

#         # feed image data to base model to obtain base feature map
#         base_feat = self.RCNN_base(im_data)

#         # feed base feature map tp RPN to obtain rois
#         rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

#         # if it is training phrase, then use ground trubut bboxes for refining
#         if self.training:
#             roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
#             rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

#             rois_label = Variable(rois_label.view(-1).long())
#             rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
#             rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
#             rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
#         else:
#             rois_label = None
#             rois_target = None
#             rois_inside_ws = None
#             rois_outside_ws = None
#             rpn_loss_cls = 0
#             rpn_loss_bbox = 0

#         rois = Variable(rois)
#         # do roi pooling based on predicted rois

#         if cfg.POOLING_MODE == 'align':
#             pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))  # [1024, 1024, 7, 7]
#         elif cfg.POOLING_MODE == 'pool':
#             pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

#         # feed pooled features to top model
#         pooled_feat = self._head_to_tail(pooled_feat)  # [1024, 2048]

#         # compute bbox offset
#         bbox_pred = self.RCNN_bbox_pred(pooled_feat)
#         if self.training and not self.class_agnostic:
#             # select the corresponding columns according to roi labels
#             bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
#             bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
#             bbox_pred = bbox_pred_select.squeeze(1)

#         # compute object classification probability
#         cls_score = self.RCNN_cls_score(pooled_feat)  # [1024, 21]
#         cls_prob = F.softmax(cls_score, 1)

#         RCNN_loss_cls = 0
#         RCNN_loss_bbox = 0

#         if self.training:
#             # classification loss
#             RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
#             # print(cls_score[:30])
#             # print(rois_label[:30])
#             # raise Exception('stop')

#             # bounding box regression L1 loss
#             RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


#         cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
#         bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

#         return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


class FasterRCNN(_fasterRCNN):
    def __init__(self, classes, num_layers=50, pretrained=False):
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        _fasterRCNN.__init__(self, classes)

    def _init_modules(self):
        resnet = resnet50()
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet. (base -> top -> head)
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        self.RCNN_top = nn.Sequential(resnet.layer4)  # 1024 -> 2048
        # build rcnn head
        self.RCNN_bbox_pred = nn.Linear(2048, 4)
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)

        # Fix blocks 
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)


    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)  # [128, 2048]
        return fc7