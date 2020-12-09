from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .iou_anchor_target_layer import _IOUAnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _PrimeRPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, gamma):
        super(_PrimeRPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _IOUAnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # self.loss_func = IOUFocalLoss()
        self.loss_func = PrimeLoss(gamma)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, all_cls_gt_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # [B, 9*2, H, W]

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)  # [B, 9*2, H, W]

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # [B, 9*4, H, W]

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            all_cls_rpn_data = self.RPN_anchor_target((rpn_cls_score.data, all_cls_gt_boxes, im_info, num_boxes))
            ##################
            # rpn_data: list of length=4
            # [0]: labels [B, 1, 9*H, W]
            # [1]: bbox_targets [B, 9*4, H, W]
            # [2]: bbox_inside_weights [B, 9*4, H, W]
            # [3]: bbox_outside_weights [B, 9*4, H, W]
            ##################

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)  # [B*RPN_BATCHSIZE, 2]

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())  # [B*RPN_BATCHSIZE]

            #############################################################
            rpn_overlap = all_cls_rpn_data[4].view(batch_size, -1)
            rpn_overlap = torch.index_select(rpn_overlap.view(-1), 0, rpn_keep.data)
            rpn_overlap = Variable(rpn_overlap)

            self.rpn_loss_cls = self.loss_func(rpn_cls_score, rpn_label, rpn_overlap)

            #############################################################

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:4]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box


class PrimeLoss(nn.Module):
    def __init__(self, gamma):
        super(PrimeLoss, self).__init__()
        self.gamma = gamma

    def forward(self, rpn_cls_score, rpn_label, rpn_overlap):
        batch_loss = F.cross_entropy(rpn_cls_score, rpn_label, reduction='none')

        zero_mask = torch.zeros_like(rpn_overlap)
        rpn_overlap_masked = torch.where(rpn_overlap >= 0.5, rpn_overlap, zero_mask)
        weights = 1. + (rpn_overlap_masked * self.gamma)
        loss = (batch_loss * weights).mean()
        # N = inputs.size(0)
        # C = inputs.size(1)
        # P = F.softmax(inputs, dim=1)

        # class_mask = inputs.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        # ids = targets.view(-1, 1)
        # class_mask.scatter_(1, ids.data, 1.)

        # probs = (P*class_mask).sum(1).view(-1,1)
        # batch_loss = -1 * probs.log()

        # overlaps += self.delta
        # weights = overlaps / overlaps.sum()

        # loss = (batch_loss.squeeze() * weights).sum()
        return loss
