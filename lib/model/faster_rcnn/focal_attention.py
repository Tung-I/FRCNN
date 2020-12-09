import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math
import time

from model.utils.config import cfg
from model.rpn.focal_rpn import _FocalRPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.faster_rcnn.resnet import resnet50


class _focalRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, n_head, n_way=2, n_shot=5, pos_encoding=True):
        super(_focalRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_head = n_head
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # pooling or align
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        # few shot rcnn head
        self.pool_feat_dim = 1024
        self.avgpool = nn.AvgPool2d(14, stride=1)
        dim_in = self.pool_feat_dim
        ################
        self.d_k = 64
        Q_list = []
        K_list = []
        V_list = []
        for i in range(n_head):
            Q_weight = nn.Linear(dim_in, self.d_k)
            K_weight = nn.Linear(dim_in, self.d_k)
            V_weight = nn.Linear(dim_in, self.d_k)
            init.normal_(Q_weight.weight, std=0.01)
            init.constant_(Q_weight.bias, 0)
            init.normal_(K_weight.weight, std=0.01)
            init.constant_(K_weight.bias, 0)
            init.normal_(V_weight.weight, std=0.01)
            init.constant_(V_weight.bias, 0)
            Q_list.append(Q_weight)
            K_list.append(K_weight)
            V_list.append(V_weight)
        self.rpn_Q_layers = nn.ModuleList(Q_list)
        self.rpn_K_layers = nn.ModuleList(K_list)
        self.rpn_V_layers = nn.ModuleList(V_list)
        Q_list = []
        K_list = []
        V_list = []
        for i in range(n_head):
            Q_weight = nn.Linear(dim_in, self.d_k)
            K_weight = nn.Linear(dim_in, self.d_k)
            V_weight = nn.Linear(dim_in, self.d_k)
            init.normal_(Q_weight.weight, std=0.01)
            init.constant_(Q_weight.bias, 0)
            init.normal_(K_weight.weight, std=0.01)
            init.constant_(K_weight.bias, 0)
            init.normal_(V_weight.weight, std=0.01)
            init.constant_(V_weight.bias, 0)
            Q_list.append(Q_weight)
            K_list.append(K_weight)
            V_list.append(V_weight)
        self.rcnn_Q_layers = nn.ModuleList(Q_list)
        self.rcnn_K_layers = nn.ModuleList(K_list)
        self.rcnn_V_layers = nn.ModuleList(V_list)

        if n_head != 1:
            self.rpn_multihead_layer = nn.Linear(n_head * 400, 400)
            self.rcnn_multihead_layer = nn.Linear(n_head * self.d_k * 49, self.d_k * 49)
            self.output_score_layer = FFN(self.d_k * 49, dim_in)
        else:
            self.output_score_layer = FFN(self.d_k * 49, dim_in)
        # rpn
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_rpn = _FocalRPN(400, 256)
        # positional encoding
        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pos_encoding_layer = PositionalEncoding()
            self.rpn_pos_encoding_layer = PositionalEncoding(max_len=400)


    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_ims, all_cls_gt_boxes=None):
        if self.training:
            self.num_of_rois = cfg.TRAIN.BATCH_SIZE
        else:
            self.num_of_rois = cfg.TEST.RPN_POST_NMS_TOP_N 
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feature extraction
        base_feat = self.RCNN_base(im_data)
        if self.training:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)
            support_feats = support_feats.view(-1, self.n_way*self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].mean(1)
            neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].mean(1)  # [B, 1024, 20, 20]
            pos_support_feat_pooled = self.avgpool(pos_support_feat)  # [B, 1024, 7, 7]
            neg_support_feat_pooled = self.avgpool(neg_support_feat)
        else:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].mean(1)
            pos_support_feat_pooled = self.avgpool(pos_support_feat)

        # attention RPN
        batch_size = pos_support_feat.size(0)
        feat_h = base_feat.size(2)
        feat_w = base_feat.size(3)
        support_mat = pos_support_feat.view(batch_size, self.pool_feat_dim, -1).transpose(1, 2)  # [B, 400, 1024]
        query_mat = base_feat.view(batch_size, self.pool_feat_dim, -1).transpose(1, 2)  # [B, h*w, 1024]
        if self.pos_encoding:
                support_mat = self.rpn_pos_encoding_layer(support_mat)
        correlation_feat = []
        for n in range(self.n_head):
            q_mat_k = self.rpn_Q_layers[n](support_mat)  # [B 400, d_k]
            k_mat_k = self.rpn_K_layers[n](support_mat)
            v_mat_k = self.rpn_V_layers[n](support_mat)
            qk_mat_k = torch.bmm(q_mat_k, k_mat_k.transpose(1, 2)) / math.sqrt(self.d_k)  # [B, 400, 400]
            qk_mat_k = F.softmax(qk_mat_k, dim=2)
            z_mat_k = torch.bmm(qk_mat_k, v_mat_k)  # [B, 400, d_k]

            q_mat_q = self.rpn_Q_layers[n](query_mat)  # [B, hw, d_k]
            k_mat_q = self.rpn_K_layers[n](query_mat)
            v_mat_q = self.rpn_V_layers[n](query_mat)
            qk_mat_q = torch.bmm(q_mat_q, k_mat_q.transpose(1, 2)) / math.sqrt(self.d_k)  # [B, hw, hw]
            qk_mat_q = F.softmax(qk_mat_q, dim=2)
            z_mat_q = torch.bmm(qk_mat_q, v_mat_q)  # [N, hw, d_k]

            correlation_feat.append(torch.bmm(z_mat_q, z_mat_k.transpose(1, 2)) / math.sqrt(self.d_k))  # [B, hw, 400]
        correlation_feat = torch.cat(correlation_feat, 2)  # [B, hw, n_head*400]
        if self.n_head != 1:
            correlation_feat = F.relu(self.rpn_multihead_layer(correlation_feat))  # [B, hw, 400]
        correlation_feat = correlation_feat.transpose(1, 2).view(batch_size, -1, feat_h, feat_w)  # [B, 400, h, w]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(correlation_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            ## rois [B, rois_per_image(128), 5]
                ### 5 is [batch_num, x1, y1, x2, y2]
            ## rois_label [B, 128]
            ## rois_target [B, 128, 4]
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

        # do roi pooling based on predicted rois, pooled_feat = [B*128, 1024, 7, 7]
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # rcnn head
        if self.training:
            bbox_pred, cls_prob, cls_score_all = self.rcnn_head(pooled_feat, pos_support_feat_pooled)
            _, neg_cls_prob, neg_cls_score_all = self.rcnn_head(pooled_feat, neg_support_feat_pooled)
            cls_prob = torch.cat([cls_prob, neg_cls_prob], dim=0)
            cls_score_all = torch.cat([cls_score_all, neg_cls_score_all], dim=0)
            neg_rois_label = torch.zeros_like(rois_label)
            rois_label = torch.cat([rois_label, neg_rois_label], dim=0)
        else:
            bbox_pred, cls_prob, cls_score_all = self.rcnn_head(pooled_feat, pos_support_feat_pooled)

        # losses
        if self.training:
            ## bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            ## classification loss, 2-way, 1:2:1
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            bg_inds = (rois_label == 0).nonzero().squeeze(-1)
            cls_score_softmax = torch.nn.functional.softmax(cls_score_all, dim=1)
            bg_cls_score_softmax = cls_score_softmax[bg_inds, :]
            bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(rois_label.shape[0] * 0.25)))
            bg_num_1 = max(1, min(fg_inds.shape[0], bg_num_0))
            _sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 1], descending=True)
            real_bg_inds = bg_inds[sorted_bg_inds]  # sort the real_bg_inds
            real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(rois_label.shape[0] * 0.5)][:bg_num_0]  # pos support
            real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(rois_label.shape[0] * 0.5)][:bg_num_1]  # neg_support
            topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
            RCNN_loss_cls = F.cross_entropy(cls_score_all[topk_inds], rois_label[topk_inds])
        else:
            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

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
        # normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


    def rcnn_head(self, pooled_feat, support_feat):
        # box regression
        bbox_pred = self.RCNN_bbox_pred(self._head_to_tail(pooled_feat))  # [B*128, 4]
        # classification
        batch_size = support_feat.size(0)
        n_roi = pooled_feat.size(0)
        support_mat = []
        query_mat = []
        for query_feat, target_feat in zip(pooled_feat.chunk(batch_size, dim=0), support_feat.chunk(batch_size, dim=0)):
            # query_feat [128, c, 7, 7], target_feat [1, c, 7, 7]
            n_channel = query_feat.size(1) 
            target_feat = target_feat.view(1, n_channel, -1).transpose(1, 2)  # [1, 49, c]
            target_feat = target_feat.repeat(query_feat.size(0), 1, 1)  # [128, 49, c]
            query_feat = query_feat.view(query_feat.size(0), n_channel, -1).transpose(1, 2)  # [128, 49, c]
            if self.pos_encoding:
                target_feat = self.pos_encoding_layer(target_feat)
                query_feat = self.pos_encoding_layer(query_feat)
            support_mat += [target_feat]
            query_mat += [query_feat]
        support_mat = torch.cat(support_mat, 0)
        query_mat = torch.cat(query_mat, 0)  # [B*128, 49, c]

        multihead_feat = []
        for n in range(self.n_head):
            _q = self.rcnn_Q_layers[n](support_mat)  # [B*128, 49, d_k]
            _k = self.rcnn_K_layers[n](query_mat)  # [B*128, 49, d_k]
            _v = self.rcnn_V_layers[n](query_mat)
            _qk = torch.bmm(_q, _k.transpose(1, 2)) / math.sqrt(self.d_k)  # [B*128, 49, 49]
            _qk = F.softmax(_qk, dim=2)
            _z = torch.bmm(_qk, _v).view(n_roi, -1)  # [B*128, 49*d_k]
            multihead_feat += [_z]
        multihead_feat = torch.cat(multihead_feat, 1)  # one head: 49*d_k, multihead: n_head*49*d_k
        if self.n_head != 1:
            multihead_feat = F.relu(self.rcnn_multihead_layer(multihead_feat))  # n_head*49*d_k -> 49*d_k
        cls_score = self.output_score_layer(multihead_feat)
        cls_prob = F.softmax(cls_score, 1)  # [B*128, 1]

        return bbox_pred, cls_prob, cls_score 


class FFN(nn.Module):
    def __init__(self, in_channel, hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=1024, max_len=49):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False).cuda()

    def forward(self, x):
        x = x + self.pe
        return x


class FocalRCNN(_focalRCNN):
    def __init__(self, classes, n_head, num_layers=50, pretrained=False, num_way=2, num_shot=5, pos_encoding=True):
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        _focalRCNN.__init__(self, classes, n_head, n_way=num_way, n_shot=num_shot, pos_encoding=pos_encoding)

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