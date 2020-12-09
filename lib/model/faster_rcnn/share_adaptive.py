import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.faster_rcnn.resnet import resnet50


class _ShareAdaptiveAttention(nn.Module):
    """ Auto-Adaptive-Attention """
    def __init__(self, classes, n_head, attention_type, shot_mode, n_way=2, n_shot=5, pos_encoding=True):
        super(_ShareAdaptiveAttention, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_head = n_head
        self.attention_type = attention_type
        self.shot_mode = shot_mode
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
        self.rpn_d = 256
        self.rcnn_d = 64

        Q_list = []
        K_list = []
        for i in range(n_head):
            Q_weight = nn.Linear(dim_in, self.rpn_d)
            K_weight = nn.Linear(dim_in, self.rpn_d)
            init.normal_(Q_weight.weight, std=0.01)
            init.constant_(Q_weight.bias, 0)
            init.normal_(K_weight.weight, std=0.01)
            init.constant_(K_weight.bias, 0)
            Q_list.append(Q_weight)
            K_list.append(K_weight)
        self.adapt_Q_layers = nn.ModuleList(Q_list)
        self.adapt_K_layers = nn.ModuleList(K_list)
        # rpn
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        if self.attention_type == 'concat':
            self.RCNN_rpn = _RPN(2048)
            self.rcnn_transform_layer = nn.Linear(2048, self.rcnn_d)
        elif self.attention_type == 'product':
            self.RCNN_rpn = _RPN(1024)
            self.rcnn_transform_layer = nn.Linear(1024, self.rcnn_d)

        if n_head != 1:
            self.rpn_multihead_layer = nn.Linear(n_head * dim_in, dim_in)
            self.rcnn_multihead_layer = nn.Linear(n_head * dim_in, dim_in)

        self.output_score_layer = FFN(self.rcnn_d * 49, dim_in)
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
            support_ims = support_ims.view(-1, support_ims.size(2), support_ims.size(3), support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)  # [B*2*shot, 1024, 20, 20]
            support_feats = support_feats.view(-1, self.n_way*self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].contiguous()  # [B, shot, 1024, 20, 20]
            neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].contiguous()
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            neg_support_feat_pooled = self.avgpool(neg_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)  # [B, shot, 1024, 7, 7]
            neg_support_feat_pooled = neg_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)
        else:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :]
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)

        # attention RPN
        batch_size = pos_support_feat.size(0)
        feat_h = base_feat.size(2)
        feat_w = base_feat.size(3)
        support_mat = pos_support_feat.transpose(0, 1).view(self.n_shot, batch_size, 1024, -1).transpose(2, 3)  # [shot, B, 400, 1024]
        query_mat = base_feat.view(batch_size, 1024, -1).transpose(1, 2)  # [B, h*w, 1024]
        attentive_support_feat = []
        for n in range(self.n_head):
            feat_to_pool = []
            for i in range(self.n_shot):
                if self.pos_encoding:
                    s_mat = self.rpn_pos_encoding_layer(support_mat[i])
                else:
                    s_mat = self.support_mat[i]
                q_mat_query = self.adapt_Q_layers[n](query_mat)
                k_mat_support = self.adapt_K_layers[n](s_mat)
                qk_mat = torch.bmm(q_mat_query, k_mat_support.transpose(1, 2)) / math.sqrt(self.rpn_d)  # [B, hw, 400]
                qk_mat = F.softmax(qk_mat, dim=2)
                z_mat_support = torch.bmm(qk_mat, s_mat)  # [B, hw, 1024]
                feat_to_pool += [z_mat_support]
            feat_to_pool = torch.stack(feat_to_pool, 0)
            if self.shot_mode == 'max':
                attentive_support_feat += [torch.max(feat_to_pool, 0)[0]]
            elif self.shot_mode == 'mean':
                attentive_support_feat += [torch.mean(feat_to_pool, 0)]
        
        attentive_support_feat = torch.cat(attentive_support_feat, 2)  # [B, hw, n_head*1024]
        if self.n_head != 1:
            attentive_support_feat = F.relu(self.rpn_multihead_layer(attentive_support_feat))  # [B, hw, 1024]
        attentive_support_feat = attentive_support_feat.transpose(1, 2).view(batch_size, -1, feat_h, feat_w)  # [B, 1024, h, w]
        
        if self.attention_type == 'concat':
            correlation_feat = torch.cat([base_feat, attentive_support_feat], 1)
        elif self.attention_type == 'product':
            correlation_feat = base_feat * attentive_support_feat

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
        n_roi = pooled_feat.size(0)
        support_mat = []
        query_mat = []
        batch_size = support_feat.size(0)
        for query_feat, target_feat in zip(pooled_feat.chunk(batch_size, dim=0), support_feat.chunk(batch_size, dim=0)):
            # query_feat [128, c, 7, 7], target_feat [1, shot, c, 7, 7]
            target_feat = target_feat.view(1, self.n_shot, 1024, -1).transpose(2, 3)  # [1, shot, 49, c]
            target_feat = target_feat.repeat(query_feat.size(0), 1, 1, 1)  # [128, shot, 49, c]
            query_feat = query_feat.view(query_feat.size(0), 1024, -1).transpose(1, 2)  # [128, 49, c]
            if self.pos_encoding:
                target_feat = self.pos_encoding_layer(target_feat.view(-1, 49, 1024)).view(-1, self.n_shot, 49, 1024)
                query_feat = self.pos_encoding_layer(query_feat)
            support_mat += [target_feat]
            query_mat += [query_feat]
        support_mat = torch.cat(support_mat, 0).transpose(0, 1)  # [shot, B*128, 49, c]
        query_mat = torch.cat(query_mat, 0)  # [B*128, 49, c]

        multihead_feat = []
        for n in range(self.n_head):
            feat_to_pool = []
            for i in range(self.n_shot):
                _q = self.adapt_Q_layers[n](query_mat)  # [B*128, 49, rcnn_d]
                _k = self.adapt_K_layers[n](support_mat[i])
                _qk = torch.bmm(_q, _k.transpose(1, 2)) / math.sqrt(self.rcnn_d)  # [B*128, 49, 49]
                _qk = F.softmax(_qk, dim=2)
                _z = torch.bmm(_qk, support_mat[i])  # [B*128, 49, c]
                feat_to_pool += [_z]
            feat_to_pool = torch.stack(feat_to_pool, 0)
            if self.shot_mode == 'max':
                multihead_feat += [torch.max(feat_to_pool, 0)[0]]
            elif self.shot_mode == 'mean':
                multihead_feat += [torch.mean(feat_to_pool, 0)]
        multihead_feat = torch.cat(multihead_feat, 2)  # [B*128, 49, n_head*c]
        if self.n_head != 1:
            multihead_feat = F.relu(self.rcnn_multihead_layer(multihead_feat))  # [B*128, 49, c]
        if self.attention_type == 'concat':
            correlation_feat = torch.cat([query_mat, multihead_feat ], 2)  # [B*128, 49, 2c]
        elif self.attention_type == 'product':
            correlation_feat = query_mat * multihead_feat   # [B*128, 49, c]
        correlation_feat = self.rcnn_transform_layer(correlation_feat)  # [B*128, 49, rcnn_d]
        cls_score = self.output_score_layer(correlation_feat.view(n_roi, -1))
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
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return x


class ShareAAARCNN(_ShareAdaptiveAttention):
    def __init__(self, classes, n_head, attention_type, shot_mode, num_layers=50, pretrained=False, num_way=2, num_shot=5, pos_encoding=True):
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        _ShareAdaptiveAttention.__init__(self, classes, n_head, attention_type, shot_mode, n_way=num_way, n_shot=num_shot, pos_encoding=pos_encoding)

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