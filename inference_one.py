import numpy as np
import random
import time
import os
import cv2
import torch
import copy
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from scipy.misc import imread, imsave
from torch.autograd import Variable
from matplotlib import pyplot as plt

from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob
from model.roi_layers import nms

from model.framework.faster_rcnn import FasterRCNN
from model.framework.fsod import FSOD
from model.framework.meta import METARCNN
from model.framework.fgn import FGN
from model.framework.dana import DAnARCNN
from model.framework.cisa import CISARCNN
import utils


def support_im_preprocess(im_list, cfg, support_im_size):
    n_of_shot = len(im_list)
    support_data_all = np.zeros((n_of_shot, 3, support_im_size, support_im_size), dtype=np.float32)
    for i, im in enumerate(im_list):
        im = im[:,:,::-1]  # rgb -> bgr
        target_size = np.min(im.shape[0:2])  # don't change the size
        im, _ = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        _h, _w = im.shape[0], im.shape[1]
        if _h > _w:
            resize_scale = float(support_im_size) / float(_h)
            unfit_size = int(_w * resize_scale)
            im = cv2.resize(im, (unfit_size, support_im_size), interpolation=cv2.INTER_LINEAR)
        else:
            resize_scale = float(support_im_size) / float(_w)
            unfit_size = int(_h * resize_scale)
            im = cv2.resize(im, (support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
        h, w = im.shape[0], im.shape[1]
        support_data_all[i, :, :h, :w] = np.transpose(im, (2, 0, 1))
    support_data = torch.from_numpy(support_data_all).unsqueeze(0)
    
    return support_data

def query_im_preprocess(im_data, cfg):
    target_size = cfg.TRAIN.SCALES[0]
    im_data, im_scale = prep_im_for_blob(im_data, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    im_data = torch.from_numpy(im_data)
    im_info = np.array([[im_data.shape[0], im_data.shape[1], im_scale]], dtype=np.float32)
    im_info = torch.from_numpy(im_info)
    gt_boxes = torch.from_numpy(np.array([0]))
    num_boxes = torch.from_numpy(np.array([0]))
    query = im_data.permute(2, 0, 1).contiguous().unsqueeze(0)
    
    return query, im_info, gt_boxes, num_boxes

def generate_pseudo_label(output_dir, sp_dir, q_im_path, model, num_shot):
    # data = list of [im, cls]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    q_im = np.asarray(Image.open(q_im_path))[:, :, :3]
    if num_shot > 1:
        final_dets = None
        for i in range(num_shot): 
            sp_im_path = os.path.join(sp_dir, f'shot_{i+1}.jpg')
            sp_im = np.asarray(Image.open(sp_im_path))[:, :, :3]
            cls_dets = run_detection(sp_im, q_im, model)
            if final_dets is not None:
                final_dets = torch.cat((final_dets, cls_dets), 0)
            else:
                final_dets = cls_dets
        _, order = torch.sort(final_dets[:, 4], 0, True)
        final_dets = final_dets[order]
        keep = nms(final_dets[:, :4], final_dets[:, 4], cfg.TEST.NMS)
        final_dets = final_dets[keep.view(-1).long()]
    else:
        sp_im_path = os.path.join(sp_dir, 'shot_1.jpg')
        sp_im = np.asarray(Image.open(sp_im_path))[:, :, :3]
        final_dets = run_detection(sp_im, q_im, model)
    return final_dets


def run_detection(sp_im, q_im, model):
    support_data = support_im_preprocess([sp_im], cfg, 320)
    query_data, im_info, gt_boxes, num_boxes = query_im_preprocess(q_im, cfg)
    data = [query_data, im_info, gt_boxes, num_boxes, support_data]
    im_data, im_info, num_boxes, gt_boxes, support_ims = utils.prepare_var()
    with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        support_ims.resize_(data[4].size()).copy_(data[4])
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    box_deltas = bbox_pred.data
    # Optionally normalize targets by a precomputed mean and stdev
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    # re-scale boxes to the origin im scale
    pred_boxes /= data[1][0][2].item()
    # do nms
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    thresh = 0.05
    inds = torch.nonzero(scores[:,1]>thresh).view(-1)
    cls_scores = scores[:,1][inds]
    cls_boxes = pred_boxes[inds, :]
    cls_dets = utils.NMS(cls_boxes, cls_scores)

    return cls_dets

def filt_boxes(dets, cls, im_size):
    if cls == 'cube':
        max_box_scale = 0.25
        thres = 0.5
    elif cls == 'can':
        max_box_scale = 0.4
        thres = 0.3
    elif cls == 'box':
        max_box_scale = 0.5
        thres = 0.3
    elif cls == 'bottle':
        max_box_scale = 0.6
        thres = 0.3
    else:
        raise Exception(f'class {cls} not defined')

    for i in range(dets.shape[0]):
        w = dets[i, 2] - dets[i, 0]
        h = dets[i, 3] - dets[i, 1]
        if h > im_size * max_box_scale or w > im_size * max_box_scale:
            dets[i, 4] = 0.
    new_dets = []
    for i in range(dets.shape[0]):
        if dets[i, 4] > thres:
            new_dets.append((dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]))
    return np.asarray(new_dets)

if __name__ == '__main__':

    args = utils.get_args('inference_one')

    cfg_from_list(['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]'])
    checkpoint_dir = os.path.join(args.load_dir, "train/checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise Exception('There is no input directory for loading network from ' + checkpoint_dir)
    load_path = os.path.join(checkpoint_dir,
        'model_{}_{}.pth'.format(args.checkepoch, args.checkpoint))
    model = utils.get_model(args.net, pretrained=False, way=1, shot=1, load_path=load_path, eval=True)
    
    cls_names = ['cube', 'can', 'box', 'bottle']
    cls_im_inds = [list(range(1000, 1010)), list(range(1010, 1020)), list(range(1020, 1030)), list(range(1030, 1040))]
    output_dir = os.path.join('/home/tony/YCB_simulation/output', args.o_dir)
    im_dir = '/home/tony/YCB_simulation/query/images'
    for cls, inds in zip(cls_names, cls_im_inds):
        flag = True
        for ind in tqdm(inds):
            q_im_path = os.path.join(im_dir, str(ind).zfill(6)+'.jpg')
            sp_dir = Path(args.sp_dir) / Path(cls)
            dets = generate_pseudo_label(output_dir, sp_dir, q_im_path, model, args.shot)
            dets = filt_boxes(dets, cls, 256)
            np.save(os.path.join(output_dir, str(ind).zfill(6)+'.npy'), dets)

            if flag:
                q_im = np.asarray(Image.open(q_im_path))[:, :, :3]
                im2show = utils.plot_box(q_im, dets, thres=0.)
                cv2.imwrite(os.path.join(output_dir, str(ind).zfill(6)+'.jpg'), im2show[:, :, ::-1])
                flag=False
