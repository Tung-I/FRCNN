import os
import sys
import numpy as np
import argparse
import time
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.fs_loader import FewShotLoader, sampler
from roi_data_layer.oracle_loader import OracleLoader
from roi_data_layer.finetune_loader import FinetuneLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.fsod_logger import FSODLogger
from utils import *
from pycocotools.coco import COCO

CWD = os.getcwd()

CLS_NAMES = ['cube', 'can', 'box', 'bottle']
# NDARR_DIR = '/home/tony/YCB_simulation/query/ndarray'
IM_DIR = '/home/tony/YCB_simulation/query/images'
# CLS_IM_INDS_SET = [ [list(range(1000, 1064)), list(range(1256, 1320)), list(range(1512, 1576)), list(range(1768, 1832))], 
#                 [list(range(1000, 1128)), list(range(1256, 1384)), list(range(1512, 1640)), list(range(1768, 1896))], 
#                 [list(range(1000, 1192)), list(range(1256, 1448)), list(range(1512, 1704)), list(range(1768, 1960))], 
#                 [list(range(1000, 1256)), list(range(1256, 1512)), list(range(1512, 1768)), list(range(1768, 2024))] ]
start1, start2, start3, start4 = 1000, 1100, 1200, 1300
CLS_IM_INDS_SET = []
for n in [16, 32, 48, 64, 80, 96]:
    CLS_IM_INDS_SET.append([list(range(start1, start1+n)), list(range(start2, start2+n)), list(range(start3, start3+n)), list(range(start4, start4+n))])

DUMP_DIR = '/home/tony/datasets/YCB2D/annotations'

def filt_boxes(dets, cls, im_size, thres):
    if cls == 'cube':
        max_box_scale = 0.25
        min_box_scale = 0
    elif cls == 'can':
        max_box_scale = 0.35
        min_box_scale = 0
    elif cls == 'box':
        max_box_scale = 0.4
        min_box_scale = 0.15
    elif cls == 'bottle':
        max_box_scale = 0.45
        min_box_scale = 0
    else:
        raise Exception(f'class {cls} not defined')

    for i in range(dets.shape[0]):
        w = dets[i, 2] - dets[i, 0]
        h = dets[i, 3] - dets[i, 1]
        if h > im_size * max_box_scale or w > im_size * max_box_scale:
            dets[i, 4] = 0.
        if h < im_size * min_box_scale or w < im_size * min_box_scale:
            dets[i, 4] = 0.
    new_dets = []
    for i in range(dets.shape[0]):
        if dets[i, 4] > thres:
            new_dets.append((dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]))
    return np.asarray(new_dets)

if __name__ == '__main__':

    args = parse_args()
    print(args)
    cfg_from_file(args.cfg_file)
    cfg_from_list(args.set_cfgs)

    args.sup_dir = 'ycb2d'
    args.pred_dir = os.path.join('/home/tony/YCB_simulation/output', f'pred{args.stage}')
    args.emb_shot = args.stage if args.stage <=5 else 5
    args.dataset = f'pseudo{args.stage}'
    args.imdb_name = f"ycb2d_pseudo{args.stage}"
    CLS_IM_INDS = CLS_IM_INDS_SET[args.stage - 1]

    # make results determinable
    random_seed = 1996
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.CUDA = True

    # network for prediction
    model = get_model(args.net, pretrained=False, way=args.way, shot=1, classes=['fg', 'bg'])
    checkpoint_dir = os.path.join(args.load_dir, "train/checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise Exception('There is no input directory for loading network from ' + checkpoint_dir)
    load_path = os.path.join(checkpoint_dir,
        'model_{}_{}.pth'.format(args.checkepoch, args.checkpoint))
    print("load checkpoint %s" % (load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    model.cuda()
    model.eval()

    # predict
    sup_dir = os.path.join(CWD, 'data/supports', args.sup_dir)
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    for cls, inds in zip(CLS_NAMES, CLS_IM_INDS):
        flag = True
        for ind in tqdm(inds):
            q_im_path = os.path.join(IM_DIR, str(ind).zfill(6)+'.jpg')
            cur_sup_dir = os.path.join(sup_dir, cls)
            dets = generate_pseudo_label(args.pred_dir, cur_sup_dir, q_im_path, model, args.emb_shot)
            dets = filt_boxes(dets, cls, 256, args.thres)
            np.save(os.path.join(args.pred_dir, str(ind).zfill(6)+'.npy'), dets)
            if flag:
                q_im = np.asarray(Image.open(q_im_path))[:, :, :3]
                im2show = plot_box(q_im, dets, thres=0.)
                cv2.imwrite(os.path.join(args.pred_dir, str(ind).zfill(6)+'.jpg'), im2show[:, :, ::-1])
                flag = False

    # generate the annotation file
    dump_path = os.path.join(DUMP_DIR, f'instances_{args.dataset}.json')
    create_annotation(args.pred_dir, CLS_NAMES, CLS_IM_INDS, dump_path)

    ################################################################################################
    stage = args.stage
    settings = [16, 32, 48, 64, 80, 96]
    setting = settings[args.stage - 1]
    coco_json_path = f'/home/tony/datasets/YCB2D/annotations/instances_replace{setting}.json'
    with open(coco_json_path, 'r') as f:
        replace_data = json.load(f)
    coco_json_path = f'/home/tony/datasets/YCB2D/annotations/instances_pseudo{stage}.json'
    with open(coco_json_path, 'r') as f:
        dense_data = json.load(f)   
    new_dict = {}
    new_dict['info'] = replace_data['info']
    new_dict['images'] = replace_data['images'] + dense_data['images']
    new_dict['licenses'] = replace_data['licenses']
    new_dict['annotations'] = replace_data['annotations'] + dense_data['annotations']
    new_dict['categories'] = replace_data['categories']
    dump_path = f'/home/tony/datasets/YCB2D/annotations/instances_stage{stage}.json'
    with open(dump_path, 'w') as f:
        json.dump(new_dict, f)

    ################################################################################################


    # # prepare checkpoint output dir
    # checkpoint_output_dir = os.path.join(args.save_dir, "train/checkpoints") 
    # if not os.path.exists(checkpoint_output_dir):
    #     os.makedirs(checkpoint_output_dir)

    # # prepare dataloader
    # cfg.TRAIN.USE_FLIPPED = args.use_flip
    # cfg.USE_GPU_NMS = True
    # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    # support_dir = os.path.join(CWD, 'data/supports', args.sup_dir)
    # dataset = FinetuneLoader(imdb, roidb, ratio_list, ratio_index, args.batch_size, \
    #                     imdb.num_classes, support_dir, training=True, num_shot=args.shot)
    
    # # dataset = FewShotLoader(roidb, ratio_list, ratio_index, args.batch_size, \
    # #                         imdb.num_classes, training=True, num_way=args.way, num_shot=args.shot)

    # train_size = len(roidb)
    # print('{:d} roidb entries'.format(len(roidb)))
    # sampler_batch = sampler(train_size, args.batch_size)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                         sampler=sampler_batch, num_workers=args.num_workers)

    # # initilize the tensor holders
    # holders = prepare_var(support=True)
    # im_data = holders[0]
    # im_info = holders[1]
    # num_boxes = holders[2]
    # gt_boxes = holders[3]
    # support_ims = holders[4]

    # # prepare the network
    # model = get_model(args.net, pretrained=False, way=args.way, shot=args.shot, classes=['fg', 'bg'])
    # print("load checkpoint %s" % (args.ori_model))
    # checkpoint = torch.load(args.ori_model)
    # model.load_state_dict(checkpoint['model'])
    # if 'pooling_mode' in checkpoint.keys():
    #     cfg.POOLING_MODE = checkpoint['pooling_mode']
    # print('load model successfully!')
    # model.cuda()
    # # model.finetune()

    # # optimizer
    # lr = cfg.TRAIN.LEARNING_RATE
    # lr = args.lr
    # params = []
    # for key, value in dict(model.named_parameters()).items():
    #     if value.requires_grad:
    #         if 'bias' in key:
    #             params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
    #                     'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
    #         else:
    #             params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    # if args.optimizer == "adam":
    #     optimizer = torch.optim.Adam(params)
    # elif args.optimizer == "sgd":
    #     optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # if args.mGPUs:
    #     model = nn.DataParallel(model)

    # # initialize logger
    # if not args.dlog:
    #     logger_save_dir = os.path.join(args.save_dir, "train")
    #     tb_logger = FSODLogger(logger_save_dir)

    # # training
    # iters_per_epoch = int(train_size / args.batch_size)
    # for epoch in range(args.start_epoch, args.max_epochs + 1):
    #     model.train()
    #     loss_temp = 0
    #     start_time = time.time()
    #     if epoch % (args.lr_decay_step + 1) == 0:
    #         adjust_learning_rate(optimizer, args.lr_decay_gamma)
    #         lr *= args.lr_decay_gamma
    #     data_iter = iter(dataloader)
    #     for step in range(iters_per_epoch):
    #         data = next(data_iter)
    #         with torch.no_grad():
    #             im_data.resize_(data[0].size()).copy_(data[0])
    #             im_info.resize_(data[1].size()).copy_(data[1])
    #             gt_boxes.resize_(data[2].size()).copy_(data[2])
    #             num_boxes.resize_(data[3].size()).copy_(data[3])
    #             support_ims.resize_(data[4].size()).copy_(data[4])

    #         model.zero_grad()

    #         rois, cls_prob, bbox_pred, \
    #         rpn_loss_cls, rpn_loss_box, \
    #         RCNN_loss_cls, RCNN_loss_bbox, \
    #         rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims)

    #         loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
    #             + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
    #         loss_temp += loss.item()

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if step % args.disp_interval == 0:
    #             end_time = time.time()
    #             if step > 0:
    #                 loss_temp /= (args.disp_interval + 1)
    #             if args.mGPUs:
    #                 loss_rpn_cls = rpn_loss_cls.mean().item()
    #                 loss_rpn_box = rpn_loss_box.mean().item()
    #                 loss_rcnn_cls = RCNN_loss_cls.mean().item()
    #                 loss_rcnn_box = RCNN_loss_bbox.mean().item()
    #                 fg_cnt = torch.sum(rois_label.data.ne(0))
    #                 bg_cnt = rois_label.data.numel() - fg_cnt
    #             else:
    #                 loss_rpn_cls = rpn_loss_cls.item()
    #                 loss_rpn_box = rpn_loss_box.item()
    #                 loss_rcnn_cls = RCNN_loss_cls.item()
    #                 loss_rcnn_box = RCNN_loss_bbox.item()
    #                 fg_cnt = torch.sum(rois_label.data.ne(0))
    #                 bg_cnt = rois_label.data.numel() - fg_cnt

    #             print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
    #                                     % (epoch, step, iters_per_epoch, loss_temp, lr))
    #             print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end_time-start_time))
    #             print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
    #                         % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

    #             info = {
    #             'loss': loss_temp,
    #             'loss_rpn_cls': loss_rpn_cls,
    #             'loss_rpn_box': loss_rpn_box,
    #             'loss_rcnn_cls': loss_rcnn_cls,
    #             'loss_rcnn_box': loss_rcnn_box
    #             }
    #             loss_temp = 0
    #             start_time = time.time()
    #     if not args.dlog:
    #         tb_logger.write(epoch, info, save_im=args.imlog)

    #     save_name = os.path.join(checkpoint_output_dir, 'model_{}_{}.pth'.format(epoch, step))
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'pooling_mode': cfg.POOLING_MODE,
    #     }, save_name)
    #     print('save model: {}'.format(save_name))


