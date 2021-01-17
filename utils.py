import argparse
import torch
import cv2
from model.utils.config import cfg
from torch.autograd import Variable
from model.framework.faster_rcnn import FasterRCNN
from model.framework.fsod import FSOD
from model.framework.meta import METARCNN
from model.framework.fgn import FGN
from model.framework.dana import DAnARCNN
from model.framework.cisa import CISARCNN
from model.roi_layers import nms


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # net and dataset
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='res50', type=str)
    parser.add_argument('--flip', dest='use_flip', help='use flipped data or not', default=False, action='store_true')
    # optimizer
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch', default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    # train setting
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=8, type=int)
    parser.add_argument('--ls', dest='large_scale', help='whether use large imag scale', action='store_true')                      
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=12, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="models", type=str)
    parser.add_argument('--ascale', dest='ascale', help='number of anchor scale', default=4, type=int)
    parser.add_argument('--ft', dest='finetune', help='finetune mode', default=False, action='store_true')
    parser.add_argument('--eval', dest='eval', help='evaluation mode', default=False, action='store_true')
    # inference one by one
    parser.add_argument('--thres', dest='thres', help='threshold of score', default=0.5, type=float)
    parser.add_argument('--o_dir', dest='output_dir', help='output_dir', default=None, type=str)
    # few shot
    parser.add_argument('--fs', dest='fewshot', help='few-shot setting', default=False, action='store_true')
    parser.add_argument('--way', dest='way', help='num of support way', default=2, type=int)
    parser.add_argument('--shot', dest='shot', help='num of support shot', default=5, type=int)
    parser.add_argument('--sp_dir', dest='support_dir', help='directory of support images', default='all', type=str) 
    # load checkpoints
    parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true', default=False)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="models", type=str)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model', default=0, type=int)
    # logger
    parser.add_argument('--dlog', dest='dlog', help='disable the logger', default=False, action='store_true')
    parser.add_argument('--imlog', dest='imlog', help='save im in the logger', default=False, action='store_true')


    args = parser.parse_args()

    # parse dataset
    if args.ascale == 3:
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.ascale == 4:
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    else:
        raise Exception(f'invalid anchor scale {args.ascale}')
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train"
        args.imdbval_name = "coco_2014_minival"
    elif args.dataset == "coco2017":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    elif args.dataset == "set1":
        args.imdb_name = "coco_60_set1"
    elif args.dataset == "set1_all_cat":
        args.imdb_name = "coco_60_set1allcat"
    elif args.dataset == "0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "ycb2d":
        args.imdb_name = "ycb2d_train"
    elif args.dataset == "ycb2d_iter":
        args.imdb_name = "ycb2d_iter"
    else:
        raise Exception(f'dataset {args.dataset} not defined')
    args.cfg_file = "cfgs/res50.yml"
    return args

def get_model(name, pretrained=True, way=2, shot=3, eval=False, classes=[]):
    if name == 'frcnn':
        model = FasterRCNN(classes, pretrained=pretrained)
    elif name == 'fsod':
        model = FSOD(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'meta':
        model = METARCNN(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'fgn':
        model = FGN(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'cisa':
        model = CISARCNN(classes, 'concat', 256, 256, pretrained=pretrained, num_way=way, num_shot=shot)
    else:
        raise Exception(f"network {name} is not defined")
    model.create_architecture()
    model.cuda()
    if eval:
        model.eval()
    return model


def prepare_var(support=False):
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if support:
        support_ims = torch.FloatTensor(1)
        support_ims = support_ims.cuda()
        support_ims = Variable(support_ims)
        return [im_data, im_info, num_boxes, gt_boxes, support_ims]
    else:
        return [im_data, im_info, num_boxes, gt_boxes]
        

def plot_box(im, boxes, thres=0.5):
    # boxes[n] = [x1, y1, x2, y2, score]
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if box[4] > thres:
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (20, 255, 20), 2)
    return im 

def NMS(boxes, scores):
    _, order = torch.sort(scores, 0, True)
    dets = torch.cat((boxes, scores.unsqueeze(1)), 1)[order]
    keep = nms(boxes[order, :], scores[order], cfg.TEST.NMS)
    dets = dets[keep.view(-1).long()]
    return dets
