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


def get_args(key):
    if key == 'inference_one': 
        args = inference_one_args()
    else:
        raise Exception(f'{key} not defined.')
    return args

def inference_one_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--sp_dir', dest='sp_dir',
                        help='directory to load supports', default=None, type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default=None, type=str)
    parser.add_argument('--net', dest='net', 
                        help='name of the model', default=None, type=str)
    parser.add_argument('--shot', dest='shot', 
                        help='number of shot', default=1, type=int)
    parser.add_argument('--thres', dest='thres', 
                        help='threshold of score', default=0.5, type=float)
    parser.add_argument('--cls', dest='cls', 
                        help='class name', default=None, type=str)
    parser.add_argument('--o_dir', dest='o_dir', 
                        help='output_dir', default=None, type=str)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model', default=0, type=int)
    args = parser.parse_args()
    return args


def get_model(name, pretrained=True, way=2, shot=3, load_path=None, eval=False):
    if name == 'frcnn':
        model = FasterRCNN([], pretrained=pretrained)
    elif name == 'fsod':
        model = FSOD([], pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'meta':
        model = METARCNN([], pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'fgn':
        model = FGN([], pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'cisa':
        model = CISARCNN([], 'concat', 256, 256, pretrained=pretrained, num_way=way, num_shot=shot)
    else:
        raise Exception(f"network {name} is not defined")
    model.create_architecture()
    print("load checkpoint %s" % (load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    if eval:
        model.eval()
    model.cuda()
    
    return model


def prepare_var():
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    support_ims = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    support_ims = support_ims.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    support_ims = Variable(support_ims)

    return im_data, im_info, num_boxes, gt_boxes, support_ims

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
