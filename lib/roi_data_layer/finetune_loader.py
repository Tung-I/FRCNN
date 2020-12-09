import numpy as np
import random
import time
import pdb
import cv2
import torch.utils.data as data
import torch
import os
from pathlib import Path
from PIL import Image
from scipy.misc import imread

from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob

from pycocotools.coco import COCO


class FTLoader(data.Dataset):
    def __init__(self, imdb, roidb, ratio_list, ratio_index, support_dir, 
                batch_size, num_classes, training=True, normalize=None, num_way=2, num_shot=5):
        self._imdb = imdb
        self._roidb = roidb
        self._num_classes = num_classes
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        #############################################################################
        # roidb:
        # {'width': 640, 'height': 484, 'boxes': array([[ 58, 152, 268, 243]], dtype=uint16), 
        # 'gt_classes': array([79], dtype=int32), flipped': False, 'seg_areas': array([12328.567], dtype=float32),
        # 'img_id': 565198, 'image': '/home/tungi/FSOD/data/coco/images/val2014/COCO_val2014_000000565198.jpg', 
        # 'max_classes': array([79]), 'max_overlaps': array([1.], dtype=float32), 'need_crop': 0}

        # name_to_coco_cls_ind = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7,
        #  	'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
        # 	'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24,
        # 	'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35,
        # 	'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
        # 	'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 
        # 	'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 
        # 	'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72,
        # 	'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 
        # 	'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}
        #############################################################################
        self.support_im_size = 320
        self.support_way = num_way
        self.support_shot = num_shot

        self.support_pool = [[] for i in range(self._num_classes)]
        self._label_to_cls_name = dict(list(zip(list(range(self._num_classes)), self._imdb.classes)))
        for _label in range(1, self._num_classes):
            cls_name = self._label_to_cls_name[_label]
            cls_dir = os.path.join(support_dir, cls_name)
            support_im_paths = [str(_p) for _p in list(Path(cls_dir).glob('*.jpg'))]
            if len(support_im_paths) == 0:
                raise Exception(f'support data not found in {cls_dir}')
            random.seed(0)  # fix the shots
            support_im_paths = random.sample(support_im_paths, k=self.support_shot)
            self.support_pool[_label].extend(support_im_paths)
        ##############################
        # given the ratio_list, we want to make the ratio same for each batch.
        # ex. [0.5, 0.5, 0.7, 0.8, 1.5, 1.6, 2., 2.] -> [0.5, 0.5, 0.7, 0.7, 1.6, 1.6, 2., 2.]
        ##############################
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)
            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1
            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

    


    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # though it is called minibatch, in fact it contains only one img here
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db)  # [n_box, 5]
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])  # (H, W, scale)
        data_height, data_width = data.size(1), data.size(2)  # [1, h, w, c]
        gt_boxes = blobs['gt_boxes']

        #################
        # support data
        #################
        support_data_all = np.zeros((self.support_way * self.support_shot, 3, self.support_im_size, self.support_im_size), dtype=np.float32)
        # positive
        current_gt_class_id = int(gt_boxes[0][4])
        pos_selected_supports = self.support_pool[current_gt_class_id]
        for i, _path in enumerate(pos_selected_supports):
            support_im = imread(_path)[:,:,::-1]  # rgb -> bgr
            target_size = np.min(support_im.shape[0:2])  # don't change the size
            support_im, _ = prep_im_for_blob(support_im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            _h, _w = support_im.shape[0], support_im.shape[1]
            if _h > _w:
                resize_scale = float(self.support_im_size) / float(_h)
                unfit_size = int(_w * resize_scale)
                support_im = cv2.resize(support_im, (unfit_size, self.support_im_size), interpolation=cv2.INTER_LINEAR)
            else:
                resize_scale = float(self.support_im_size) / float(_w)
                unfit_size = int(_h * resize_scale)
                support_im = cv2.resize(support_im, (self.support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
            h, w = support_im.shape[0], support_im.shape[1]
            support_data_all[i, :, :h, :w] = np.transpose(support_im, (2, 0, 1)) 
        # negative
        neg_cls = []
        for cls_id in range(1, self._num_classes):
            if cls_id != current_gt_class_id:
                neg_cls.append(cls_id)
        random.seed(index)
        neg_cls_idx = random.sample(neg_cls, k=1)[0]
        neg_selected_supports = self.support_pool[neg_cls_idx]
        for i, _path in enumerate(neg_selected_supports):
            support_im = imread(_path)[:,:,::-1]  # rgb -> bgr
            target_size = np.min(support_im.shape[0:2])  # don't change the size
            support_im, _ = prep_im_for_blob(support_im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            _h, _w = support_im.shape[0], support_im.shape[1]
            if _h > _w:
                resize_scale = float(self.support_im_size) / float(_h)
                unfit_size = int(_w * resize_scale)
                support_im = cv2.resize(support_im, (unfit_size, self.support_im_size), interpolation=cv2.INTER_LINEAR)
            else:
                resize_scale = float(self.support_im_size) / float(_w)
                unfit_size = int(_h * resize_scale)
                support_im = cv2.resize(support_im, (self.support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
            h, w = support_im.shape[0], support_im.shape[1]
            support_data_all[i + self.support_shot, :, :h, :w] = np.transpose(support_im, (2, 0, 1)) 

        support = torch.from_numpy(support_data_all)
    
        #################
        # query data
        #################
        # padding the input image to fixed size for each group 
        # if the image need to crop, crop to the target size.
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        ratio = self.ratio_list_batch[index]
        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # no need to crop, or after cropping
        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                                data_width, 3).zero_()
            padding_data[:data_height, :, :] = data[0]
            # update im_info [[H, W, scale]]
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                                int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        # filt boxes
        fs_gt_boxes = gt_boxes.clone()
       
        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

        #
        not_keep = (fs_gt_boxes[:,0] == fs_gt_boxes[:,2]) | (fs_gt_boxes[:,1] == fs_gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        fs_gt_boxes_padding = torch.FloatTensor(self.max_num_box, fs_gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            fs_gt_boxes = fs_gt_boxes[keep]
            num_boxes = min(fs_gt_boxes.size(0), self.max_num_box)
            fs_gt_boxes_padding[:num_boxes,:] = fs_gt_boxes[:num_boxes]
        else:
            num_boxes = 0

        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        
        # to sum up, data in diffenent batches may have different size, 
        # but will have same size in the same batch

        return padding_data, im_info, fs_gt_boxes_padding, num_boxes, support_data_all, gt_boxes_padding

    def __len__(self):
        return len(self._roidb)