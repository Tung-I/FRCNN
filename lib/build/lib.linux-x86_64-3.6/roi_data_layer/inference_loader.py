"""The data layer used during training to train a Fast R-CNN network.
"""
import numpy as np
import random
import time
import pdb
import cv2
import torch.utils.data as data
import torch
from pathlib import Path
from PIL import Image
from scipy.misc import imread

from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob

from pycocotools.coco import COCO


class InferenceLoader(data.Dataset):
	def __init__(self, imdb, roidb, ratio_list, ratio_index, support_dir, \
				batch_size, num_classes, num_shot=5, training=True, normalize=None):
		self._imdb = imdb
		self._roidb = roidb
		self._num_classes = num_classes
		# we make the height of image consistent to trim_height, trim_width
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
		#############################################################################

		#############################################################################
		# {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 8, 16: 9, 17: 10, 18: 11, 
		#  19: 12, 20: 13, 21: 14, 44: 15, 62: 16, 63: 17, 64: 18, 67: 19, 72: 20}
		#############################################################################
		self._class_to_ind = dict(list(zip(self._imdb.classes, list(range(self._num_classes)))))
		coco_getCatIds = [0, 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
		self._class_to_coco_cat_id = dict(list(zip([c_n for c_n in self._imdb.classes], coco_getCatIds)))
		self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls], self._class_to_ind[cls])
										for cls in self._imdb.classes[1:]])
		self.class_ind_to_coco_cat_id = dict([(self._class_to_ind[cls], self._class_to_coco_cat_id[cls])
										for cls in self._imdb.classes[1:]])

		self.support_pool = [[] for i in range(self._num_classes)]
		for class_ind in range(self._num_classes):
			if class_ind == 0:
				continue
			coco_cat_id = self.class_ind_to_coco_cat_id[class_ind]
			support_files = [str(p) for p in list(Path(support_dir).glob(str(coco_cat_id) + '_*.jpg'))]
			self.support_pool[class_ind] += support_files
			
		self.support_im_size = 320
		self.testing_shot = num_shot
    


	def __getitem__(self, index):
		# testing
		index_ratio = index
		# though it is called minibatch, in fact it contains only one img here
		minibatch_db = [self._roidb[index_ratio]]

		# load query
		blobs = get_minibatch(minibatch_db)
		data = torch.from_numpy(blobs['data'])
		im_info = torch.from_numpy(blobs['im_info'])  # (H, W, scale)
		data_height, data_width = data.size(1), data.size(2)
		data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
		im_info = im_info.view(3)
		gt_boxes = torch.from_numpy(blobs['gt_boxes'])
		num_boxes = gt_boxes.size(0)
		
		# get supports
		support_data_all = np.zeros((self.testing_shot, 3, self.support_im_size, self.support_im_size), dtype=np.float32)
		current_gt_class_id = int(gt_boxes[0][4])
		pool = self.support_pool[current_gt_class_id]
		random.seed(index)

		selected_supports = random.sample(pool, k=self.testing_shot)
		for i, _path in enumerate(selected_supports):
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
		supports = torch.from_numpy(support_data_all)


		return data, im_info, gt_boxes, num_boxes, supports

	def __len__(self):
		return len(self._roidb)