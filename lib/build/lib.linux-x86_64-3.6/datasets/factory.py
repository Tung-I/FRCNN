# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.coco_split import coco_split
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.fsod import fsod
from datasets.simulated import simulated
from datasets.episode import episode
from datasets.ocid import ocid
from datasets.coco_finetune import coco_finetune

import numpy as np

# coco 20 evaluation
for year in ['set1', 'set2']:
  for split in ['3way', '5way']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# vis
for year in ['set1', 'set2', 'set3', 'set4']:
  for split in ['vis']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# coco 20 evaluation
for year in ['set1', 'set2', 'set3', 'set4']:
  for split in ['20']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# coco 60 training
for year in ['set1', 'set2', 'set3', 'set4', 'set1allcat']:
  for split in ['60']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# episode
for year in ['novel', 'base', 'val']:
  for n in range(600): 
    split = 'ep' + str(n)
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: episode(split, year))




# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# # Set up coco_2017_<split>
# for year in ['2017']:
#   for split in ['train', 'val']:
#     name = 'coco_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: coco(split, year))

# # Set up coco_2014_cap_<split>
# for year in ['2014']:
#   for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
#     name = 'coco_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up fsod
# for year in [None]:
#   for split in ['train', 'test']:
#     name = 'fsod_{}'.format(split)
#     __sets[name] = (lambda split=split, year=year: fsod(split))

# Set up coco_2015_<split>
# for year in ['2015']:
#   for split in ['test', 'test-dev']:
#     name = 'coco_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
# for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
#     for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# # set up image net.
# for split in ['train', 'val', 'val1', 'val2', 'test']:
#     name = 'imagenet_{}'.format(split)
#     devkit_path = 'data/imagenet/ILSVRC/devkit'
#     data_path = 'data/imagenet/ILSVRC'
#     __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
