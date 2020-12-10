from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.coco_split import coco_split
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.episode import episode

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


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
