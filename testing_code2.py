# from models import *
# from train import *
import numpy as np
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"
from multi_object_datasets import multi_dsprites

data_path = '/data/sliangal/clevr_with_masks_clevr_with_masks_train.tfrecords'

batch_size = 1
dataset = multi_dsprites.dataset(data_path, 'colored_on_colored')
# batched_dataset = dataset.batch(batch_size)  # optional batching
# iterator = batched_dataset.make_one_shot_iterator()
# data = iterator.get_next()
dataset = dataset.enumerate(start=0)
for element in dataset:
  print(element)
exit()