from models import *
# from train import *
import numpy as np
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"


IMG_SHAPE = (400, 400, 3)

vgg16 = tf.keras.applications.VGG16(input_shape = IMG_SHAPE,
                                    include_top=False,
                                    weights='imagenet')
# vgg16.summary()
feat_extractor = tf.keras.Sequential()
for i in range(10):
    feat_extractor.add(vgg16.get_layer(index=i))
feat_extractor.summary()
exit()
# exit()
# layer0 = vgg16.get_layer(index=0)
# layer1 = vgg16.get_layer(index=1)

img = tf.zeros([1] + list(IMG_SHAPE))
# print('img :', img.shape)
# out = layer0(img)
# print('layer0: ', out.shape)
# out = layer1(out)
# print('layer1: ', out.shape)
feat = vgg16(img)
print(feat.shape)
upsampled = tf.image.resize(feat, [400, 400])
print(upsampled.shape)
