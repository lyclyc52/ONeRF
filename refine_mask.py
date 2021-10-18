import torch
import torch.nn as nn
import scipy
from scipy import ndimage
import imageio
from KM_testing import *
from PIL import Image
import cv2


input_size = 400
datadir = 'data/nerf_synthetic/clevr_depth'
images, poses, depth_maps, render_poses, hwf, i_split = load_data(
            datadir, True, 1, size = input_size)


N_imgs=49
images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
# images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)




val = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48]



# for i in range(5):
#     im = imageio.imread('./results/testing_9/imgs/val_{:06d}_slot1.jpg'.format(i))
#     dilation = ndimage.binary_dilation(im)
#     erode = ndimage.binary_erosion(dilation)
#     # dilation = (dilation / 255.).astype(np.float32)
#     origin = images[val[i]]
#     imageio.imsave('./results/testing_9/imgs/dilation{:d}.png'.format(i), to8b(erode[...,None]))


for i in range(len(val)):
    im = imageio.imread('./results/testing_11/first_cluster/val_{:06d}_slot1.jpg'.format(i))
    
    dilation = ndimage.binary_dilation(im)
    dilation = ndimage.binary_dilation(dilation)
    erode = ndimage.binary_erosion(dilation)
    erode = ndimage.binary_erosion(erode)
    origin = images[val[i]]
    erode = (erode / 255.).astype(np.float32)
    erode = tf.compat.v1.image.resize_area(erode[None, ..., None], [input_size, input_size]).numpy()

    imageio.imsave('./results/testing_11/mask_refine/input{:d}.png'.format(i), to8b(erode[0, ...] * origin * 255.))
    imageio.imsave('./results/testing_11/mask_refine/mask{:d}.png'.format(i), to8b(erode[0]))
