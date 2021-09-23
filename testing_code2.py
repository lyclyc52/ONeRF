from model_torch import *
# from train import *
import numpy as np
# import tensorflow as tf
import os
import torch
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"]="8"
from load_blender import *
from torchvision.transforms import Normalize
from run_nerf import config_parser

vgg_preprocess = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

parser = config_parser()
args = parser.parse_args()

images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
images = torch.from_numpy(images).permute([0, 3, 1, 2])[:, 0:3, ...]
img = images[0]
print(torch.min(img), torch.max(img))
img = vgg_preprocess(img)
print(torch.min(img), torch.max(img))
