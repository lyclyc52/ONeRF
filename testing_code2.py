from models import *
# from train import *
import numpy as np
import tensorflow as tf
import os
import torch
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
