import torch 
from model_torch import *

H, W, focal = 100,100,100.
depth =torch.zeros([4,H,W])
im = torch.zeros([4,H,W,3])
c2w = torch.zeros([4,4,4])
model = Encoder_Decoder_nerf([H, W, focal])
model.process(im, depth, c2w)
# model3(pts)