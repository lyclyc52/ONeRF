import tensorflow as tf

from models import *


c2w=np.zeros([3,4,4])

H=400
W=400
focal=100
get_rays(H,W,focal,c2w)

image=np.zeros([3,H,W,3])









