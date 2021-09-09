import tensorflow as tf

from models import *


c2w=np.zeros([3,4,4])
c2w=tf.cast(c2w, dtype=tf.float32)

H=400
W=400
focal=100
get_rays(H,W,focal,c2w)

image=np.zeros([3,H,W,3])
depth=tf.cast(image, dtype=tf.float32)
depth=np.zeros([3,H,W])
depth=tf.cast(depth, dtype=tf.float32)
encoder = Encoder(cam_param=[H,W,focal])
# encoder.set_input(depth, c2w)

x = encoder(image,depth,c2w)
print(x.shape)

x= tf.reshape(x, [-1,64])

attention=SlotAttention()
slots,attention=attention(x)
print(slots.shape)
print(attention.shape)









