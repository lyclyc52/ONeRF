import tensorflow as tf

from models import *


c2w=np.zeros([3,4,4])
c2w=tf.cast(c2w, dtype=tf.float32)

H=400
W=400
focal=100


image=np.zeros([3,H,W,3])
image=tf.cast(image, dtype=tf.float32)
depth=np.zeros([3,H,W])
depth=tf.cast(depth, dtype=tf.float32)

N_samples=64

pose = [H,W,focal]

p,z,r,_=sampling_points(pose, c2w, is_selection=True)
model = Encoder_Decoder_nerf(pose)
x=model(image, depth, c2w,p,z,r)

print(p.shape)
print(z.shape)
print(r.shape)
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16













