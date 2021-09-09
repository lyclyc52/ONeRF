from models import *
from train import *

hwf = (400, 400, 400)

# fn = Encoder_Decoder(cam_param=hwf, num_slots=5, num_iterations=3, resolution=(400, 400), 
#     decoder_initial_size=(25, 25))

images = tf.ones((10, 400, 400, 3))
depth_maps = tf.ones((10, 400, 400))
poses = tf.ones((10, 4, 4))
train(images, depth_maps, hwf, poses, num_slots=5, num_iters=3)
