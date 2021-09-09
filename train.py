from run_nerf_helpers import *
from models import *

def train(images, depth_maps, hwf, poses, num_slots, num_iters):
    # images: (N, H, W, c)
    model = build_model(hwf, num_slots, num_iters)

    train_iters = 10000

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            recon, rgbs, masks, slots = model(images, depth_maps, poses)

            loss = img2mse(recon, images)

            ## update
    