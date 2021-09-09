from run_nerf_helpers import *
from models import *

def train(images, depth_maps, hwf, poses, num_slots, num_iters):
    # images: (N, H, W, C)
    # depth_maps : (N, H, W)
    # poses: (N, 4, 4)

    model = build_model(hwf, num_slots, num_iters, data_shape=images.shape)
    trainable_vars = model.trainable_weights

    lrate = 10e-4
    # if args.lrate_decay > 0:
    #     lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
    #                                                            decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    train_iters = 2

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            recon, rgbs, masks, slots = model([images, depth_maps, poses])

            loss = img2mse(recon, images)
        
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))