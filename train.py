import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"
from run_nerf_helpers import *
from models import *
from run_nerf import *


def save_weights(net, i):
    path = os.path.join(
        'weights', '{:06d}.npy'.format(i))
    np.save(path, net.get_weights())
    print('saved weights at', path)

def train(images, depth_maps, hwf, poses, num_slots, num_iters,
        N_save=100, N_img=100, train_iters=100000):
    # images: (N, H, W, C)
    # depth_maps : (N, H, W)
    # poses: (N, 4, 4)

    model = build_model(hwf, num_slots, num_iters, data_shape=images.shape)
    trainable_vars = model.trainable_weights

    lrate = 1e-4
    # if args.lrate_decay > 0:
    #     lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
    #                                                            decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            recon, rgbs, masks, slots = model([images, depth_maps, poses])

            loss = img2mse(recon, images)
        
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        print('iter', i)

        if i % N_save == 0:
            save_weights(model, i)
        
        if i % N_img == 0:
            imageio.imwrite(os.path.join('./imgs/{:06d}.jpg'.format(i)), recon)
            for j in range(rgbs.shape[0]):
                imageio.imwrite(os.path.join(
                    './imgs/{:06d}_{:03d}.jpg'.format(i, j)), rgbs[j])

def main():
    # train args
    N_save = 100
    N_img = 100
    train_iters = 10000
    # train args end

    depth_maps = np.load('depth.npy')

    parser = config_parser()
    args = parser.parse_args()

    N_train = depth_maps.shape[0]

    images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip)

    train(images[0:N_train, :, :, 0:3], depth_maps, hwf, poses[0:N_train], 2, 3, N_save, N_img=N_img, train_iters=train_iters)

if __name__ == '__main__':
    main()