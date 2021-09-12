import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"
from run_nerf_helpers import *
from models import *
from run_nerf import *


def save_weights_npy(net, i):
    path = os.path.join(
        'weights', '{:06d}.npy'.format(i))
    np.save(path, net.get_weights())
    print('saved weights at', path)

def load_weights_npy(dir, model):
    ckpts = [os.path.join(dir, f) for f in sorted(os.listdir(os.path.join(dir)))]
    if len(ckpts) > 0:
        ckpt = ckpts[-1]
        print('reload from: ', ckpt)
        model.set_weights(np.load(ckpt, allow_pickle=True))

def train(images, depth_maps, hwf, poses, num_slots, num_iters=3,
        N_save=100, N_img=100, train_iters=100000, lrate=1e-4):
    # images: (N, H, W, C)
    # depth_maps : (N, H, W)
    # poses: (N, 4, 4)

    os.makedirs('./imgs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    model = build_model(hwf, num_slots, num_iters, data_shape=images[0:1].shape)
    trainable_vars = model.trainable_weights
    # if args.lrate_decay > 0:
    #     lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
    #                                                            decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    ckpt = tf.train.Checkpoint(
        network=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory='./checkpoints', max_to_keep=None)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    # load_weights_npy('./weights', model)

    N_imgs = images.shape[0]

    for i in range(20001, train_iters):
        t = np.random.randint(0, N_imgs)
        with tf.GradientTape() as tape:
            recon, rgbs, masks, slots = model([images[t:t+1], depth_maps[t:t+1], poses[t:t+1]])

            loss = img2mse(recon, images[t])
        
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        print('iter: {:06d}, img: {:03d}, loss: {:f}'.format(i, t, loss))

        if i % N_save == 0:
            # save_weights_npy(model, i)
            ckpt_manager.save(i)
        
        if i % N_img == 0:
            imageio.imwrite(os.path.join('./imgs/{:06d}.jpg'.format(i)), to8b(recon))
            for j in range(rgbs.shape[0]):
                imageio.imwrite(os.path.join(
                    './imgs/{:06d}_{:03d}_{:03d}.jpg'.format(i, j, t)), to8b(rgbs[j]))
        if i == 0:
            for j in range(images.shape[0]):
                imageio.imwrite(os.path.join('./imgs/GT_{:03d}.jpg'.format(j)), images[j])

def main():
    # train args
    N_save = 200
    N_img = 20
    train_iters = 50000
    lrate = 5e-4
    # train args end

    depth_maps = np.load('depth.npy')

    parser = config_parser()
    args = parser.parse_args()

    # N_train = depth_maps.shape[0] - 10
    N_train = 20

    images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip)

    train(images[0:N_train, :, :, 0:3], depth_maps[0:N_train], hwf, 
            poses[0:N_train], 3, 3, N_save, N_img=N_img, train_iters=train_iters,
            lrate=lrate)

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()