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





def raw2outputs(raw, z_vals, rays_d):
    def raw2alpha(raw, dists): return 1.0 - tf.exp(-raw * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = tf.concat(
        [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
        axis=-1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    # Extract RGB of each sample position along each ray.
    rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]


    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # [N_rays, N_samples]
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

    # Computed weighted color of each sample along each ray.
    rgb_map = tf.reduce_sum(
        weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

    return rgb_map

def generate_rgb(raws, masked_raws, z_vals, rays_d, pts_shape, num_slots):
    B, P, N_samples,_ = pts_shape
    raws = tf.reshape(raws,[B, P, N_samples, 4])
    masked_raws = tf.reshape(masked_raws,[num_slots, B, P, N_samples, 4])
    # unmasked_raws = tf.reshape(unmasked_raws,[num_slots, B, P, N_samples, 4])
    # masks = tf.reshape(masks,[num_slots, B, P, N_samples, 1])

    rgb_maps = raw2outputs(raws, z_vals, rays_d)
    z_vals = tf.tile(z_vals[tf.newaxis,...], [num_slots, 1, 1, 1])
    rays_d = tf.tile(rays_d[tf.newaxis,...], [num_slots, 1, 1, 1])
    slots_rgb_maps = raw2outputs(masked_raws, z_vals, rays_d)

    return rgb_maps, slots_rgb_maps


def train_with_nerf(images, depth_maps, hwf, poses, num_slots, num_iters=3,
        N_save=100, N_img=100, train_iters=100000, lrate=1e-4, N_samples=64, chunk=512):

    os.makedirs('./imgs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    H, W, focal = hwf

    model = build_model(hwf, num_slots, num_iters, data_shape=images[0:1].shape, chunk=chunk, use_nerf=True)

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


    print('Start training')
    for i in range(0, train_iters):
        t = np.random.randint(0, N_imgs, 4)
        input_images, input_depths, input_poses = images[t], depth_maps[t], poses[t]


        
        with tf.GradientTape() as tape:
            points, z_vals, rays_d, select_inds = sampling_points(hwf, input_poses, N_samples, is_selection=True)
            pts_shape = points.shape
            training = mask = np.array([True])
            raws, masked_raws, unmasked_raws, masks = model([input_images, input_depths, input_poses, points, training])
            rgbs,_ = generate_rgb(raws, masked_raws, z_vals, rays_d, pts_shape, num_slots)
            loss_rgb = tf.gather_nd(input_images, select_inds, batch_dims = 1)
            loss = img2mse(rgbs, loss_rgb)
        
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))


        print('iter: {:06d}, loss: {:f}'.format(i, loss))

        if i % N_save == 0:
            # save_weights_npy(model, i)
            ckpt_manager.save(i)
        
        if i % N_img == 0:
            print('Saving images')
            val = np.random.randint(0, N_imgs, 4)
            val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]

            val_points, val_z_vals, val_rays_d = sampling_points(hwf, val_poses, N_samples)

            val_pts_shape = val_points.shape
            training = np.array([False])



            for c in range(0, val_points.shape[1], chunk):
                raws_c, masked_raws_c, unmasked_raws_c, masks_c = model([val_images, val_depths, val_poses, \
                                val_points[:, c:c+chunk,...], training])
                if c == 0:
                    val_raws = raws_c
                    val_slots_raws = masked_raws_c
                else:
                    val_raws = tf.concat([val_raws, raws_c], axis=-2)
                    val_slots_raws = tf.concat([val_slots_raws, masked_raws_c], axis=-2)

            val_rgb, val_slots_rgb = generate_rgb(val_raws, val_slots_raws, val_z_vals[0:1], val_rays_d[0:1], val_pts_shape, num_slots)
            
            print(val_rgb.shape)
            exit()
            val_rgb = tf.reshape(val_rgb, [4, H, W, 3])
            val_slots_rgb = tf.reshape(val_slots_rgb, [num_slots, 4, H, W, 3])

            imageio.imwrite(os.path.join('./imgs/val_{:06d}.jpg'.format(i)), to8b(val_rgb[0]))
            imageio.imwrite(os.path.join('./imgs/GT_{:06d}.jpg'.format(i)), val_images[0])

            for j in range(val_slots_rgb.shape[0]):
                imageio.imwrite(os.path.join(
                    './imgs/val_{:06d}_slot{:03d}.jpg'.format(i, j)), to8b(val_slots_rgb[j][0]))
            print('Done')
                








def main():
    # train args
    N_save = 200
    N_img = 100
    train_iters = 50000
    lrate = 5e-4
    # train args end

    depth_file = 'data/nerf_synthetic/clevr_100_2objects/all_depths.npy'
    depth_maps = np.load(depth_file)

    parser = config_parser()
    args = parser.parse_args()

    N_train = depth_maps.shape[0]

    images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip)

    train_with_nerf(images[0:N_train, :, :, 0:3], depth_maps[0:N_train], hwf, 
            poses[0:N_train], 3, 3, N_save, N_img=N_img, train_iters=train_iters,
            lrate=lrate)

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()