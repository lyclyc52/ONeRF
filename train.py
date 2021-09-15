import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='1,2'
from run_nerf_helpers import *
from models import *
from run_nerf import *
# import tensorflow.keras.applications.vgg16.VGG16 as VGG16


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



def train(images, depth_maps, hwf, poses, num_slots, num_iters=3, N_print=500,
        N_save=2000, N_img=1000, train_iters=1000000, lrate=1e-4):
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

        if i % N_print == 0:
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








def train_with_nerf(images, depth_maps, hwf, poses, num_slots, num_iters=3, N_print=500,
        N_save=2000, N_img=1000, train_iters=1000000, lrate=5e-4, N_samples=64, chunk=1024*64 , N_selection=64):

    os.makedirs('./imgs_1', exist_ok=True)
    os.makedirs('./checkpoints_1', exist_ok=True)

    H, W, focal = hwf

    model = build_model(hwf, num_slots, num_iters, data_shape=images[0:1].shape, chunk=chunk, use_nerf=True)


    # if args.lrate_decay > 0:
    #     lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
    #                                                            decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    ckpt = tf.train.Checkpoint(
        network=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory='./checkpoints_1', max_to_keep=None)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    # load_weights_npy('./weights', model)

    N_imgs = images.shape[0]

    # basemodel = VGG16(include_top = False)    
    # vgg16 = tf.keras.Sequential(basemodel.layers[:16])
    # for layer in vgg16.layers[:10]:
    #     layer.trainable = False


    print('Start training')
    for i in range(0, train_iters):
        t = np.random.randint(0, N_imgs)
        t = 0
        input_images, input_depths, input_poses = images[t], depth_maps[t], poses[t]

        input_images, input_depths, input_poses = input_images[None,...], input_depths[None,...], input_poses[None,...]
        with tf.GradientTape() as tape:

            points, z_vals, rays_d, select_inds = sampling_points(hwf, input_poses, N_samples, is_selection=True, N_selection=N_selection)
            pts_shape = points.shape
            training = mask = np.array([True])
            raws, masked_raws, unmasked_raws, masks = model(input_images, input_depths, input_poses, points, training)

            rgbs,_ = generate_rgb(raws, masked_raws, z_vals, rays_d, pts_shape, num_slots)

            rgbs = tf.reshape(rgbs, [1, N_selection, N_selection, 3])

            select_inds_x, select_inds_y = select_inds 
            loss_rgb = input_images[:, select_inds_x:select_inds_x+64, select_inds_y:select_inds_y+64,...]
            loss = img2mse(rgbs, loss_rgb)
        

        trainable_vars = model.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))



        if i % N_print == 0:
            print('iter: {:06d}, img: {:03d}, loss: {:f}'.format(i, t, loss))


        if i % N_save == 0:
            # save_weights_npy(model, i)
            ckpt_manager.save(i)
        
        if i % N_img == 0:
           
            val = np.random.randint(0, N_imgs)
            val = 0
            val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]
            val_images, val_depths, val_poses = val_images[None,...], val_depths[None,...], val_poses[None,...]
            val_points, val_z_vals, val_rays_d = sampling_points(hwf, val_poses, N_samples)

            val_pts_shape = val_points.shape
            training = np.array([False])



            for c in range(0, val_points.shape[1], chunk):
                raws_c, masked_raws_c, unmasked_raws_c, masks_c = model(val_images, val_depths, val_poses, \
                                val_points[:, c:c+chunk,...], training)
                if c == 0:
                    val_raws = raws_c
                    val_slots_raws = masked_raws_c
                else:
                    val_raws = tf.concat([val_raws, raws_c], axis=-2)
                    val_slots_raws = tf.concat([val_slots_raws, masked_raws_c], axis=-2)

            val_rgb, val_slots_rgb = generate_rgb(val_raws, val_slots_raws, val_z_vals[0:1], val_rays_d[0:1], val_pts_shape, num_slots)
            

            val_rgb = tf.reshape(val_rgb, [1, H, W, 3])
            val_slots_rgb = tf.reshape(val_slots_rgb, [num_slots, 1, H, W, 3])

            print('Save images')

            imageio.imwrite(os.path.join('./imgs_1/val_{:06d}.jpg'.format(i)), to8b(val_rgb[0]))
            imageio.imwrite(os.path.join('./imgs_1/GT_{:06d}.jpg'.format(i)),  to8b(val_images[0]))

            for j in range(val_slots_rgb.shape[0]):
                imageio.imwrite(os.path.join(
                    './imgs_1/val_{:06d}_slot{:03d}.jpg'.format(i, j)), to8b(val_slots_rgb[j][0]))
            print('Done')

                








def main():
    # train args
    # N_save = 200
    # N_img = 100
    # train_iters = 50000
    # lrate = 5e-4
    # train args end

    depth_file = 'data/nerf_synthetic/clevr_100_2objects/all_depths.npy'
    depth_maps = np.load(depth_file)
    depth_maps = depth_maps[..., None]
    depth_maps = tf.compat.v1.image.resize_area(depth_maps, [128, 128]).numpy()
    depth_maps = tf.squeeze(depth_maps, axis=-1).numpy()


    parser = config_parser()
    args = parser.parse_args()

    N_train = depth_maps.shape[0]

    images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip)

    train_with_nerf(images[0:N_train, :, :, 0:3], depth_maps[0:N_train], hwf, 
            poses[0:N_train], 3, 3)

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()