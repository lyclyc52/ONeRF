import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='7, 8'
from model_torch import *
from load_blender import *
from run_nerf_helpers import *
from run_nerf import *
import torch



depth_file = 'data/nerf_synthetic/clevr_100_2objects/all_depths.npy'
depth_maps = np.load(depth_file)
depth_maps = depth_maps[..., None]
depth_maps = tf.compat.v1.image.resize_area(depth_maps, [128, 128]).numpy()
depth_maps = tf.squeeze(depth_maps, axis=-1).numpy()

weights_dir = './weights_new_loss'
imgs_dir = './imgs_new_loss'
os.makedirs(imgs_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

parser = config_parser()
args = parser.parse_args()

N_train = depth_maps.shape[0]

images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)


model = Encoder_Decoder_nerf(hwf)


start_iter = 0
# model.load_weights(weights_dir,2000)


train_iters = 10000000

N_print=10
N_save = 100
N_imgs = 20
N_save_img = 100
N_input_img = 1

images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)

print('Start training')
for i in range(0, train_iters):
    t = np.random.randint(0, N_imgs, N_input_img)
    input_images, input_depths, input_poses = images[t], depth_maps[t], poses[t]
    loss = model.update_grad(input_images, input_depths, input_poses, i)
    
    if i % N_print == 0:
        print('iter: {:06d},  loss: {:f}'.format(i, loss))


    if i % N_save == 0:
        model.save_weights(weights_dir, i)



    if i % N_save_img == 0: 
        val = np.random.randint(0, N_imgs, N_input_img)
        val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]
        with torch.no_grad():
            rgb, slots_rgb = model.forward(val_images, val_depths, val_poses, isTrain=False)
            rgb, slots_rgb = rgb.numpy(), slots_rgb.numpy()

            val_images = val_images.numpy()
            imageio.imwrite(os.path.join(imgs_dir, 'val_{:06d}.jpg'.format(i)), to8b(rgb[0]))
            for s in range(model.num_slots):
                imageio.imwrite(os.path.join(imgs_dir, 'val_{:06d}_s{:02d}.jpg'.format(i, s)), to8b(slots_rgb[s].squeeze(0)))
