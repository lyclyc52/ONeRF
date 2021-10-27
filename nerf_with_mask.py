from object_segmentation_helper import *
from model_torch import *

datadir = 'data/nerf_synthetic/clevrtex_bg2'
input_size = 128
images, poses, depth_maps, render_poses, hwf, i_split = load_data(
            datadir, size = input_size)


mask_dir = 'results/testing_4/segmentation'

val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]
images, poses = images[val], poses[val]



masks = []
for i in range(15):
    slot1 = imageio.imread(os.path.join(mask_dir, 'val_000500_r_{:d}_slot1.jpg'.format(i)))
    slot2 = imageio.imread(os.path.join(mask_dir, 'val_000500_r_{:d}_slot2.jpg'.format(i)))
    slot1 = slot1 / 255.
    slot2 = slot2 / 255.

    slot3 = 1. - (slot1+slot2)
    masks.append(np.stack([slot1, slot2, slot3]))


masks = np.stack(masks)


images, masks, poses =  torch.from_numpy(images), torch.from_numpy(masks), torch.from_numpy(poses)



model = Multi_Nerf(hwf)


img_dir = './results/testing_4/imgs'


start_iter = 0
train_iters = 10000000
for i in range(start_iter, train_iters):
    t = np.random.randint(0, 15, 1)
    input_images, input_masks, input_poses = images[t], masks[t], poses[t]
    loss = model.update_grad(input_images, input_masks, input_poses, i)

    if i % N_print == 0:
        print('iter: {:06d},  loss: {:f}'.format(i, loss))


    if i % N_save == 0:
        model.save_weights(weights_dir, i)



    if i % N_img == 0: 
        val = np.random.randint(0, 20, 1)
        # val = [0, 3, 25, 39]
        # val = [0, 1, 2, 3]

        # check = 0
        val_images, val_masks, val_poses = images[val], masks[val], poses[val]
        with torch.no_grad():
            rgb, masked_rgb_slots, unmasked_rgb_slots = model.forward(val_images, val_masks, val_poses, isTrain=False)
            rgb = rgb.cpu().numpy()
            for s in range(len(masked_rgb_slots)):
                masked_rgb_slots[s] = masked_rgb_slots[s].cpu().numpy()
                unmasked_rgb_slots[s] = unmasked_rgb_slots[s].cpu().numpy()
            val_images = val_images.cpu().numpy()
            # print(attn.shape)
            print('Saving images')
            imageio.imwrite(os.path.join(img_dir, 'val_{:06d}.jpg'.format(i)), to8b(rgb[check]))
            for j in range(len(masked_rgb_slots)):
                imageio.imwrite(os.path.join(img_dir, 'val_{:06d}_masked_slot{:01d}.jpg'.format(i,j)), to8b(masked_rgb_slots[j][check]))
                imageio.imwrite(os.path.join(img_dir, 'val_{:06d}_unmasked_slot{:01d}.jpg'.format(i,j)), to8b(unmasked_rgb_slots[j][check]))

            print('Done') 
