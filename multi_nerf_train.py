import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
from model_multi_nerf import Multiple_NeRF
from load_blender_torch import load_blender_data
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import torchvision.transforms.functional as TF
def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

img_dir = './data/nerf_synthetic/shape_tex'
device = 'cuda:0'
img_size = 400
# N_imgs = 1
img_ids = [4, 5, 6, 23, 24, 30, 33, 40, 41, 42, 43, 45, 46, 48, 58]
# img_ids.remove(25)
# img_ids.remove(45)
# img_ids.remove(46)
# img_ids.remove(42)
i_print = 50
i_save = 500
i_vis = 400
save_img_dir = 'test_shapetex_imgs'
save_weight_dir = 'test_shapetex_weights'

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_weight_dir, exist_ok=True)

# test mask
# prefix='./masks_2d/segmentation/val_000550__slot'

# for k in range(16):
#     if k == 7 or k == 13:
#         continue
#     mask = imageio.imread(f'{prefix}{k}.jpg')
#     H, W, _ = mask.shape
#     mask = mask.reshape([-1, 3])
#     kmeans = KMeans(n_clusters=4, random_state=42).fit(mask)
#     cluster_map = kmeans.labels_
#     cluster_map = cluster_map.reshape((H, W, 1))
#     mask1 = (cluster_map == 2).squeeze()
#     mask2 = (cluster_map == 3).squeeze()

#     mask2 = ndimage.binary_erosion(mask2, structure=np.ones((2,2)))
#     mask1 = mask1.astype(np.uint8) * 255
#     mask2 = mask2.astype(np.uint8) * 255
#     imageio.imsave(f'./masks_2d/mask_final/mask_final_{k:04d}_s1.png', mask1)
#     imageio.imsave(f'./masks_2d/mask_final/mask_final_{k:04d}_s2.png', mask2)
# exit()



masks = []
mask_prefix = 'data/nerf_synthetic/shape_tex/masks/val_000950_r_'
for k in range(15):
    # if k == 7 or k == 13 or k ==11 or k==14:
    #     continue
    # mask1 = imageio.imread(f'./masks_2d/mask_final/mask_final_{k:04d}_s1.png').astype(np.float32)
    # mask2 = imageio.imread(f'./masks_2d/mask_final/mask_final_{k:04d}_s2.png').astype(np.float32)
    mask1 = imageio.imread(mask_prefix+f'{k}_slot0.jpg')
    mask2 = imageio.imread(mask_prefix+f'{k}_slot1.jpg')
    mask1 = (mask1 > 128).astype(np.float32)
    mask2 = (mask2 > 128).astype(np.float32)
    mask3 = (1 - mask1 - mask2)

    # ax1 = plt.subplot(3, 1, 1)
    # ax2 = plt.subplot(3, 1, 2)
    # ax3 = plt.subplot(3, 1, 3)
    # ax1.imshow(mask1, cmap='gray')
    # ax2.imshow(mask2, cmap='gray')
    # ax3.imshow(mask3, cmap='gray')
    # plt.show()
    # exit()

    mask = np.stack([mask1, mask2, mask3], axis=0)
    masks.append(mask)

masks = np.array(masks)

images, poses, render_poses, hwf, i_split = load_blender_data(
            img_dir, False, 1, size=img_size)
images = images[img_ids, ...]
poses = poses[img_ids]

images, poses = images.to(device), torch.from_numpy(poses).to(device)
# select images and masks
# masks = np.load(os.path.join('test_clusters', 'cluster_maps.npy'))
masks = torch.from_numpy(masks).to(device)
masks = TF.resize(masks, img_size, interpolation=0)
model = Multiple_NeRF(images, poses, hwf, N_nerf=3, masks=masks)


for i in range(1000000):
    loss, loss_recon, loss_overlap, loss1, loss2 = model.update()

    if i % i_print == 0:
        print('='*50)
        print(i)
        print('loss:         ', loss)
        print('loss_recon:   ', loss_recon)
        print('loss_overlap: ', loss_overlap)
        print('loss1:        ', loss1)
        print('loss2:        ', loss2)
        print('='*50)

    if i % i_vis == 0:
        img, img_locals = model.validate()
        img = to8b(img.detach().cpu().numpy())
        imageio.imwrite(os.path.join(save_img_dir, '{:08d}_combined.jpg'.format(i)), img)
        for s, img_local in enumerate(img_locals, start=0):
            img_local = to8b(img_local.detach().cpu().numpy())
            imageio.imwrite(os.path.join(
                save_img_dir, '{:08}_local{:02d}.jpg'.format(i, s)), img_local)

    if i % i_save == 0:
        model.save_weights(os.path.join(save_weight_dir, '{:08d}_weight.pt'.format(i)), i)
