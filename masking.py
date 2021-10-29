import numpy as np
import imageio
val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]



for i in range(1,16):
    img_file = './data/nerf_synthetic/clevr_1/train/r_{:d}.png'.format(i)
    img=imageio.imread(img_file)

    mask_file = 'results/testing_1/segmentation/val_000100_r_{:d}_slot3.png'.format(i-1)
    mask = imageio.imread(mask_file)

    mask = mask /255.
    mask = mask[..., None]

    output = img * mask

    output_file = 'data/nerf_synthetic/clevr_1/training/r_{:d}.png'.format(i)
    imageio.imwrite(output_file, output)

