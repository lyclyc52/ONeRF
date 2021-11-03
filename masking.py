import numpy as np
import imageio
# val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]
val =[0,1,2,9,10,11,12,13,14,16,17,24,25,27,28]


for i in range(15):
    img_file = './data/nerf_synthetic/clevr_c_d4/train/r_{:d}.png'.format(val[i]+1)
    img=imageio.imread(img_file)

    mask_file1 = 'results/testing_2/segmentation/val_000200_r_{:d}_slot3.png'.format(i)
    mask1 = imageio.imread(mask_file1)

    mask1 = mask1 /255.
    mask1 = mask1[..., None]


    mask_file2 = 'results/testing_2/segmentation/val_000200_r_{:d}_slot2.png'.format(i)
    mask2 = imageio.imread(mask_file2)

    mask2 = mask2 /255.
    mask2 = mask2[..., None]

    output = img * (mask1+mask2)

    output_file = 'data/nerf_synthetic/clevr_c_d4/training/r_{:d}.png'.format(val[i]+1)
    imageio.imwrite(output_file, output)

