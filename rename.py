import os 
val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]
for i in range(100,0, -1):

    t = 'data/nerf_synthetic/clevr_1/train2/r_{:d}.png'.format(i-1)
    s = 'data/nerf_synthetic/clevr_1/train2/r_{:d}.png'.format(i)
    os.rename(t, s)