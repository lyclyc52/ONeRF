import os 
val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]
for i in range(45):
    t = 'data/nerf_synthetic/clevr_c_d4/train/{:03d}.png'.format(i)
    s = 'data/nerf_synthetic/clevr_c_d4/train/r_{:d}.png'.format(i+1)
    os.rename(t, s)