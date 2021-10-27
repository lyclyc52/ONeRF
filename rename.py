import os 
val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60]
for i in val:
    t = 'data/nerf_synthetic/clevr_c_d2/images/{:03}.png'.format(i)
    s = 'data/nerf_synthetic/clevr_c_d2/images/r_{:d}.png'.format(i)
    os.rename(t, s)