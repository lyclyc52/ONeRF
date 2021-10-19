import os 
for i in range(63):
    t = 'data/nerf_synthetic/clevr_same/val/CLEVR_new_{:06}.png'.format(i)
    s= 'data/nerf_synthetic/clevr_same/val/r_{:d}.png'.format(i)
    os.rename(t, s)