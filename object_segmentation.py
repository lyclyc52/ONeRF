from object_segmentation_helper import *
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

base_dir = './results/testing_3'


datadir = 'data/nerf_synthetic/clevr_bg6'


input_size = 400
images, poses, depth_maps, render_poses, hwf, i_split = load_data(
            datadir, size = input_size)


first_cluster_dir = os.path.join(base_dir, 'first_cluster')
os.makedirs(first_cluster_dir, exist_ok=True)

mask_refine_dir = os.path.join(base_dir, 'mask_refine')
os.makedirs(mask_refine_dir, exist_ok=True)

segmentation_dir = os.path.join(base_dir, 'segmentation')
os.makedirs(segmentation_dir, exist_ok=True)


N_imgs =100
images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]



device = torch.device("cuda:0" )




# image = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48]
# val = [0, 3, 25, 39]
# val = [t for t in range(iter*2,iter*2+4)]
# val = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48] #for simple clevr



# val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60] #for  clevrtex

val = [ 4, 5, 6, 23, 24, 30, 33, 40, 41, 42, 43, 45, 46, 48, 58] #for  clevrtex
# val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# val = [0,1,2, 9,10,11,12,13,14,16,17,24,25,27,28]
print(val)


val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]




cluster_size = 100
cluster_images = tf.compat.v1.image.resize_area(val_images, [cluster_size, cluster_size]).numpy()
cluster_depth = tf.compat.v1.image.resize_area(val_depths[...,None], [cluster_size, cluster_size]).numpy()
cluster_depth = cluster_depth[...,0]


cluster_images, cluster_depth, val_poses = torch.from_numpy(cluster_images), torch.from_numpy(cluster_depth), torch.from_numpy(val_poses)



H,W,focal = hwf
_,H,W,_ = cluster_images.shape
hwf= [H,W,focal]


# val_images, val_depths, val_poses = val_images.to(device), val_depths.to(device), val_poses.to(device)


with torch.no_grad():
    cluster_images, cluster_depth, val_poses = cluster_images.to(device), cluster_depth.to(device), val_poses.to(device)
    f_extractor = Encoder_VGG(hwf, device=device)
    f_extractor.to(device)
    print(cluster_images.shape)
    print(cluster_depth.shape)
    print(val_poses.shape)
    f = f_extractor(cluster_images, cluster_depth, val_poses)
    B, H, W, C = f.shape
    f = f.reshape([-1, C])
    f_p = f[...,C-3:]
    f = f[...,:C-3]

    w = .5

    attn_logits = KM_clustering(f, f_p, w, device)
    attn = attn_logits.softmax(dim=-1)
    attn = attn.reshape([B,H,W,2])
    attn = attn.permute([0,3,1,2])

    attn = attn.cpu().numpy()
    cluster_images = cluster_images.cpu().numpy()
    # print(iter)

    seg_inputs = []


    for b in range(B):
        for s in range(2):
            imageio.imwrite(os.path.join(first_cluster_dir, 'val_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s]))
            imageio.imwrite(os.path.join(first_cluster_dir, 'masked_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s][...,None]*cluster_images[b]))


        im = to8b(attn[b][1])
        dilation = ndimage.binary_dilation(im)
        dilation = ndimage.binary_dilation(dilation, iterations=1)
        # erode = ndimage.binary_erosion(dilation, iterations=1)

        origin = images[val[b]]
        erode = (dilation / 255.).astype(np.float32)
        erode = tf.compat.v1.image.resize_area(erode[None, ..., None], [input_size, input_size]).numpy()



        seg_input =  to8b(erode[0, ...] * origin * 255.)
        seg_inputs.append(torch.from_numpy(seg_input))
        imageio.imsave(os.path.join(mask_refine_dir,'seg_input{:d}.png'.format(b)),seg_input)
        imageio.imsave(os.path.join(mask_refine_dir,'mask{:d}.png').format(b), to8b(erode[0] * 255.))



# seg_inputs = torch.stack(seg_inputs)
# seg_inputs = seg_inputs/255.

# print(seg_inputs.shape)



# torch.cuda.empty_cache()





# imgs = []
# for i in range(15):
#     fname = os.path.join(mask_refine_dir, 'seg_input{:01d}.png'.format(i))
#     imgs.append(imageio.imread(fname))


# imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)

# imgs= imgs[...,:3]
# seg_inputs = torch.from_numpy(imgs)


# model, target_im = train(seg_inputs, base_dir)


# cluster =  np.unique(im_target)


# target_im = target_im.cpu().numpy()
# final_masks = []
# for c in range(cluster.shape[0]):
#     masks = (target_im==cluster[c])
#     cluster_masks = []
#     for b in range(B):
#         im = mask[b]
#         dilation = ndimage.binary_dilation(im, iterations= 1)
#         erode = ndimage.binary_erosion(im, iterations= 4)
#         cluster_masks.append(erode)
#     cluster_masks = torch.stack(cluster_masks)
#     final_masks.append(cluster_masks)


# final_masks = torch.stack(final_masks, dim=-1)



# init_bg_mask = attn[0][0]
# init_bg_mask = tf.compat.v1.image.resize_area(init_bg_mask[None, ..., None], [input_size, input_size]).numpy()
# init_bg_mask = init_bg_mask[0,...,0]

# inter_area = []
# for c in range(cluster.shape[0]):
#     intersection = init_bg_mask*final_masks[0,...,c]
#     inter_area.append(np.sum(intersection))

# max_value = max(inter_area) 
# max_index = number_list.index(max_value) 




# val_images = torch.from_numpy(val_images)
# val_images = val_images.to(device)


# output = model( val_images )
# inti_slot = val_images.shape[1]
# output = output.permute([0,2,3,1]).reshape( [-1, inti_slot] )



# cluster_index = []
# for c in range(cluster.shape[0]):
#     cur_m = final_masks[..., c]
#     cur_m = cur_m.reshape(-1) 
#     cluster_index.append(cur_m)





# for i in range(5):
#     t = []
#     print('sample')
#     for j in range(cluster.shape[0]):
#         size = cluster_index[j].shape[0]
#         index = torch.randint(size, (size//10,))
#         c_class = cluster.shape[0][j]
#         c_class = c_class[index]
#         t.append(c_class.mean(dim=0))
#     for j in range(cluster.shape[0]):
#         print(torch.norm(t[j]-t[max_index]))

