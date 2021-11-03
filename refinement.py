import torch
import imageio
from object_segmentation_helper import *





base_dir = './results/testing_clevrtex_room'
datadir = 'data/nerf_synthetic/clevrtex_room'
image_dir = os.path.join(base_dir, 'segmentation')
nerf_dir = os.path.join(base_dir, 'nerf_mask')
model_dir = os.path.join(base_dir, 'model')
use_nerf = False
val = [8,9, 23, 24, 28,29, 31, 32, 37, 42, 43, 46, 48, 53,54]



masks = []
slot_n = [0,1,5]
for j in slot_n:
    slot = []
    for i in range(15):
        image_file = os.path.join(image_dir, 'val_000200_r_{:d}_slot{:d}.png'.format(i,j))
        slot.append(imageio.imread(image_file))
    slot = np.stack(slot, 0)
    masks.append(slot)
masks = np.stack(masks,0)
masks = torch.from_numpy(masks)
masks = masks / 255.

images, poses, depth_maps, render_poses, hwf, i_split = load_data(datadir)
images = images[...,:3]
# val = [ 4, 5, 6, 23, 24, 30, 33, 40, 41, 42, 43, 45, 46, 48, 58] #for  clevrtex

print(val)

val_images = images[val]
val_images = torch.from_numpy(val_images)
val_images = val_images.permute([0,3,1,2])


model_slot = 18
model = MyNet(3, num_slot = model_slot)
model.load_state_dict(torch.load(model_dir))

scores = model(val_images)
scores = scores.permute([0,2,3,1])
scores = scores.reshape([-1, model_slot])

mu = []
for j in range(len(slot_n)):
    slot = masks[j]
    slot = slot.reshape(-1)
    index = (slot>0.5)
    in_mask_score = scores[index]
    mu.append(in_mask_score.mean(dim=0))


background_mask = torch.ones_like(masks[0])
for j in range(len(slot_n)):
    out_mask = 1.-masks[j]
    background_mask = out_mask * background_mask



background_mask = background_mask.reshape(-1)
index = (background_mask>0.5)
out_mask_score = scores[index]
mu.append(out_mask_score.mean(dim=0))


mu = torch.stack(mu)


expand_out_mask_score = out_mask_score[:, None, :]
expand_out_mask_score = expand_out_mask_score.expand([-1, mu.shape[0], -1])


for _ in range(10):
    expand_mu = mu[None, ...]
    k = torch.exp(-torch.sum((expand_out_mask_score - expand_mu)**2, dim=-1))

    z = k/(1e-6 + k.sum(dim=1, keepdim=True))
    update = torch.matmul(out_mask_score.T, z /(1e-6 + z.sum(dim = 0 , keepdim=True)))
    mu[-1] = update[:, -1]


final_score = torch.nn.functional.softmax(z, dim=1)

slot1_mask = masks[0].clone()
slot1_mask_shape = slot1_mask.shape
slot1_mask = slot1_mask.reshape(-1)
slot1_mask[index] = final_score[:,0]
# max_score,_ = torch.max(final_score, dim=1)
# max_index = (final_score[:,1] + 1e-7>= max_score)
# max_index = max_index.float()
# new_score = final_score[:,1] * max_index




if use_nerf:
    nerf_masks = []
    for i in range(15):
        mask_file = os.path.join(nerf_dir, 'acc1_slot{:d}.png'.format(i))
        nerf_masks.append(imageio.imread(mask_file))
    nerf_masks = np.stack(nerf_masks, 0)
    nerf_masks = nerf_masks / 255.


slot1_mask = slot1_mask.reshape(slot1_mask_shape)
slot1_mask = slot1_mask.detach().cpu().numpy()
slot1_mask = 0.5 * slot1_mask + 0.5 * nerf_masks
slot1_mask = slot1_mask > 0.5
masks = masks.detach().cpu().numpy()
for i in range(15):
    image_file = os.path.join(base_dir, 'refinement', 'mask{:d}_slot1.png'.format(i))
    image_file1 = os.path.join(base_dir, 'refinement', 'outcome{:d}.png'.format(i))
    imageio.imwrite(image_file , to8b(slot1_mask[i]))
    imageio.imwrite(image_file1 ,to8b(images[val[i]] * slot1_mask[i][..., None]))


