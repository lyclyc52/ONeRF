import torch
import imageio
from object_segmentation_helper import *



def _l2norm(inp, dim):
    '''Normlize the inp tensor with l2-norm.
    Returns a tensor where each sub-tensor of input along the given dim is 
    normalized such that the 2-norm of the sub-tensor is equal to 1.
    Arguments:
        inp (tensor): The input tensor.
        dim (int): The dimension to slice over to get the ssub-tensors.
    Returns:
        (tensor) The normalized tensor.
    '''
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

base_dir =  'results/testing_8'
datadir = 'data/nerf_synthetic/clevr_bg6'
image_dir = os.path.join(base_dir, 'segmentation')

masks = []
for j in range(2):
    slot = []
    for i in range(15):
        image_file = os.path.join(image_dir, 'val_000950_r_{:d}_slot{:d}.jpg'.format(i,j))
        slot.append(imageio.imread(image_file))
    slot = np.stack(slot, 0)
    masks.append(slot)
masks = np.stack(masks,0)
masks = torch.from_numpy(masks)
masks = masks / 255.

images, poses, depth_maps, render_poses, hwf, i_split = load_data(datadir)
images = images[...,:3]
val = [ 4, 5, 6, 23, 24, 30, 33, 40, 41, 42, 43, 45, 46, 48, 58] #for  clevrtex
print(val)

val_images = images[val]
val_images = torch.from_numpy(val_images)
val_images = val_images.permute([0,3,1,2])

model = MyNet(3, num_slot = 12)
model.load_state_dict(torch.load('./results/testing_8/model'))

scores = model(val_images)
scores = scores.permute([0,2,3,1])
scores = scores.reshape([-1, 12])

mu = []
for j in range(2):
    slot = masks[j]
    slot = slot.reshape(-1)
    index = (slot>0.5)
    in_mask_score = scores[index]
    mu.append(in_mask_score.mean(dim=0))


background_mask = torch.ones_like(masks[0])
for j in range(2):
    out_mask = 1.-masks[j]
    background_mask = out_mask * background_mask


background_mask = background_mask.reshape(-1)
index = (background_mask>0.5)
out_mask_score = scores[index]
mu.append(out_mask_score.mean(dim=0))


mu = torch.stack(mu)


expand_out_mask_score = out_mask_score[:, None, :]
expand_out_mask_score = expand_out_mask_score.expand([-1, 3, -1])


for _ in range(10):
    expand_mu = mu[None, ...]
    k = torch.exp(-torch.sum((expand_out_mask_score - expand_mu)**2, dim=-1))

    z = k/(1e-6 + k.sum(dim=1, keepdim=True))
    update = torch.matmul(out_mask_score.T, z /(1e-6 + z.sum(dim = 0 , keepdim=True)))
    mu[2] = update[:, 2]


final_score = torch.nn.functional.softmax(z, dim=1)




slot1_mask = masks[1].clone()
slot1_mask_shape = slot1_mask.shape
slot1_mask = slot1_mask.reshape(-1)

max_score,_ = torch.max(final_score, dim=1)
max_index = (final_score[:,1] + 1e-7>= max_score)
max_index = max_index.float()
new_score = final_score[:,1] * max_index

slot1_mask[index] = new_score


slot1_mask = slot1_mask.reshape(slot1_mask_shape)
slot1_mask = slot1_mask.detach().cpu().numpy()
slot1_mask = slot1_mask > 0.35
masks = masks.detach().cpu().numpy()
for i in range(15):
    image_file = os.path.join(base_dir, 'refinement', 'slot{:d}.png'.format(i))
    image_file1 = os.path.join(base_dir, 'refinement', 'origin{:d}.png'.format(i))
    imageio.imwrite(image_file , to8b(slot1_mask[i]))
    imageio.imwrite(image_file1 ,to8b(masks[1][i]))

