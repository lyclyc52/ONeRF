import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='5,6'
from model_clustering import *
from load_blender import *
from run_nerf_helpers import *
from run_nerf import *
import torch

import time
seed = int(time.time())
torch.manual_seed(seed)

def load_data(basedir, half_res=False, testskip=1, size=-1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))


        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    for i in i_split[0]:
        dname = os.path.join(basedir, 'depth', 'depth_{}.png'.format(i) )
        all_depths.append(imageio.imread(fname))

    all_depths = (np.array(all_depths) / 255. * 30).astype(np.float32)

    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]

    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # if basedir=='./data/nerf_synthetic/clevr_100_2objects':
    focal = 875.
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    

    if size > 0:
        imgs = tf.compat.v1.image.resize_area(imgs, [size, size]).numpy()
        all_depths = tf.compat.v1.image.resize_area(all_depths, [size, size]).numpy()
        H = H * size//H
        W = W * size//W
        focal = focal * size/ 400.

    all_depths = all_depths[..., 0]
        
    return imgs, poses, all_depths, render_poses, [H, W, focal], i_split



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


# class EMAU(nn.Module):
#     '''The Expectation-Maximization Attention Unit (EMAU).
#     Arguments:
#         c (int): The input and output channel number.
#         k (int): The number of the bases.
#         stage_num (int): The iteration number for EM.
#     '''
#     def __init__(self, c, k, stage_num=6):
#         super(EMAU, self).__init__()
#         self.stage_num = stage_num

#         slots = torch.randn(C, 3)
#         slots = self._l2norm(slots, dim=1)
#         self.register_buffer('slots', slots)

#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             norm_layer(c))        
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
 

#     def forward(self, x):
#         idn = x
#         # The first 1x1 conv
#         x = self.conv1(x)

#         # The EM Attention
#         b, c, h, w = x.size()
#         x = x.permute([0,2,3,1])

#         x = f.reshape([-1,c])

#         slots = slots.to(device)
#         with torch.no_grad():
#             for i in range(10):
#                 z = torch.matmul(f, slots)      # NxK
#                 z = F.softmax(z, dim=-1)                 # NxK
#                 z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
#                 slots = torch.matmul(f.T, z_)       # CxK
#                 slots = _l2norm(slots, dim=0)


#         # attn_logits = torch.matmul(f, slots)
#         # attn = attn_logits.softmax(dim=-1)
#         # print(attn.shape)
#         # attn = attn.reshape([B,H,W,3])
#         # attn = attn.permute([0,3,1,2])
                
#         # !!! The moving averaging operation is writtern in train.py, which is significant.


#         z_t = z.permute(1, 0)            # k * n
#         x = mu.matmul(z_t)                  # c * n
#         x = x.permute([1,0])
#         x = x.reshape([b, h, w, c ])
#         x = x.permute([0,3,1,2])              # b * c * h * w
#         x = F.relu(x, inplace=True)

#         # The second 1x1 conv
#         x = self.conv2(x)
#         x = x + idn
#         x = F.relu(x, inplace=True)

#         return x, slots




def main():
    weights_dir = './results/testing_3/weights'
    img_dir = './results/testing_11/imgs'

    input_size = 128

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)



    datadir = 'data/nerf_synthetic/clevr_depth'


    images, poses, depth_maps, render_poses, hwf, i_split = load_data(
                datadir, True, 1, size = input_size)


    N_imgs=49
    images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
    images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)


    device = torch.device("cuda:0" )
    f_extractor = Encoder_VGG(hwf, device=device)
    f_extractor.to(device)

    C = 259
    num_slots = 2
    slots = torch.randn(num_slots, C-3)
    position = torch.randn(num_slots, 3)
    slots = slots.to(device)
    position = position.to(device)

    # image = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48]
    for iter in range(1):
        # val = [0, 3, 25, 39]
        # val = [t for t in range(iter*2,iter*2+4)]
        val = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48]
        print(val)
        val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]
        val_images, val_depths, val_poses = val_images.to(device), val_depths.to(device), val_poses.to(device)
        print(val_images.shape)
        print(val_depths.shape)
        f = f_extractor(val_images, val_depths, val_poses)
        B, H, W, C = f.shape


        f = f.reshape([-1, C])

        f_p = f[...,C-3:]
        f = f[...,:C-3]


        # normal_f = nn.LayerNorm(C-3)
        # normal_f.to(device)
        # f = normal_f(f)



        w = 3.
        # slots = slots.T
        # position = position.T

        # with torch.no_grad():
        #     for i in range(20):
        #         z = torch.matmul(f, slots)      # NxK
        #         z_p = torch.matmul(f_p, position) 
        #         z = z + w * z_p
        #         z = F.softmax(z, dim=-1)                 # NxK
        #         z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
        #         slots = torch.matmul(f.T, z_)       # CxK
        #         slots = _l2norm(slots, dim=0) 
        #         position = torch.matmul(f_p.T, z_)
        #         position = _l2norm(position, dim=0) 


        # attn_logits = torch.matmul(f, slots)
        # attn = attn_logits.softmax(dim=-1)
        # print(attn.shape)
        # attn = attn.reshape([B,H,W,num_slots])
        # attn = attn.permute([0,3,1,2])





        for i in range(50):
            z=[]
            z_p=[]
            for j in range(num_slots):
                z.append(torch.sum((f - slots[j])**2, dim=-1))    # N
                if j!= num_slots-1:
                    z_p.append(torch.sum((f_p - position[j])**2, dim=-1)) # N
                # z_p.append(torch.sum((f_p - position[j])**2, dim=-1))


            z = torch.stack(z)
            z_p = torch.stack(z_p)
            z = z.T
            z_p = z_p.T 



            bg_p = torch.zeros([z.shape[0],1])
            bg_p = bg_p.to(device)
            z_p = torch.cat([z_p,bg_p],dim=1)

            score = z+w*z_p
            ignore, index = torch.min(score,1)
            
            for j in range(num_slots):
                slots[j] = torch.mean(f[index==j], dim=0)
                if j!= num_slots-1:
                    position[j]  = torch.mean(f_p[index==j], dim=0)

                # position[j] = torch.mean(f_p[index==j], dim=0)


        attn_logits = score
        attn = attn_logits.softmax(dim=-1)
        
        attn = attn.reshape([B,H,W,num_slots])
        attn = attn.permute([0,3,1,2])

        attn = attn.cpu().numpy()
        val_images = val_images.cpu().numpy()
        # print(iter)
        for b in range(B):
            for s in range(num_slots):
                imageio.imwrite(os.path.join(img_dir, 'val_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s]))
                imageio.imwrite(os.path.join(img_dir, 'masked_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s][...,None]*val_images[b]))
