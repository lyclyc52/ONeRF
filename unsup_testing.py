import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='5,6'
from model_clustering import *
from load_blender import *
from run_nerf_helpers import *
from run_nerf import *
import torch



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



class MyNet(nn.Module):
    def __init__(self,input_dim, slot_d=64, num_slot=3):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, slot_d, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(slot_d)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.nConv = 2
        for i in range(self.nConv-1):
            self.conv2.append( nn.Conv2d(slot_d, slot_d, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(slot_d) )
        self.conv3 = nn.Conv2d(slot_d, num_slot, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(num_slot)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

depth_file = 'data/nerf_synthetic/clevr_100_2objects/all_depths.npy'
depth_maps = np.load(depth_file)
depth_maps = depth_maps[..., None]
depth_maps = tf.compat.v1.image.resize_area(depth_maps, [256, 256]).numpy()
depth_maps = tf.squeeze(depth_maps, axis=-1).numpy()

weights_dir = './results/testing_3/weights'
img_dir = './results/testing_7/imgs'
basedir = './results/testing_9/imgs'


os.makedirs(weights_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)


parser = config_parser()
args = parser.parse_args()

N_train = depth_maps.shape[0]


num_slot = 20

# images, poses, render_poses, hwf, i_split = load_blender_data(
#             args.datadir, args.half_res, args.testskip, size = 256)
# N_imgs=100
# images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
# images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)
loss_fn = torch.nn.CrossEntropyLoss()


loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

device = torch.device("cuda:0" )
# f_extr = Encoder_VGG(hwf, device=device)
# f_extr.to(device)

model = MyNet( 3, num_slot=num_slot )
model.to(device)

B,H,W = 4, 128, 128# 4, images.shape[1], images.shape[2]

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
label_colours = np.random.randint(255,size=(num_slot,3))

HPy_target = torch.zeros(B, H-1, W, num_slot)
HPz_target = torch.zeros(B, H, W-1, num_slot)
HPy_target = HPy_target.to(device)
HPz_target = HPz_target.to(device)


imgs = []
for i in range(4):
    fname = os.path.join(basedir, 'masked_{:06d}_slot{:01d}.jpg'.format(i,1))
    imgs.append(imageio.imread(fname))


imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)

imgs= imgs[...,:3]
imgs = torch.from_numpy(imgs)
imgs = imgs.to(device)

for i in range(1000):
    optimizer.zero_grad()
    

    # val = np.random.randint(0, 20, )
    # val = [0, 3, 25, 39]
    # val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]
    # val_images, val_depths, val_poses = val_images.to(device), val_depths.to(device), val_poses.to(device)
    # f = f_extr(val_images, val_depths, val_poses)
    # f = f.permute([0,3,1,2])

    # f = val_images.permute([0,3,1,2])
    f = imgs.permute([0,3,1,2])
    output = model( f )

    output = output.permute([0,2,3,1]).reshape( [-1, num_slot] )






    outputHP = output.reshape( [B,H,W, num_slot] )
    HPy = outputHP[:,1:, :, :] - outputHP[:,0:-1, :, :]
    HPz = outputHP[:,:, 1:, :] - outputHP[:,:, 0:-1, :]
    lhpy = loss_hpy(HPy,HPy_target)
    lhpz = loss_hpz(HPz,HPz_target)


    ignore, target = torch.max( output, 1 )


    loss = loss_fn(output, target) + (lhpy + lhpz)

    print(i)
    print(loss)

    
    loss.backward()
    optimizer.step()
    if i % 50 ==0:
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( [B,H,W,3] ).astype( np.uint8 )
        for b in range(B):
            imageio.imwrite(os.path.join(img_dir, 'val_{:06d}__slot{:01d}.jpg'.format(i,b)), im_target_rgb[b])



print(np.unique(im_target).shape)