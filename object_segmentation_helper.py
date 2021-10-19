import torch 
import imageio
import scipy
from scipy import ndimage


from model_clustering import *
from load_blender import *
from run_nerf_helpers import *
from run_nerf import *
from PIL import Image
import cv2



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


def KM_clustering(f, f_p, w, device):


    num_slots = 2
    slots = torch.randn(num_slots, f.shape[-1])
    position = torch.randn(num_slots, f_p.shape[-1])
    slots = slots.to(device)
    position = position.to(device)

    for i in range(100):
        z=[]
        z_p=[]
        for j in range(num_slots):
            z.append(torch.sum((f - slots[j])**2, dim=-1))    # N
            if j!= num_slots-1:
                z_p.append(torch.sum((f_p - position[j])**2, dim=-1)) # N

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

 

    return score




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


def train(imgs, base_dir):

    device = torch.device("cuda:1" )

    B,H,W,_ = imgs.shape

    num_slot = 10
    model = MyNet( 3, num_slot=num_slot )
    model.to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    HPy_target = torch.zeros(B, H-1, W, num_slot)
    HPz_target = torch.zeros(B, H, W-1, num_slot)
    HPy_target = HPy_target.to(device)
    HPz_target = HPz_target.to(device)


    img_dir = os.path.join(base_dir, 'segmentation')
    model_path = os.path.join(base_dir, 'model')

    for i in range(1000):
        optimizer.zero_grad()
        
        f = imgs.permute([0,3,1,2])
        f = f.to(device)
        output = model( f )

        output = output.permute([0,2,3,1]).reshape( [-1, num_slot] )


        outputHP = output.reshape( [B,H,W, num_slot] )
        HPy = outputHP[:,1:, :, :] - outputHP[:,0:-1, :, :]
        HPz = outputHP[:,:, 1:, :] - outputHP[:,:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)


        ignore, target = torch.max( output, 1 )


        loss = loss_fn(output, target) + (lhpy + lhpz)


        
        loss.backward()
        optimizer.step()
        if i % 50 ==0 and i>0:
            print(i)
            print(loss)

            im_target = target.data.cpu().numpy()
            im_target = im_target.reshape([B,H,W])


            cluster = np.unique(im_target)

            
            print(cluster.shape[0])

            for c in range(cluster.shape[0]):
                for b in range(B):
                    mask = (im_target[b] == cluster[c])
                    mask = mask.astype(int)
                    mask = mask * 255
                    mask = mask[..., None]
                    mask = mask.astype(np.uint8)
                    imageio.imwrite(os.path.join(img_dir, 'val_{:06d}_r_{:1d}_slot{:01d}.jpg'.format(i,b,c)), mask)


            torch.save(model.state_dict(), model_path)

    return model, im_target