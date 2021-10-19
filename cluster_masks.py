import torch 
import imageio
import scipy
from scipy import ndimage

from KM_testing import *
from PIL import Image
import cv2




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



class SobelOperator(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/4
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_x.weight.requires_grad = False

        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_y.weight.requires_grad = False

    def forward(self, x):

        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b*c, 1, h, w)

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


os.environ["CUDA_VISIBLE_DEVICES"]='8'


num_slots = 6
masks = []
for i in range(15):
    image_dir = 'results/testing_7/mask_refine/seg_input{:d}.png'.format(i)
    image = imageio.imread(image_dir)
    image_mask = []
    for j in range(num_slots):
        mask_dir= 'results/testing_7/segmentation/val_000950_r_{:d}_slot{:d}.jpg'.format(i,j)

        im = imageio.imread(mask_dir)

        # 
        # dilation = dilation.astype(np.uint8)

        image_mask.append(erode)


        imageio.imwrite('results/testing_7/remove_noise/r_{:d}_slot{:d}.jpg'.format(i,j), erode[...,None] * image)

    masks.append(image_mask)
    # image_dir = 'results/testing_11/mask_refine/input{:d}.png'.format(i)
    # image = imageio.imread(image_dir)

input_size = 400



masks = np.array(masks)
masks = np.transpose(masks, [0,2,3,1])
# masks = masks.astype(np.float32)


device = torch.device("cuda:0" )

datadir = 'data/nerf_synthetic/clevrtex'
images, poses, depth_maps, render_poses, hwf, i_split = load_data(
            datadir, True, 1, size = input_size)




model = MyNet( 3, num_slot=10 )
model.load_state_dict(torch.load('./results/testing_3/model'))
model.to(device)

N_imgs=70
images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)


# val = [0, 2, 3, 5, 22, 23, 24, 25, 39, 40, 41, 42, 43, 45, 46, 48] #for simple clevr

val = [0, 3, 4, 23, 24, 40, 41, 42, 43, 45, 46, 48, 58, 59, 60] #for  clevrtex




val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]
val_images, val_depths, val_poses = val_images.to(device), val_depths.to(device), val_poses.to(device)
masks = torch.from_numpy(masks)
masks = masks.to(device)


val_images = val_images.permute([0,3,1,2])
output = model( val_images )
output = output.permute([0,2,3,1]).reshape( [-1, 10] )

cur_m1 = masks[..., 0]
cur_m1 = cur_m1.reshape(-1)
class1 = output[cur_m1]
print(class1.mean(dim=0))
# print(torch.var(class1, dim=0))

cur_m2 = masks[..., 1]
cur_m2 = cur_m2.reshape(-1)
class2 = output[cur_m2]
print(class2.mean(dim=0))
# print(torch.var(class2, dim=0))



cur_m3 = masks[..., 2]
cur_m3 = cur_m3.reshape(-1)
class3 = output[cur_m3]
print(class3.mean(dim=0))
# print(torch.var(class3, dim=0))


cur_m4 = masks[..., 3]
cur_m4 = cur_m4.reshape(-1)
class4 = output[cur_m4]
print(class4.mean(dim=0))
# print(torch.var(class4, dim=0))


a = [class1, class2, class3, class4]


for i in range(5):
    t = []
    for j in range(4):
        size = a[j].shape[0]
        index = torch.randint(size, (size//10,))
        c_class = a[j]
        c_class =c_class[index]
        t.append(c_class.mean(dim=0))
    print(torch.norm(t[0]-t[3]))
    print(torch.norm(t[1]-t[3]))
    print(torch.norm(t[2]-t[3]))
    print('sample')
exit()
# f_extractor = Encoder_VGG(hwf, device=device)
# f_extractor.to(device)
# f = f_extractor(val_images, val_depths, val_poses)
# B, H, W, C = f.shape


# f = f.reshape([-1, C])

# f = f[...,:C-3]

# slots = []
# slots_size = []
# for i in range(num_slots):
#     cur_m = masks[..., i]
#     cur_m =cur_m.reshape(-1)
#     cur_s = f[cur_m]

#     slots.append(cur_s)
#     slots_size.append(cur_s.shape[0])

# # slots = torch.stack(slots)

# num_sample = min(slots_size) // 4
# print(num_sample)

# sample_pts = []
# for i in range(num_slots):
#     index = torch.randint(slots_size[i], (num_sample,))
#     pts = slots[i][index]
#     sample_pts.append(pts)

# sample_pts = torch.cat(sample_pts)
# print(sample_pts.shape)


# num_cluster = 2
# cluster = torch.randn(num_cluster, C-3)
# cluster = cluster.to(device)

# for i in range(50):
#     z=[]

#     for j in range(num_cluster):
#         z.append(torch.sum((sample_pts - cluster[j])**2, dim=-1))    # N


#     z = torch.stack(z)



#     score = z.T
    
#     ignore, index = torch.min(score,1)
    
#     for j in range(num_slots):
#         slots[j] = torch.mean(sample_pts[index==j], dim=0)

# print(score.shape)

# final_cluster = []
# for i in range(0, num_sample* num_slots, num_sample):
#     slot_pts_score = score[i : i + num_sample ]
#     ignore, index = torch.min(slot_pts_score,1)
#     count = []
#     for j in range(num_cluster):
#         print(torch.sum(index==j))
#         count.append(torch.sum(index==j))
#     max_value = max(count)
#     final_cluster.append(count.index(max_value))

# print(final_cluster)

