import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import json
import imageio
import configargparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

selects = [0, 2, 3, 4, 5, 22, 42]
# configs
parser = configargparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=128,
                        help='image size')
parser.add_argument('--gpu', type=int, default=7)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

def load_blender_data(basedir, size=-1, device='cuda:0'):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        for frame in meta['frames'][::1]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        all_imgs.append(imgs)
    
    imgs = np.concatenate(all_imgs, 0)

    imgs = torch.from_numpy(imgs).to(device).permute([0,3,1,2])[:, :3, ...]
    if size > 0:
        imgs = TF.resize(imgs, size=size)
        
    return imgs

# load img
img_size = args.img_size
N_imgs = 50
img_dir = './data/nerf_synthetic/clevr_100_2objects'
device = 'cuda:0'
images = load_blender_data(img_dir, size=img_size)[:N_imgs]


# vgg process
vgg_features = vgg16(pretrained=True).features
feature_extractor = nn.Sequential()
for i in range(16):
    feature_extractor.add_module(str(i), vgg_features[i])
feature_extractor = feature_extractor.to(device)

os.makedirs('clusters_2d', exist_ok=True)
print('make dir cluster_2d...')

cluster_maps = []
for select in selects:
    x = TF.normalize(images[[select]], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = feature_extractor(x)
    x = F.interpolate(x, scale_factor=4) # todo: check mode
    feature_maps = x.permute([0,2,3,1])

    # feat = feature_maps[select]
    feat = feature_maps
    feat = feat.reshape((-1, feat.shape[-1]))

    feat = feat.detach().cpu().numpy()


    # img = images[2].permute([1, 2, 0]).detach().cpu().numpy()
    # feat = StandardScaler().fit_transform(feat)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(feat)

    cluster_map = kmeans.labels_
    cluster_map = cluster_map.reshape((img_size, img_size, 1))

    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    # ax1.imshow(images[select].permute([1, 2, 0]).detach().cpu().numpy())
    # ax2.imshow(cluster_map)
    if len(cluster_map[cluster_map==0]) < len(cluster_map[cluster_map==1]):
        cluster_map = 1 - cluster_map
    cluster_maps.append(cluster_map)
    plt.imsave(os.path.join('clusters_2d', 'mask_{:2d}.jpg'.format(select)), 
            cluster_map.squeeze())

cluster_maps = np.array(cluster_maps)
np.save(os.path.join('clusters_2d', 'cluster_maps'),  cluster_maps)

print('finish')