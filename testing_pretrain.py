from run_nerf import *
import os 
from models import *

os.environ['CUDA_VISIBLE_DEVICES']='2,3'
parser = config_parser()
args = parser.parse_args()

images, poses, render_poses, hwf, i_split = load_blender_data(
    args.datadir, args.half_res, args.testskip)


print('Loaded blender', images.shape,
        render_poses.shape, hwf, args.datadir)
i_train, i_val, i_test = i_split

near = 4.
far = 16.

if args.white_bkgd:
    images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
else:
    images = images[..., :3]

H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]



basedir = args.basedir
expname = args.expname
os.makedirs(os.path.join(basedir, expname), exist_ok=True)
f = os.path.join(basedir, expname, 'args.txt')
with open(f, 'w') as file:
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))
if args.config is not None:
    f = os.path.join(basedir, expname, 'config.txt')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())

# Create nerf model
render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
    args)

bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}

render_kwargs_test.update(bds_dict)



testimgdir = 'tesing_rgb'



file_name = 'all_depths.npy'


N_samples=128

target = images[0:2]
pose = poses[0:2, :3, :4]

pts, z_vals, rays_d = sampling_points(hwf, pose, is_selection=False, near=near, far=far, N_samples=N_samples)
print(pts.shape)
print(z_vals.shape)
print(rays_d.shape)

pts_shape = pts.shape
input_pts = tf.reshape(pts , [-1,N_samples,3])
input_rays_d = tf.reshape(rays_d, [-1, 3])

chunk=1024*64
for i in range(0,input_pts.shape[0],chunk):
    raw_c = render_kwargs_test['network_query_fn'](input_pts[i:i+chunk], input_rays_d[i:i+chunk], render_kwargs_test['network_fn'])
    if i==0:
        raw=raw_c
    else:
        raw=tf.concat([raw,raw_c],axis=0)
masked_raw = raw[None,...]


rgb,_ = generate_rgb(raw, masked_raw, z_vals, rays_d, pts_shape, 1)

rgb = tf.reshape(rgb, [2, H, W ,3])
imageio.imwrite(os.path.join('./testing_output/GT_{:03d}.jpg'.format(0)), to8b(rgb[0]))
imageio.imwrite(os.path.join('./testing_output/GT_{:03d}.jpg'.format(1)), to8b(rgb[1]))