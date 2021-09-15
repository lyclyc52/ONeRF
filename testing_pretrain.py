from run_nerf import *
import os 

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

for i in i_train:
    target = images[i]
    pose = poses[i, :3, :4]

    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                    **render_kwargs_test)
    disp = 1/disp
    
    if i == 0:
        all_depths = disp[None, ...]
    else:
        all_depths = tf.concat([all_depths, disp[None, ...]], axis=0)     
    np.save(file_name, all_depths)   
    print('Finish '+str(i))