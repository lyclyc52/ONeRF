import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="8"
from run_nerf_helpers import *
from run_nerf import *

N_train = 20

parser = config_parser()
args = parser.parse_args()

images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
near = 4.
far = 16.
render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

print('render')

rgbs, disps = render_path(
                poses[0:N_train], hwf, args.chunk, render_kwargs_test)


depth_maps = 1 / disps
print(depth_maps.shape)
np.save('depth.npy', depth_maps)
exit()