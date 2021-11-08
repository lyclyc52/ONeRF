from run_nerf import *
def render_rays_acc(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=True,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2acc(raw_m, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)


        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        

        
        alpha = []
        raw_density = tf.zeros_like(raw_m[0][..., 3])
        num_r = len(raw_m)
        for i in range(num_r):
            raw_density += raw_m[i][..., 3]
            alpha.append(raw2alpha(raw_m[i][..., 3] + noise, dists))

        alpha_total = raw2alpha(raw_density + noise, dists)  # [N_rays, N_samples]
 
        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]

        weights_total = alpha_total * \
            tf.math.cumprod(1.-alpha_total + 1e-10, axis=-1, exclusive=True)

        weights = []
        for i in range(num_r):
            weights.append(alpha[i] * \
                tf.math.cumprod(1.-alpha_total + 1e-10, axis=-1, exclusive=True))
    


        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_maps = []
        for i in range(num_r):
            acc_maps.append(tf.reduce_sum(weights[i], -1)) 

        return acc_maps, weights_total

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.

    num_m = len(network_fn)
    raw_m = []
    for i in range(num_m):
        raw_m.append(network_query_fn(pts, viewdirs, network_fn[i]))
        sigma = tf.nn.relu(raw_m[i][..., 3:])
        raw_m[i] = tf.concat([raw_m[i][..., :3], sigma], -1)

    # raw0 =   # [N_rays, N_samples, 4]
    # raw1 = network_query_fn(pts, viewdirs, network_fn[1])



    acc_maps, weights = raw2acc(
        raw_m, z_vals, rays_d)


    if N_importance > 0:
        raw_m_c = raw_m
        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine

        raw_m = []
        for i in range(num_m):
            raw_m.append(network_query_fn(pts, viewdirs, run_fn[i]))
            sigma = tf.nn.relu(raw_m[i][..., 3:])
            raw_m[i] = tf.concat([raw_m[i][..., :3], sigma], -1)


        acc_maps, weights = raw2acc(
            raw_m, z_vals, rays_d)

    ret = {'acc_maps': acc_maps}


    return ret



def batchify_rays_acc(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_acc(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            num_m = len(ret[k])
            if k not in all_ret:
                all_ret[k] = []
                for t in range(num_m):
                    all_ret[k].append([])
            for t in range(num_m):
                all_ret[k][t].append(ret[k][t])
    
    for k in all_ret:
        all_item = []
        num_m =len(all_ret[k])
        for n in range(num_m):
            all_item.append(tf.concat(all_ret[k][n], 0))
            print(all_item[n].shape)

        all_ret = {k: all_item}
    return all_ret


def render_acc(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays_acc(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k][0].shape[1:])
        all_item = []
        num_m = len(all_ret[k])
        for i in range(num_m):
            all_item.append(tf.reshape(all_ret[k][i], k_sh))
        all_ret[k] = all_item

    # k_extract = ["acc_map", "acc_map1"]
    ret_list = all_ret["acc_maps"]

    return ret_list


def refine():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)

    i_train, i_val, i_test = i_split

    near = 2.
    far = 25.

    if args.white_bkgd:
        images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
    else:
        images = images[..., :3]


    if args.render_test:
        render_poses = np.array(poses[i_test])

    args.expname = "clevrtex_room3_test_sofa"
    render_kwargs_train1, render_kwargs_test1, start, grad_vars1, models1 = create_nerf(
        args)

    args.expname = "clevrtex_room3_test_desk"
    render_kwargs_train2, render_kwargs_test2, start, grad_vars, models2 = create_nerf(
        args)


    # args.expname = "clevrtex_room_test_desk"
    # render_kwargs_train3, render_kwargs_test3, start, grad_vars, models3 = create_nerf(
    #     args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train1.update(bds_dict)
    render_kwargs_test1.update(bds_dict)
    render_kwargs_train2.update(bds_dict)
    render_kwargs_test2.update(bds_dict)



    network_fn = []
    network_fn.append(render_kwargs_test1["network_fn"])
    network_fn.append(render_kwargs_test2["network_fn"])
    # network_fn.append(render_kwargs_test3["network_fn"])
    network_fine = []
    network_fine.append(render_kwargs_test1["network_fine"])
    network_fine.append(render_kwargs_test2["network_fine"])
    # network_fine.append(render_kwargs_test3["network_fine"])



    render_kwargs_test1["network_fn"] = network_fn
    render_kwargs_test1["network_fine"] = network_fine

    # i_test = [0,2,7,8,9,15,16,22,23,27,28,31,32,35,37,38,40,43,46,47,48,53,54,55,57]
    i_test = [2,3,7,8,9,12, 15,16, 22,23,24, 27,28,31, 32,35, 37, 38, 40,43, 46,47, 48,51,52]

    N_rand = args.N_rand



    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]




    nerf_mask_dir = '/data/yliugu/ONeRF/results/testing_clevrtex_room3/nerf_mask'
    for i in range(len(i_test)):
        # img_i = np.random.choice(i_test)
        img_i = i_test[i]
        target = images[img_i]
        pose = poses[img_i, :3, :4]

        acc = render_acc(H, W, focal, chunk=args.chunk, c2w=pose,
                                        **render_kwargs_test1)


        imageio.imwrite(nerf_mask_dir+'/acc_slot0_r_{:d}.png'.format(i), to8b(acc[0]))
        imageio.imwrite(nerf_mask_dir+'/acc_slot1_r_{:d}.png'.format(i), to8b(acc[1]))
        # imageio.imwrite(nerf_mask_dir+'/acc_slot2_r_{:d}.png'.format(i), to8b(acc[0]))
    # imageio.imwrite('rgb2.png'.format(i), to8b(rgb2))
    # imageio.imwrite('acc2.png'.format(i), to8b(acc2))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='6,7'
    refine()

