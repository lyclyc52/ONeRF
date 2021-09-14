import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from run_nerf_helpers import get_embedder


def get_position(ray_o, ray_d, depth):
    '''
    input: ray origin, ray direction, depth

    output: 3d position of each points
    '''

    depth = depth[...,tf.newaxis]

    position = depth*ray_d+ray_o
    return position

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    dirs = dirs[tf.newaxis,...]
    # N = c2w.shape[0]
    # dirs = tf.tile(dirs, [N, 1, 1, 1])

    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], -1)
    rays_o = c2w[:,:3, -1]
    rays_o = rays_o[:,np.newaxis, np.newaxis, :]

    rays_o = tf.broadcast_to(rays_o, tf.shape(rays_d))

    return rays_o, rays_d


def sampling_points(cam_param, c2w, N_samples=64, near=4., far=14., is_selection=False, N_selection=512):
    H, W, focal = cam_param
    batch_size = c2w.shape[0]

    rays_o, rays_d = get_rays(H, W, focal, c2w) 

    t_vals = tf.linspace(0., 1., N_samples)

    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals[tf.newaxis, tf.newaxis, tf.newaxis, :]
    z_vals = tf.broadcast_to(z_vals, [batch_size, H, W, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    if is_selection:
        coords = tf.stack(tf.meshgrid(
                tf.range(H), tf.range(W), indexing='ij'), -1)
        coords = tf.reshape(coords, [-1, 2])
        select_inds = np.random.choice(
            coords.shape[0], size=[N_selection], replace=False)
        # pts = tf.reshape(pts,[batch_size, -1, N_samples, 3])
        # pts = tf.reshape(pts,[batch_size, -1, N_samples])


        select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
        select_inds = tf.tile(select_inds[tf.newaxis,...],[batch_size,1,1])
       


        pts = tf.gather_nd(pts, select_inds, batch_dims = 1)
        z_vals = tf.gather_nd(z_vals, select_inds, batch_dims = 1)
        rays_d = tf.gather_nd(rays_d, select_inds, batch_dims = 1)



        return pts, z_vals, rays_d, select_inds

    pts = tf.reshape(pts,[batch_size, -1, N_samples, 3])
    z_vals = tf.reshape(z_vals,[batch_size, -1, N_samples])
    rays_d = tf.reshape(rays_d,[batch_size, -1, 3])
    return pts, z_vals, rays_d


def preprocess_pts(points, num_slots):
    points_bg = points
    points_fg = tf.tile(points[tf.newaxis,...],[num_slots-1, 1, 1, 1, 1])

    points_bg = tf.reshape(points_bg, [-1,3])
    points_fg = tf.reshape(points_fg, [num_slots-1, -1, 3])

    return points_bg, points_fg





class Encoder(layers.Layer):
    def __init__(self, cam_param, input_c=3, layers_c=64, position_emb=True, position_emb_dim=16):
        super().__init__()
        self.input_c=input_c
        self.layers_c=layers_c
        self.position_emb=position_emb
        self.position_emb_dim=position_emb_dim

        self.H, self.W, self.focal=cam_param
        if self.position_emb:
            self.input_c = self.input_c + position_emb_dim

        self.encoder_cnn = tf.keras.Sequential([
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu")
        ], name="encoder_cnn")
                                        
    
    # def set_input(self,:
    #     self.d=d
    #     self.c2w=c2w

    def call(self, x, d, c2w):
        '''
        input:  x: images  Bx3xHxW
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4

        output: feature maps  BxDxHxW
        '''

        if self.position_emb:

            rays_o, rays_d = get_rays(self.H, self.W, self.focal, c2w) 
            position = get_position(rays_o, rays_d, d)
            x = tf.concat([x, position], axis=-1)

        x = self.encoder_cnn(x)  # BxDxHxW
        return x




class SlotAttention(layers.Layer):
    def __init__(self, num_iterations=2, num_slots=3, slot_dim=64, mlp_hidden_dim=64, epsilon=1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mlp_hidden_size = mlp_hidden_dim
        self.epsilon = epsilon

        self.norm_inputs = layers.LayerNormalization()

        self.slots_mu_fg = self.add_weight(
            initializer="glorot_uniform",
            shape=[ 1, self.slot_dim],
            dtype=tf.float32,
            name="slots_mu_fg")
        self.slots_log_sigma_fg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, self.slot_dim],
            dtype=tf.float32,
            name="slots_log_sigma_fg")


        self.slots_mu_bg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, self.slot_dim],
            dtype=tf.float32,
            name="slots_mu_bg")
        self.slots_log_sigma_bg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, self.slot_dim],
            dtype=tf.float32,
            name="slots_log_sigma_bg")
 
        self.k = layers.Dense(self.slot_dim, use_bias=False, name="k")
        self.v = layers.Dense(self.slot_dim, use_bias=False, name="v")
        self.q_fg = tf.keras.Sequential([layers.LayerNormalization(),
                                         layers.Dense(self.slot_dim, use_bias=False)],
                                         name="q_fg")

        self.q_bg = tf.keras.Sequential([layers.LayerNormalization(),
                                         layers.Dense(self.slot_dim, use_bias=False)],
                                         name="q_bg")

        self.gru_fg = layers.GRUCell(self.slot_dim)
        self.gru_bg = layers.GRUCell(self.slot_dim)

        self.mlp_fg = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.slot_dim)
        ], name="mlp_fg")
        self.mlp_bg = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.slot_dim)
        ], name="mlp_bg")


    def call(self, x):

        """
        input:
            feat: visual feature with position information, NxC
        output: slots: KxC, attn: KxN
        """

        # Initialize slots
        x = self.norm_inputs(x)
        slots_fg = self.slots_mu_fg + tf.exp(self.slots_log_sigma_fg) * tf.random.normal(
                [self.num_slots-1, self.slot_dim])

        slots_bg = self.slots_mu_bg + tf.exp(self.slots_log_sigma_bg) * tf.random.normal(
                [1, self.slot_dim])

        k_input = self.k(x)
        v_input = self.v(x)

        for _ in range(self.num_iterations):
            slots_prev_bg = slots_bg
            slots_prev_fg = slots_fg

            q_slots_fg = self.q_fg(slots_fg)
            q_slots_fg *= self.slot_dim ** -0.5
            q_slots_bg = self.q_bg(slots_bg)
            q_slots_bg *= self.slot_dim ** -0.5


            attn_logits_fg = tf.matmul(k_input, tf.transpose(q_slots_fg))
            attn_logits_bg = tf.matmul(k_input, tf.transpose(q_slots_bg))
            attn_logits = tf.concat([attn_logits_bg, attn_logits_fg], axis=-1)
            attn = tf.nn.softmax(attn_logits, axis=-1)

            attn_bg, attn_fg = attn[:, 0:1], attn[:, 1:]  # Nx1, Nx(K-1)


            
            weight_bg = attn_bg + self.epsilon
            weight_bg /= tf.reduce_sum(weight_bg, axis=-2, keepdims=True)
            updates_bg = tf.matmul(tf.transpose(weight_bg), v_input)

            weight_fg = attn_fg + self.epsilon
            weight_fg /= tf.reduce_sum(weight_fg, axis=-2, keepdims=True)
            updates_fg = tf.matmul(tf.transpose(weight_fg), v_input)
            


            slots_bg, _ = self.gru_bg(
                updates_bg,
                [slots_prev_bg]
            )
            slots_fg, _ = self.gru_fg(
                updates_fg,
                [slots_prev_fg]
            )

            slots_bg += self.mlp_bg(slots_bg)
            slots_fg += self.mlp_fg(slots_fg)
    
        slots = tf.concat([slots_bg, slots_fg], axis=0)

        return slots, tf.transpose(attn)


class Decoder(layers.Layer):
    def __init__(self, position_emb=True,
            decoder_initial_size=(25, 25)):
        super().__init__()

        # self.H, self.W, self.focal=cam_param
        self.position_emb = position_emb

        self.decoder_initial_size = decoder_initial_size
        # self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)
        
        if position_emb:
            self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

        self.decoder_cnn = tf.keras.Sequential([
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(2, 2), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                64, 5, strides=(1, 1), padding="SAME", activation="relu"),
            layers.Conv2DTranspose(
                4, 3, strides=(1, 1), padding="SAME", activation=None)
        ], name="decoder_cnn")
    
    def call(self, x):
        
        if self.position_emb:
            x = self.decoder_pos(x)
        x = self.decoder_cnn(x)

        return x #(B, H, W, 4)

class Decoder_nerf(layers.Layer):
    def __init__(self, position_emb_dim=10, mlp_hidden_size=64):
        super().__init__()

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim
        self.embedding_fn, self.out_dim = get_embedder(self.position_emb_dim, 0)

        self.mlp_hidden_size = mlp_hidden_size


        self.decoder_mlp_bg_layer_0 = tf.keras.Sequential([
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
        ], name="decoder_mlp_bg_layer_0")

        self.decoder_mlp_bg_layer_1 = tf.keras.Sequential([
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(4, activation="relu"),
        ], name="decoder_mlp_bg_layer_1")


        self.decoder_mlp_fg_layer_0 = tf.keras.Sequential([
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
        ], name="decoder_mlp_fg_layer_0")

        self.decoder_mlp_fg_layer_1 = tf.keras.Sequential([
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(self.mlp_hidden_size, activation="relu"),
            layers.Dense(4, activation="relu"),
        ], name="decoder_mlp_fg_layer_1")

    def call(self, points_bg, points_fg, slots):
        '''
        inputs: points_bg: points sampled in background   Nx3
                points_fg: points sampled in objects  (K-1)xNx3
                slots: generated by slot attention model  KxC

        outputs: rgb colors and volume density of bg and fg
        '''

        K, C = slots.shape
        N = points_bg.shape[0]

        
        slots_bg = slots[0:1, :]  # 1xC
        slots_fg = slots[1:, :]  # (K-1)xC

        slots_bg =  tf.tile(slots_bg[:, None, :], [1, N, 1])
        slots_fg =  tf.tile(slots_fg[:, None, :], [1, N, 1])
        points_bg = tf.reshape(points_bg, [-1,3])


        encoded_points_bg = self.embedding_fn(points_bg)
        encoded_points_bg = tf.reshape(encoded_points_bg, [N, self.out_dim])
        encoded_points_bg = encoded_points_bg[None, ...]

        
        points_fg = tf.reshape(points_fg, [-1,3])
        encoded_points_fg = self.embedding_fn(points_fg)
        encoded_points_fg = tf.reshape(encoded_points_fg, [K-1, N,  self.out_dim])


        input_bg = tf.concat([encoded_points_bg, slots_bg], axis=-1)
        input_bg = input_bg[0]
        input_fg = tf.concat([encoded_points_fg, slots_fg], axis=-1)
        input_fg = tf.reshape(input_fg, [-1, self.out_dim + C])

        tmp = self.decoder_mlp_bg_layer_0(input_bg)
        raw_bg = self.decoder_mlp_bg_layer_1(tf.concat([input_bg, tmp], axis=-1))
        raw_bg = raw_bg[tf.newaxis, ...]

        tmp = self.decoder_mlp_fg_layer_0(input_fg)
        raw_fg = self.decoder_mlp_fg_layer_1(tf.concat([input_fg,tmp], axis=-1))
        raw_fg = tf.reshape(raw_fg, [K-1, N, 4])


        all_raws = tf.concat([raw_bg, raw_fg], axis=0)  # KxNx4

        raw_masks = tf.nn.relu(all_raws[:, :, -1:])  # KxNx1
        masks = raw_masks / (tf.math.reduce_sum(raw_masks,axis=0) + 1e-5)  # KxNx1
        raw_rgb = (tf.math.tanh(all_raws[:, :, :3]) + 1) / 2
        raw_sigma = raw_masks

        unmasked_raws = tf.concat([raw_rgb, raw_sigma], axis=2)  # KxNx4
        masked_raws = unmasked_raws * masks
        raws = tf.math.reduce_sum(masked_raws, axis=0)

        return raws, masked_raws, unmasked_raws, masks 





def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = layers.Dense(hidden_size, use_bias=True)
    self.grid = build_grid(resolution)

  def call(self, inputs):
    ## input: (B, H, W, 64)
    ## added: (1, H, W, 64)
    return inputs + self.dense(self.grid)


class Encoder_Decoder(layers.Layer):
    def __init__(self, cam_param, num_slots, num_iterations, resolution, 
            decoder_initial_size=(25, 25)):

        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.resolution = resolution

        self.encoder = Encoder(cam_param)

        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_dim=64,
            mlp_hidden_dim=128)
        
        self.decoder_initial_size = decoder_initial_size
        self.decoder = Decoder(
            decoder_initial_size=self.decoder_initial_size)
    
    def call(self, images, depth_maps, c2ws):
        '''
        input:  images: images  BxHxWx3
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4
        '''
        x = self.encoder(images, depth_maps, c2ws) # (B,H',W',C)

        # (N*H*W, C)
        x = tf.reshape(x, [x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])

        slots, attn = self.slot_attention(x) # (N_slots, slot_size)

        # (N_slots, W_init, H_init, slot_size)
        broadcast = tf.tile(slots[:, None, None, :], 
            [1, self.decoder_initial_size[0], self.decoder_initial_size[1], 1])

        out = self.decoder(broadcast) #(1*N_slots, W, H, 4)

        rgbs, masks = tf.split(out, [3, 1], axis=-1)
        masks = tf.nn.softmax(masks, axis=0) # softmax over N_slots

        recon = tf.reduce_sum(rgbs * masks, axis=0)  # Recombine image.

        return recon, rgbs, masks, slots



class Encoder_Decoder_nerf(tf.keras.Model):
    def __init__(self, cam_param, num_slots = 3, num_iterations = 2):

        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.cam_param = cam_param

        self.encoder = Encoder(cam_param)

        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_dim=64,
            mlp_hidden_dim=128)
        

        self.decoder = Decoder_nerf()


    def call(self, images, depth_maps, c2ws, points, training):
        '''
        input:  images: images  BxHxWx3
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4
        '''
        # if training[0]==1:
        #     images, depth_maps, c2ws = images[0:1], depth_maps[0:1], c2ws[0:1]


        x = self.encoder(images, depth_maps, c2ws) # (B,H',W',C)
        x = tf.reshape(x, [-1, x.shape[-1]])

        slots, attn = self.slot_attention(x) # (N_slots, slot_size)
        
        points_bg, points_fg = preprocess_pts(points, self.num_slots)


        raws, masked_raws, unmasked_raws, masks = self.decoder(points_bg, points_fg, slots)



        # points sampling
        return raws, masked_raws, unmasked_raws, masks





def build_model(hwf, num_slots, num_iters, data_shape, N_samples=64, chunk=512, use_nerf=False):
    
    N, H, W, C = data_shape
    resolution = (H, W)

    N=4

    images = tf.keras.Input((H, W, C), batch_size=4)
    depth_maps = tf.keras.Input((H, W), batch_size=4)
    c2ws = tf.keras.Input((4, 4), batch_size=4)
    points = tf.keras.Input((chunk, N_samples, 3), batch_size=4)
    training = tf.keras.Input((1,))

    if use_nerf:
        outputs = Encoder_Decoder_nerf(hwf, num_slots, num_iters)(images, depth_maps, c2ws, points, training)
        model = tf.keras.Model(inputs=[images, depth_maps, c2ws, points, training], outputs=outputs)

    else:
        outputs = Encoder_Decoder(hwf, num_slots, num_iters, resolution)(images, depth_maps, c2ws)
        model = tf.keras.Model(inputs=[images, depth_maps, c2ws], outputs=outputs)




    
    # slots = tf.keras.Input((64,), batch_size=300)
    # outputs = SlotAttention()(slots)
    # model = tf.keras.Model(inputs=images, outputs=outputs)

    return model


