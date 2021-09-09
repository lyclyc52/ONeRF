import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers




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
    N = c2w.shape[0]
    dirs = tf.broadcast_to(dirs,[N,dirs.shape[1],dirs.shape[2],dirs.shape[3]])

    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], -1)
    rays_o = c2w[:,:3, -1]
    rays_o = rays_o[:,np.newaxis, np.newaxis, :]

    rays_o = tf.broadcast_to(rays_o, tf.shape(rays_d))

    return rays_o, rays_d


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

        print(x.shape)
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

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

class Encoder_Decoder(layers.Layer):
    def __init__(self, cam_param, num_slots, num_iterations, resolution):

        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.resolution = resolution

        self.encoder = Encoder(cam_param)
        
        self.layer_norm = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64)
        ], name="feedforward")

        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=64,
            mlp_hidden_size=128)
        
        self.decoder_initial_size = (25, 25)
        self.decoder = Decoder(
            decoder_initial_size=self.decoder_initial_size)
    
    def call(self, images, depth_maps, poses):
        '''
        input:  images: images  BxHxWx3
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4
        '''
        feature_maps = self.encoder(images, depth_maps, poses) # (B,H',W',C)

        x = tf.reshape(x, [x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])
        x = x[None, ...] # (1, N*H*W, C)

        slots = self.slot_attention(x) # (1, N_slots, slot_size)

        #(1*N_slots, W_init, H_init, slot_size)
        out = spatial_broadcast(slots, self.decoder_initial_size)

        out = self.decoder(out) #(1*N_slots, W, H, 4)

        rgbs, masks = tf.split(out, [3, 1], axis=-1)
        masks = tf.nn.softmax(masks, axis=0) # softmax over N_slots

        recon = tf.reduce_sum(rgbs * masks, axis=0)  # Recombine image.

        return recon, rgbs, masks, slots

def build_model(hwf, num_slots, num_iters, data_shape):
    
    N, H, W, C = data_shape
    resolution = (H, W)

    fn = Encoder_Decoder(cam_param=hwf, num_slots, num_iters, resolution)

    image = tf.keras.Input(data_shape, batch_size=None)
    outputs = fn(image)
    model = tf.keras.Model(inputs=image, outputs=outputs)
    
    return model
