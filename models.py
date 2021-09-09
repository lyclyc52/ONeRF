import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers




def get_position(ray_o, ray_d, depth):
    '''
    input: ray origin, ray direction, depth

    output: 3d position of each points
    '''

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
        self.input_c=input_c
        self.layers_c=layers_c
        self.position_emb=position_emb
        self.position_emb_dim=position_emb_dim

        self.H, self.W, self.focal=cam_param
        if self.position_emb:
            self.encoder_layer_0 = layers.Conv2D(input_c + position_emb_dim, kernel_size=5, padding="SAME", activation="relu"),
        else:
            self.encoder_layer_0 = layers.Conv2D(input_c, kernel_size=5, padding="SAME", activation="relu"),

        self.encoder_cnn = tf.keras.Sequential([
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu"),
            layers.Conv2D(layers_c, kernel_size=5, padding="SAME", activation="relu")
        ], name="encoder_cnn")
                                        
        

    def call(self, x, d, c2w):
        '''
        input:  x: images  Bx3xHxW
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4

        output: feature maps  BxDxHxW
        '''

        if self.position_emb:

            rays_o, rays_d = get_rays(self.H, self.W, self.focal, self.c2w) 

            position = get_position(ray_o, rays_d, d)

            x = tf.concat([x, position], axis=-1)

        x = self.encoder_layer_0(x)
        x = self.encoder_cnn(x)  # BxDxHxW
        return x




class SlotAttention(layers.Layer):
    def __init__(self, num_iterations, num_slots, slot_dim, mlp_hidden_dim, epsilon=1e-8):
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mlp_hidden_size = mlp_hidden_dim
        self.epsilon = epsilon

        self.norm_inputs = layers.LayerNormalization()

        self.slots_mu_fg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_dim],
            dtype=tf.float32,
            name="slots_mu_fg")
        self.slots_log_sigma_fg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_dim],
            dtype=tf.float32,
            name="slots_log_sigma_fg")


        self.slots_mu_bg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_dim],
            dtype=tf.float32,
            name="slots_mu_bg")
        self.slots_log_sigma_bg = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_dim],
            dtype=tf.float32,
            name="slots_log_sigma_bg")
 
        self.k = layers.Dense(self.slot_size, use_bias=False, name="k")
        self.v = layers.Dense(self.slot_size, use_bias=False, name="v")
        self.q_fg = tf.keras.Sequential([layers.LayerNormalization(),
                                         layers.Dense(self.slot_size, use_bias=False)],
                                         name="q_fg")

        self.q_bg = tf.keras.Sequential([layers.LayerNormalization(),
                                         layers.Dense(self.slot_size, use_bias=False)],
                                         name="q_bg")

        self.gru_fg = layers.GRUCell(self.slot_size)
        self.gru_bg = layers.GRUCell(self.slot_size)

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

            q_slot_fg = self.q_fg(slot_fg)
            q_slot_fg *= self.slot_dim ** -0.5
            q_slot_bg = self.q_bg(slot_bg)
            q_slot_bg *= self.slot_dim ** -0.5


            attn_logits_fg = tf.matmul(k_input, q_slot_fg.T)
            attn_logits_bg = tf.matmul(k_input, q_slot_bg.T)
            attn_logits = tf.concat([attn_logits_bg, attn_logits_fg], axis=0)
            attn = tf.nn.softmax(attn_logits, axis=-1)

            attn_bg, attn_fg = attn[0:1, :], attn[1:, :]  # 1xN, (K-1)xN


            
            weight_bg = attn_bg + self.epsilon
            weight_bg /= tf.reduce_sum(weight_bg, axis=-2, keepdims=True)
            updates_bg = tf.matmul(weight_bg.T, v)

            weight_fg = attn_fg + self.epsilon
            weight_fg /= tf.reduce_sum(weight_fg, axis=-2, keepdims=True)
            updates_fg = tf.matmul(weight_fg.T, v)
            


            slot_bg, _ = self.gru_bg(
                updates_bg,
                [slot_prev_bg]
            )
            slot_fg, _ = self.gru_fg(
                updates_fg,
                [slot_prev_fg]
            )

            slot_bg += self.mlp_bg(slot_bg)
            slot_fg += self.mlp_fg(slot_fg)
    
    slots = tf.concat([])

    return slots, attn






