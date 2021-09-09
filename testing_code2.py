from models import *

decoder_initial_size = (25, 25)
decoder = Decoder(
            decoder_initial_size=decoder_initial_size)

slots = tf.ones((1, 5, 64))
out = spatial_broadcast(slots, decoder_initial_size)
out = decoder(out) #(1*N_slots, W, H, 4)
rgbs, masks = tf.split(out, [3, 1], axis=-1)
masks = tf.nn.softmax(masks, axis=0) # softmax over N_slots

recon = tf.reduce_sum(rgbs * masks, axis=0)  # Recombine image.
print(recon.shape, rgbs.shape)