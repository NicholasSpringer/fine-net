import tensorflow as tf
from MathworksLoader import MathworksLoader
from model import FingNet

N_BATCHES = 20
N_IDENTITIES = 10
N_ANCHOR_PER_IDENTITY = 3
N_POS_PER_ANCHOR = 2

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

ALPHA = 1
LAMBDA = 0
D_LATENT = 300

LEARNING_RATE = 1e-3

loader = MathworksLoader(IMAGE_HEIGHT, IMAGE_WIDTH)
loader.load_fingerprints('./data', 0.6)

model = FingNet(ALPHA, LAMBDA, D_LATENT)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for i in range(N_BATCHES):
    x_a, x_p, x_n = loader.create_batch(
        N_IDENTITIES, N_ANCHOR_PER_IDENTITY, N_POS_PER_ANCHOR, True)

    with tf.GradientTape() as tape:
        z_a = model(x_a)
        z_p = model(x_p)
        z_n = model(x_n)
        z_a = loader.repeat_latent_for_triplets(z_a, N_POS_PER_ANCHOR, D_LATENT)
        z_n = loader.repeat_latent_for_triplets(z_n, N_POS_PER_ANCHOR, D_LATENT)
        loss = model.loss_function(z_a, z_p, z_n)
        print(loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))