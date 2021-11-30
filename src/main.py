import tensorflow as tf
from MathworksLoader import MathworksLoader
from eval import stats
from model import FingNet
from triplets import create_triplets_batch

N_BATCHES = 50
N_IDENTITIES = 
N_ANCHOR_PER_IDENTITY = 3
N_POS_PER_ANCHOR = 2

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

ALPHA = 1
LAMBDA = 1
D_LATENT = 300

LEARNING_RATE = .1

loader = MathworksLoader(IMAGE_HEIGHT, IMAGE_WIDTH)
loader.load_fingerprints('./data', 0.6)
identities_x_train = loader.train_fingerprints

model = FingNet(ALPHA, LAMBDA, D_LATENT)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
for i in range(N_BATCHES):
    with tf.GradientTape() as tape:
        identities_z_train = model.call_on_identities(
            identities_x_train, training=True)
        z_a, z_p, z_n = create_triplets_batch(
            identities_z_train, N_IDENTITIES, N_ANCHOR_PER_IDENTITY, N_POS_PER_ANCHOR)
        loss = model.loss_function(z_a, z_p, z_n)
        print(loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
stats(model, loader.train_fingerprints)

#tf.keras.models.save_model(model, './models/fing')
