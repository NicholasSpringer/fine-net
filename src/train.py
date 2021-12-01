import tensorflow as tf
import numpy as np
from time import time
from MathworksLoader import MathworksLoader
from model import FingNet
from triplets import create_triplets_batch

N_BATCHES = 1
N_IDENTITIES = 10
N_ANCHOR_PER_IDENTITY = 3
N_POS_PER_ANCHOR = 2

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

ALPHA = 0.3
LAMBDA = 1
D_LATENT = 300

LEARNING_RATE = 0.1


def sample_prints(identities_x, n_identities, n_prints_per_identity):
    # Sample anchors from positives
    sample_identities_indices = np.random.permutation(np.arange(identities_x.shape[0]))[
        :n_identities
    ]
    # Make same size as prints indices
    sample_identities_indices = tf.expand_dims(sample_identities_indices, 1)
    sample_identities_indices = tf.repeat(
        sample_identities_indices, n_prints_per_identity, axis=1
    )
    samples_prints_indices = np.empty([n_identities, identities_x.shape[1]])
    for i in range(n_identities):
        samples_prints_indices[i] = np.random.permutation(
            np.arange(identities_x.shape[1])
        )[:n_prints_per_identity]
    sample_indices = tf.stack(
        [sample_identities_indices, samples_prints_indices], axis=2
    )
    sample_identities_x = tf.gather_nd(identities_x, sample_indices)

    return sample_identities_x


def train(model, optimizer, identities_x_train):
    for i in range(N_BATCHES):
        sample_identities_x = sample_prints(
            identities_x_train, N_IDENTITIES, N_ANCHOR_PER_IDENTITY
        )
        with tf.GradientTape() as tape:
            sample_identities_z = model.call_on_identities(
                sample_identities_x, training=True
            )
            z_a, z_p, z_n = create_triplets_batch(sample_identities_z, N_POS_PER_ANCHOR)
            loss = model.loss_function(z_a, z_p, z_n)
            print(f"Batch: {i}, Loss: {loss.numpy()}")
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == "__main__":
    loader = MathworksLoader(IMAGE_HEIGHT, IMAGE_WIDTH)
    loader.load_fingerprints("./data", 0.6)
    identities_x_train = loader.train_fingerprints

    model = FingNet(ALPHA, LAMBDA, D_LATENT)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train(model, optimizer, identities_x_train)
    model.save_weights("./models/fing")
