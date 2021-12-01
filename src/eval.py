import tensorflow as tf
import matplotlib as plt
import numpy as np
from MathworksLoader import MathworksLoader

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from model import FingNet
from knn import knn_negative, knn_positive


def stats(model, identities_x):
    identities_z = model.call_on_identities(identities_x, training=False)

    closest_neg_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    closest_pos_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    for identity_idx in range(identities_z.shape[0]):
        for print_idx in range(identities_z.shape[1]):
            closest_pos_z = knn_positive(
                identity_idx, print_idx, identities_z, 1)
            closest_neg_z = knn_negative(
                identity_idx, print_idx, identities_z, 1)
            z = identities_z[identity_idx, print_idx, :]
            closest_neg_dist[identity_idx, print_idx] = tf.norm(
                closest_neg_z - z)
            closest_pos_dist[identity_idx, print_idx] = tf.norm(
                closest_pos_z - z)
    print(f"Average closest negative distance: {np.mean(closest_neg_dist)}")
    print(f"Average closest positive distance: {np.mean(closest_pos_dist)}")


# Adapted from: https://datascientyst.com/get-list-of-n-different-colors-names-python-pandas/
def generate_n_hex_colors(n: int):
    colors = [""] * n
    for i in range(n):
        colors[i] = "#%06x" % np.random.randint(0, 0xFFFFFF)
    return colors


def show_tsne_visualization(training_z: tf.Tensor, testing_z: tf.Tensor):
    training_z = training_z[:10]
    testing_z = testing_z[:10]

    tsne = TSNE(n_components=2, random_state=0)

    if training_z.shape[0] != testing_z.shape[0]:
        raise ValueError(
            "Training and testing sets must have the same number of identities"
        )
    num_identities = training_z.shape[0]

    # Concatenate the training/testing latent spaces so we run the same t-SNE on both
    # (Recall that t-SNE uses non-convex optimization, so we can't run it on both independently)
    #
    # Shape: (50, 5, 300)
    all_z = tf.concat([training_z, testing_z], axis=1)
    print(f"all_z shape: {all_z.shape}")

    # Shape: (250, 2)
    all_z_2d = tsne.fit_transform(tf.reshape(all_z, [-1, all_z.shape[-1]]))
    print(f"all_z_2d shape: {all_z_2d.shape}")

    # Separate the training/testing within the 2D space
    # Should be of shape: (50, 5, 2)
    all_2d = tf.reshape(all_z_2d, [all_z.shape[0], all_z.shape[1], 2])
    print(f"all_2d shape: {all_2d.shape}")

    # Shape: (50, 3, 2), (50, 2, 2)
    training_2d, testing_2d = tf.split(
        all_2d, [training_z.shape[1], testing_z.shape[1]], axis=1
    )
    print(
        f"training_2d shape: {training_2d.shape}, testing_2d shape: {testing_2d.shape}"
    )

    colors = generate_n_hex_colors(num_identities)

    # Plot the training set
    for i, color in zip(range(num_identities), colors):
        plt.scatter(
            training_2d[i, :, 0],
            training_2d[i, :, 1],
            c=color,
            label=f"Identity {i}",
            alpha=0.5,
        )

    # Plot the testing set
    for i, color in zip(range(num_identities), colors):
        # Don't include a label for the testing set because it's already there from training
        plt.scatter(
            testing_2d[i, :, 0],
            testing_2d[i, :, 1],
            c=color,
            alpha=0.5,
            marker="s",
        )

    plt.legend()
    plt.show()


if __name__ == '__main__':
    loader = MathworksLoader(200, 200)
    loader.load_fingerprints("./data", 0.6)
    model = FingNet(0.3, 0, 300)
    model.load_weights('./models/fing')
    stats(model, loader.train_fingerprints)
    stats(model, loader.test_fingerprints)
    show_tsne_visualization(
        model.call_on_identities(loader.train_fingerprints, training=True),
        model.call_on_identities(loader.test_fingerprints, training=True),
    )
