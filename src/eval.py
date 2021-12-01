import tensorflow as tf
import matplotlib as plt
import numpy as np
from MathworksLoader import MathworksLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from model import FingNet
from knn import knn_negative, knn_positive


def stats(model, identities_x):
    identities_z = model.call_on_identities(identities_x, training=False)

    closest_neg_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    closest_pos_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    for identity_idx in range(identities_z.shape[0]):
        for print_idx in range(identities_z.shape[1]):
            closest_pos_z = knn_positive(identity_idx, print_idx, identities_z, 1)
            closest_neg_z = knn_negative(identity_idx, print_idx, identities_z, 1)
            z = identities_z[identity_idx, print_idx, :]
            closest_neg_dist[identity_idx, print_idx] = tf.norm(closest_neg_z - z)
            closest_pos_dist[identity_idx, print_idx] = tf.norm(closest_pos_z - z)
    print(f"Average closest negative distance: {np.mean(closest_neg_dist)}")
    print(f"Average closest positive distance: {np.mean(closest_pos_dist)}")


# Adapted from: https://datascientyst.com/get-list-of-n-different-colors-names-python-pandas/
def generate_n_hex_colors(n: int):
    return """#000000
#00FF00
#0000FF
#FF0000
#01FFFE
#FFA6FE
#FFDB66
#006401
#010067
#95003A
#007DB5
#FF00F6
#FFEEE8
#774D00
#90FB92
#0076FF
#D5FF00
#FF937E
#6A826C
#FF029D
#FE8900
#7A4782
#7E2DD2
#85A900
#FF0056
#A42400
#00AE7E
#683D3B
#BDC6FF
#263400
#BDD393
#00B917
#9E008E
#001544
#C28C9F
#FF74A3
#01D0FF
#004754
#E56FFE
#788231
#0E4CA1
#91D0CB
#BE9970
#968AE8
#BB8800
#43002C
#DEFF74
#00FFC6
#FFE502
#620E00
#008F9C
#98FF52
#7544B1
#B500FF
#00FF78
#FF6E41
#005F39
#6B6882
#5FAD4E
#A75740
#A5FFD2
#FFB167
#009BFF
#E85EBE
""".split(
        "\n"
    )[
        :n
    ]


def show_tsne_visualization(training_z: tf.Tensor, testing_z: tf.Tensor):
    training_z = training_z[:25]
    testing_z = testing_z[:25]

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
    all_z = np.concatenate([training_z, testing_z], axis=1)
    print(f"all_z shape: {all_z.shape}")

    # Reshape all_z to (250, 300)
    flat_z = np.reshape(all_z, [-1, all_z.shape[-1]])

    # Before TSNE, do some PCA to reduce the dimensionality
    pca = PCA(n_components=10)
    # Shape: (250, 100)
    reduced_z = pca.fit_transform(flat_z)

    # Shape: (250, 2)
    all_z_2d = tsne.fit_transform(reduced_z)
    print(f"all_z_2d shape: {all_z_2d.shape}")

    # Separate the training/testing within the 2D space
    # Should be of shape: (50, 5, 2)
    all_2d = np.reshape(all_z_2d, [all_z.shape[0], all_z.shape[1], 2])
    print(f"all_2d shape: {all_2d.shape}")

    # Shape: (50, 3, 2), (50, 2, 2)
    training_2d, testing_2d = np.split(all_2d, [training_z.shape[1]], axis=1)
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


# Should show nice clusters
def test_positive_tsne_visualization(model, identities_x):
    training_normal = np.empty((50, 3, 300))
    testing_normal = np.empty((50, 3, 300))

    for i in range(50):
        training_normal[i] = np.random.normal(i * 2, 1, (3, 300))
        testing_normal[i] = np.random.normal(i * 2, 1, (3, 300))

    show_tsne_visualization(training_normal, testing_normal)


# Should show nonsense
def test_random_tsne_visualization():
    training_random = np.empty((50, 3, 300))
    testing_random = np.empty((50, 3, 300))

    for i in range(50):
        training_random[i] = np.random.randint(-50, 50, (3, 300))
        testing_random[i] = np.random.randint(-50, 50, (3, 300))

    show_tsne_visualization(training_random, testing_random)


if __name__ == "__main__":
    loader = MathworksLoader(200, 200)
    loader.load_fingerprints("./data", 0.6)
    model = FingNet(0.3, 0, 300)
    model.load_weights("./models/fing")
    # stats(model, loader.train_fingerprints)
    # stats(model, loader.test_fingerprints)
    show_tsne_visualization(
        model.call_on_identities(loader.train_fingerprints, training=True),
        model.call_on_identities(loader.test_fingerprints, training=True),
    )
