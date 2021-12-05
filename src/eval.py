import tensorflow as tf
import matplotlib as plt
import numpy as np
from MathworksLoader import MathworksLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

from model import FineNet
from knn import knn, knn_negative, knn_positive


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


# Returns the optimal k value for fingerprint classification. Inclusive.
def optimal_k(model, identities_x_train, k_range) -> int:
    k_lower, k_upper = k_range
    num_identities = identities_x_train.shape[0]
    prints_per_identity = identities_x_train.shape[1]

    # First, assemble two tensors with the embeddings and labels
    # [identities, prints_per_identities, latent_d]
    identity_embeddings = model.call_on_identities(identities_x_train)
    # [identities*prints_per_identities, latent_d]
    embeddings = tf.reshape(
        identity_embeddings, [-1, identity_embeddings.shape[-1]]
    ).numpy()

    # Assemble the kdTree.
    labels = tf.repeat(
        tf.range(0, num_identities), [prints_per_identity] * num_identities
    ).numpy()

    best_k = k_lower
    best_accuracy = 0.0

    for k in range(k_lower, k_upper):
        accuracy = compare(None, None, embeddings, labels, k)
        print(f"Accuracy for k={k}: {accuracy}")
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    print(f"Returning best k of {best_k} with best accuracy of {best_accuracy}")
    return best_k


# train_embeddings: [num_identities x train_prints x latent_d]
# test_embeddings:  [num_identities x test_prints  x latent_d]
def accuracy(train_embeddings, test_embeddings, k) -> float:
    num_train_prints_per_identity = train_embeddings.shape[1]
    num_identities = train_embeddings.shape[0]

    # Put together our k-D tree
    train_prints = tf.reshape(train_embeddings, [-1, train_embeddings.shape[-1]])
    kd = KDTree(train_prints)

    # Get test data shaped up correctly
    test_prints = tf.reshape(test_embeddings, [-1, test_embeddings.shape[-1]])
    num_test_prints_per_iden = test_embeddings.shape[1]

    # Assemble the labels
    train_labels = tf.repeat(
        tf.range(0, num_identities), [num_train_prints_per_identity] * num_identities
    ).numpy()
    test_labels = tf.repeat(
        tf.range(0, num_identities), [num_test_prints_per_iden] * num_identities
    ).numpy()

    return compare(train_prints, train_labels, test_prints, test_labels, k)


# Uses the train embeddings and labels to use as the neighbors. Runs KNN on the test embeddings.
# Returns accuracy on the test embeddings.
#
# If we're seeing a fingerprint for the first time, we might not have any training data. In this
# case, for every fingerprint in test, we'll find its k nearest neighbors within the test space.
# If we have test embeddings {1, 2, 5, 6} and we query for 1, we'll make sure _not_ to return 1 (just
# 2, 5, or 6; based on k). If no training data is provided, train_{embeds|labels} should both be
# set to None.
#
# Parameters:
#   train_embeds are [num_prints_train x latent_d] or None
#   train_labels are [num_prints x 1] or None
#   test_embeds are [num_prints_test x latent_d]
#   train_labels are [num_prints_test x 1]
#   should_omit_self is explained blow.
#   k is the k to use for KNN.
#
def compare(train_embeds, train_labels, test_embeds, test_labels, k=1) -> float:
    if (train_embeds is None and train_labels is not None) or (
        train_embeds is not None and train_labels is None
    ):
        raise Exception("train embeds and labels must both be None or both be lists")

    only_testing = train_embeds is None and train_labels is None
    kd = KDTree(test_embeds if only_testing else train_embeds)

    num_correct = 0
    for i, test_embed in enumerate(test_embeds):
        # If we query a test embed on the testing embeds, we'll get ourself back! We'll remove it
        # later, but because we'll remove it, we need to query for 1 more than k.
        actual_k = k + 1 if only_testing else k
        _, knn_indices = kd.query(test_embed, k=actual_k)
        if type(knn_indices) is int:
            knn_indices = [knn_indices]

        if only_testing:
            # Remove i, the current index, from the closest neighbors
            assert len(knn_indices) == k + 1
            knn_indices = np.delete(knn_indices, np.where(knn_indices == i))
            assert len(knn_indices) == k
            guessed_labels = test_labels[knn_indices]
        else:
            # Find the most common *train* label for the found train indices
            guessed_labels = train_labels[knn_indices]
            if type(guessed_labels) is np.int32:
                guessed_labels = np.array([guessed_labels])

        most_guessed_label = np.bincount(guessed_labels).argmax()
        if most_guessed_label == test_labels[i]:
            num_correct += 1

    return num_correct / len(test_embeds)


if __name__ == "__main__":
    loader = MathworksLoader(200, 200)
    loader.load_fingerprints("./data", 0.6, 0.75)
    model = FineNet(0.3, 0, 300)
    model.load_weights("./models/fing")

    # The model provides the fingerprint embedding, but classification still needs to be done.
    # We find the optimal k. It will be between 1 and model.train_fingerprints.shape[0], inclusive.
    best_k = optimal_k(model, loader.train_fingerprints, (1, 8))

    test_accuracy = accuracy(
        model.call_on_identities(loader.train_fingerprints),
        model.call_on_identities(loader.test_fingerprints),
        best_k,
    )
    print(f"Accuracy on testing set: {test_accuracy}")

    # stats(model, loader.train_fingerprints)
    # stats(model, loader.test_fingerprints)
    show_tsne_visualization(
        model.call_on_identities(loader.train_fingerprints, training=True),
        model.call_on_identities(loader.test_fingerprints, training=True),
    )
