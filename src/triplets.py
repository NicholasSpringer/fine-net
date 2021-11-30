import numpy as np
import tensorflow as tf

from knn import knn_negative, knn_positive


def create_triplets(identities_z, identity_idx: int, n_anchors: int, n_pos_per_anchor: int):
    if identities_z.shape[1] < n_anchors:
        raise Exception(
            f"Provided anchor cardinality argument, {n_anchors}, " +
            f"exceeds number examples, {identities_z.shape[1]} " +
            f"for current identity_idx={identity_idx}"
        )

    # Sample anchors from positives
    triplet_anchor_indices = np.random.permutation(
        np.arange(identities_z.shape[1]))[:n_anchors]

    triplet_anchors_z = tf.gather(
        identities_z[identity_idx], triplet_anchor_indices)
    # Repeat anchors to align with positives
    triplet_anchors_z = tf.repeat(triplet_anchors_z, n_pos_per_anchor, axis=0)

    triplet_pos_list = []
    triplet_neg_list = []
    for i, anchor_idx in enumerate(triplet_anchor_indices):
        triplet_pos_list.append(knn_positive(
            identity_idx, anchor_idx, identities_z, n_pos_per_anchor))
        triplet_neg_list.append(knn_negative(
            identity_idx, anchor_idx, identities_z, 1))
    triplet_pos_z = tf.concat(triplet_pos_list, 0)
    triplet_neg_z = tf.concat(triplet_neg_list, 0)
    # Repeat negatives to align with positives
    triplet_neg_z = tf.repeat(triplet_neg_z, n_pos_per_anchor, axis=0)

    n_triplets = n_anchors * n_pos_per_anchor
    d_latent = identities_z.shape[-1]
    assert (
        [n_triplets, d_latent] == triplet_anchors_z.shape == triplet_pos_z.shape == triplet_neg_z.shape
    )
    return triplet_anchors_z, triplet_pos_z, triplet_neg_z


def create_triplets_batch(identities_z, n_identities: int, n_anchor_per_ident: int, n_pos_per_anchor: int):
    """
    Creates a batch of triplets.
    """
    d_latent = identities_z.shape[-1]
    n_triplets = n_identities * n_anchor_per_ident * n_pos_per_anchor
    n_triplets_per_identity = n_anchor_per_ident * n_pos_per_anchor
    # Sample identities
    anchor_identity_indices = np.random.permutation(
        np.arange(identities_z.shape[0]))[:n_identities]
    anchors_list = []
    pos_list = []
    neg_list = []
    for i, identity_idx in enumerate(anchor_identity_indices):
        anchors, positives, negatives = create_triplets(
            identities_z, identity_idx, n_anchor_per_ident, n_pos_per_anchor)
        anchors_list.append(anchors)
        pos_list.append(positives)
        neg_list.append(negatives)

    z_a = tf.concat(anchors_list, 0)
    z_p = tf.concat(pos_list, 0)
    z_n = tf.concat(neg_list, 0)

    n_triplets = n_identities * n_anchor_per_ident * n_pos_per_anchor
    d_latent = identities_z.shape[-1]
    assert (
        [n_triplets, d_latent] == z_a.shape == z_p.shape == z_n.shape
    )
    return z_a, z_p, z_n
    # return tf.cast(z_a, dtype=tf.float32), tf.cast(z_p, dtype=tf.float32), tf.cast(z_n, dtype=tf.float32)
