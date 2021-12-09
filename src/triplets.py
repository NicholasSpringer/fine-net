import numpy as np
import tensorflow as tf

from knn import knn_negative, knn_positive


def create_triplets(identities_z, identity_idx: int, n_pos_per_anchor: int):
    n_anchors = identities_z.shape[1]
    if identities_z.shape[1] < n_anchors:
        raise Exception(
            f"Provided anchor cardinality argument, {n_anchors}, "
            + f"exceeds number examples, {identities_z.shape[1]} "
            + f"for current identity_idx={identity_idx}"
        )
    triplet_anchors_z = identities_z[identity_idx]
    # Repeat anchors to align with positives
    triplet_anchors_z = tf.repeat(triplet_anchors_z, n_pos_per_anchor, axis=0)

    triplet_pos_list = []
    triplet_neg_list = []
    for anchor_idx in range(n_anchors):
        triplet_pos_list.append(
            knn_positive(identity_idx, anchor_idx, identities_z, n_pos_per_anchor)
        )
        triplet_neg_list.append(knn_negative(identity_idx, anchor_idx, identities_z, 1))
    triplet_pos_z = tf.concat(triplet_pos_list, 0)
    triplet_neg_z = tf.concat(triplet_neg_list, 0)
    # Repeat negatives to align with positives
    triplet_neg_z = tf.repeat(triplet_neg_z, n_pos_per_anchor, axis=0)

    return triplet_anchors_z, triplet_pos_z, triplet_neg_z


def create_triplets_batch(identities_z, n_pos_per_anchor: int):
    """
    Creates a batch of triplets.
    """
    n_identities = identities_z.shape[0]
    # Sample identities
    anchors_list = []
    pos_list = []
    neg_list = []
    for identity_idx in range(n_identities):
        anchors, positives, negatives = create_triplets(
            identities_z, identity_idx, n_pos_per_anchor
        )
        anchors_list.append(anchors)
        pos_list.append(positives)
        neg_list.append(negatives)

    z_a = tf.concat(anchors_list, 0)
    z_p = tf.concat(pos_list, 0)
    z_n = tf.concat(neg_list, 0)

    return z_a, z_p, z_n
