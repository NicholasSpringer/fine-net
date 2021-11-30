try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from re import T
import tensor_annotations.tensorflow as ttf
import tensorflow as tf

from AbstractLoader import ArbitraryNumPrints, Height, Width


def knn(
    target_z: ttf.Tensor3[Literal[1], Height, Width],
    candidates_z: ttf.Tensor3[ArbitraryNumPrints, Height, Width],
    k: int,
) -> ttf.Tensor3[ArbitraryNumPrints, Height, Width]:
    diffs = candidates_z - \
        tf.repeat(tf.expand_dims(target_z, 0), candidates_z.shape[0], axis=0)
    distances = tf.norm(diffs, axis=1)
    # Get k closest
    k_closest_indices = tf.math.top_k(-1 *
                                      distances, k=k, sorted=False).indices
    return tf.gather(candidates_z, k_closest_indices)


def knn_positive(identity_idx, print_idx, identities_z, k):
    target_z = identities_z[identity_idx, print_idx, :]
    all_pos_z, _ = split_positive_negative(identity_idx, identities_z)
    pos_candidates_z = tf.concat(
        [all_pos_z[:print_idx], all_pos_z[print_idx+1:]], 0)
    return knn(target_z, pos_candidates_z, k)


def knn_negative(identity_idx, print_idx, identities_z, k):
    target_z = identities_z[identity_idx, print_idx, :]
    _, all_neg_z = split_positive_negative(identity_idx, identities_z)
    return knn(target_z, all_neg_z, k)


def split_positive_negative(identity_idx, identities_z):
    all_pos_z = identities_z[identity_idx]
    negative_mask = 1 - tf.one_hot(identity_idx, identities_z.shape[0])
    all_neg_z = tf.boolean_mask(identities_z, negative_mask, axis=0)
    # Flatten negative examples
    all_neg_z = tf.reshape(
        all_neg_z, (-1, identities_z.shape[-1]))

    return all_pos_z, all_neg_z
