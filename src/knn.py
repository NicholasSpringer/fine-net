try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from re import T
import tensor_annotations.tensorflow as ttf
import tensorflow as tf

from AbstractLoader import ArbitraryNumPrints, Height, Width


def knn(
    target_x: ttf.Tensor3[Literal[1], Height, Width],
    candidates_x: ttf.Tensor3[ArbitraryNumPrints, Height, Width],
    k: int,
    model: tf.keras.Model,
) -> ttf.Tensor3[ArbitraryNumPrints, Height, Width]:
    target_x = tf.reshape(target_x, [1, target_x.shape[0], target_x.shape[1], 1])
    target_z = tf.cast(model.call(target_x), tf.double)
    candidates_x = tf.expand_dims(candidates_x, 3)
    candidates_z = tf.cast(model.call(candidates_x), tf.double)
    diffs = candidates_z - tf.repeat(target_z, candidates_z.shape[0], axis=0)
    distances = tf.norm(diffs, axis=1)
    # Get k closest
    k_closest_indices = tf.math.top_k(-1 *
                                      distances, k=k, sorted=False).indices
    return tf.gather(candidates_x, k_closest_indices)


def knn_positive(identity_idx, print_idx, identities_x, k, model):
    target_x = identities_x[identity_idx, print_idx, :, :]
    all_pos_x, _ = split_positive_negative(identity_idx, identities_x)
    pos_candidates_x = tf.concat(
        [all_pos_x[:print_idx], all_pos_x[print_idx+1:]], 0)
    return knn(target_x, pos_candidates_x, k, model)


def knn_negative(identity_idx, print_idx, identities_x, k, model):
    target_x = identities_x[identity_idx, print_idx, :, :]
    _, all_neg_x = split_positive_negative(identity_idx, identities_x)
    return knn(target_x, all_neg_x, k, model)


def split_positive_negative(identity_idx, identities_x):
    all_pos_x = identities_x[identity_idx]
    negative_mask = 1 - tf.one_hot(identity_idx, identities_x.shape[0])
    all_neg_x = tf.boolean_mask(identities_x, negative_mask, axis=0)
    # Flatten negative examples
    all_neg_x = tf.reshape(
        all_neg_x, (-1, identities_x.shape[-2], identities_x.shape[-1]))

    return all_pos_x, all_neg_x
