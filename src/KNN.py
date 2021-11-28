from typing import Literal
import tensor_annotations.tensorflow as ttf
import tensorflow as tf

from AbstractLoader import ArbitraryNumPrints, Height, Width


def KNN(
    data: ttf.Tensor3[ArbitraryNumPrints, Height, Width],
    anchor: ttf.Tensor3[Literal[1], Height, Width],
    model: tf.keras.Model,
    k: int,
) -> ttf.Tensor3[ArbitraryNumPrints, Height, Width]:
    data_in_latent_space = tf.cast(model.call(data), tf.double)
    anchor_in_latent_space = tf.cast(model.call(anchor), tf.double)

    distances = tf.norm(data_in_latent_space, axis=1) - tf.norm(
        anchor_in_latent_space, axis=1
    )

    # Return the kth_largest, so multiply by -1
    kth_largest_indices = tf.math.top_k(-1 * distances, k=k, sorted=True).indices
    return tf.gather(data, kth_largest_indices)
