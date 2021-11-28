import os

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from PIL import Image

from AbstractLoader import (
    AbstractLoader,
    Height,
    PlaceholderModel,
    TotalPrintsPerIdentity,
    TrainOrTestPrintsPerIdentity,
    UniqueIdentities,
    Width,
)
from KNN import KNN


class MathworksLoader(AbstractLoader):
    def __init__(self, image_height: int, image_width: int) -> None:
        self.image_height = image_height
        self.image_width = image_width

        self.train_fingerprints = None
        self.test_fingerprints = None

    def load_fingerprints(self, dir: str, train_test_split: float) -> None:
        data_from_fs = list(os.walk(dir))
        if len(data_from_fs[0][1]) != 0:
            # Directory contains subdirectories; must be root. Warn, but ignore.
            print(f"Found entry with subdirectories {data_from_fs[0][1]}. Skipping.")
            data_from_fs = data_from_fs[1:]

        # Sort the subdirectories by lexicographic order so that we can test more easily
        # i.e. if we have [c, a, b] then we want [a, b, c]
        data_from_fs.sort(key=lambda x: x[0])

        # Length of the sub-directories tells us how many unique identities we have
        num_identities = len(data_from_fs)
        # The last entry of the first tuple should tell us how many fingerprints per identity
        num_fingerprints_per_identity = len(data_from_fs[0][-1])

        # Initialize the memory upfront to avoid resizing later on
        all_fingerprints = np.empty(
            (
                num_identities,
                num_fingerprints_per_identity,
                self.image_height,
                self.image_width,
            )
        )

        for i, (subdir, _, fingerprints) in enumerate(data_from_fs):
            # Initialize the fingerprints for this identity up-front.
            fingerprints_for_identity: np.ndarray = np.empty(
                (num_fingerprints_per_identity, self.image_height, self.image_width)
            )

            # Set each fingerprint for the current identity to what we read
            for j, fingerprint_name in enumerate(fingerprints):
                path_to_fingerprint = os.path.join(subdir, fingerprint_name)
                image_data = np.asarray(Image.open(path_to_fingerprint))

                fingerprints_for_identity[j] = image_data

            all_fingerprints[i] = fingerprints_for_identity

        # Use all_fingerprints to populate the test/train tensors
        all_fingerprints_tensor: ttf.Tensor4[
            UniqueIdentities, TotalPrintsPerIdentity, Height, Width
        ] = tf.convert_to_tensor(all_fingerprints)

        # Do the splitting into the training and testing.
        # Can be done by:
        #   1. Split on identities: so keep 80% of the identities for training, and 20% for testing
        #   2. Split on fingerprints: so keep 80% of the fingerprints for training, and 20% for testing
        num_train = round(num_fingerprints_per_identity * train_test_split)
        num_test = num_fingerprints_per_identity - num_train

        self.train_fingerprints, self.test_fingerprints = tf.split(
            all_fingerprints_tensor, [num_train, num_test], axis=1
        )

    def create_triplets_for_identity(
        self,
        identity: int,
        num_anchors: int,
        k: int,
        is_training: bool,
        model: tf.keras.Model = PlaceholderModel,
    ) -> ttf.Tensor4:  # Tensor4[num_anchors, k+1, self.image_height, self.image_width]
        data = self.train_fingerprints if is_training else self.test_fingerprints

        data_for_identity: ttf.Tensor3[
            TrainOrTestPrintsPerIdentity, Height, Width
        ] = data[identity]

        mask = 1 - tf.one_hot(identity, len(data))
        data_without_identity = tf.boolean_mask(data, mask, axis=0)
        flattened_data_without_identity = tf.reshape(
            # data_without_identity is (NumIdentities-1, NumPrintsPerIdentity, Height, Width)
            # We want (NumIdentities-1*NumPrintsPerIdentity, Height, Width). This is the same as
            # tf.reshape(data_without_identity, (-1, Height, Width)).
            #
            # What this gets us is a tensor that has everybody's fingerprints BUT the current
            # identity's fingerprints. We'll then use this in the negative KNN step.
            #
            # *tf.shape(...)[-2:] is the last two dimensions of the tensor.
            data_without_identity,
            (-1, *tf.shape(data_without_identity).numpy()[-2:]),
        )

        if len(data_for_identity) < num_anchors:
            raise Exception(
                f"Provided anchor cardinality argument, {num_anchors}, exceeds number examples, {len(data_for_identity)} for current identity={identity}"
            )

        triplets = np.empty((num_anchors, k + 1, self.image_height, self.image_width))
        # TODO: Randomly choose the anchors. For now, just take the first Np.
        anchors = data_for_identity[:num_anchors]

        # For each anchor, determine the k closest neighbors from data_for_identity.
        for idx, anchor in enumerate(anchors):
            # anchors is [num_anchors, Height, Width]
            # anchor should be [1, Height, Width], but from the loop, it is just [Height, Width]
            anchor = tf.expand_dims(anchor, axis=0)

            # Expected shape: [k, image_height, image_width]
            positive_examples = KNN(data_for_identity, anchor, model, k=k)
            # Expected shape: [1, image_height, image_width]
            negative_example = KNN(flattened_data_without_identity, anchor, model, k=1)

            # Concatenate the positive and negative examples together.
            concatenated_examples = tf.concat(
                [positive_examples, negative_example], axis=0
            )
            triplets[idx] = concatenated_examples

        return triplets


if __name__ == "__main__":
    # Load the fingerprints
    loader = MathworksLoader()
    loader.load_fingerprints("./data/", 0.8)
