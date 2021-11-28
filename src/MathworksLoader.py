import os

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from PIL import Image
from tensorflow.python.ops.gen_math_ops import zeta_eager_fallback

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
            print(
                f"Found entry with subdirectories {data_from_fs[0][1]}. Skipping.")
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
                grayscale_image_data = np.sum(image_data, 2) / (3 * 255)

                fingerprints_for_identity[j] = grayscale_image_data

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

    def split_positive_negative(self, identity, data):
        all_pos: ttf.Tensor3[
            TrainOrTestPrintsPerIdentity, Height, Width
        ] = data[identity]

        negative_mask = 1 - tf.one_hot(identity, len(data))
        all_neg = tf.boolean_mask(data, negative_mask, axis=0)
        # Flatten negative examples
        all_neg = tf.reshape(
            all_neg, (-1, self.image_height, self.image_width))

        return all_pos, all_neg

    def create_triplets_for_identity(
        self,
        identity: int,
        num_anchors: int,
        k: int,
        is_training: bool,
        model: tf.keras.Model = PlaceholderModel,
    ) -> ttf.Tensor4:  # Tensor4[num_anchors, k+1, self.image_height, self.image_width]
        data = self.train_fingerprints if is_training else self.test_fingerprints
        all_pos, all_neg = self.split_positive_negative(identity, data)

        if all_pos.shape[0] < num_anchors:
            raise Exception(
                f"Provided anchor cardinality argument, {num_anchors}, exceeds number examples, {len(all_pos)} for current identity={identity}"
            )

        # Sample anchors from positives
        triplet_anchor_indices = np.random.permutation(
            np.arange(all_pos.shape[0]))[:num_anchors]
        triplet_anchors = tf.gather(all_pos, triplet_anchor_indices)
        triplet_anchors = np.reshape(
            triplet_anchors, [-1, 1, self.image_height, self.image_width])

        triplet_pos = np.empty(
            (num_anchors, k, self.image_height, self.image_width)
        )
        triplet_neg = np.empty(
            (num_anchors, 1, self.image_height, self.image_width))
        for i, anchor_idx in enumerate(triplet_anchor_indices):
            # Remove anchor from positive candidates
            triplet_pos_candidates = tf.concat(
                [all_pos[:anchor_idx], all_pos[anchor_idx+1:]], 0)

            triplet_pos[i] = KNN(
                triplet_pos_candidates, triplet_anchors[i], model, k=k)
            triplet_neg[i] = KNN(
                all_neg, triplet_anchors[i], model, k=1
            )

        assert (
            triplet_anchors.shape[0] == triplet_pos.shape[0] == triplet_neg.shape[0]
        )
        triplet_pos = tf.convert_to_tensor(triplet_pos)
        triplet_neg = tf.convert_to_tensor(triplet_neg)
        return triplet_anchors, triplet_pos, triplet_neg

    def create_batch(
        self, n_identities: int, n_anchor_per_ident: int, n_pos_per_anchor: int, is_training: bool, model: tf.keras.Model = PlaceholderModel,
    ) -> ttf.Tensor2:
        """
        Creates a batch of triplets.
        """
        data = self.train_fingerprints if is_training else self.test_fingerprints

        # Sample identities
        anchor_identities = np.random.permutation(
            np.arange(data.shape[0]))[:n_identities]

        x_a = np.empty(
            (n_identities, n_anchor_per_ident, 1,
             self.image_height, self.image_width)
        )
        x_p = np.empty(
            (n_identities, n_anchor_per_ident, n_pos_per_anchor,
             self.image_height, self.image_width)
        )
        x_n = np.empty(
            (n_identities, n_anchor_per_ident, 1,
             self.image_height, self.image_width)
        )

        for i, identity_index in enumerate(anchor_identities):
            anchors, positives, negatives = self.create_triplets_for_identity(
                identity_index, n_anchor_per_ident, n_pos_per_anchor, is_training, model
            )

            x_a[i] = anchors
            x_p[i] = positives
            x_n[i] = negatives

        x_a = tf.reshape(x_a, [-1, self.image_height, self.image_width, 1])
        x_p = tf.reshape(x_p, [-1, self.image_height, self.image_width, 1])
        x_n = tf.reshape(x_n, [-1, self.image_height, self.image_width, 1])
        return x_a, x_p, x_n

    def repeat_latent_for_triplets(self, z, n_pos_per_anchor, d_latent):
        # Repeat anchor or negative latent outputs to make first dimension size n_triplets
        z = tf.reshape(z, [-1, 1, d_latent])
        z = tf.repeat(z, n_pos_per_anchor, axis=1)
        z = tf.reshape(z, [-1, d_latent])
        return z
        

if __name__ == "__main__":
    # Load the fingerprints
    loader = MathworksLoader()
    loader.load_fingerprints("./data/", 0.8)
