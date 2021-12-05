import os
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from PIL import Image
from math import ceil

from AbstractLoader import (
    AbstractLoader,
    Height,
    TotalPrintsPerIdentity,
    UniqueIdentities,
    Width,
)


class MathworksLoader(AbstractLoader):
    def __init__(self, image_height: int, image_width: int) -> None:
        self.image_height = image_height
        self.image_width = image_width

        self.train_fingerprints = None
        self.test_fingerprints = None

    # fingerprints: [num_identities, prints_per_identity, height, width]
    def _generate_partials(self, fingerprints: tf.Tensor, ratio: float):
        h, w = self.image_height, self.image_width
        r = ratio

        # We'll need this later when we reshape at the end
        num_identities = fingerprints.shape[0]

        # [num_identities*prints_per_identity, height, width, 1]
        prints_no_identity = tf.cast(
            tf.reshape(fingerprints, [-1, h, w, 1]), tf.float32
        )

        boxes = tf.constant(
            [
                [0, 0, r, r],  # Top-left box
                [0, 1 - r, r, 0],  # Top-right box
                [1 - r, 0, 1, r],  # Bottom-left box
                [1 - r, 1 - r, 1, 1],  # Bottom-right box
            ]
        )

        partial_prints = tf.map_fn(
            fn=lambda t: tf.image.crop_and_resize(
                tf.expand_dims(t, axis=0),
                boxes=boxes,
                box_indices=tf.constant([0, 0, 0, 0]),
                crop_size=tf.constant([self.image_height, self.image_width]),
            ),
            elems=prints_no_identity,
        )

        # Remove the channel dimension we added earlier
        partial_prints = tf.squeeze(partial_prints)

        identities_with_partials = tf.reshape(
            partial_prints, [num_identities, -1, h, w]
        )

        return identities_with_partials

    def load_fingerprints(
        self, dir: str, train_test_split: float, partial_ratio=None
    ) -> None:
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

        train_fingerprints, test_fingerprints = tf.split(
            all_fingerprints_tensor, [num_train, num_test], axis=1
        )

        if partial_ratio is not None:
            # Existing dimensions: [num_identities x prints_per_identity x height x width]
            # What we end up getting: [num_identities x 4*prints_per_identity x height x width]

            partial_train = self._generate_partials(train_fingerprints, partial_ratio)
            partial_test = self._generate_partials(test_fingerprints, partial_ratio)

            self.train_fingerprints = partial_train
            self.test_fingerprints = partial_test
        else:
            self.train_fingerprints = train_fingerprints
            self.test_fingerprints = test_fingerprints


if __name__ == "__main__":
    # Load the fingerprints
    loader = MathworksLoader()
    loader.load_fingerprints("./data/", 0.8)
