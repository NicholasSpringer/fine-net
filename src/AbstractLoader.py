import typing
from abc import ABC, abstractmethod

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from tensor_annotations import axes

# Custom tensor type annotations.
UniqueIdentities = typing.NewType("UniqueIdentities", axes.Axis)
TotalPrintsPerIdentity = typing.NewType("TotalPrintsPerIdentity", axes.Axis)
TrainOrTestPrintsPerIdentity = typing.NewType("TrainOrTestPrintsPerIdentity", axes.Axis)
ArbitraryNumPrints = typing.NewType("ArbitraryNumPrints", axes.Axis)
# SomePrints = ttf.Tensor3[] typing.NewType("SomePrints", axes.Axis)

# Will have to change once we introduce a notion of partial fingerprints.
Height, Width = axes.Height, axes.Width


class AbstractLoader(ABC):
    # Used so that we can efficiently pre-allocate memory
    image_height: int = None
    image_width: int = None

    train_fingerprints: ttf.Tensor4[
        UniqueIdentities, TrainOrTestPrintsPerIdentity, Height, Width
    ] = None
    test_fingerprints: ttf.Tensor4[
        UniqueIdentities, TrainOrTestPrintsPerIdentity, Height, Width
    ] = None

    @abstractmethod
    def __init__(image_height: int, image_width: int):
        pass

    @abstractmethod
    def load_fingerprints(self, dir: str, train_test_split: float) -> None:
        """
        Loads the fingerprints in the given directory. Assumes that each sub-directory of the given
        directory pertains to a unique identity. As such, structure should be:

        dir/
        ├── identity_1/
        │   ├── 1_1.jpg
        │   ├── 1_2.jpg
        │   ├── ...
        │   └── 1_n.jpg
        ├── identity_2/
        │   ├── 2_1.jpg
        │   ├── 2_2.jpg
        │   ├── ...
        │   └── 2_n.jpg
        └── ...

        train_test_split is a float between 0 and 1 that specifies what fraction of data goes
        towards training data vs. testing data. A value of 0 means that all data is testing. A value
        of 1 means that all data is training. Something around 0.8 should be reasonable.
        """
        pass

    # @abstractmethod
    # def create_triplets_for_identity(
    #     self,
    #     identity: int,
    #     Np: int,
    #     k: int,
    #     is_training: bool,
    #     model: tf.keras.Model,
    # ) -> ttf.Tensor2:
    #     """
    #     Creates a tensor that contains all the triplets for the given identity.
    #     Note that below, A = Np.

    #     Implementations of this function must:
    #         1. Choose A anchors for the given identity
    #         2. For each anchor, choose the k nearest fingerprints of the same identity
    #         3. For each anchor, find the closest negative fingerprint of a different identity
    #         4. Create triplets from the <anchors.positives> concatted with the <negative> anchor
    #         5. The result should be a tensor of shape: [A, k+1]. The "+1" is for the negative fingerprint

    #     The model is passed in so that we can compute the latent space for each fingerprint, when
    #     doing KNN.

    #     If is_training is true, self.train_fingerprints should be used. Otherwise, use the testing
    #     fingerprints, self.test_fingerprints.
    #     """
    #     pass

    # @abstractmethod
    # def create_batch(self, Nc: int, Np: int, k: int, is_training: bool) -> ttf.Tensor2:
    #     """
    #     Creates a batch of triplets.
    #     """
    #     pass


class PlaceholderModel(tf.keras.Model):
    def call(input):
        return tf.convert_to_tensor(np.tile(np.array([[1, 2, 3]]), (input.shape[0], 1)))
