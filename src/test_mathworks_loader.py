import os
import shutil

import numpy as np
from PIL import Image

from MathworksLoader import MathworksLoader

# For 3 unique identities, will create 5 test images, that are 2x2 images, for each identity

# Identity 1
fingerprint_1_1 = np.array([[1, 2], [3, 4]])
fingerprint_1_2 = np.array([[1, 2], [5, 6]])
fingerprint_1_3 = np.array([[1, 2], [7, 8]])
fingerprint_1_4 = np.array([[1, 2], [9, 10]])
fingerprint_1_5 = np.array([[3, 4], [1, 2]])

# Identity 2
fingerprint_2_1 = np.array([[100, 101], [3, 4]])
fingerprint_2_2 = np.array([[100, 101], [5, 6]])
fingerprint_2_3 = np.array([[100, 101], [7, 8]])
fingerprint_2_4 = np.array([[100, 101], [9, 10]])
fingerprint_2_5 = np.array([[3, 4], [100, 102]])

# Identity 3
fingerprint_3_1 = np.array([[200, 201], [3, 4]])
fingerprint_3_2 = np.array([[200, 201], [5, 6]])
fingerprint_3_3 = np.array([[200, 201], [7, 8]])
fingerprint_3_4 = np.array([[200, 201], [9, 10]])
fingerprint_3_5 = np.array([[3, 4], [200, 202]])

# Assemble the identities
identity_1 = np.array(
    [
        fingerprint_1_1,
        fingerprint_1_2,
        fingerprint_1_3,
        fingerprint_1_4,
        fingerprint_1_5,
    ]
).astype(np.uint8)

identity_2 = np.array(
    [
        fingerprint_2_1,
        fingerprint_2_2,
        fingerprint_2_3,
        fingerprint_2_4,
        fingerprint_2_5,
    ]
).astype(np.uint8)

identity_3 = np.array(
    [
        fingerprint_3_1,
        fingerprint_3_2,
        fingerprint_3_3,
        fingerprint_3_4,
        fingerprint_3_5,
    ]
).astype(np.uint8)

# Identity data
data_to_serialize = np.array(
    [
        ("identity_1", identity_1),
        ("identity_2", identity_2),
        ("identity_3", identity_3),
    ]
)

# Convert each identity to an image and write to a temporary directory

# Clean up past work in the temporary directory
shutil.rmtree("/tmp/mathworks_test")

# Add all the images to the temporary directory
os.mkdir("/tmp/mathworks_test")
for identity_name, identity_data in data_to_serialize:
    print("Writing {}".format(identity_name))
    identity_path = os.path.join("/tmp/mathworks_test", identity_name)
    os.mkdir(identity_path)

    for i, fingerprint in enumerate(identity_data):
        print("Writing {}".format(i))
        image = Image.fromarray(fingerprint)
        image.save(os.path.join(identity_path, f"{i}.jpg"))

print("Done writing images\n\n")

#################################################################
# Now, we test whether our MathworksLoader can load the images ##
#################################################################
loader = MathworksLoader(2, 2)

# For all train: make sure that we have 3 identities, 5 fingerprints each, that are all 2x2 images
loader.load_fingerprints("/tmp/mathworks_test", 1)
assert loader.train_fingerprints.shape == (3, 5, 2, 2)

# For an 80/20 split, make sure that we have 4 fingerprints for train; 1 for test
loader.load_fingerprints("/tmp/mathworks_test", 0.8)
assert loader.train_fingerprints.shape == (3, 4, 2, 2)
assert loader.test_fingerprints.shape == (3, 1, 2, 2)

# Now, we'll create triplets for each identity
print(loader.train_fingerprints, loader.test_fingerprints)
identity_1_triplets = loader.create_triplets_for_identity(0, 1, 2, True)
print("\n\n")
print(identity_1_triplets)
