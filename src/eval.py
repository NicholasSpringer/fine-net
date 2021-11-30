# write in here calculation of stats for evaluation (distances within a class, closest to other classes, etc)
# as well as tsne
import tensorflow as tf
import matplotlib as plt
import numpy as np

from knn import knn_negative, knn_positive


def stats(model, identities_x):
    identities_z = model.call_on_identities(identities_x, training=False)

    closest_neg_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    closest_pos_dist = np.empty([identities_z.shape[0], identities_z.shape[1]])
    for identity_idx in range(identities_z.shape[0]):
        for print_idx in range(identities_z.shape[1]):
            closest_pos_z = knn_positive(identity_idx, print_idx, identities_z, 1)
            closest_neg_z = knn_negative(identity_idx, print_idx, identities_z, 1)
            z = identities_z[identity_idx, print_idx, :]
            closest_neg_dist[identity_idx, print_idx] = tf.norm(closest_neg_z - z)
            closest_pos_dist[identity_idx, print_idx] = tf.norm(closest_pos_z - z)
    print(f'Average closest negative distance: {np.mean(closest_neg_dist)}')
    print(f'Average closest positive distance: {np.mean(closest_pos_dist)}')



        
