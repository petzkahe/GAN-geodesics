# calculate pca for minst training data

import numpy as np
import tensorflow as tf


def get_pca_subspace(_data,_dim,_dir):

	mean_per_pixel = tf.reshape(tf.reduce_mean(_data, axis=0),shape= (1,-1))

	_data_normalized = _data - mean_per_pixel

	singular_values,left_singular_vectors,right_singular_vectors = tf.linalg.svd(_data_normalized)

	with tf.Session() as sess:
		S,U,V, _mean_per_pixel = sess.run([singular_values,left_singular_vectors,right_singular_vectors, mean_per_pixel])
	
	subspace_map = V[:,:_dim]
	np.save(_dir + 'right_singular_vectors',V)
	np.save(_dir + 'mean_per_pixel', _mean_per_pixel)


	return subspace_map, _mean_per_pixel
