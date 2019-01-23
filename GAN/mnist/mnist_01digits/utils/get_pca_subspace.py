# calculate pca for minst training data

import numpy as np
import tensorflow as tf


def get_pca_subspace(_data,_dim,_dir):



	singular_values,left_singular_vectors,right_singular_vectors = tf.linalg.svd(_data)

	with tf.Session() as sess:
		S,U,V = sess.run([singular_values,left_singular_vectors,right_singular_vectors])
	
	subspace_map = V[:,:_dim]
	np.save(_dir + 'right_singular_vectors',V)

	return subspace_map
