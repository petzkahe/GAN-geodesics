# calculate pca for minst training data

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

dim_reduced = 2

dataset = input_data.read_data_sets( 'MNIST_data' )

training_data = dataset.train.images
training_labels = dataset.train.labels



training_data_zero_one = np.array([training_data[key] for (key, label) in enumerate(training_labels) if int(label) == 0 or int(label) == 1])
training_labels_zero_one = np.array([training_labels[key] for (key, label) in enumerate(training_labels) if int(label) == 0 or int(label) == 1])


print(training_data_zero_one.shape)

training_data = training_data_zero_one
training_labels = training_labels_zero_one




singular_values,left_singular_vectors,right_singular_vectors = tf.linalg.svd(training_data)

with tf.Session() as sess:
		S,U,V = sess.run([singular_values,left_singular_vectors,right_singular_vectors])
		print(S.shape,U.shape,V.shape)


np.save('svd_right_save',V)

# map into subspace

#V = np.load('svd_right_save.npy')


subspace_map = V[:,:dim_reduced]
training_data_subspace = np.matmul(training_data,subspace_map)

print(training_data_subspace.shape)

fig,ax = plt.subplots()
ax.scatter(training_data_subspace[:,0],training_data_subspace[:,1],c=training_labels)
for i, txt in enumerate(training_labels):
    ax.annotate(txt, (training_data_subspace[i,0], training_data_subspace[i,1]))
plt.savefig('pca_test.png', bbox_inches='tight' )


training_data_reconstructed = np.matmul(training_data_subspace[0:24,:],np.transpose(subspace_map))

plt.figure( figsize=(12, 10) )
gs = gridspec.GridSpec( 5, 5 )
for j, generated_image in enumerate( training_data_reconstructed ):
    ax = plt.subplot( gs[j] )
    ax.set_xticks( [] )
    ax.set_yticks( [] )
    #ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
    c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
    plt.colorbar(c)
plt.savefig('pca_reconstruction_test.png', bbox_inches='tight' )
plt.close()


# read dataset into matrix
# apply svd
# save matrices in data structure