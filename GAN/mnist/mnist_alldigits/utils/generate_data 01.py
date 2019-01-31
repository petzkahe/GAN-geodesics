from GAN.mnist.mnist_alldigits.BIGAN_Learning.config_BIGAN import *
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

dataset = input_data.read_data_sets( 'MNIST_data' )
    
training_data = dataset.train.images
training_labels = dataset.train.labels

training_data = np.array([training_data[key] for (key, label) in enumerate(training_labels) if int(label) == 0 or int(label) == 1])
#training_labels = np.array([training_labels[key] for (key, label) in enumerate(training_labels) if int(label) == 0 or int(label) == 1])

counter = 0

#
# # generate real samples
def generate_real_data(n_batch):
#
	if counter+n_batch > int(training_data.shape[0]):
		training_data = np.random.shuffle(training_data)
		counter = 0
		print('Epoch finished and counter reset')

	data = training_data[counter:counter+n_batch,:]
	counter += n_batch

	return data
#     while True:
#         data, _ = dataset.train.next_batch(n_batch)
#         yield data




def generate_latent_data(_n_batch):



    if latent_distribution == "uniform":
        latent = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                               size=[_n_batch, dim_latent] ).astype( 'float32' )
    elif latent_distribution == "Gaussian":
        latent = np.random.normal( size=[_n_batch, dim_latent] ).astype( 'float32' )

    else:
        raise Exception()

    return latent



