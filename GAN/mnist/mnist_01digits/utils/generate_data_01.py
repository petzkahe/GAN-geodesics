from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *
from GAN.mnist.mnist_01digits.main_config import *
import numpy as np

counter = 0

train_data = np.load(results_directory + 'Data/selected_train_data.npy')
train_labels = np.load(results_directory + 'Data/selected_train_labels.npy')


# # generate real samples
def generate_real_data(n_batch):
#
	global counter
	global train_data
	global train_labels
	
	if counter+n_batch > int(train_data.shape[0]):
		rng_state = np.random.get_state()
		np.random.shuffle(train_data)
		np.random.set_state(rng_state)
		np.random.shuffle(train_labels)

		counter = 0
		
	_data = train_data[counter:counter+n_batch,:]
	counter += n_batch

	return _data





def generate_latent_data(_n_batch):



	if latent_distribution == "uniform":
		latent = np.random.uniform( low=latent_min_value, high=latent_max_value,
											   size=[_n_batch, dim_latent] ).astype( 'float32' )
	elif latent_distribution == "Gaussian":
		latent = np.random.normal( size=[_n_batch, dim_latent] ).astype( 'float32' )

	else:
		raise Exception()

	return latent



