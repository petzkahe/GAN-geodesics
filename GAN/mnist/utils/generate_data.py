from tensorflow.examples.tutorials.mnist import input_data
from GAN.mnist.BIGAN_Learning.config_BIGAN import *
import numpy as np

# dataset = input_data.read_data_sets('MNIST_data')
#
# # generate real samples
# def generate_real_data(n_batch):
#
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



