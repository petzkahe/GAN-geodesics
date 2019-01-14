import matplotlib
import numpy as np
import matplotlib.gridspec as gridspec

matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.mnist.Geodesic_Learning.config_geodesic_mnist import *
from GAN.mnist.BIGAN_Learning.config_BIGAN import *
from GAN.mnist.Geodesic_Learning.config_geodesic_mnist import *


def plot_sample_space(samples, iteration_step):
    plt.figure( figsize=(12, 10) )
    gs = gridspec.GridSpec( 5, 5 )
    for j, generated_image in enumerate( samples ):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        #ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r' )
        plt.colorbar(c)
    plt.savefig('{}/frame_{}.png'.format(log_directory, iteration_step), bbox_inches='tight' )
    plt.close()

    return None


def plot_geodesic(geodesics_in_sample_space, method):

    plt.clf()

    plt.figure( figsize=(12, 10))
    gs = gridspec.GridSpec(n_geodesics,n_interp_selected)



    filter = [j*increment for j in range(n_interp_selected-1)]+[n_interpolations_points_geodesic]


    selected_geodesics_in_sample_space = geodesics_in_sample_space[filter,:,:]

    selected_geodesics_in_sample_space_vectorized=np.reshape(np.transpose(selected_geodesics_in_sample_space, (2, 0, 1)),(n_geodesics *n_interp_selected, dim_data) )
    # first all interpolation points for first geodesic, then second, etc


    for j, generated_image in enumerate(selected_geodesics_in_sample_space_vectorized):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        # ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
        plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r' )
    plt.savefig('{}/geodesics_{}.png'.format(log_directory_geodesics, method), bbox_inches='tight' )
    plt.close()

    return None


