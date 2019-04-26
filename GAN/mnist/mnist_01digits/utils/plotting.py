import matplotlib
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.mlab import griddata


matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.mnist.mnist_01digits.Geodesic_Learning.config_geodesic_mnist import *
from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *
from GAN.mnist.mnist_01digits.Geodesic_Learning.config_geodesic_mnist import *
from GAN.mnist.mnist_01digits.main_config import *


def plot_sample_space_01(samples,disc, iteration_step,_dir):
    plt.figure( figsize=(12, 10) )
    gs = gridspec.GridSpec( 5, 5 )
    for j, generated_image in enumerate( samples ):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        ax.set_title( 'D(x,z) = {}'.format( int(disc[j]*100)/100.0 ) )
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r')
        #c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
        plt.colorbar(c)
    plt.savefig('{}/frame_{}.png'.format(_dir + log_directory_01, iteration_step), bbox_inches='tight' )
    plt.close()

    return None

def plot_geodesic(geodesics_in_sample_space, disc_values, method, _dir):

    plt.clf()

    plt.figure( figsize=(15, 10))
    gs = gridspec.GridSpec(n_geodesics,n_interp_selected)

    filter = [j*increment for j in range(n_interp_selected-1)]+[n_interpolations_points_geodesic]


    selected_geodesics_in_sample_space = geodesics_in_sample_space[filter,:,:]
    selected_disc_values = disc_values[filter,:]

    selected_geodesics_in_sample_space_vectorized=np.reshape(np.transpose(selected_geodesics_in_sample_space, (2, 0, 1)),(n_geodesics *n_interp_selected, dim_data) )
    # first all interpolation points for first geodesic, then second, etc
    selected_disc_values_vectorized = np.reshape(np.transpose(selected_disc_values, (1,0)), (n_geodesics*n_interp_selected))

    for j, generated_image in enumerate(selected_geodesics_in_sample_space_vectorized):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )

        if np.isnan(selected_disc_values_vectorized[j]):
            ax.set_title( "NaN")
        else:
            ax.set_title(int( selected_disc_values_vectorized[j] * 100 ) / 100.0)

        # ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r')
        plt.colorbar(c)
        #plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
    
    plt.savefig('{}/geodesics_{}.png'.format(_dir, method), bbox_inches='tight' )
    plt.close()

    return None

def plot_geodesics_in_pca_space(curves,method,geodesics_suppl_dict, _dir):

    reals,labels = geodesics_suppl_dict['reals']
    latent_background_pca,latent_background_discriminator = geodesics_suppl_dict["latent_background"]


    plt.clf()

    fig,ax = plt.subplots(figsize=(15, 10))
    #mesh = ax.pcolormesh(np.linspace(pca_grid_minima[0], pca_grid_maxima[0], n_pca_grid_per_dimension),
    #               np.linspace(pca_grid_minima[1], pca_grid_maxima[1], n_pca_grid_per_dimension),
    #               np.transpose(background), cmap='gray',vmin=0,vmax=.8)
    #plt.colorbar(mesh)

    x = latent_background_pca[:,0]
    y = latent_background_pca[:,1]
    z = latent_background_discriminator[:,0]
    xi = np.linspace(min(x),max(x),n_pca_grid_per_dimension)
    yi = np.linspace(min(y),max(y),n_pca_grid_per_dimension)
    zi = griddata(x,y,z,xi,yi,interp='linear')
    #CS = ax.contour(xi, yi, zi, 5, linewidths=0.5, colors='k')
    #CS = ax.contourf(xi, yi, zi, 100, vmax=(1), vmin=0)
    CS = ax.contourf(xi, yi, zi, 100)
    fig.colorbar(CS)  # draw colorbar


    #plot reals and labels
    ax.scatter(reals[:,0],reals[:,1],c=labels,s=5,cmap='inferno')
    #for i, txt in enumerate(labels):
    #    ax.annotate(txt, (reals[i,0], reals[i,1]))

    #plot curves
    for i in range(1,curves.shape[2]):
        plt.plot(curves[:, 0, i], curves[:, 1, i],
                    'r.-')
    #plt.scatter(geodesics_in_latent[:, 0, 0], geodesics_in_latent[:, 1, 0],
    #                color='yellow', marker='.', s=1)
    plt.plot(curves[:, 0, 0], curves[:, 1, 0], 'k.-')
    
    plt.savefig('{}/geodesics_in_pca_{}.png'.format(_dir,method), bbox_inches='tight' )


