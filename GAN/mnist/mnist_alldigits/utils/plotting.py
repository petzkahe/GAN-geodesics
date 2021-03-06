import matplotlib
import numpy as np
import matplotlib.gridspec as gridspec
import os

matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.mnist.mnist_alldigits.Geodesic_Learning.config_geodesic_mnist import *
from GAN.mnist.mnist_alldigits.BIGAN_Learning.config_BIGAN import *
from GAN.mnist.mnist_alldigits.Geodesic_Learning.config_geodesic_mnist import *


def plot_sample_space(samples, iteration_step):
    plt.figure( figsize=(12, 10) )
    gs = gridspec.GridSpec( 5, 5 )
    for j, generated_image in enumerate( samples ):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        #ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
        plt.colorbar(c)
    plt.savefig('{}/frame_{}.png'.format(log_directory, iteration_step), bbox_inches='tight' )
    plt.close()

    return None

def plot_sample_space_01(samples, iteration_step):
    plt.figure( figsize=(12, 10) )
    gs = gridspec.GridSpec( 5, 5 )
    for j, generated_image in enumerate( samples ):
        ax = plt.subplot( gs[j] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        #ax.set_title( 'guess = {}, true = {}'.format( arg_max, true ) )
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
        plt.colorbar(c)
    plt.savefig('{}/frame_{}.png'.format(log_directory_01, iteration_step), bbox_inches='tight' )
    plt.close()

    return None

def plot_geodesic(geodesics_in_sample_space, method):

    plt.clf()

    plt.figure( figsize=(15, 10))
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
        c = plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r')
        plt.colorbar(c)
        #plt.imshow( generated_image.reshape( 28, 28 ), cmap='Greys_r', vmin=0, vmax=1)
    
    plt.savefig('{}/geodesics_{}.png'.format(log_directory_geodesics, method), bbox_inches='tight' )
    plt.close()

    return None

def plot_geodesics_in_pca_space(curves,method,reals,labels):

    plt.clf()
    #plot reals and labels
    fig,ax = plt.subplots(figsize=(15, 10))
    ax.scatter(reals[:,0],reals[:,1],c=labels,s=10)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (reals[i,0], reals[i,1]))

    #plot curves
    for i in range(1,curves.shape[2]):
        plt.plot(curves[:, 0, i], curves[:, 1, i],
                    'r.-')
    #plt.scatter(geodesics_in_latent[:, 0, 0], geodesics_in_latent[:, 1, 0],
    #                color='yellow', marker='.', s=1)
    plt.plot(curves[:, 0, 0], curves[:, 1, 0], 'k.-')
    
    plt.savefig('{}/geodesics_in_pca_{}.png'.format(log_directory_geodesics,method), bbox_inches='tight' )





def make_videos(video_dict):
    # video_dict[method] = [video_frames_sample]

    for n in range(n_geodesics):
        frame={}

        for i  in range(n_video_frames + 1):
            for method in methods:

                frame[method] = np.reshape(  np.array(video_dict[method])[0,i,:,n], (dim_data)) # video_dict[method] of shape = 1, n_video_frames + 1, dim_data, batch

            plt.clf()
            plt.figure( figsize=(10, 7))

            for j, method in enumerate(methods):
                ax = plt.subplot(1,3,j+1)
                #ax.set_xticks([])
                #ax.set_yticks([])
                ax.set_title( '{}'.format(method))
                plt.imshow(frame[method].reshape(28, 28), cmap='Greys_r')

            if not os.path.exists( '{}/video_frames/geodesic_{}'.format( log_directory_geodesics, n) ):
                os.makedirs( '{}/video_frames/geodesic_{}'.format( log_directory_geodesics, n) )


            if i in range(10):
                plt.savefig( '{}/video_frames/geodesic_{}/image-00{}.png'.format( log_directory_geodesics, n, i ), bbox_inches='tight' )
            elif i in range(10,100):
                plt.savefig( '{}/video_frames/geodesic_{}/image-0{}.png'.format( log_directory_geodesics, n, i ), bbox_inches='tight' )
            else:
                plt.savefig( '{}/video_frames/geodesic_{}/image-{}.png'.format( log_directory_geodesics,n, i ), bbox_inches='tight' )
            plt.close()


        os.system("ffmpeg -framerate {} -i {}/video_frames/geodesic_{}/image-%03d.png {}/video_{}.webm".format(25,log_directory_geodesics, n ,log_directory_geodesics, n))
        print("Video {} done".format(n))
    return None


    






