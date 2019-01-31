from GAN.mnist.mnist_alldigits.utils.plotting import make_videos
from GAN.mnist.mnist_alldigits.Geodesic_Learning.geodesic_graph_mnist import *


video_dict={}

start = np.random.uniform( low=latent_min_value, high=latent_max_value, size=[1, dim_latent, n_geodesics] ).astype('float32' )
end = np.random.uniform( low=latent_min_value, high=latent_max_value, size=[1, dim_latent, n_geodesics] ).astype('float32' )


BIGAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="BIGAN" )
model_saver = tf.train.Saver(BIGAN_parameters)
    
    # To save the polynomial coefficients
coefficient_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Geodesics" )
coefficients_saver = tf.train.Saver( coefficient_parameters )

with tf.Session() as sess:
    
    

    for method in methods:
        sess.run( tf.global_variables_initializer() )

        model_saver.restore( sess, tf.train.latest_checkpoint( '../BIGAN_Learning/trained_model/' ) )

        coefficients_saver.restore(sess, 'trained_polynomials/{}'.format( method ) )
        video_frames_sample= sess.run(video_frames_in_sample_space, feed_dict={z_start: start, z_end: end} )
        video_dict[method] = [video_frames_sample]

make_videos(video_dict)