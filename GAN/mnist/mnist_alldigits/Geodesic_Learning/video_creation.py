from GAN.mnist.mnist_alldigits.utils.plotting import make_videos
from GAN.mnist.mnist_alldigits.Geodesic_Learning.geodesic_graph_mnist import *


video_dict={}

start, end = np.load('trained_polynomials/latent_start_end_points.npy')


BIGAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="BIGAN" )
model_saver = tf.train.Saver(BIGAN_parameters)
    

coefficient_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Geodesics" )
coefficients_saver = tf.train.Saver( coefficient_parameters )

with tf.Session() as sess:

    sess.run( tf.global_variables_initializer() )

    model_saver.restore( sess, tf.train.latest_checkpoint( '../BIGAN_Learning/trained_model/' ) )


    for method in methods:

        coefficients_saver.restore(sess, 'trained_polynomials/{}/{}'.format( method, method ) )
        video_frames_sample= sess.run(video_frames_in_sample_space, feed_dict={z_start: start, z_end: end} )
        video_dict[method] = [video_frames_sample]

make_videos(video_dict)