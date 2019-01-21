from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.geodesic_graph_poly_nn import *
from GAN.swiss_roll.utils.plotting import plot_geodesic_test
from GAN.swiss_roll.utils.generate_data import *

NN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Curve_net" )
saver = tf.train.Saver( NN_parameters )

GAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN" )
model_saver = tf.train.Saver( GAN_parameters )

with tf.Session() as session:
    session.run( tf.global_variables_initializer() )
    saver.restore( session, tf.train.latest_checkpoint( 'trained_NN/' ) )
    model_saver.restore( session, tf.train.latest_checkpoint( '../../GAN_Learning/trained_model/' ) )

    test_input=np.array([-0.1,0.8,0.4,0.5]).reshape(1,2*dim_latent)

    curves_in_latent_space_value, curves_in_sample_space_value = session.run([curves_in_latent_space,curves_in_sample_space],feed_dict={z_in : test_input})

    generate_real_samples = generate_real_data()
    real_samples = generate_real_samples.__next__()

    plot_geodesic_test( real_samples, curves_in_latent_space_value, curves_in_sample_space_value, "test")