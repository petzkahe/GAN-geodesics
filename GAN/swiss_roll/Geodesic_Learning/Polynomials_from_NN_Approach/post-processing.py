from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.geodesic_graph_poly_nn import *
from GAN.swiss_roll.utils.plotting import plot_geodesic
from GAN.swiss_roll.utils.generate_data import *
from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.compute_geodesics_poly_nn import create_grid

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

    # Calculate discriminator background for sample space
    sample_grid_vectorized = create_grid( sample_grid_minima, sample_grid_maxima, n_discriminator_grid_sample )
    [disc_values_over_sample_grid_vectorized] = session.run( [disc_values_on_real],
                                                             feed_dict={data_real: sample_grid_vectorized} )
    disc_values_over_sample_grid = disc_values_over_sample_grid_vectorized.reshape(
        (n_discriminator_grid_sample, n_discriminator_grid_sample) )

    # Calculate discriminator background for latent space
    latent_grid_vectorized = create_grid( latent_grid_minima, latent_grid_maxima, n_discriminator_grid_latent )
    [disc_values_over_latent_grid_vectorized] = session.run( [disc_values_on_generated],
                                                             feed_dict={data_latent: latent_grid_vectorized} )
    disc_values_over_latent_grid = disc_values_over_latent_grid_vectorized.reshape(
        (n_discriminator_grid_latent, n_discriminator_grid_latent) )


    suppl_dict={}
    suppl_dict["disc_values_over_sample_grid"] = disc_values_over_sample_grid
    suppl_dict["disc_values_over_latent_grid"] = disc_values_over_latent_grid

    generate_real_samples = generate_real_data()
    real_samples = generate_real_samples.__next__()


    plot_geodesic( real_samples, curves_in_latent_space_value, curves_in_sample_space_value, "test", suppl_dict)