from GAN.swiss_roll.geodesic_graph import *

with tf.variable_scope('geodesic'):

    disc_values_curves_sample_space_vectorized = Discriminator(curves_in_sample_space_vectorized)

    disc_values_curves_sample_space = tf.transpose(tf.reshape(disc_values_curves_sample_space_vectorized, shape=(n_geodesics, degree_polynomial_geodesic_latent + 1)), perm=(1, 0))

    curves_in_sample_space = tf.transpose(tf.reshape(curves_in_sample_space_vectorized, shape=(n_geodesics, degree_polynomial_geodesic_latent + 1, dim_data)),
                                          perm = [1,2,0])

    diff_square_vector = tf.reduce_sum(tf.square(curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :]), axis=1)


    objective_vector_proposed = tf.divide(diff_square_vector, disc_values_curves_sample_space[1:,:])
    objective_vector_Jacobian = diff_square_vector

    if penalty == True:
        geodesic_penalty = tf.reduce_max(diff_square_vector)  # maximum of norm difference in sample space
        penalty_hyper_param = 100.
    else:
        geodesic_penalty = 0
        penalty_hyper_param = 0

    geodesic_objective_function_proposed = tf.reduce_sum(objective_vector_proposed) + penalty_hyper_param * geodesic_penalty
    geodesic_objective_function_Jacobian = tf.reduce_sum(objective_vector_Jacobian) + penalty_hyper_param * geodesic_penalty




