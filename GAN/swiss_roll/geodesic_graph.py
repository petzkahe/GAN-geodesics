from GAN.swiss_roll.GAN_graph import *
from GAN.swiss_roll.config_geodesics import *

import numpy as np

# run gan-graph here?

with tf.variable_scope("Geodesics"):

    z_start = tf.placeholder(tf.float32, shape=[1, dim_latent, n_geodesics], name='z_start')
    z_end = tf.placeholder(tf.float32, shape=[1, dim_latent, n_geodesics], name='z_end')

    coefficients_initializations = np.zeros(shape=(degree_polynomial_geodesic_latent - 1, dim_latent, n_geodesics), dtype='float32')
    coefficients = tf.Variable(initial_value=coefficients_initializations, name='coefficients')


    def parametrize_curve(z_start, z_end, interpolation_degree, no_geodesic_interpolations):

        constant_part = tf.reshape(z_start, shape=(1, dim_latent, n_geodesics))
        linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
            1, dim_latent, n_geodesics)) - tf.reshape(tf.reduce_sum(coefficients,
                                                                    axis=0), shape=(1, dim_latent, n_geodesics))

        coefficients_vector = tf.concat([constant_part, linear_part, coefficients], axis=0)

        # Initialize parameter variable of size interpolation_degree times dimensions_noise space

        interpolation_matrix_entries = np.zeros(shape=(no_geodesic_interpolations + 1, interpolation_degree + 1))
        for i in range(no_geodesic_interpolations + 1):
            for j in range(interpolation_degree + 1):
                interpolation_matrix_entries[i, j] = (float(i) / no_geodesic_interpolations) ** j
        interpolation_matrix = tf.constant(interpolation_matrix_entries,
                                           shape=(no_geodesic_interpolations + 1, interpolation_degree + 1),
                                           dtype='float32')

        geodesic_points_in_z_matrix = tf.matmul(
            tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, interpolation_degree + 1]),
            tf.transpose(interpolation_matrix, perm=[1, 0]))

        geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
                                            shape=[n_geodesics, dim_latent, no_geodesic_interpolations + 1])
        geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0])

        return geodesic_points_in_z


    curves_in_latent_space = parametrize_curve(z_start, z_end, degree_polynomial_geodesic_latent, n_interpolations_points_geodesic)
    curves_in_latent_space_vectorized = tf.reshape(tf.transpose(curves_in_latent_space, perm=[2, 0, 1]),
                                                   shape=(n_geodesics * (n_interpolations_points_geodesic + 1), dim_latent))

with tf.variable_scope("GAN"):

    curves_in_sample_space_vectorized = Generator(curves_in_latent_space_vectorized)
    disc_values_curves_sample_space_vectorized = Discriminator(curves_in_sample_space_vectorized)

with tf.variable_scope("Geodesics"):

    disc_values_curves_sample_space = tf.transpose(tf.reshape(disc_values_curves_sample_space_vectorized, shape=(n_geodesics, n_interpolations_points_geodesic + 1)), perm=(1, 0))

    curves_in_sample_space = tf.transpose(tf.reshape(curves_in_sample_space_vectorized, shape=(n_geodesics,n_interpolations_points_geodesic+ 1, dim_data)),
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
