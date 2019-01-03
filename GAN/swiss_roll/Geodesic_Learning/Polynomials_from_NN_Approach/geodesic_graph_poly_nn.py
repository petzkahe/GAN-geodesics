from GAN.swiss_roll.GAN_Learning.GAN_graph import *
from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.config_geodesics_poly_nn import *

import numpy as np

###########################################
#
#
#       PLACEHOLDER
#
###########################################

with tf.variable_scope("Geodesics"):
    z_in = tf.placeholder(tf.float32, shape=[None, 2 * dim_latent], name='z_in')


###########################################
#
#
#       NETWORK for path
#
###########################################


def parametrize_line(z_in, n_geodesic_interpolations):
    z_start = z_in[:, 0:2]
    z_end = z_in[:, 2:]

    constant_part = tf.reshape(z_start, shape=(1, dim_latent, tf.shape(z_in)[0]))

    linear_part = tf.reshape(z_end, shape=(1, dim_latent, tf.shape(z_in)[0])) - tf.reshape(z_start, shape=(
        1, dim_latent, tf.shape(z_in)[0]))

    coefficients_vector = tf.concat([constant_part, linear_part], axis=0)

    interpolation_matrix_entries = np.zeros( shape=(n_geodesic_interpolations + 1, 2))
    for i in range(n_geodesic_interpolations + 1):
        for j in range(2):
            interpolation_matrix_entries[i, j] = (float(i) / (n_geodesic_interpolations + 1)) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(n_geodesic_interpolations + 1, 2),
                                        dtype='float32' )

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, 2]),
        tf.transpose(interpolation_matrix, perm=[1, 0]))

    geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
                                         shape=[tf.shape(z_in)[0], dim_latent, n_geodesic_interpolations + 1] )
    geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0]) # shape = (n_geodesic_interpol+1, dim_latent, batch)

    geodesic_points_in_z_vectorized = tf.reshape(tf.transpose(geodesic_points_in_z, perm=[2, 0, 1] ),
                                                   shape=(
                                                       tf.shape( z_in )[0] * (n_geodesic_interpolations + 1),
                                                       dim_latent))

    return geodesic_points_in_z, geodesic_points_in_z_vectorized


def parametrize_curve(z_in, interpolation_degree, n_geodesic_interpolations,_coefficients_from_network):
    z_start = tf.transpose(z_in[:, 0:dim_latent]) #shape=(dim_latent,batch)
    z_end = tf.transpose(z_in[:, dim_latent:]) #shape=(dim_latent,batch)

    # shape(coefficients_from_network) = (batch, (interpol_degree-1)*dim_latent, where we assume that first come both dimensions for the quadratic degree coeffficients, then two dimensions for third degree coefficients,etc
    coefficients_from_network_matrix = tf.transpose(tf.reshape(coefficients_from_network, shape=(tf.shape(z_in)[0], interpolation_degree-1, dim_latent)), perm = [1,2,0]) # now shape=(interpol-1,dim_latent,batch)

    constant_part = tf.reshape(z_start, shape=(1, dim_latent, tf.shape(z_in)[0]))  # reshaping just shifts for for concatenation

    linear_part = tf.reshape(z_end - z_start - tf.reduce_sum(coefficients_from_network_matrix, axis=0), shape=(1, dim_latent, tf.shape(z_in)[0])) # reshaping just shifts for for concatenation

    coefficients_vector = tf.concat([constant_part, linear_part, coefficients_from_network_matrix], axis=0)  #shape=(interpol+1,dim_latent,batch)

    # Initialize parameter variable of size interpolation_degree times dimensions_noise space

    interpolation_matrix_entries = np.zeros(shape=(n_geodesic_interpolations + 1, interpolation_degree + 1))
    for i in range(n_geodesic_interpolations + 1 ):
        for j in range(interpolation_degree + 1 ):
            interpolation_matrix_entries[i, j] = (float(i) / (n_geodesic_interpolations + 1)) ** j
    interpolation_matrix = tf.constant(interpolation_matrix_entries,
                                        shape=(n_geodesic_interpolations + 1, interpolation_degree + 1),
                                        dtype='float32')

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, interpolation_degree + 1]),
        tf.transpose(interpolation_matrix, perm=[1, 0])) #shape=matmul(shape((batch * dim_latent), interpol+1),shape((interpolation_degree + 1, n_geodesic_interpolations + 1))= shape(batch*dim_latent,n_geodesic_interp+1) where  first come all interpolation points for for all dimensions of batch1, then all dimesnions for batch 2, etc

    geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
                                         shape=[tf.shape( z_in )[0], dim_latent, n_geodesic_interpolations + 1])
    geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0]) # shape=(n_geodesic_interpol+1, dim_latent, batch)

    geodesic_points_in_z_vectorized = tf.reshape( tf.transpose(geodesic_points_in_z, perm=[2, 0, 1]),
                                                  shape=[
                                                      tf.shape(z_in)[0] * (n_geodesic_interpolations + 1),
                                                      dim_latent]) # where first come all interpolation points for first batch item, then for second batch item, etc

    return geodesic_points_in_z, geodesic_points_in_z_vectorized


def Curve_net(z_in):
    with tf.variable_scope( "Curve_net", reuse=tf.AUTO_REUSE ):
        output = layer.ReLuLayer(dim_latent * 2, dim_nn_curve_net, z_in, "Layer.1" )
        output = layer.ReLuLayer(dim_nn_curve_net, dim_nn_curve_net, output, "Layer.2" )
        output = layer.LinearLayer(dim_nn_curve_net, (degree_polynomial_geodesic_latent - 1) * dim_latent, output,
                                    "Layer.3" )
    return output


###########################################
#
#
#       Curves in latent and sample space
#
###########################################


lines_in_latent_space, lines_in_latent_space_vectorized = parametrize_line(z_in, n_interpolations_points_geodesic)

coefficients_from_network = Curve_net(z_in)
curves_in_latent_space, curves_in_latent_space_vectorized = parametrize_curve(z_in, degree_polynomial_geodesic_latent,
                                                                               n_interpolations_points_geodesic,
                                                                               coefficients_from_network)

with tf.variable_scope( "GAN" ):
    curves_in_sample_space_vectorized = Generator(curves_in_latent_space_vectorized) #shape=(batch*n_geodesic_interp, dim_data)
    disc_values_curves_sample_space_vectorized = Discriminator(curves_in_sample_space_vectorized)

    lines_in_sample_space_vectorized = Generator(lines_in_latent_space_vectorized)

disc_values_curves_sample_space = tf.transpose(
    tf.reshape(disc_values_curves_sample_space_vectorized,
                shape=(tf.shape(z_in)[0], n_interpolations_points_geodesic + 1)),
    perm=(1, 0)) # shape=(n_interpolations_points_geodesic + 1,batch)

curves_in_sample_space = tf.transpose(tf.reshape(curves_in_sample_space_vectorized, shape=(
    tf.shape( z_in )[0], n_interpolations_points_geodesic + 1, dim_data)),
                                       perm=[1, 2, 0]) # shape=(n_interpol_points_geodesic+1, dim_data, batch)

lines_in_sample_space = tf.transpose(
    tf.reshape(lines_in_sample_space_vectorized,
                shape=(tf.shape( z_in )[0], n_interpolations_points_geodesic + 1, dim_data)),
    perm=[1, 2, 0] )

###########################################
#
#
#       LOSS
#
###########################################


diff_square_vector = tf.reduce_sum(tf.square(curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :]),
                                    axis=1) #shape=(n_interp_points_geodesic, batch)

small_eps = 0.01

disc_values_curves_sample_space = tf.exp(tf.multiply(0.5, tf.add(tf.log(disc_values_curves_sample_space[1:, :]), tf.log(disc_values_curves_sample_space[:-1, :]))))
denominator = tf.clip_by_value(tf.add(disc_values_curves_sample_space, small_eps), small_eps, 0.4 + small_eps)

denominator = tf.multiply(denominator, denominator)

objective_vector_proposed = tf.divide(diff_square_vector, denominator)

objective_vector_Jacobian = diff_square_vector

if penalty == True:
    geodesic_penalty = tf.reduce_max( diff_square_vector )  # maximum of norm difference in sample space
    penalty_hyper_param = 100.
else:
    geodesic_penalty = 0
    penalty_hyper_param = 0

geodesic_objective_per_geodesic_proposed = tf.reduce_sum(objective_vector_proposed, axis=0)
geodesic_objective_function_proposed = tf.reduce_sum(
    geodesic_objective_per_geodesic_proposed) + penalty_hyper_param * geodesic_penalty  # + \

geodesic_objective_per_geodesic_Jacobian = tf.reduce_sum( objective_vector_Jacobian, axis=0)
geodesic_objective_function_Jacobian = tf.reduce_sum(
    geodesic_objective_per_geodesic_Jacobian) + penalty_hyper_param * geodesic_penalty

all_variables = tf.trainable_variables()
parameters_curve_net = [entry for entry in all_variables if "Curve_net" in entry.name]
