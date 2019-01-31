from GAN.swiss_roll.GAN_Learning.GAN_graph import *
from GAN.swiss_roll.Geodesic_Learning.Neural_Network_Approach.config_geodesics_nn import *

import numpy as np

###########################################
#
#
#       PLACEHOLDER
#
###########################################

with tf.variable_scope( "Geodesics" ):
    z_in = tf.placeholder( tf.float32, shape=[None, 2 * dim_latent], name='z_in' )


###########################################
#
#
#       NETWORK for path
#
###########################################


def parametrize_line(z_in, n_geodesic_interpolations):
    z_start = tf.transpose(z_in[:,0:2])
    z_end = tf.transpose(z_in[:,2:])

    constant_part = tf.reshape( z_start, shape=(1, dim_latent, tf.shape(z_in)[0]) )

    linear_part = tf.reshape( z_end, shape=(1, dim_latent, tf.shape(z_in)[0]) ) - tf.reshape( z_start, shape=(
        1, dim_latent, tf.shape(z_in)[0]) )

    coefficients_vector = tf.concat( [constant_part, linear_part], axis=0 )

    interpolation_matrix_entries = np.zeros( shape=(n_geodesic_interpolations + 1, 2) )
    for i in range( n_geodesic_interpolations + 1 ):
        for j in range( 2 ):
            interpolation_matrix_entries[i, j] = (float( i ) / (n_geodesic_interpolations + 1)) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(n_geodesic_interpolations + 1, 2),
                                        dtype='float32' )

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape( tf.transpose( coefficients_vector, perm=[2, 1, 0] ), shape=[-1, 2] ),
        tf.transpose( interpolation_matrix, perm=[1, 0] ) )

    geodesic_points_in_z_t = tf.reshape( geodesic_points_in_z_matrix,
                                         shape=[tf.shape(z_in)[0], dim_latent, n_geodesic_interpolations + 1] )
    geodesic_points_in_z = tf.transpose( geodesic_points_in_z_t, perm=[2, 1, 0] )

    # _n_batch = tf.shape(z_in)[0]
    # print(_n_batch.get_shape().as_list())
    # z_interp_init = np.zeros(tf.shape(z_in)[0].value,dim_latent,n_geodesic_interpolations+1)
    # z_interp.shape()
    # z_interp = tf.constant(z_interp_init)
    # t_vec = tf.constant(np.linspace(0,1,n_geodesic_interpolations+1))
    # for t in range(t_vec):
    #     z_interp[:,:,t] = tf.add( tf.multiply(z_start,t) , tf.multiply(z_end,(1-t)) )

    # geodesic_points_in_z = tf.transpose(z_interp,perm=[2,1,0])



    return geodesic_points_in_z


def Curve_net(z_in):
    with tf.variable_scope( "Curve_net", reuse=tf.AUTO_REUSE ):
        output = layer.ReLuLayer( dim_latent * 2, dim_nn_curve_net, z_in, "Layer.1" )
        output = layer.ReLuLayer( dim_nn_curve_net, dim_nn_curve_net, output, "Layer.2" )
        output = layer.LinearLayer( dim_nn_curve_net, 2 * (n_interpolations_points_geodesic - 1), output, "Layer.3" )
        output = tf.multiply(tf.add(tf.nn.sigmoid(output),-0.5),2)

    return output


# constant_part = tf.reshape(z_start, shape=(1, dim_latent, n_geodesics))

# if do_shared_variables:
#     linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
#         1, dim_latent, n_geodesics)) - tf.reshape(tf.reduce_sum(coefficients_shared,
#                                                              axis=0), shape=(1, dim_latent, n_geodesics))
#     print(1)
#     coefficients_vector = tf.concat([constant_part, linear_part, coefficients_shared], axis=0)

# else:
#     linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
#         1, dim_latent, n_geodesics)) - tf.reshape(tf.reduce_sum(coefficients,
#                                                              axis=0), shape=(1, dim_latent, n_geodesics))

#     coefficients_vector = tf.concat([constant_part, linear_part, coefficients], axis=0)

# # Initialize parameter variable of size interpolation_degree times dimensions_noise space

# interpolation_matrix_entries = np.zeros(shape=(n_geodesic_interpolations + 1, interpolation_degree + 1))
# for i in range(n_geodesic_interpolations + 1):
#     for j in range(interpolation_degree + 1):
#         interpolation_matrix_entries[i, j] = (float(i) / (n_geodesic_interpolations + 1)) ** j
# interpolation_matrix = tf.constant(interpolation_matrix_entries,
#                                     shape=(n_geodesic_interpolations + 1, interpolation_degree + 1),
#                                    dtype='float32')

# geodesic_points_in_z_matrix = tf.matmul(
#     tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, interpolation_degree + 1]),
#     tf.transpose(interpolation_matrix, perm=[1, 0]))

# geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
#                                     shape=[n_geodesics, dim_latent, n_geodesic_interpolations + 1])
# geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0])


def reformat_curve(_z_in, _z_out):
    curve = tf.concat( [_z_in[:, 0:2], _z_out, _z_in[:, 2:]], axis=1 )

    geodesic_points_in_z = tf.transpose( tf.reshape( curve, shape=[tf.shape(curve)[0], n_interpolations_points_geodesic + 1, dim_latent] ),
                                         perm=[1, 2, 0] )

    geodesic_points_in_z_vectorized = tf.reshape( tf.transpose( geodesic_points_in_z, perm=[2, 0, 1] ),
                                                  shape=[tf.shape(curve)[0]*(n_interpolations_points_geodesic+1), dim_latent] )

    return geodesic_points_in_z, geodesic_points_in_z_vectorized


###########################################
#
#
#       Curves in latent and sample space
#
###########################################


lines_in_latent_space = parametrize_line(z_in, n_interpolations_points_geodesic )
lines_in_latent_space_vectorized = tf.reshape( tf.transpose( lines_in_latent_space, perm=[2, 0, 1] ),
                                               shape=(
                                               tf.shape(z_in)[0] * (n_interpolations_points_geodesic + 1), dim_latent) )

curves_from_network = Curve_net( z_in )
curves_in_latent_space, curves_in_latent_space_vectorized = reformat_curve( z_in, curves_from_network )

with tf.variable_scope( "GAN" ):
    curves_in_sample_space_vectorized = Generator( curves_in_latent_space_vectorized )
    disc_values_curves_sample_space_vectorized = Discriminator( curves_in_sample_space_vectorized )

    lines_in_sample_space_vectorized = Generator( lines_in_latent_space_vectorized )

disc_values_curves_sample_space = tf.transpose(
    tf.reshape( disc_values_curves_sample_space_vectorized, shape=(tf.shape(z_in)[0], n_interpolations_points_geodesic + 1) ),
    perm=(1, 0) )

curves_in_sample_space = tf.transpose( tf.reshape( curves_in_sample_space_vectorized, shape=(
tf.shape(z_in)[0], n_interpolations_points_geodesic + 1, dim_data) ),
                                       perm=[1, 2, 0] )

lines_in_sample_space = tf.transpose(
    tf.reshape( lines_in_sample_space_vectorized, shape=(tf.shape(z_in)[0], n_interpolations_points_geodesic + 1, dim_data) ),
    perm=[1, 2, 0] )

###########################################
#
#
#       LOSS
#
###########################################


diff_square_vector = tf.reduce_sum( tf.square( curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :] ),
                                    axis=1 )
diff_square_vector_latent = tf.reduce_sum(tf.square(curves_in_latent_space[1:, :, :] - curves_in_latent_space[:-1, :, :]), axis=1)


diff_square_mean, diff_square_variance = tf.nn.moments(diff_square_vector,axes=[0]) 
small_eps = 0.01

if True:
    disc_values_curves_sample_space = tf.exp( tf.multiply( 0.5,
                                                           tf.add( tf.log( disc_values_curves_sample_space[1:, :] ),
                                                                   tf.log(
                                                                       disc_values_curves_sample_space[:-1, :] ) ) ) )
    denominator = tf.clip_by_value( tf.add( disc_values_curves_sample_space, small_eps ), small_eps, 0.4 + small_eps )
else:
    denominator = tf.clip_by_value( tf.add( disc_values_curves_sample_space[1:, :], small_eps ), small_eps,
                                    0.4 + small_eps )

# denominator = tf.Print(denominator,[denominator])


denominator = tf.multiply( denominator, denominator )

# objective_vector_proposed = tf.divide(1, denominator)
# objective_vector_proposed = tf.divide(diff_square_vector_latent, denominator)

# hyper_lambda = 100000.0
objective_vector_proposed = tf.divide( diff_square_vector, denominator )

objective_vector_Jacobian = diff_square_vector
# objective_vector_Jacobian = diff_square_vector_latent

# if method == "proposed"


if penalty == True:
    geodesic_penalty = tf.reduce_max( diff_square_vector )  # maximum of norm difference in sample space
    penalty_hyper_param = 100.
else:
    geodesic_penalty = 0
    penalty_hyper_param = 0

geodesic_objective_per_geodesic_proposed = tf.reduce_sum( objective_vector_proposed, axis=0 )
# geodesic_objective_per_geodesic_proposed = tf.Print(geodesic_objective_per_geodesic_proposed,[geodesic_objective_per_geodesic_proposed])
geodesic_objective_function_proposed = tf.reduce_sum(
    geodesic_objective_per_geodesic_proposed ) + penalty_hyper_param * geodesic_penalty #+ \
 #   tf.multiply(hyper_lambda,tf.reduce_sum(diff_square_variance))
# geodesic_objective_function_proposed = tf.Print(geodesic_objective_function_proposed,[geodesic_objective_function_proposed])

# geodesic_objective_function_proposed = tf.reduce_sum(geodesic_objective_function_proposed) + penalty_hyper_param * geodesic_penalty

geodesic_objective_per_geodesic_Jacobian = tf.reduce_sum( objective_vector_Jacobian, axis=0 )
# geodesic_objective_per_geodesic_Jacobian = tf.Print(geodesic_objective_per_geodesic_Jacobian,[geodesic_objective_per_geodesic_Jacobian])
geodesic_objective_function_Jacobian = tf.reduce_sum(
    geodesic_objective_per_geodesic_Jacobian ) + penalty_hyper_param * geodesic_penalty 
    #+ tf.multiply(hyper_lambda,tf.reduce_sum(diff_square_variance))
# geodesic_objective_function_Jacobian = tf.reduce_sum(geodesic_objective_function_Jacobian) + penalty_hyper_param * geodesic_penalty


# tf.summary.scalar("geodesic_objective_function_proposed",geodesic_objective_function_proposed)
# for iter in range(min(n_batch_size_curve_net,10)):
#     tf.summary.scalar("geodesic_objective_per_geodesic_proposed_" + str(iter),geodesic_objective_per_geodesic_proposed[iter])


all_variables = tf.trainable_variables()
parameters_curve_net = [entry for entry in all_variables if "Curve_net" in entry.name]
