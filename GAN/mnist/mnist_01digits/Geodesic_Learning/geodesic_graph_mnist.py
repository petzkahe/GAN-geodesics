from GAN.mnist.mnist_01digits.BIGAN_Learning.BIGAN_graph import *
from GAN.mnist.mnist_01digits.Geodesic_Learning.config_geodesic_mnist import *
import numpy as np


def safe_log(x):
    return tf.log( x + 1e-8 )

with tf.variable_scope( "Geodesics" ):
    z_start = tf.placeholder( tf.float32, shape=[1, dim_latent, n_geodesics], name='z_start' )
    z_end = tf.placeholder( tf.float32, shape=[1, dim_latent, n_geodesics], name='z_end' )

    if sampling_geodesic_coefficients == "zeros":
        coefficients_initializations = np.zeros(shape=(degree_polynomial_geodesic_latent - 1, dim_latent, n_geodesics),
                                                 dtype='float32' )
    elif sampling_geodesic_coefficients == "uniform":
        coefficients_initializations = np.random.uniform(-initialization_value_coefficients, initialization_value_coefficients, size=(
        degree_polynomial_geodesic_latent - 1, dim_latent, n_geodesics) ).astype( "float32" )
    else:
        raise Exception(
            "sampling method {} for geodesic coefficients unknown".format( sampling_geodesic_coefficients ) )

    coefficients = tf.Variable( initial_value=coefficients_initializations, name='coefficients' )



def parametrize_line(z_start, z_end, n_geodesic_interpolations):
    constant_part = tf.reshape(z_start, shape=(1, dim_latent, n_geodesics))

    linear_part = tf.reshape( z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape( z_start, shape=(
        1, dim_latent, n_geodesics) )

    coefficients_vector = tf.concat( [constant_part, linear_part], axis=0 )

    interpolation_matrix_entries = np.zeros( shape=(n_geodesic_interpolations + 1, 2) )
    for i in range( n_geodesic_interpolations + 1 ):
        for j in range( 2 ):
            interpolation_matrix_entries[i, j] = (float( i ) / (n_geodesic_interpolations + 1)) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(n_geodesic_interpolations + 1, 2),
                                        dtype='float32')

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape( tf.transpose( coefficients_vector, perm=[2, 1, 0] ), shape=[-1, 2]),
        tf.transpose( interpolation_matrix, perm=[1, 0] ) )

    geodesic_points_in_z_t = tf.reshape( geodesic_points_in_z_matrix,
                                         shape=[n_geodesics, dim_latent, n_geodesic_interpolations + 1] )
    geodesic_points_in_z = tf.transpose( geodesic_points_in_z_t, perm=[2, 1, 0] )

    return geodesic_points_in_z


def parametrize_curve(z_start, z_end, interpolation_degree, n_geodesic_interpolations):
    factor1 = [2.0 ** i - 1.0 for i in range( 2, degree_polynomial_geodesic_latent + 1 )]
    factor1_tensor = tf.reshape( tf.constant( factor1 ), shape=(degree_polynomial_geodesic_latent - 1, 1, 1) )
    factor2 = [2.0 ** i - 2.0 for i in range( 2, degree_polynomial_geodesic_latent + 1 )]
    factor2_tensor = tf.reshape( tf.constant( factor2 ), shape=(degree_polynomial_geodesic_latent - 1, 1, 1) )

    linear_part = tf.reshape( z_end, shape=(1, dim_latent, n_geodesics) ) - tf.reshape( z_start, shape=(
    1, dim_latent, n_geodesics) ) - tf.reshape( tf.reduce_sum( tf.multiply( factor1_tensor, coefficients ), axis=0 ),
                                                shape=(1, dim_latent, n_geodesics) )

    constant_part = 2 * tf.reshape( z_start, shape=(1, dim_latent, n_geodesics) ) - tf.reshape( z_end, shape=(
    1, dim_latent, n_geodesics) ) + tf.reshape( tf.reduce_sum( tf.multiply( factor2_tensor, coefficients ), axis=0 ),
                                                shape=(1, dim_latent, n_geodesics) )

    coefficients_vector = tf.concat( [constant_part, linear_part, coefficients], axis=0 )

    # Initialize parameter variable of size interpolation_degree times dimensions_noise space

    interpolation_matrix_entries = np.zeros( shape=(n_geodesic_interpolations + 1, interpolation_degree + 1) )
    for i in range( n_geodesic_interpolations + 1 ):
        for j in range( interpolation_degree + 1 ):
            interpolation_matrix_entries[i, j] = (1.0 + float( i ) / (n_geodesic_interpolations + 1)) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(n_geodesic_interpolations + 1, interpolation_degree + 1),
                                        dtype='float32' )

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape( tf.transpose( coefficients_vector, perm=[2, 1, 0] ), shape=[-1, interpolation_degree + 1] ),
        tf.transpose( interpolation_matrix, perm=[1, 0] ) )

    geodesic_points_in_z_t = tf.reshape( geodesic_points_in_z_matrix,
                                         shape=[n_geodesics, dim_latent, n_geodesic_interpolations + 1] )
    geodesic_points_in_z = tf.transpose( geodesic_points_in_z_t, perm=[2, 1, 0] )

    return geodesic_points_in_z


curves_in_latent_space = parametrize_curve( z_start, z_end, degree_polynomial_geodesic_latent,
                                            n_interpolations_points_geodesic )
curves_in_latent_space_vectorized = tf.reshape( tf.transpose( curves_in_latent_space, perm=[2, 0, 1] ),
                                                shape=(
                                                n_geodesics * (n_interpolations_points_geodesic + 1), dim_latent) )

lines_in_latent_space = parametrize_line( z_start, z_end, n_interpolations_points_geodesic )
lines_in_latent_space_vectorized = tf.reshape( tf.transpose( lines_in_latent_space, perm=[2, 0, 1] ),
                                               shape=(
                                               n_geodesics * (n_interpolations_points_geodesic + 1), dim_latent) )

with tf.variable_scope("BIGAN",reuse=True):
    curves_in_sample_space_vectorized = Generator( curves_in_latent_space_vectorized )
    disc_values_curves_sample_space_vectorized = Discriminator(curves_in_sample_space_vectorized, curves_in_latent_space_vectorized)

    lines_in_sample_space_vectorized = Generator( lines_in_latent_space_vectorized )

disc_values_curves_sample_space = tf.transpose(
    tf.reshape( disc_values_curves_sample_space_vectorized, shape=(n_geodesics, n_interpolations_points_geodesic + 1) ),
    perm=(1, 0) )

curves_in_sample_space = tf.transpose( tf.reshape( curves_in_sample_space_vectorized, shape=(
n_geodesics, n_interpolations_points_geodesic + 1, dim_data) ),
                                       perm=[1, 2, 0] )

lines_in_sample_space = tf.transpose(
    tf.reshape( lines_in_sample_space_vectorized, shape=(n_geodesics, n_interpolations_points_geodesic + 1, dim_data) ),
    perm=[1, 2, 0] )

diff_square_vector = tf.reduce_sum( tf.square( curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :] ),
                                    axis=1 )
diff_square_vector_linear = tf.reduce_sum( tf.square( lines_in_sample_space[1:, :, :] - lines_in_sample_space[:-1, :, :] ),
                                    axis=1 )

# diff_abs_vector = tf.reduce_sum(tf.abs(curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :]), axis=1)

diff_square_vector_latent = tf.reduce_sum(
    tf.square( curves_in_latent_space[1:, :, :] - curves_in_latent_space[:-1, :, :] ), axis=1 )

# penalty for leaving domain
#out_of_domain_penalty = tf.add(
#    tf.exp( tf.clip_by_value( tf.add( tf.abs( curves_in_latent_space ), -1 ), 0, np.infty ) ), -1 )

small_eps = 0.01

disc_values_curves_sample_space = tf.exp( tf.multiply( 0.5, tf.add( safe_log( disc_values_curves_sample_space[1:, :] ),
                                                                    safe_log(
                                                                        disc_values_curves_sample_space[:-1, :] ) ) ) )
denominator = tf.clip_by_value( tf.add( disc_values_curves_sample_space, small_eps ), small_eps, 0.8 + small_eps )

# denominator = tf.Print(denominator,[denominator])


denominator = tf.multiply( denominator, denominator )

# objective_vector_proposed = tf.divide(1, denominator)
# objective_vector_proposed = tf.divide(diff_square_vector_latent, denominator)

# objective_vector_proposed = tf.divide(diff_square_vector,denominator)
objective_vector_proposed = hyper_lambda*(0.8 + small_eps) ** 2 / n_interpolations_points_geodesic * tf.divide( 1.0,
                                                                                                   denominator ) + tf.multiply(
    diff_square_vector, float( n_interpolations_points_geodesic ) )

objective_vector_Jacobian = tf.multiply( diff_square_vector, float( n_interpolations_points_geodesic ) )

objective_vector_linear = tf.multiply( diff_square_vector_linear, float( n_interpolations_points_geodesic ) )

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
    geodesic_objective_per_geodesic_proposed ) + penalty_hyper_param * geodesic_penalty # + tf.reduce_sum( out_of_domain_penalty )

geodesic_objective_per_geodesic_Jacobian = tf.reduce_sum( objective_vector_Jacobian, axis=0 )
geodesic_objective_function_Jacobian = tf.reduce_sum(
    geodesic_objective_per_geodesic_Jacobian ) + penalty_hyper_param * geodesic_penalty # + tf.reduce_sum( out_of_domain_penalty )


geodesic_objective_per_geodesic_linear = tf.reduce_sum( objective_vector_linear, axis=0 )

# Illustrate curves in PCA-space:

#   calculate pca for some real mnist samples of two/all classes
real_samples_for_pca = tf.placeholder(tf.float32,shape=[n_batch_pca,dim_data],name='real_samples_for_pca')
subspace_map = tf.placeholder(tf.float32,shape=[dim_data,dim_pca],name='subspace_map')
real_samples_in_pca_space = tf.matmul(real_samples_for_pca,subspace_map)

# Calculate pca for geodesics
curves_in_pca_space_vectorized = tf.matmul(curves_in_sample_space_vectorized,subspace_map)
curves_in_pca_space = tf.transpose( tf.reshape( curves_in_pca_space_vectorized, shape=(
n_geodesics, n_interpolations_points_geodesic + 1, dim_pca) ),
                                       perm=[1, 2, 0] )

lines_in_pca_space_vectorized = tf.matmul(lines_in_sample_space_vectorized,subspace_map)
lines_in_pca_space = tf.transpose( tf.reshape( lines_in_pca_space_vectorized, shape=(
n_geodesics, n_interpolations_points_geodesic + 1, dim_pca) ),
                                       perm=[1, 2, 0] )


gridpoints_in_pca_space = tf.placeholder(tf.float32,shape=[n_pca_gridpoints,dim_pca],name='gridpoints_in_pca_space')
gridpoints_in_sample_space = tf.matmul(gridpoints_in_pca_space, tf.transpose(subspace_map))
with tf.variable_scope("BIGAN",reuse=True):
    gridpoints_in_latent_space = Encoder(gridpoints_in_sample_space)
    gridpoints_discriminated = Discriminator(gridpoints_in_sample_space,gridpoints_in_latent_space)


start_custom_in_pca = tf.placeholder(tf.float32,shape=[1,dim_pca],name='start_custom_in_pca')
end_custom_in_pca = tf.placeholder(tf.float32,shape=[1,dim_pca],name='end_custom_in_pca')
start_custom_in_sample_space = tf.matmul(start_custom_in_pca,tf.transpose(subspace_map))
end_custom_in_sample_space = tf.matmul(end_custom_in_pca,tf.transpose(subspace_map))
with tf.variable_scope("BIGAN",reuse=True):
    start_custom_in_latent_space = Encoder(start_custom_in_sample_space)
    end_custom_in_latent_space = Encoder(end_custom_in_sample_space)







#   setup grid in pca-space, map these into sample space, encode into latent space, and send both into dicriminator
#   plot cmap of grid of disc points, labeled real samples, and geodesics
#   select endpoints
#
#
#
#

#tf.summary.scalar( "geodesic_objective_function_proposed", geodesic_objective_function_proposed )
#for iter in range( min( n_geodesics, 10 ) ):
#    tf.summary.scalar( "geodesic_objective_per_geodesic_proposed_" + str( iter ),
#                       geodesic_objective_per_geodesic_proposed[iter] )




