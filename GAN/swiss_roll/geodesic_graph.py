from GAN.swiss_roll.GAN_graph import *
from GAN.swiss_roll.config_geodesics import *

import numpy as np



with tf.variable_scope("Geodesics"):

    z_start = tf.placeholder(tf.float32, shape=[1, dim_latent, n_geodesics], name='z_start')
    z_end = tf.placeholder(tf.float32, shape=[1, dim_latent, n_geodesics], name='z_end')

    if sampling_geodesic_coefficients == "zeros":
        coefficients_initializations = np.zeros(shape=(degree_polynomial_geodesic_latent - 1, dim_latent, n_geodesics), dtype='float32')
    elif sampling_geodesic_coefficients == "uniform":
        coefficients_initializations = np.random.uniform(-2,2 , size=(degree_polynomial_geodesic_latent - 1, dim_latent, n_geodesics)).astype("float32")
    elif sampling_geodesic_coefficients == "grid":

        tmp_grid = np.zeros((n_loss_grid, n_loss_grid, 2), dtype='float32')
        tmp_grid[:, :, 0] = np.linspace(coefficient_range[0], coefficient_range[1], n_loss_grid)[:, None]
        # for zero: for any second entry linspace runs over first coordinate
        tmp_grid[:, :, 1] = np.linspace(coefficient_range[0], coefficient_range[1], n_loss_grid)[None, :]
        # for one: for any first entry linspace runs over second coordinate
        coefficients_initializations = np.transpose(tmp_grid.reshape((1,-1,2)),(0,2,1))
        print(coefficients_initializations)
        
        # calculate a grid and then resize to correct format
    else:
        raise Exception("sampling method {} for geodesic coefficients unknown".format(sampling_geodesic_coefficients))

    

    if do_shared_variables:
    ###########
        zero_init = np.zeros(shape=(degree_polynomial_geodesic_latent - 1, dim_latent, 1),dtype='float32')
        coefficients = tf.Variable(initial_value=zero_init, name='coefficients')
        coefficients_offset = tf.constant(coefficients_initializations)
        coefficients_shared = tf.add(tf.tile(coefficients,(1,1,n_geodesics)),coefficients_offset)
    ##############
    else:
        coefficients = tf.Variable(initial_value=coefficients_initializations, name='coefficients')
    
def parametrize_line(z_start, z_end, n_geodesic_interpolations):
    constant_part = tf.reshape(z_start, shape=(1, dim_latent, n_geodesics))

    linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
        1, dim_latent, n_geodesics))

    coefficients_vector = tf.concat([constant_part, linear_part], axis=0)

    interpolation_matrix_entries = np.zeros(shape=(n_geodesic_interpolations + 1, 2))
    for i in range(n_geodesic_interpolations + 1):
        for j in range(2):
            interpolation_matrix_entries[i, j] = (float(i) / (n_geodesic_interpolations +1 ) ) ** j
    interpolation_matrix = tf.constant(interpolation_matrix_entries,
                                       shape=(n_geodesic_interpolations + 1, 2),
                                       dtype='float32')

    geodesic_points_in_z_matrix = tf.matmul(
        tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, 2]),
        tf.transpose(interpolation_matrix, perm=[1, 0]))

    geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
                                        shape=[n_geodesics, dim_latent, n_geodesic_interpolations + 1])
    geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0])

    return geodesic_points_in_z




def parametrize_curve(z_start, z_end, interpolation_degree, n_geodesic_interpolations):

        constant_part = tf.reshape(z_start, shape=(1, dim_latent, n_geodesics))
        
        if do_shared_variables:
            linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
                1, dim_latent, n_geodesics)) - tf.reshape(tf.reduce_sum(coefficients_shared,
                                                                     axis=0), shape=(1, dim_latent, n_geodesics))
            print(1)
            coefficients_vector = tf.concat([constant_part, linear_part, coefficients_shared], axis=0)
        
        else:
            linear_part = tf.reshape(z_end, shape=(1, dim_latent, n_geodesics)) - tf.reshape(z_start, shape=(
                1, dim_latent, n_geodesics)) - tf.reshape(tf.reduce_sum(coefficients,
                                                                     axis=0), shape=(1, dim_latent, n_geodesics))

            coefficients_vector = tf.concat([constant_part, linear_part, coefficients], axis=0)
        
        # Initialize parameter variable of size interpolation_degree times dimensions_noise space

        interpolation_matrix_entries = np.zeros(shape=(n_geodesic_interpolations + 1, interpolation_degree + 1))
        for i in range(n_geodesic_interpolations + 1):
            for j in range(interpolation_degree + 1):
                interpolation_matrix_entries[i, j] = (float(i) / (n_geodesic_interpolations + 1)) ** j
        interpolation_matrix = tf.constant(interpolation_matrix_entries,
                                            shape=(n_geodesic_interpolations + 1, interpolation_degree + 1),
                                           dtype='float32')

        geodesic_points_in_z_matrix = tf.matmul(
            tf.reshape(tf.transpose(coefficients_vector, perm=[2, 1, 0]), shape=[-1, interpolation_degree + 1]),
            tf.transpose(interpolation_matrix, perm=[1, 0]))

        geodesic_points_in_z_t = tf.reshape(geodesic_points_in_z_matrix,
                                            shape=[n_geodesics, dim_latent, n_geodesic_interpolations + 1])
        geodesic_points_in_z = tf.transpose(geodesic_points_in_z_t, perm=[2, 1, 0])

        return geodesic_points_in_z


curves_in_latent_space = parametrize_curve(z_start, z_end, degree_polynomial_geodesic_latent, n_interpolations_points_geodesic)
curves_in_latent_space_vectorized = tf.reshape(tf.transpose(curves_in_latent_space, perm=[2, 0, 1]),
                                                   shape=(n_geodesics * (n_interpolations_points_geodesic + 1), dim_latent))

lines_in_latent_space = parametrize_line(z_start,z_end,n_interpolations_points_geodesic)
lines_in_latent_space_vectorized = tf.reshape(tf.transpose(lines_in_latent_space, perm=[2, 0, 1]),
                                                   shape=(n_geodesics * (n_interpolations_points_geodesic + 1), dim_latent))


with tf.variable_scope("GAN"):
    curves_in_sample_space_vectorized = Generator(curves_in_latent_space_vectorized)
    disc_values_curves_sample_space_vectorized = Discriminator(curves_in_sample_space_vectorized)

    lines_in_sample_space_vectorized = Generator(lines_in_latent_space_vectorized)
    

disc_values_curves_sample_space = tf.transpose(tf.reshape(disc_values_curves_sample_space_vectorized, shape=(n_geodesics, n_interpolations_points_geodesic + 1)), perm=(1, 0))

curves_in_sample_space = tf.transpose(tf.reshape(curves_in_sample_space_vectorized, shape=(n_geodesics,n_interpolations_points_geodesic+ 1, dim_data)),
                                      perm = [1,2,0])

lines_in_sample_space = tf.transpose(tf.reshape(lines_in_sample_space_vectorized, shape=(n_geodesics,n_interpolations_points_geodesic+ 1, dim_data)),
                                     perm = [1,2,0])

diff_square_vector = tf.reduce_sum(tf.square(curves_in_sample_space[1:, :, :] - curves_in_sample_space[:-1, :, :]), axis=1)
diff_square_vector_latent = tf.reduce_sum(tf.square(curves_in_latent_space[1:, :, :] - curves_in_latent_space[:-1, :, :]), axis=1)

small_eps = 0.01

if True:
    disc_values_curves_sample_space = tf.exp(tf.multiply(0.5,tf.add(tf.log(disc_values_curves_sample_space[1:,:]),tf.log(disc_values_curves_sample_space[:-1,:]))))
    denominator = tf.clip_by_value(tf.add(disc_values_curves_sample_space,small_eps), small_eps,0.4+small_eps)
else:
    denominator = tf.clip_by_value(tf.add(disc_values_curves_sample_space[1:,:],small_eps), small_eps,0.4+small_eps)

#denominator = tf.Print(denominator,[denominator])


denominator = tf.multiply(denominator,denominator)

#objective_vector_proposed = tf.divide(1, denominator)
#objective_vector_proposed = tf.divide(diff_square_vector_latent, denominator)

objective_vector_proposed = tf.divide(diff_square_vector,denominator)

objective_vector_Jacobian = diff_square_vector

#if method == "proposed"


if penalty == True:
    geodesic_penalty = tf.reduce_max(diff_square_vector)  # maximum of norm difference in sample space
    penalty_hyper_param = 100.
else:
    geodesic_penalty = 0
    penalty_hyper_param = 0

geodesic_objective_per_geodesic_proposed = tf.reduce_sum(objective_vector_proposed,axis=0) 
#geodesic_objective_per_geodesic_proposed = tf.Print(geodesic_objective_per_geodesic_proposed,[geodesic_objective_per_geodesic_proposed])
geodesic_objective_function_proposed = tf.reduce_sum(geodesic_objective_per_geodesic_proposed) + penalty_hyper_param * geodesic_penalty
#geodesic_objective_function_proposed = tf.Print(geodesic_objective_function_proposed,[geodesic_objective_function_proposed])

#geodesic_objective_function_proposed = tf.reduce_sum(geodesic_objective_function_proposed) + penalty_hyper_param * geodesic_penalty

geodesic_objective_per_geodesic_Jacobian = tf.reduce_sum(objective_vector_Jacobian,axis=0) 
#geodesic_objective_per_geodesic_Jacobian = tf.Print(geodesic_objective_per_geodesic_Jacobian,[geodesic_objective_per_geodesic_Jacobian])
geodesic_objective_function_Jacobian = tf.reduce_sum(geodesic_objective_per_geodesic_Jacobian) + penalty_hyper_param * geodesic_penalty
#geodesic_objective_function_Jacobian = tf.reduce_sum(geodesic_objective_function_Jacobian) + penalty_hyper_param * geodesic_penalty


tf.summary.scalar("geodesic_objective_function_proposed",geodesic_objective_function_proposed)
for iter in range(min(n_geodesics,10)):
    tf.summary.scalar("geodesic_objective_per_geodesic_proposed_" + str(iter),geodesic_objective_per_geodesic_proposed[iter])




