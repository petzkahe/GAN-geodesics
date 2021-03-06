from GAN.mnist.mnist_alldigits.Geodesic_Learning.geodesic_graph_mnist import *
from tensorflow.examples.tutorials.mnist import input_data
import os

def set_up_training(objective_Jacobian, objective_proposed):

    train_Jacobian = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=adam_beta1,
        beta2=adam_beta2
    ).minimize(
        objective_Jacobian,
        var_list=coefficients
    )

    train_proposed = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=adam_beta1,
        beta2=adam_beta2
    ).minimize(
        objective_proposed,
        var_list=coefficients
    )

    return train_Jacobian, train_proposed


def find_geodesics(method, start, end, sess, training, train_writer, V):
    if method == "proposed":
        print('Proposed method training')
        for iteration in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )
            if iteration % 500 == 0:
                print(str(int(iteration/n_train_iterations_geodesics*100.0)) + ' %')

        curves_in_sample_space_value, _objective_values, curves_in_pca_space_value= sess.run(
            [curves_in_sample_space, geodesic_objective_per_geodesic_proposed, curves_in_pca_space],
            feed_dict={z_start: start, z_end: end,subspace_map:V} )



    elif method == "Jacobian":
        print('Jacobian method training')
        for iteration in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )
            if iteration % 500 == 0:
                print(str(int(iteration/n_train_iterations_geodesics*100.0)) + ' %')

        curves_in_sample_space_value, _objective_values, curves_in_pca_space_value = sess.run(
            [curves_in_sample_space, geodesic_objective_per_geodesic_Jacobian, curves_in_pca_space],
            feed_dict={z_start: start, z_end: end,subspace_map:V} )



    elif method == "linear":
        curves_in_sample_space_value, _objective_values, curves_in_pca_space_value = sess.run(
            [lines_in_sample_space, geodesic_objective_per_geodesic_linear, lines_in_pca_space],
            feed_dict={z_start: start, z_end: end, subspace_map:V} )


    else:
        raise Exception( "method {} unknown".format( method ) )
    return curves_in_sample_space_value, _objective_values, curves_in_pca_space_value


def create_grid(_min, _max,_n):

    #  Creates a grid in sample space
    _grid = np.zeros((_n, _n, 2), dtype='float32')
    _grid[:, :, 0] = np.linspace(_min[0], _max[0], _n)[:, None]
    # for zero: for any second entry linspace runs over first coordinate
    _grid[:, :, 1] = np.linspace(_min[1], _max[1], _n)[None, :]
    # for one: for any first entry linspace runs over second coordinate
    _grid_vectorized = _grid.reshape((-1, 2))  # gives list of points of all combinations

    return _grid_vectorized

##################################################################################################
#########################################################################################################


def compute_geodesics(latent_start, latent_end):
    BIGAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="BIGAN" )
    model_saver = tf.train.Saver(BIGAN_parameters)

    # To save the polynomial coefficients
    coefficient_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Geodesics" )
    coefficients_saver = tf.train.Saver(coefficient_parameters)

    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training( geodesic_objective_function_Jacobian,
                                                                        geodesic_objective_function_proposed )

    dataset = input_data.read_data_sets( 'MNIST_data' )
    V = np.load('../utils/svd_right_save.npy')
    _subspace_map = V[:,:dim_pca]

    with tf.Session() as session:

        # methods=["linear", "Jacobian", "proposed"]
        suppl_dict = {}
        dict = {}

        for method in methods:
            session.run( tf.global_variables_initializer() )

            model_saver.restore( session, tf.train.latest_checkpoint( '../BIGAN_Learning/trained_model/' ) )


            if method == "Jacobian":
                curves_in_sample_space_value, objective_values, curves_in_pca_space_value = find_geodesics(method, latent_start, latent_end, session, train_geodesic_Jacobian, None, _subspace_map )
                print( 'Jacobian done!' )
            elif method == "proposed":
                train_writer = tf.summary.FileWriter( './graphs', session.graph )
                curves_in_sample_space_value, objective_values, curves_in_pca_space_value = find_geodesics(method, latent_start, latent_end, session, train_geodesic_proposed, None, _subspace_map )
                print( 'Proposed done!' )

            elif method == "linear":
                curves_in_sample_space_value, objective_values, curves_in_pca_space_value = find_geodesics(
                    method, latent_start,
                    latent_end,
                    session,
                    None, None, _subspace_map )



            else:
                raise Exception( "method {} unknown".format( method ) )

            dict[method] = [curves_in_sample_space_value, objective_values, curves_in_pca_space_value]


            if not os.path.exists( 'trained_polynomials/{}'.format(method) ):
                os.makedirs( 'trained_polynomials/{}'.format(method) )
            coefficients_saver.save( session, 'trained_polynomials/{}/{}'.format(method,method) )

            #variables_names = [v.name for v in tf.trainable_variables()]
            #values = session.run( variables_names )

        # get pca of reals,labels
        # generate reals with labels,
        reals,labels = dataset.train.next_batch( n_batch_pca )
        print(reals.shape)
        # load PCA
        
        
        # feed into session
        
        reals_in_pca = session.run(real_samples_in_pca_space, feed_dict={real_samples_for_pca: reals, subspace_map: _subspace_map} )
        suppl_dict['reals'] = [reals_in_pca,labels]

        np.save( 'trained_polynomials/latent_start_end_points', [latent_start, latent_end] )


        return dict, suppl_dict




