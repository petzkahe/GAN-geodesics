from GAN.swiss_roll.Geodesic_Learning.Neural_Network_Approach.geodesic_graph_nn import *


def set_up_training(objective_Jacobian, objective_proposed):
    # train_Jacobian = tf.train.GradientDescentOptimizer(
    #     learning_rate=learning_rate_geodesics
    #     ).minimize(objective_Jacobian,var_list=coefficients)

    # train_proposed = tf.train.GradientDescentOptimizer(
    #     learning_rate=learning_rate_geodesics
    #     ).minimize(objective_proposed,var_list=coefficients)

    train_Jacobian = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=adam_beta1,
        beta2=adam_beta2
    ).minimize(
        objective_Jacobian,
        var_list=parameters_curve_net
    )

    train_proposed = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=adam_beta1,
        beta2=adam_beta2
    ).minimize(
        objective_proposed,
        var_list=parameters_curve_net
    )

    return train_Jacobian, train_proposed


def find_geodesics_nn(method, sess, training, train_writer):
    if method == "proposed":

        print( 'Proposed method training...' )
        for iteraton in range( n_train_iterations_geodesics ):
            start_end = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[n_batch_size_curve_net, 2 * dim_latent] ).astype( 'float32' )
            _ = sess.run( [training], feed_dict={z_in: start_end} )
            if iteraton % 100 == 99:
                print( iteraton / n_train_iterations_geodesics * 100.0 )

        start_end = np.concatenate( [z_start_center, z_end_center], axis=0 ).reshape( 1, 2 * dim_latent )

        _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values = sess.run(
            [curves_in_latent_space, curves_in_sample_space, geodesic_objective_per_geodesic_proposed],
            feed_dict={z_in: start_end} )

    elif method == "Jacobian":

        print( 'Jacobian method training...' )
        for iteraton in range( n_train_iterations_geodesics ):
            start_end = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[n_batch_size_curve_net, 2 * dim_latent] ).astype( 'float32' )
            _ = sess.run( [training], feed_dict={z_in: start_end} )
            if iteraton % 100 == 99:
                print( iteraton / n_train_iterations_geodesics * 100.0 )

        start_end = np.concatenate( [z_start_center, z_end_center], axis=0 ).reshape( 1, 2 * dim_latent )

        _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values = sess.run(
            [curves_in_latent_space, curves_in_sample_space, geodesic_objective_per_geodesic_Jacobian],
            feed_dict={z_in: start_end} )

    elif method == "linear":

        start_end = np.concatenate( [z_start_center, z_end_center], axis=0 ).reshape( 1, 2 * dim_latent )

        _curves_in_latent_space_value, _curves_in_sample_space_value = sess.run(
            [lines_in_latent_space, lines_in_sample_space], feed_dict={z_in: start_end} )
        _objective_values = None

    else:
        raise Exception( "method {} unknown".format( method ) )
    return _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values


def create_grid(_min, _max, _n):
    #  Creates a grid in sample space
    _grid = np.zeros( (_n, _n, 2), dtype='float32' )
    _grid[:, :, 0] = np.linspace( _min[0], _max[0], _n )[:, None]
    # for zero: for any second entry linspace runs over first coordinate
    _grid[:, :, 1] = np.linspace( _min[1], _max[1], _n )[None, :]
    # for one: for any first entry linspace runs over second coordinate
    _grid_vectorized = _grid.reshape( (-1, 2) )  # gives list of points of all combinations

    return _grid_vectorized


##################################################################################################
#########################################################################################################


def compute_geodesics_nn(latent_start_end):
    GAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN" )
    model_saver = tf.train.Saver( GAN_parameters )

    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training( geodesic_objective_function_Jacobian,
                                                                        geodesic_objective_function_proposed )
    with tf.Session() as session:

        # methods=["linear", "Jacobian", "proposed"]

        dict = {}
        suppl_dict = {}

        if do_no_training:
            session.run( tf.global_variables_initializer() )

            model_saver.restore( session, tf.train.latest_checkpoint( 'trained_model/' ) )

            # calculate loss function for each initial variable value on grid and return

            _loss_per_geodesic_proposed, _loss_per_geodesic_Jacobian = session.run(
                [geodesic_objective_per_geodesic_proposed, geodesic_objective_per_geodesic_Jacobian],
                feed_dict={z_in: latent_start_end} )

            # raise Exception("So far ({}) so good".format(do_no_training))
            # reshape vector into matrix
            loss_surface_proposed = _loss_per_geodesic_proposed.reshape( n_loss_grid, n_loss_grid )
            dict['loss_surface_proposed'] = loss_surface_proposed

            # reshape vector into matrix
            loss_surface_Jacobian = _loss_per_geodesic_Jacobian.reshape( n_loss_grid, n_loss_grid )
            dict['loss_surface_Jacobian'] = loss_surface_Jacobian

        else:

            for method in methods:
                session.run( tf.global_variables_initializer() )

                model_saver.restore( session, tf.train.latest_checkpoint( 'trained_model/' ) )

                if method == "Jacobian":
                    curves_in_latent_space_value, curves_in_sample_space_value, objective_values = find_geodesics_nn(
                        method, session, train_geodesic_Jacobian, None )
                    print( 'Jacobian done!' )
                elif method == "proposed":
                    train_writer = tf.summary.FileWriter( './graphs', session.graph )
                    curves_in_latent_space_value, curves_in_sample_space_value, objective_values = find_geodesics_nn(
                        method, session, train_geodesic_proposed, train_writer )
                    print( 'Proposed done!' )

                elif method == "linear":
                    curves_in_latent_space_value, curves_in_sample_space_value, objective_values = find_geodesics_nn(
                        method, session, None, None )



                else:
                    raise Exception( "method {} unknown".format( method ) )

                dict[method] = [curves_in_latent_space_value, curves_in_sample_space_value, objective_values]

                variables_names = [v.name for v in tf.trainable_variables()]
                values = session.run( variables_names )

            # Calculate discriminator background for sample space
            sample_grid_vectorized = create_grid( sample_grid_minima, sample_grid_maxima, n_discriminator_grid_sample )
            [disc_values_over_sample_grid_vectorized] = session.run( [disc_values_on_real],
                                                                     feed_dict={data_real: sample_grid_vectorized} )
            disc_values_over_sample_grid = disc_values_over_sample_grid_vectorized.reshape(
                (n_discriminator_grid_sample, n_discriminator_grid_sample) )
            suppl_dict["disc_values_over_sample_grid"] = disc_values_over_sample_grid

            # Calculate discriminator background for latent space
            latent_grid_vectorized = create_grid( latent_grid_minima, latent_grid_maxima, n_discriminator_grid_latent )
            [disc_values_over_latent_grid_vectorized] = session.run( [disc_values_on_generated],
                                                                     feed_dict={data_latent: latent_grid_vectorized} )
            disc_values_over_latent_grid = disc_values_over_latent_grid_vectorized.reshape(
                (n_discriminator_grid_latent, n_discriminator_grid_latent) )
            suppl_dict["disc_values_over_latent_grid"] = disc_values_over_latent_grid

    return dict, suppl_dict
