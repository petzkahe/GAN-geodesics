from GAN.swiss_roll.geodesic_graph import *


def set_up_training(objective_Jacobian, objective_proposed):

    

    train_Jacobian = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_geodesics
        ).minimize(objective_Jacobian,var_list=coefficients)

    train_proposed = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_geodesics
        ).minimize(objective_proposed,var_list=coefficients)


    # train_Jacobian = tf.train.AdamOptimizer(
    #     learning_rate=learning_rate_geodesics,
    #     beta1=adam_beta1,
    #     beta2=adam_beta2
    # ).minimize(
    #     objective_Jacobian,
    #     var_list=coefficients
    # )

    # train_proposed = tf.train.AdamOptimizer(
    #     learning_rate=learning_rate_geodesics,
    #     beta1=adam_beta1,
    #     beta2=adam_beta2
    # ).minimize(
    #     objective_proposed,
    #     var_list=coefficients
    # )


    return train_Jacobian, train_proposed



def find_geodesics(method, start, end, sess, training,train_writer):

    if method=="proposed":

        for iteraton in range(n_train_iterations_geodesics):
            if False:
                merge = tf.summary.merge_all()
                summary,_ = sess.run([merge,training], feed_dict={z_start : start, z_end: end})
                train_writer.add_summary(summary, iteraton)
            else:
                _ = sess.run([training], feed_dict={z_start : start, z_end: end})
                

        _curves_in_latent_space_value, _curves_in_sample_space_value = sess.run(
            [curves_in_latent_space, curves_in_sample_space],
            feed_dict={z_start: start, z_end: end})

    elif method == "Jacobian":

        for iteraton in range(n_train_iterations_geodesics):
            _ = sess.run([training], feed_dict={z_start : start, z_end: end})


        _curves_in_latent_space_value, _curves_in_sample_space_value = sess.run(
            [curves_in_latent_space, curves_in_sample_space],
            feed_dict={z_start: start, z_end: end})

    elif method == "linear":
        _curves_in_latent_space_value, _curves_in_sample_space_value = sess.run(
            [lines_in_latent_space, lines_in_sample_space], feed_dict={z_start: start, z_end: end})
    
    else:
        raise Exception("method {} unknown".format(method))

    return _curves_in_latent_space_value, _curves_in_sample_space_value


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



    GAN_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN")
    model_saver = tf.train.Saver(GAN_parameters)


    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training(geodesic_objective_function_Jacobian,
                                                                       geodesic_objective_function_proposed)
    with tf.Session() as session:


        # methods=["linear", "Jacobian", "proposed"]

        dict = {}


        for method in methods:
            session.run(tf.global_variables_initializer())

            model_saver.restore(session, tf.train.latest_checkpoint('trained_model/'))

            #######################################################3333
            ########### DELETE ME WHEN DONE CHECKING STUFF
            #############################################################


            _curves_in_latent_space_value, _curves_in_sample_space_value = session.run(
                [curves_in_latent_space, curves_in_sample_space],
                feed_dict={z_start: latent_start, z_end: latent_end})

            dict["before"] = [_curves_in_latent_space_value, _curves_in_sample_space_value]


            #################################################################
            #################################################################
            #################################################################




            if method == "Jacobian":
                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start, latent_end, session, train_geodesic_Jacobian,None)

            elif method == "proposed":
                train_writer  =  tf.summary.FileWriter( './graphs',session.graph)
                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start, latent_end,
                                                                                            session, train_geodesic_proposed,train_writer)

                
            elif method == "linear":
                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start,
                                                                                            latent_end,
                                                                                            session,
                                                                                            None,None)



            else:
                raise Exception("method {} unknown".format(method))


            dict[method] = [curves_in_latent_space_value, curves_in_sample_space_value]

            variables_names = [v.name for v in tf.trainable_variables()]
            values = session.run(variables_names)


        suppl_dict = {}

        # Calculate discriminator background for sample space 
        sample_grid_vectorized = create_grid(sample_grid_minima, sample_grid_maxima,n_discriminator_grid_sample)
        [disc_values_over_sample_grid_vectorized] = session.run([disc_values_on_real], feed_dict={data_real: sample_grid_vectorized})
        disc_values_over_sample_grid = disc_values_over_sample_grid_vectorized.reshape((n_discriminator_grid_sample, n_discriminator_grid_sample))
        suppl_dict["disc_values_over_sample_grid"] = disc_values_over_sample_grid


        # Calculate discriminator background for latent space
        latent_grid_vectorized = create_grid(latent_grid_minima, latent_grid_maxima,n_discriminator_grid_latent)
        [disc_values_over_latent_grid_vectorized] = session.run([disc_values_on_generated], feed_dict={data_latent: latent_grid_vectorized})
        disc_values_over_latent_grid = disc_values_over_latent_grid_vectorized.reshape((n_discriminator_grid_latent, n_discriminator_grid_latent))
        suppl_dict["disc_values_over_latent_grid"] = disc_values_over_latent_grid



    return dict , suppl_dict