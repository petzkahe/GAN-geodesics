from GAN.swiss_roll.geodesic_graph import *

def set_up_training(objective_Jacobian, objective_proposed):


    train_Jacobian = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        objective_Jacobian,
        var_list=coefficients
    )

    train_proposed = tf.train.AdamOptimizer(
        learning_rate=learning_rate_geodesics,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        objective_proposed,
        var_list=coefficients
    )


    return train_Jacobian, train_proposed



def find_geodesics(method, start, end, sess, training):

    if method == "Jacobian" or  method=="proposed":

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


def create_sample_grid(_min, _max):

    #  Creates a grid in sample space
    x_grid = np.zeros((n_discriminator_grid, n_discriminator_grid, 2), dtype='float32')
    x_grid[:, :, 0] = np.linspace(_min[0], _max[0], n_discriminator_grid)[:, None]
    # for zero: for any second entry linspace runs over first coordinate
    x_grid[:, :, 1] = np.linspace(_min[1], _max[1], n_discriminator_grid)[None, :]
    # for one: for any first entry linspace runs over second coordinate
    x_grid = x_grid.reshape((-1, 2))  # gives list of points of all combinations

    return x_grid


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

                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start, latent_end, session, train_geodesic_Jacobian)

            elif method == "proposed":

                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start, latent_end,
                                                                                            session, train_geodesic_proposed)

            elif method == "linear":
                curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, latent_start,
                                                                                            latent_end,
                                                                                            session,
                                                                                            None)



            else:
                raise Exception("method {} unknown".format(method))


            dict[method] = [curves_in_latent_space_value, curves_in_sample_space_value]

            variables_names = [v.name for v in tf.trainable_variables()]
            values = session.run(variables_names)


        suppl_dict = {}

        sample_grid_vectorized = create_sample_grid(sample_grid_minima, sample_grid_maxima)

        # Pass grid through Discriminator
        [disc_values_over_sample_grid_vectorized] = session.run([disc_values_on_real], feed_dict={data_real: sample_grid_vectorized})

        disc_values_over_sample_grid = disc_values_over_sample_grid_vectorized.reshape((n_discriminator_grid, n_discriminator_grid))

        suppl_dict["disc_values_over_sample_grid"] = disc_values_over_sample_grid


    return dict , suppl_dict