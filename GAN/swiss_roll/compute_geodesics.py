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

    if method == "Jacobian":

        for iteraton in range(n_train_iterations_geodesics):
            _ = sess.run([training], feed_dict={z_start : start, z_end: end})


        _curves_in_latent_space_value, _curves_in_sample_space_value = sess.run(
            [curves_in_latent_space, curves_in_sample_space],
            feed_dict={z_start: start, z_end: end})

    ##### Add methods here and in config


    else:
        raise Exception("method {} unknown".format(method))

    return _curves_in_latent_space_value, _curves_in_sample_space_value


def compute_geodesics(z_start, z_end):



    GAN_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN")
    model_saver = tf.train.Saver(GAN_parameters)


    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training(geodesic_objective_function_Jacobian,
                                                                       geodesic_objective_function_proposed)
    with tf.Session() as session:


        # methods=["linear", "Jacobian", "proposed"]

        dict = {}


        for method in methods:
            session.run(tf.initialize_all_variables())

            model_saver.restore(session, tf.train.latest_checkpoint('trained_model/'))

            curves_in_latent_space_value, curves_in_sample_space_value = find_geodesics(method, z_start, z_end, session,train_geodesic_Jacobian)

            dict[method] = [curves_in_latent_space_value, curves_in_sample_space_value]

            variables_names = [v.name for v in tf.trainable_variables()]
            values = session.run(variables_names)


        # session run of fakes = grid of latent points, gives fakes_sample_space, discriminator values, and possibly jacobian metric
        # supplementary_dict = above stuff

    return dict #, supplementary_dict