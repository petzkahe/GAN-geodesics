from GAN.mnist.Geodesic_Learning.geodesic_graph_mnist import *

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


def find_geodesics(method, start, end, sess, training, train_writer):
    if method == "proposed":

        for iteraton in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )

        curves_in_sample_space_value, _objective_values = sess.run(
            [curves_in_sample_space, geodesic_objective_per_geodesic_proposed],
            feed_dict={z_start: start, z_end: end} )

    elif method == "Jacobian":

        for iteraton in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )

        curves_in_sample_space_value, _objective_values = sess.run(
            [curves_in_sample_space, geodesic_objective_per_geodesic_Jacobian],
            feed_dict={z_start: start, z_end: end} )

    elif method == "linear":
        curves_in_sample_space_value, _objective_values = sess.run(
            [lines_in_sample_space, geodesic_objective_per_geodesic_linear],
            feed_dict={z_start: start, z_end: end} )
        # _objective_values = None

    else:
        raise Exception( "method {} unknown".format( method ) )
    return curves_in_sample_space_value, _objective_values




##################################################################################################
#########################################################################################################


def compute_geodesics(latent_start, latent_end):
    BIGAN_parameters = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="BIGAN" )
    model_saver = tf.train.Saver(BIGAN_parameters)

    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training( geodesic_objective_function_Jacobian,
                                                                        geodesic_objective_function_proposed )
    with tf.Session() as session:

        # methods=["linear", "Jacobian", "proposed"]

        dict = {}

        for method in methods:
            session.run( tf.global_variables_initializer() )

            model_saver.restore( session, tf.train.latest_checkpoint( '../BIGAN_Learning/trained_model/' ) )


            if method == "Jacobian":
                curves_in_sample_space_value, objective_values = find_geodesics(method, latent_start, latent_end, session, train_geodesic_Jacobian, None )
                print( 'Jacobian done!' )
            elif method == "proposed":
                train_writer = tf.summary.FileWriter( './graphs', session.graph )
                curves_in_sample_space_value, objective_values = find_geodesics(
                    method, latent_start, latent_end,
                    session, train_geodesic_proposed, train_writer )
                print( 'Proposed done!' )

            elif method == "linear":
                curves_in_sample_space_value, objective_values = find_geodesics(
                    method, latent_start,
                    latent_end,
                    session,
                    None, None )



            else:
                raise Exception( "method {} unknown".format( method ) )

            dict[method] = [curves_in_sample_space_value, objective_values]

            #variables_names = [v.name for v in tf.trainable_variables()]
            #values = session.run( variables_names )

        return dict