from GAN.mnist.mnist_01digits.Geodesic_Learning.geodesic_graph_mnist import *
from tensorflow.examples.tutorials.mnist import input_data
from GAN.mnist.mnist_01digits.main_config import *



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


def find_geodesics(method, start, end, sess, training, train_writer, V, _mean_per_pixel):
    if method == "proposed":
        print('Proposed method training')
        for iteration in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )
            if iteration % 100 == 0:
                print(str(int(iteration/n_train_iterations_geodesics*100.0)) + ' %')

        curves_in_sample_space_value, _disc_values_curves_sample_space, _objective_values, curves_in_pca_space_value = sess.run(
            [curves_in_sample_space, disc_values_curves_sample_space, geodesic_objective_per_geodesic_proposed, curves_in_pca_space],
            feed_dict={z_start: start, z_end: end,subspace_map:V, mean_per_pixel:_mean_per_pixel} )


    elif method == "Jacobian":
        print('Jacobian method training')
        for iteration in range( n_train_iterations_geodesics ):
            _ = sess.run( [training], feed_dict={z_start: start, z_end: end} )
            if iteration % 100 == 0:
                print(str(int(iteration/n_train_iterations_geodesics*100.0)) + ' %')

        curves_in_sample_space_value, _disc_values_curves_sample_space, _objective_values, curves_in_pca_space_value = sess.run(
            [curves_in_sample_space, disc_values_curves_sample_space, geodesic_objective_per_geodesic_Jacobian, curves_in_pca_space],
            feed_dict={z_start: start, z_end: end,subspace_map:V, mean_per_pixel: _mean_per_pixel} )

    elif method == "linear":
        curves_in_sample_space_value, _disc_values_curves_sample_space, _objective_values, curves_in_pca_space_value = sess.run(
            [lines_in_sample_space, disc_values_curves_sample_space, geodesic_objective_per_geodesic_linear, lines_in_pca_space],
            feed_dict={z_start: start, z_end: end, subspace_map:V, mean_per_pixel:_mean_per_pixel} )
        # _objective_values = None

    else:
        raise Exception( "method {} unknown".format( method ) )
    return curves_in_sample_space_value, _disc_values_curves_sample_space, _objective_values, curves_in_pca_space_value


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

    train_geodesic_Jacobian, train_geodesic_proposed = set_up_training( geodesic_objective_function_Jacobian,
                                                                        geodesic_objective_function_proposed )

    training_data = np.load(results_directory + 'Data/selected_train_data.npy')
    training_labels = np.load(results_directory + 'Data/selected_train_labels.npy')

    V = np.load(results_directory + 'PCA/right_singular_vectors.npy')
    _subspace_map = V[:,:dim_pca]
    _mean_per_pixel = np.load( results_directory + 'PCA/mean_per_pixel.npy' )

    with tf.Session() as session:

        # methods=["linear", "Jacobian", "proposed"]
        suppl_dict = {}
        dict = {}

        for method in methods:
            session.run( tf.global_variables_initializer() )

            model_saver.restore( session, tf.train.latest_checkpoint( results_directory + 'BIGAN/trained_model/' ) )
            print(results_directory)
            
            _update_ops = session.run(update_ops, feed_dict={data_real: np.ones((1,dim_data)), data_latent: np.ones((1,dim_latent)), isTrain: True})

            #" print(_update_ops)
       

            if method == "Jacobian": 
                curves_in_sample_space_value, disc_values_curves_sample_space, objective_values, curves_in_pca_space_value = find_geodesics(method, latent_start, latent_end, session, train_geodesic_Jacobian, None, _subspace_map, _mean_per_pixel )
                print( 'Jacobian done!' )
            elif method == "proposed":
                #train_writer = tf.summary.FileWriter( './graphs', session.graph )
                curves_in_sample_space_value, disc_values_curves_sample_space, objective_values, curves_in_pca_space_value = find_geodesics(
                    method, latent_start, latent_end,
                    session, train_geodesic_proposed, None, _subspace_map, _mean_per_pixel )
                print( 'Proposed done!' )

            elif method == "linear":
                curves_in_sample_space_value, disc_values_curves_sample_space, objective_values, curves_in_pca_space_value = find_geodesics(
                    method, latent_start,
                    latent_end,
                    session,
                    None, None, _subspace_map, _mean_per_pixel )



            else:
                raise Exception( "method {} unknown".format( method ) )

            dict[method] = [curves_in_sample_space_value, disc_values_curves_sample_space, objective_values, curves_in_pca_space_value]
            
            #variables_names = [v.name for v in tf.trainable_variables()]
            #values = session.run( variables_names )

        # get pca of reals,labels
        # generate reals with labels,

        reals = training_data[:n_batch_pca,:]
        #reals = training_data
        labels = training_labels[:n_batch_pca]
        #labels = training_labels
        
        # feed into session
        
        reals_in_pca = session.run(real_samples_in_pca_space, feed_dict={real_samples_for_pca: reals, subspace_map: _subspace_map, mean_per_pixel: _mean_per_pixel} )
        suppl_dict['reals'] = [reals_in_pca,labels]

        
        print(_update_ops)
        # Latent space background = discriminator backgropund of samples coming from the latent space

            # generate random points in latent space
        latent_points = np.random.uniform( low=latent_points_minima, high=latent_points_maxima,
                                           size=[n_latent_background, dim_latent] ).astype('float32' )
            # output points in pca, and discriminator values from session
        pca_points,discriminator_points = session.run([points_in_pca_space, points_discriminated],
            feed_dict={points_in_latent_space:latent_points, subspace_map: _subspace_map, mean_per_pixel: _mean_per_pixel})
            # add to suppl_dict
        print('Mean is ' + str(np.mean(discriminator_points)))
        suppl_dict["latent_background"] = [pca_points,discriminator_points]



        return dict, suppl_dict



