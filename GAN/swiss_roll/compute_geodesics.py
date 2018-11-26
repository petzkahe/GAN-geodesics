from GAN.swiss_roll.geodesic_objective_graph import *


saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    saver.restore(session, tf.train.latest_checkpoint('trained_model/'))

    train_geodesic_Jacobian, train_geodesic_propose = set_up_training(geodesic_objective_Jacobian,
                                                                      geodesic_objective_proposed)

    # methods=["linear", "Jacobian", "proposed"]
    methods = ["Jacobian"]

    for method in methods:
        geodesic_points_in_z_value, geodesic_points_in_sample_space_value = find_geodesic(method, z_start_values,
                                                                                          z_end_values)
