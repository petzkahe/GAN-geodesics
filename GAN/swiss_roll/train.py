import os
import sys
from GAN.swiss_roll.GAN_graph import *
from GAN.swiss_roll.generate_data import *
from GAN.swiss_roll.plotting import plot_sample_space

sys.path.append(os.getcwd())
# sys.path = A list of strings that specifies the search path for modules.
# Initialized from the environment variable PYTHONPATH, plus an installation-dependent default.
# import random
matplotlib.use('Agg')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot

if os.path.exists(log_directory):
    raise Exception("The log_directory ({}) exists and should not be overwritten".format(log_directory))
else:
    os.makedirs(log_directory)
    print("Log directory is set to {}".format(log_directory))

    # Training

    train_critic = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9,
        name='Adam.Discriminator'
    ).minimize(
        objective_discriminator,
        var_list=parameters_discriminator
    )

    train_generator = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9,
        name='Adam.Generator'
    ).minimize(
        objective_generator,
        var_list=parameters_generator
    )

#####################################################################
#####################################################################
# Train loop

saver = tf.train.Saver(tf.trainable_variables())

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # soft placement allows cpu operations when ops not defined for gpu
    session.run(tf.initialize_all_variables())
    generate_batch_real_data = generate_real_data()

    for iteration in range(n_GAN_iterations):
        # Train generator
        if iteration > 0:

            for j in range(n_generator_inner):
                if latent_distribution == "uniform":
                    batch_latent_data = np.random.uniform(low=latent_min_value, high=latent_max_value,
                                                          size=[n_batch_size, dim_latent]).astype('float32')
                if latent_distribution == "Gaussian":
                    batch_latent_data = np.random.normal(size=[n_batch_size, dim_latent]).astype('float32')

                _objective_generator, _ = session.run([objective_generator, train_generator],
                                                      feed_dict={data_latent: batch_latent_data})

        # Train critic
        if iteration > 2000:
            n_discriminator_inner = 1
        for j in range(n_discriminator_inner):
            batch_real_data = generate_batch_real_data.__next__()
            if latent_distribution == "uniform":
                batch_latent_data = np.random.uniform(low=latent_min_value, high=latent_max_value,
                                                      size=[n_batch_size, dim_latent]).astype('float32')
            if latent_distribution == "Gaussian":
                batch_latent_data = np.random.normal(size=[n_batch_size, dim_latent]).astype('float32')

            _objective_critic, _ = session.run([objective_discriminator, train_critic],
                                               feed_dict={data_real: batch_real_data, data_latent: batch_latent_data})

        if iteration % 1000 == 999:
            batch_latent_data = generate_latent_data()
            batch_generated_data, batch_disc_value_on_generated = session.run(
                [data_generated, disc_values_on_generated],
                feed_dict={data_latent: batch_latent_data}
            )
            plot_sample_space(batch_real_data, batch_generated_data, iteration)

    # Save the model
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    saver.save(session, 'trained_model/swissGAN')
