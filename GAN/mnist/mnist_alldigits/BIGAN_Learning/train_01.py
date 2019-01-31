import os
from GAN.mnist.mnist_alldigits.BIGAN_Learning.BIGAN_graph import *
from GAN.mnist.mnist_alldigits.utils.generate_data_01 import *
from GAN.mnist.mnist_alldigits.utils.plotting import *



# matplotlib.use('Agg')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot

if os.path.exists(log_directory_01):
    raise Exception("The log_directory_01 ({}) exists and should not be overwritten".format(log_directory_01))
else:
    os.makedirs(log_directory_01)
    print("Log directory is set to {}".format(log_directory_01))

    # Training

    train_discriminator = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.999,
        name='Adam.Discriminator'
    ).minimize(
        objective_discriminator,
        var_list=parameters_discriminator
    )

    train_generator_encoder = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.999,
        name='Adam.Generator'
    ).minimize(
        objective_generator_encoder,
        var_list=parameters_generator_encoder
    )

#####################################################################
#####################################################################
# Train loop

saver = tf.train.Saver(tf.trainable_variables())

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # soft placement allows cpu operations when ops not defined for gpu
    session.run(tf.initialize_all_variables())

    
   
    for iteration in range(n_BIGAN_iterations):
        # Train generator and encoder
        if iteration > 0:
            if iteration>n_BIGAN_iterations/2.0 and learning_rate>1e-6:
                learning_rate=learning_rate*0.999

            for j in range(n_generator_encoder_inner):
                batch_latent_data = generate_latent_data(n_batch_size)
                batch_real_data = generate_real_data( n_batch_size )
                
                _objective_generator, _ = session.run([objective_generator_encoder, train_generator_encoder],
                                                      feed_dict={data_real: batch_real_data, data_latent: batch_latent_data})

        # Train critic
        if iteration > 2000:
            n_discriminator_inner = 1

        for j in range(n_discriminator_inner):
            batch_real_data = generate_real_data( n_batch_size )
            batch_latent_data = generate_latent_data( n_batch_size )

            _objective_discriminator, _ = session.run( [objective_discriminator, train_discriminator],
                                                feed_dict={data_real: batch_real_data, data_latent: batch_latent_data})

        if iteration % 1000 == 999:

            batch_latent_data = generate_latent_data(25)
            batch_real_data = generate_real_data( 25 )
            batch_generated_data = session.run(data_generated,
                feed_dict={data_latent: batch_latent_data}
            )
            plot_sample_space_01(batch_generated_data, iteration)
            plot_sample_space_01( batch_real_data, iteration+1 )

    # Save the model
    if not os.path.exists('trained_model_01'):
        os.makedirs('trained_model_01')
    saver.save(session, 'trained_model_01/mnistBIGAN')
