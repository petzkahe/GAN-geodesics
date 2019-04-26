import os

from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *

if which_gan == 'BIGAN':
    from GAN.mnist.mnist_01digits.BIGAN_Learning.BIGAN_graph import *
elif which_gan == 'DCWGAN':
    from GAN.mnist.mnist_01digits.BIGAN_Learning.DCWGAN_graph import *
    print('Went here!')

from GAN.mnist.mnist_01digits.utils.generate_data_01 import *
from GAN.mnist.mnist_01digits.utils.plotting import *

    # Training
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
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


# matplotlib.use('Agg')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
def train_BIGAN(n_epochs,n_training_examples,_dir):



    saver = tf.train.Saver(max_to_keep=5)
    #saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=50)
    #vars_to_save = [tf.trainable_variables(), tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
    #saver = tf.train.Saver(vars_to_save, max_to_keep=50)
    #print(vars_to_save)



    n_BIGAN_iterations = int(n_epochs*n_training_examples/n_batch_size)


    if os.path.exists(_dir + log_directory_01) == False:
        os.makedirs(_dir + log_directory_01)
        print("Log directory is set to {}".format(log_directory_01))


    

    #####################################################################
    #####################################################################
    # Train loop

    
    global n_discriminator_inner
    global n_generator_encoder_inner
    global learning_rate
    
    reduction_factor = (learning_rate_final/learning_rate)**int(4.0/n_BIGAN_iterations)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # soft placement allows cpu operations when ops not defined for gpu
        session.run(tf.initialize_all_variables())

        if not os.path.exists(_dir + 'trained_model'):
            os.makedirs(_dir + 'trained_model')

        print('Trained epochs:')
        
        for iteration in range(n_BIGAN_iterations):
           
            # Train generator and encoder
            if iteration > 0:
                

                for j in range(n_generator_encoder_inner):
                    batch_latent_data = generate_latent_data(n_batch_size)
                    batch_real_data = generate_real_data( n_batch_size )
                
                    if which_gan == 'DCWGAN':
                        _objective_generator, _ = session.run([objective_generator_encoder, train_generator_encoder],
                                                      feed_dict={data_real: batch_real_data, data_latent: batch_latent_data, isTrain: True})
                    else:
                        _objective_generator, _ = session.run([objective_generator_encoder, train_generator_encoder],
                                                      feed_dict={data_real: batch_real_data, data_latent: batch_latent_data, isTrain: False})
                    
            # Train critic
            
            for j in range(n_discriminator_inner):
                batch_real_data = generate_real_data( n_batch_size )
                batch_latent_data = generate_latent_data( n_batch_size )

                if which_gan == 'DCWGAN':
                    _objective_discriminator, _ = session.run( [objective_discriminator, train_discriminator],
                                                feed_dict={data_real: batch_real_data, data_latent: batch_latent_data, isTrain: True})
                else:
                    _objective_discriminator, _ = session.run( [objective_discriminator, train_discriminator],
                                                feed_dict={data_real: batch_real_data, data_latent: batch_latent_data, isTrain: False})
                
            if iteration > 0 and iteration % 100 == 0:
                print('Epoch: ' + str(int(iteration*n_batch_size/n_training_examples)) + '/' + str(n_epochs) 
                        + ', D_loss: ' + str( int(_objective_discriminator*100)/100.)
                        + ', G_loss: ' + str( int(_objective_generator*100)/100.) )


            # Decrease learning rate
            if which_gan == 'BIGAN':

                if iteration>int(n_BIGAN_iterations/2.0) and learning_rate>learning_rate_final:
                        learning_rate=learning_rate*reduction_factor


            # Check discriminator values and plot logs

            if iteration % 1000 == 999:

                n_to_average = 250
                n_examples_to_plot = 25
                batch_real_data = generate_real_data( n_to_average )
                batch_disc_on_real,batch_disc_on_real_mean = session.run([disc_values_on_real,disc_values_on_real_mean],feed_dict={data_real:batch_real_data,isTrain:False})
                batch_latent_data = generate_latent_data(n_to_average)
                batch_generated_data,batch_disc_on_generated, batch_disc_on_generated_mean = session.run([data_generated,disc_values_on_generated,disc_values_on_generated_mean],
                                                feed_dict={data_latent: batch_latent_data,isTrain: False})
                 
                plot_sample_space_01(batch_generated_data[:n_examples_to_plot,:],batch_disc_on_generated[:n_examples_to_plot], iteration,_dir)
                plot_sample_space_01( batch_real_data[:n_examples_to_plot,:],batch_disc_on_real[:n_examples_to_plot], iteration+1,_dir)


                if which_gan == 'BIGAN':

                    disc_value_diff = batch_disc_on_real_mean - batch_disc_on_generated_mean
                    if  disc_value_diff > 0.8 and n_generator_encoder_inner < 8:
                        n_generator_encoder_inner = n_generator_encoder_inner+1 
                        print('Number of generator updates increased to: ' + str(n_generator_encoder_inner))

                    elif disc_value_diff < 0.65 and n_generator_encoder_inner > 1:
                        n_generator_encoder_inner = n_generator_encoder_inner-1
                        print('Number of generator updates decreased to: ' + str(n_generator_encoder_inner))

                    if  disc_value_diff < 0.1 and n_discriminator_inner < 8:
                        n_discriminator_inner = n_discriminator_inner+1
                        print('Number of discriminator updates increased to: ' + str(n_discriminator_inner))
                    
                    elif disc_value_diff > 0.35 and n_discriminator_inner > 1:
                        n_discriminator_inner = n_discriminator_inner-1
                        print('Number of discriminator updates decreased to: ' + str(n_discriminator_inner))
                    
                    if iteration> n_BIGAN_iterations/2.0:
                        print('Learning rate is now 10^(' + str(np.log10(learning_rate)) + ').')

            # Save checkpoints

            if iteration > 0 and iteration % 2000 == 0:
                saver.save(session, _dir + 'trained_model/incomplete_mnistBIGAN_' + str(int(iteration)))
        
        # Save the final model
        
        saver.save(session, _dir + 'trained_model/mnistBIGAN')
