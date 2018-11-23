import tensorflow as tf
import GAN.swiss_roll.nn_layer as layer
from GAN.swiss_roll.config_GAN import *




################################################################
################################################################
with tf.variable_scope("GAN", reuse=tf.AUTO_REUSE):

    def Generator(noise):

        with tf.variable_scope("Generator.1", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(dim_latent, dim_nn, noise, "Generator.1")
        with tf.variable_scope("Generator.2", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(dim_nn, 2 * dim_nn, output, "Generator.2")
        with tf.variable_scope("Generator.3", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(2 * dim_nn, dim_nn, output, "Generator.3")
        with tf.variable_scope("Generator.4", reuse=tf.AUTO_REUSE):
            output = layer.LinearLayer(dim_nn, 2, output, "Generator.4")
        return output

    def Discriminator(inputs):
        with tf.variable_scope("Discriminator.1", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(2, 64, inputs,"Discriminator.1")
        with tf.variable_scope("Discriminator.2", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(64, dim_nn, output, "Discriminator.2")
        with tf.variable_scope("Discriminator.3", reuse=tf.AUTO_REUSE):
            output = layer.ReLuLayer(dim_nn, 2 * dim_nn, output, "Discriminator.3")
        with tf.variable_scope("Discriminator.4", reuse=tf.AUTO_REUSE):
                output = layer.ReLuLayer(2 * dim_nn, dim_nn, output, "Discriminator.4")
                output = layer.LinearLayer(dim_nn, 1, output, 'Discriminator.5')
                output = tf.nn.sigmoid(output)
            # output = tf.Print(output, [output])
        return output

    #####################################################################
    #####################################################################
    # Placeholder

    data_real = tf.placeholder(tf.float32, shape=[None, 2], name='reals')

    data_latent = tf.placeholder(tf.float32, shape=[None, dim_latent], name='noise')


    #####################################################################
    #####################################################################
    # Build graph

    # Get fake data by passing it through Generator
    data_generated = Generator(data_latent)

    disc_values_on_real = Discriminator(data_real)
    disc_values_on_generated = Discriminator(data_generated)

    # Objectives

    objective_discriminator = -tf.reduce_mean(tf.log(disc_values_on_real)) - tf.reduce_mean(tf.log(1 - disc_values_on_generated))
    objective_generator = - tf.reduce_mean(tf.log(disc_values_on_generated))

    #Variables

    all_variables = tf.trainable_variables()

    parameters_discriminator = [entry for entry in all_variables if "Discriminator" in entry.name]
    parameters_generator = [entry for entry in all_variables if "Generator" in entry.name]

