import tensorflow as tf
import GAN.mnist.mnist_01digits.BIGAN_Learning.nn_layer as layer
from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *
import numpy as np

with tf.variable_scope("BIGAN", reuse=tf.AUTO_REUSE):

    def Generator(noise,isTrain = True):
        output = layer.ReLuLayer(dim_latent, dim_nn, noise, "Generator.1")
        output = layer.ReLuLayerWithBatchN(dim_nn, dim_nn, output, isTrain, "Generator.2")
        output = layer.LinearLayer(dim_nn, dim_data, output, "Generator.3")
        #output = layer.SigmoidLayer(dim_nn, dim_data, output, "Generator.3")
        #output = layer.TanhLayer(dim_nn, dim_data, output, "Generator.3")/2. + .5
        return output

  
    def Discriminator(inputs, noise, isTrain = True):
        output = tf.concat([inputs,noise],axis=1)
        output = layer.LeakyReLuLayerWithBatchN(dim_data+dim_latent, dim_nn_disc, output, isTrain, "Discriminator.1")
        #output = layer.LeakyReLuLayerWithBatchN(dim_nn_disc, dim_nn_disc, output, "Discriminator.2")
        output = layer.SigmoidLayer(dim_nn_disc, 1, output,"Discriminator.3")
        return output


    def Encoder(inputs, isTrain = True):
        output = layer.LeakyReLuLayer( dim_data, dim_nn, inputs, "Encoder.1" )
        output = layer.LeakyReLuLayerWithBatchN( dim_nn, dim_nn, output, isTrain, "Encoder.2" )
        output = layer.LinearLayer( dim_nn, dim_latent, output, "Encoder.3" )
        #output = layer.TanhLayer( dim_nn, dim_latent, output, "Encoder.3" )
        return output


    def safe_log(x):
        return tf.log( x + 1e-8 )

    #####################################################################
    #####################################################################
    # Placeholder


    data_real = tf.placeholder(tf.float32, shape=[None, dim_data], name='reals')
    data_latent = tf.placeholder(tf.float32, shape=[None, dim_latent], name='latent')
    isTrain = tf.placeholder(dtype=tf.bool)



    #####################################################################
    #####################################################################
    # Build graph


    data_generated = Generator(data_latent, isTrain)
    data_encoded = Encoder(data_real, isTrain)
    disc_values_on_real = Discriminator(data_real, data_encoded, isTrain)
    disc_values_on_real_mean = tf.reduce_mean(disc_values_on_real)
    disc_values_on_generated = Discriminator(data_generated, data_latent, isTrain)
    disc_values_on_generated_mean = tf.reduce_mean(disc_values_on_generated)

    # Objectives

    objective_discriminator = -tf.reduce_mean(safe_log(disc_values_on_real)) - tf.reduce_mean(safe_log(1 - disc_values_on_generated))
    objective_generator_encoder = - tf.reduce_mean(safe_log(disc_values_on_generated)) - tf.reduce_mean(safe_log(1 - disc_values_on_real))


    #Variables

    all_variables = tf.trainable_variables()

    parameters_discriminator = [entry for entry in all_variables if "Discriminator" in entry.name]
    parameters_generator_encoder = [entry for entry in all_variables if "Generator" in entry.name or "Encoder" in entry.name]


