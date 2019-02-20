import tensorflow as tf
import GAN.mnist.mnist_01digits.BIGAN_Learning.nn_layer as layer
from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *
import numpy as np

with tf.variable_scope("BIGAN", reuse=tf.AUTO_REUSE):

    def Generator(noise):
        '''
        input dimz -> reshape to 4-dim tensor
        1x1xdimz   -> lrelu + deconv + batchnorm
        7x7x128    -> lrelu + deconv + batchnorm
        14x14x64   -> lrelu + deconv + batchnorm
        28x28x32   -> lrelu + deconv + batchnorm
        28x28x1    -> reshape into 28**2
        784 output
        
        '''
        input = tf.reshape(noise,shape=[-1,1,1,dim_latent])
        output = layer.LeakyReLuDeconvLayerWithBatchNN(input, 128,[7,7],strides=(1,1),padding='valid',name='Generator.1')
        output = layer.LeakyReLuDeconvLayerWithBatchNN(output,64, [7,7],strides=(2,2),padding='same',name='Generator.2')
        output = layer.LeakyReLuDeconvLayerWithBatchNN(output,32, [7,7],strides=(2,2),padding='same',name='Generator.3')
        output = layer.LeakyReLuDeconvLayer(output,1,  [7,7],strides=(1,1),padding='same', name='Generator.4')
        output = tf.reshape(output,shape=[-1,28**2])

        return output

  
    def Discriminator(inputs):
        '''
        input 28**2 -> reshape to 4-dim tensor
        28x28x1   -> lrelu + conv + batchnorm
        14x14x32    -> lrelu + conv + batchnorm
        7x7x64   -> lrelu + conv + batchnorm
        1x1x128   -> dense + sigmoid
        1x1x1 output
        
        '''
        input = tf.reshape(inputs,shape=[-1,28,28,1])
        output = layer.LeakyReLuConvLayer(input, 32, [7,7],strides=(2,2),padding='same',name='Discriminator.1')
        output = layer.LeakyReLuConvLayerWithBatchNN(output,64, [7,7],strides=(2,2),padding='same',name='Discriminator.2')
        output = layer.LeakyReLuConvLayerWithBatchNN(output,128,[7,7],strides=(2,2),padding='valid',name='Discriminator.3')
        output = tf.layers.flatten(output)
        output = layer.LinearLayer(128,1,output,'Discriminator.4')        

        return output


    # def Encoder(inputs):
    #     output = layer.LeakyReLuLayerWithBatchN( dim_data, dim_nn, inputs, "Encoder.1" )
    #     output = layer.LeakyReLuLayerWithBatchN( dim_nn, dim_nn, output, "Encoder.2" )
    #     output = layer.LinearLayer( dim_nn, dim_latent, output, "Encoder.3" )
    #     #output = layer.TanhLayer( dim_nn, dim_latent, output, "Encoder.3" )
    #     return output


    def safe_log(x):
        return tf.log( x + 1e-8 )

    #####################################################################
    #####################################################################
    # Placeholder


    data_real = tf.placeholder(tf.float32, shape=[None, dim_data], name='reals')
    data_latent = tf.placeholder(tf.float32, shape=[None, dim_latent], name='latent')



    #####################################################################
    #####################################################################
    # Build graph


    data_generated = Generator(data_latent)
    #data_encoded = Encoder(data_real)
    disc_values_on_real = Discriminator(data_real)
    disc_values_on_real_mean = tf.reduce_mean(disc_values_on_real)
    disc_values_on_generated = Discriminator(data_generated)
    disc_values_on_generated_mean = tf.reduce_mean(disc_values_on_generated)

    # Objectives

    hyperparameter = 10
    #alpha = tf.random_uniform(shape=[batch_size,1,1,1],minval=0., maxval=1.)
    
    gradients = tf.gradients(disc_values_on_real, data_real)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
    
    D_loss_fake = tf.reduce_mean(disc_values_on_generated)
    D_loss_real = -tf.reduce_mean(disc_values_on_real) + hyperparameter*gradient_penalty

    objective_generator_encoder = -tf.reduce_mean(disc_values_on_generated) #+ tf.reduce_mean(disc_values_on_real)


    objective_discriminator = D_loss_real + D_loss_fake


    #Variables

    all_variables = tf.trainable_variables()

    parameters_discriminator = [entry for entry in all_variables if "Discriminator" in entry.name]
    parameters_generator_encoder = [entry for entry in all_variables if "Generator" in entry.name or "Encoder" in entry.name]


