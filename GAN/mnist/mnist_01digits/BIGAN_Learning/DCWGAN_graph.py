import tensorflow as tf
import GAN.mnist.mnist_01digits.BIGAN_Learning.nn_layer as layer
from GAN.mnist.mnist_01digits.BIGAN_Learning.config_BIGAN import *
import numpy as np

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

with tf.variable_scope("BIGAN", reuse=tf.AUTO_REUSE):


    def Generator(noise,isTrain = True):
        '''
        input dimz -> reshape to 4-dim tensor
        1x1xdimz   -> lrelu + deconv + batchnorm
        7x7x128    -> lrelu + deconv + batchnorm
        14x14x64   -> lrelu + deconv + batchnorm
        28x28x32   -> lrelu + deconv + batchnorm
        28x28x1    -> reshape into 28**2
        784 output
        
        
        #print(noise)
        #inputs = tf.reshape(noise,shape=[-1,1,1,dim_latent])
        inputs = noise
        print('Generator:')
        print('Input: ' + str(inputs.get_shape()))
        
        output = layer.LeakyReLuDeconvLayerWithBatchNN(inputs, 128,[7,7],strides=(1,1),padding='valid',name='Generator.1')
        print('Layer 1: ' + str(output.get_shape()))
        output = layer.LeakyReLuDeconvLayerWithBatchNN(output,64, [7,7],strides=(2,2),padding='same',name='Generator.2')
        print('Layer 2: ' + str(output.get_shape()))
        output = layer.LeakyReLuDeconvLayerWithBatchNN(output,32, [7,7],strides=(2,2),padding='same',name='Generator.3')
        print('Layer 3: ' + str(output.get_shape()))
        output = layer.DeconvLayer(output,                    1 , [7,7],strides=(1,1),padding='same', name='Generator.4')
        output = tf.nn.tanh(output)
        print('Output: ' + str(output.get_shape()))
        #output = tf.reshape(output,shape=[-1,28**2])
        #utput = (output + 1)/2.
        '''

    
        print('Generator:')
        print('Input:' + str(noise.get_shape()))

        
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(noise, 128, [7, 7], strides=(1, 1), padding='valid',name='Generator.1')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain,name='Generator.1.batch_nn'), 0.2)
        #lrelu1 = x
        print('1st layer: ' + str(conv1.get_shape()))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [7, 7], strides=(2, 2), padding='same',name='Generator.2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain,name='Generator.2.batch_nn'), 0.2)
        print('2nd layer: ' + str(conv2.get_shape()))

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 32, [7, 7], strides=(2, 2), padding='same',name='Generator.3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain,name='Generator.'), 0.2)
        print('3rd layer: ' + str(conv3.get_shape()))

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 1, [7, 7], strides=(1, 1), padding='same',name='Generator.4')
        #lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        print('4th layer: ' + str(conv4.get_shape()))

        # output layer
        #conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(1, 1), padding='same')
        #print('5th layer: ' + str(conv5.get_shape()))
        output = tf.nn.tanh(conv4)/2. + 0.5

        return output

  
    def Discriminator(inputs,isTrain = True):
        '''
        input 28**2 -> reshape to 4-dim tensor
        28x28x1   -> lrelu + conv + batchnorm
        14x14x32    -> lrelu + conv + batchnorm
        7x7x64   -> lrelu + conv + batchnorm
        1x1x128   -> dense + sigmoid
        1x1x1 output
        
        
        print('Discriminator:')
        #inputs = inputs*2. - 1
        #inputs = tf.reshape(inputs,shape=[-1,28,28,1])
        print('Input: ' + str(inputs.get_shape()))
        output = layer.LeakyReLuConvLayer(inputs, 32, [7,7],strides=(2,2),padding='same',name='Discriminator.1')
        print('Layer 1: ' + str(output.get_shape()))
        output = layer.LeakyReLuConvLayerWithBatchNN(output,64, [7,7],strides=(2,2),padding='same',name='Discriminator.2')
        print('Layer 2: ' + str(output.get_shape()))
        output = layer.LeakyReLuConvLayerWithBatchNN(output,128,[7,7],strides=(2,2),padding='valid',name='Discriminator.3')
        print('Layer 3: ' + str(output.get_shape()))
        output = tf.layers.flatten(output)
        print('Flatten: ' + str(output.get_shape()))
        output = layer.LinearLayer(128,1,output,'Discriminator.4')        
        print('Output: ' + str(output.get_shape()))
        
        inputs = tf.layers.flatten(inputs)
        output = layer.LeakyReLuLayerWithBatchN(dim_data, dim_nn_disc, inputs,"Discriminator.1")
        output = layer.LeakyReLuLayerWithBatchN(dim_nn_disc, dim_nn_disc, output, "Discriminator.2")
        output = layer.LinearLayer(dim_nn_disc, 1, output, "Discriminator.3")
        '''
        inputs = (inputs - 0.5)*2.
      
        print('Discriminator:')
        print('Input:' + str(inputs.get_shape()))

        conv1 = tf.layers.conv2d(inputs, 32, [7, 7], strides=(2, 2), padding='same',name='Discriminator.1')
        lrelu1 = lrelu(conv1, 0.2)
        print('1st layer: ' + str(conv1.get_shape()))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 64, [7, 7], strides=(2, 2), padding='same',name='Discriminator.2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain,name='Discriminator.2.batch_nn'), 0.2)
        print('2nd layer: ' + str(conv2.get_shape()))

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 128, [7, 7], strides=(1, 1), padding='valid',name='Discriminator.3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, trainable = True, training=isTrain,name='Discriminator.3.batch_nn'), 0.2)
        print('3rd layer: ' + str(conv3.get_shape()))
        
        # 4th hidden layer
        #conv4 = tf.layers.conv2d(lrelu3, 1, [7, 7], strides=(2, 2), padding='same')
        #lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        #lrelu4 = lrelu3
        flat = tf.reshape(lrelu3,shape=[-1,128])
        print('Flattening: ' + str(flat.get_shape()))

        dense = tf.layers.dense(flat,1,name='Discriminator.4')

        print('4th layer: ' + str(dense.get_shape()))

        # output layer
        #conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        #print('5th layer: ' + str(conv5.get_shape()))

        output = dense #tf.nn.sigmoid(conv5)



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


    #data_real = tf.placeholder(tf.float32, shape=[None, dim_data], name='reals')
    #data_latent = tf.placeholder(tf.float32, shape=[None, dim_latent], name='latent')

    data_real = tf.placeholder(tf.float32, shape=[None, 28**2], name='reals')
    data_latent = tf.placeholder(tf.float32, shape=[None, dim_latent], name='latent')
    isTrain = tf.placeholder(dtype=tf.bool)

    #####################################################################
    #####################################################################
    # Build graph

    data_real_reshaped = tf.reshape(data_real,shape=[-1,28,28,1])  
    data_latent_reshaped = tf.reshape(data_latent,shape=[-1,1,1,dim_latent])

    data_generated_reshaped = Generator(data_latent_reshaped,isTrain)
    data_generated = tf.reshape(data_generated_reshaped,shape=[-1,28**2])
    #data_encoded = Encoder(data_real)
    disc_values_on_real = Discriminator(data_real_reshaped,isTrain)
    disc_values_on_real_mean = tf.reduce_mean(disc_values_on_real)
    disc_values_on_generated = Discriminator(data_generated_reshaped,isTrain)
    disc_values_on_generated_mean = tf.reduce_mean(disc_values_on_generated)

    # Objectives

    hyperparameter = 10
    #alpha = tf.random_uniform(shape=[batch_size,1,1,1],minval=0., maxval=1.)
    
    gradients = tf.gradients(disc_values_on_real, data_real_reshaped)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
    
    D_loss_fake = tf.reduce_mean(disc_values_on_generated)
    D_loss_real = -tf.reduce_mean(disc_values_on_real) + hyperparameter*gradient_penalty

    objective_discriminator = D_loss_real + D_loss_fake

    objective_generator_encoder = -tf.reduce_mean(disc_values_on_generated) #+ tf.reduce_mean(disc_values_on_real)


    
    #Variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    all_variables = tf.trainable_variables()

    parameters_discriminator = [entry for entry in all_variables if "Discriminator" in entry.name]
    parameters_generator_encoder = [entry for entry in all_variables if "Generator" in entry.name or "Encoder" in entry.name]
    