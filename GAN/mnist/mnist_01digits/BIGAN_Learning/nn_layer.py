import numpy as np
import tensorflow as tf


def ReLuLayer(dim_latent, out_dim, input, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=- np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')
    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param()

    # assumes that input is of right size
    output = tf.matmul(input, weights)

    output = tf.nn.bias_add(output, biases)

    output = tf.nn.relu(output)

    return output

def ReLuLayerWithBatchN(dim_latent, out_dim, input, isTrain, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=- np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')
    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param()

    # assumes that input is of right size
    output = tf.matmul(input, weights)

    output = tf.nn.bias_add(output, biases)

    output = tf.layers.batch_normalization(output, training = isTrain, name=name+".batch_nn")

    output = tf.nn.relu(output)

    return output


def LinearLayer(dim_latent, out_dim, input, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=-np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')

    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param

    # assumes that input is of right size
    output = tf.matmul(input, weights)
    # else:
    #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
    #    result = tf.matmul(reshaped_inputs, weight)
    #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    output = tf.nn.bias_add(output, biases)

    return output

def SigmoidLayer(dim_latent, out_dim, input, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=-np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')

    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param

    # assumes that input is of right size
    output = tf.matmul(input, weights)
    # else:
    #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
    #    result = tf.matmul(reshaped_inputs, weight)
    #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    output = tf.nn.bias_add(output, biases)

    output = tf.nn.sigmoid(output)

    return output


def TanhLayer(dim_latent, out_dim, input, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=-np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')

    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param

    # assumes that input is of right size
    output = tf.matmul(input, weights)
    # else:
    #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
    #    result = tf.matmul(reshaped_inputs, weight)
    #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    output = tf.nn.bias_add(output, biases)

    output = tf.nn.tanh(output)

    return output


def LeakyReLuLayer(dim_latent, out_dim, input, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=-np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')

    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param

    # assumes that input is of right size
    output = tf.matmul(input, weights)
    # else:
    #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
    #    result = tf.matmul(reshaped_inputs, weight)
    #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    output = tf.nn.bias_add(output, biases)

    output = tf.nn.leaky_relu(output,alpha=0.2)

    return output

def LeakyReLuLayerWithBatchN(dim_latent, out_dim, input, isTrain, name):

    # initalize weights

    weight_initializations = np.random.uniform(
        low=-np.sqrt(2. / dim_latent),
        high=np.sqrt(2. / dim_latent),
        size=(dim_latent, out_dim)).astype('float32')

    bias_initializations = np.zeros(out_dim, dtype='float32')

    weights = tf.get_variable(name + ".W", initializer=weight_initializations)
    biases = tf.get_variable(name + ".b", initializer=bias_initializations)

    # To enable parameter sharing might actually need here the tflib library with lib.param

    # assumes that input is of right size
    output = tf.matmul(input, weights)
    # else:
    #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
    #    result = tf.matmul(reshaped_inputs, weight)
    #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    output = tf.nn.bias_add(output, biases)

    output = tf.layers.batch_normalization(output, training = isTrain, name=name + ".batch_nn")
    output = tf.nn.leaky_relu(output,alpha=0.2)

    return output

'''
def DeconvLayer(input, n_filters, filter_size,strides=(1,1),padding='valid', name='Default'):

    output = tf.layers.conv2d_transpose(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    
    return output

def ConvLayer(input, n_filters, filter_size,strides=(1,1),padding='valid', name='Default'):

    output = tf.layers.conv2d(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    

    return output

def LeakyReLuDeconvLayer(input, n_filters, filter_size,strides=(1,1),padding='valid', leak = 0.2, name='Default'):

    conv = tf.layers.conv2d_transpose(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    output = tf.nn.leaky_relu(conv,leak)

    return output

def LeakyReLuDeconvLayerWithBatchNN(input, n_filters, filter_size,strides=(1,1),padding='valid', leak = 0.2, name='Default'):

    conv = tf.layers.conv2d_transpose(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    bnorm = tf.layers.batch_normalization(conv, training = True, momentum = 0.9, name = name + ".batch_nn")    
    output = tf.nn.leaky_relu(bnorm,leak)
    
    return output

def LeakyReLuConvLayer(input, n_filters, filter_size,strides=(1,1),padding='valid', leak = 0.2, name='Default'):

    conv = tf.layers.conv2d(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    output = tf.nn.leaky_relu(conv,leak)

    return output

def LeakyReLuConvLayerWithBatchNN(input, n_filters, filter_size,strides=(1,1),padding='valid', leak = 0.2, name='Default'):

    conv = tf.layers.conv2d(input, n_filters, filter_size, strides=strides, padding=padding,name=name)
    bnorm = tf.layers.batch_normalization(conv, training = True, momentum = 0.9, name = name + ".batch_nn") 
    output = tf.nn.leaky_relu(bnorm,leak)
    
    return output
'''
