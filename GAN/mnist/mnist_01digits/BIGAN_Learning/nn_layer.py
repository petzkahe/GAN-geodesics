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

def ReLuLayerWithBatchN(dim_latent, out_dim, input, name):

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

    output = tf.layers.batch_normalization(output)

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

def LeakyReLuLayerWithBatchN(dim_latent, out_dim, input, name):

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

    output = tf.layers.batch_normalization(output)
    output = tf.nn.leaky_relu(output,alpha=0.2)

    return output

