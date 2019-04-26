dim_latent = 32
#dim_latent = 15
dim_data = 28 * 28

dim_nn= 512
dim_nn_disc = 256
# Generator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> dim_latent
# Discriminator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> 1

n_batch_size = 128

which_gan = 'DCWGAN'
#which_gan = 'BIGAN'

n_discriminator_inner = 1
n_generator_encoder_inner = 1



log_directory_01 = "logs"

learning_rate = 1e-4
learning_rate_final = 1e-8

latent_distribution="uniform"
#For uniform latent_distribution, define the range
latent_max_value = 1.0
latent_min_value = -1.0
