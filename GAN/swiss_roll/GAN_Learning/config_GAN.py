dim_latent = 2
dim_data = 2

dim_nn= 512
# Generator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> dim_latent
# Discriminator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> 1

n_batch_size = 256

n_GAN_iterations = 13000
n_discriminator_inner = 5
n_generator_inner = 1

n_latent_grid = 128


log_directory = "logs"

learning_rate = 1e-4

latent_distribution="uniform"
#For uniform latent_distribution, define the range
latent_max_value = 1.0
latent_min_value = -1.0

