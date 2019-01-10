dim_latent = 32
dim_data = 28 * 28

dim_nn= 256
# Generator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> dim_latent
# Discriminator NN is then of form dim_latent -> dim_nn -> dim_nn*2 -> dim_nn -> 1

n_batch_size = 64



n_BIGAN_iterations = 1000001
n_discriminator_inner = 1
n_generator_encoder_inner = 1



log_directory = "logs"

learning_rate = 1e-5

latent_distribution="uniform"
#For uniform latent_distribution, define the range
latent_max_value = 1.0
latent_min_value = -1.0
