from GAN.swiss_roll.GAN_Learning.config_GAN import *

methods = ["linear", "Jacobian", "proposed"]

learning_rate_geodesics = 1e-3
adam_beta1 = 0.5
adam_beta2 = 0.9

n_train_iterations_geodesics = 4000

n_geodesics = 10
n_interpolations_points_geodesic = 128  # 128 # or 1024??
degree_polynomial_geodesic_latent = 4

n_discriminator_grid_sample = 128
sample_grid_minima = [-3., -3.]
sample_grid_maxima = [3., 3.]

n_discriminator_grid_latent = 512
latent_grid_minima = [-2., -3.]
latent_grid_maxima = [3., 2.]

endpoint_initialization_mode = "random"
# z_start_center = [-0.75,-0.75]
# z_end_center = [0.5,-0.75]
z_start_center = [-0.5, 0]
z_end_center = [0.5, -.2]
# endpoint_initialization_mode ="horizontal_grid"
n_endpoint_clusters = 1
z_start_center

sampling_geodesic_coefficients = "uniform"
# sampling_geodesic_coefficients = "zeros"


n_batch_size_curve_net = 128
dim_nn_curve_net = 2* (degree_polynomial_geodesic_latent-2)*dim_latent







penalty = False

log_directory_geodesics = 'logs_geo'


do_no_training = False

