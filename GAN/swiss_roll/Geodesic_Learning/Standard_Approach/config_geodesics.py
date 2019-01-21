methods = ["linear", "Jacobian", "proposed"]

learning_rate_geodesics = 1e-2
adam_beta1 = 0.5
adam_beta2 = 0.9

n_train_iterations_geodesics = 20000

n_repeat = 5
n_geodesic_endpoints = 10
n_geodesics = n_repeat * n_geodesic_endpoints

n_interpolations_points_geodesic = 512
degree_polynomial_geodesic_latent = 6
initialization_coefficients = 0.1

n_discriminator_grid_sample = 128
sample_grid_minima = [-2, -3]
sample_grid_maxima = [2.5, 3]

n_discriminator_grid_latent = 512
latent_grid_minima = [-2, -2]
latent_grid_maxima = [2, 2]

endpoint_initialization_mode = "custom_repeat" # Options: random / custom / random_repeat / custom_repeat
# z_start_center = [-0.75,-0.75]
# z_end_center = [0.5,-0.75]
z_start_center = [-0.5, 0.5]
z_end_center = [0.3, 0.5]


sampling_geodesic_coefficients = "uniform" # Options: uniform / zeros

hyper_param_discriminator= 100.0

n_batch_size_curve_net = 128

penalty = False

log_directory_geodesics = 'logs_geo'

