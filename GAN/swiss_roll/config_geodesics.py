methods = ["Jacobian", "proposed", "linear"]

learning_rate_geodesics = 1e-3

n_geodesics = 16
n_interpolations_points_geodesic = 128 #128 # or 1024??
degree_polynomial_geodesic_latent = 4

n_train_iterations_geodesics = 2000

n_discriminator_grid_sample = 128
sample_grid_minima = [-3.,-3.]
sample_grid_maxima = [3.,3.]

n_discriminator_grid_latent = 256
latent_grid_minima = [-2.,-3.]
latent_grid_maxima = [3.,2.]

endpoint_initialization_mode ="custom"
#endpoint_initialization_mode ="horizontal_grid"
n_endpoint_clusters = 1

sampling_geodesic_coefficients = "uniform"
#sampling_geodesic_coefficients = "zeros"


penalty = False

log_directory_geodesics = 'logs_geo'


