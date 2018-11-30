methods = ["Jacobian", "proposed", "linear"]

learning_rate_geodesics = 1e-2

n_geodesics = 20
n_interpolations_points_geodesic = 256 #128 # or 1024??
degree_polynomial_geodesic_latent = 4

n_train_iterations_geodesics = 500

n_discriminator_grid = 128
sample_grid_minima = [-3.,-3.]
sample_grid_maxima = [3.,3.]

endpoint_initialization_mode ="random"
#endpoint_initialization_mode ="horizontal_grid"

sampling_geodesic_coefficients = "uniform"
#sampling_geodesic_coefficients = "zeros"


penalty = False

log_directory_geodesics = 'logs_geo'


