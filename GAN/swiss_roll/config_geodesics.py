methods = ["Jacobian", "proposed"]

learning_rate_geodesics = 1e-1

n_geodesics = 100
n_interpolations_points_geodesic = 1024 #128 # or 1024??
degree_polynomial_geodesic_latent = 4

n_train_iterations_geodesics = 500

#endpoint_initialization_mode ="random"
endpoint_initialization_mode ="horizontal_grid"

sampling_geodesic_coefficients = "uniform"


penalty = False

log_directory_geodesics = 'logs_geo'