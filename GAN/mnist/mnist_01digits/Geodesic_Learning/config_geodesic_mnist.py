methods = ["linear", "Jacobian", "proposed"]

learning_rate_geodesics = 1e-2
adam_beta1 = 0.5
adam_beta2 = 0.999

n_train_iterations_geodesics = 5000

n_geodesics = 5 
n_interpolations_points_geodesic_approx = 128 # 128 # or 1024??
degree_polynomial_geodesic_latent = 3

n_interp_selected = 15

increment=int((n_interpolations_points_geodesic_approx)/(n_interp_selected-1.0))
n_interpolations_points_geodesic = (n_interp_selected-1)*increment

hyper_lambda = 100


dim_pca = 2

n_batch_pca = 1000
n_pca_grid_per_dimension = 100
n_pca_gridpoints = n_pca_grid_per_dimension**dim_pca
pca_grid_minima = [-20,-10]
pca_grid_maxima = [5,10]


n_latent_background = 100000
latent_points_minima = -2
latent_points_maxima = 2


endpoint_initialization_mode = "custom_random"
pca_start = [[-8,-2]]
pca_end = [[5,5]]
# z_start_center = [-0.75,-0.75]
# z_end_center = [0.5,-0.75]



sampling_geodesic_coefficients = "uniform"
initialization_value_coefficients=0.5
#sampling_geodesic_coefficients = "zeros"

penalty=False

log_directory_geodesics = 'logs_geo'