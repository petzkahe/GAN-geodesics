methods = ["linear","Jacobian","proposed"]

learning_rate_geodesics = 1e-2
adam_beta1 = 0.9
adam_beta2 = 0.99

n_train_iterations_geodesics = 2000

n_geodesics = 1
n_interpolations_points_geodesic = 512 #128 # or 1024??
degree_polynomial_geodesic_latent = 4


n_discriminator_grid_sample = 128
sample_grid_minima = [-3.,-3.]
sample_grid_maxima = [3.,3.]

n_discriminator_grid_latent = 256
latent_grid_minima = [-2.,-3.]
latent_grid_maxima = [3.,2.]

endpoint_initialization_mode ="custom"
# z_start_center = [-0.75,-0.75]
# z_end_center = [0.5,-0.75]
z_start_center = [-0.5,0]
z_end_center = [0.5,-.2]
#endpoint_initialization_mode ="horizontal_grid"
n_endpoint_clusters = 1
z_start_center

#sampling_geodesic_coefficients = "uniform"
sampling_geodesic_coefficients = "zeros"


penalty = False

log_directory_geodesics = 'logs_geo'


