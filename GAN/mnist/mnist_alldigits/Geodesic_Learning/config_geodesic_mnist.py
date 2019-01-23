methods = ["linear", "Jacobian", "proposed"]

learning_rate_geodesics = 1e-2
adam_beta1 = 0.5
adam_beta2 = 0.999

n_train_iterations_geodesics = 10000

n_geodesics = 5
n_interpolations_points_geodesic_approx = 256 # 128 # or 1024??
degree_polynomial_geodesic_latent = 3

n_interp_selected = 15

increment=int((n_interpolations_points_geodesic_approx)/(n_interp_selected-1.0))
n_interpolations_points_geodesic = (n_interp_selected-1)*increment

hyper_lambda = 100.0


dim_pca = 2
n_batch_pca = 1000



endpoint_initialization_mode = "random"
# z_start_center = [-0.75,-0.75]
# z_end_center = [0.5,-0.75]



sampling_geodesic_coefficients = "uniform"
initialization_value_coefficients=0.5
#sampling_geodesic_coefficients = "zeros"

penalty=False

log_directory_geodesics = 'logs_geo'

n_video_frames = 256 # must be below 1000