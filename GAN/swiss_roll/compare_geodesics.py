from GAN.swiss_roll.config_geodesics import *
from GAN.swiss_roll.generate_data import *
from GAN.swiss_roll.compute_geodesics import compute_geodesics
from GAN.swiss_roll.plotting import plot_geodesic

import os

if os.path.exists(log_directory_geodesics):
    raise Exception("The directory ({}) exists and should not be overwritten".format(log_directory_geodesics))
else:
    os.makedirs(log_directory_geodesics)
    print("Log directory for geodesics is set to {}".format(log_directory_geodesics))


def initialize_endpoints_of_curve(initialization_mode):

    if initialization_mode == "random":

        z_start_value = np.random.uniform(low=latent_min_value, high=latent_max_value, size=[1, dim_latent, n_geodesics]).astype(
            'float32')
        z_end_value = np.random.uniform(low=latent_min_value, high=latent_max_value, size=[1, dim_latent, n_geodesics]).astype('float32')

    else:
        raise Exception("Initialization_mode {} not known".format(initialization_mode))

    return z_start_value, z_end_value





z_start_values, z_end_values = initialize_endpoints_of_curve(endpoint_initialization_mode)
generate_real_samples = generate_real_data()
real_samples = generate_real_samples.__next__()


geodesics_dict = compute_geodesics(z_start_values, z_end_values)


# returns a dictionary of results
# key = method
# value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value


for method in methods:
    curves_in_latent_space_value, curves_in_sample_space_value = geodesics_dict[method]

    plot_geodesic(real_samples, curves_in_latent_space_value, curves_in_sample_space_value,method)



#plots





# plot_geodesic(real_samples, geodesic_points_in_z_value, geodesic_points_in_sample_space_value, method)


