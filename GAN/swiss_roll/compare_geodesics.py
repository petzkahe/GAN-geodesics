from GAN.swiss_roll.config_geodesics import *
from GAN.swiss_roll.generate_data import *


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



#train geodesics
#returns geodesics for all methods

#plots





# plot_geodesic(real_samples, geodesic_points_in_z_value, geodesic_points_in_sample_space_value, method)


