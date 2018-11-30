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

    elif initialization_mode == "horizontal_grid":

        z_start_value = np.zeros( [1, dim_latent, n_geodesics]).astype(
            'float32')
        y_axis = np.linspace(latent_min_value, latent_max_value, n_geodesics)
        x_axis = np.ones(n_geodesics)
        z_start_value[0,0,:]=latent_min_value*x_axis
        z_start_value[0,1,:]=y_axis

        z_end_value = np.zeros([1, dim_latent, n_geodesics]).astype(
            'float32')
        z_end_value[0,0,:]=latent_max_value*x_axis
        z_end_value[0,1,:]=y_axis

    elif initialization_mode == "vertical_grid":

        z_start_value = np.zeros( [1, dim_latent, n_geodesics]).astype(
            'float32')
        x_axis = np.linspace(latent_min_value, latent_max_value, n_geodesics)
        y_axis = np.ones(n_geodesics)
        z_start_value[0,0,:]=x_axis
        z_start_value[0,1,:]=latent_min_value*y_axis

        z_end_value = np.zeros([1, dim_latent, n_geodesics]).astype(
            'float32')
        z_end_value[0,0,:]=x_axis
        z_end_value[0,1,:]= latent_max_value*y_axis


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


#######################################################3333
########### DELETE ME WHEN DONE CHECKING STUFF
#############################################################

curves_in_latent_space_value, curves_in_sample_space_value = geodesics_dict["before"]

plot_geodesic(real_samples, curves_in_latent_space_value, curves_in_sample_space_value,"before")

############################################################
################################################################
#################################################################

#plots





# plot_geodesic(real_samples, geodesic_points_in_z_value, geodesic_points_in_sample_space_value, method)


