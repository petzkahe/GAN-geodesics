from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.config_geodesics_poly_nn import *
from GAN.swiss_roll.utils.generate_data import *
from GAN.swiss_roll.Geodesic_Learning.Polynomials_from_NN_Approach.compute_geodesics_poly_nn import \
    compute_geodesics_poly_nn
from GAN.swiss_roll.utils.plotting import plot_geodesic

import os

if os.path.exists(log_directory_geodesics):
    # raise Exception("The directory ({}) exists and should not be overwritten".format(log_directory_geodesics))
    pass
else:
    os.makedirs(log_directory_geodesics)
    print("Log directory for geodesics is set to {}".format(log_directory_geodesics))


def initialize_endpoints_of_curve(initialization_mode):
    if initialization_mode == "random":
        z_in_value = np.random.uniform(low=latent_min_value, high=latent_max_value,
                                       size=[n_batch_size_curve_net, 2 * dim_latent]).astype('float32')

    else:
        raise Exception("Initialization_mode {} not known".format(initialization_mode))

    return z_in_value


def sort_geodesics(_geodesics_dict):
    for meth in methods:
        if meth != 'linear':
            _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values = _geodesics_dict[meth]
            sorted_indices = np.argsort(_objective_values)
            _curves_in_latent_space_value = _curves_in_latent_space_value[:, :, sorted_indices]
            _curves_in_sample_space_value = _curves_in_sample_space_value[:, :, sorted_indices]
            _objective_values = _objective_values[sorted_indices]
            _geodesics_dict[meth] = _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values
    return _geodesics_dict


z_in = initialize_endpoints_of_curve(endpoint_initialization_mode)

generate_real_samples = generate_real_data()
real_samples = generate_real_samples.__next__()

geodesics_dict, suppl_dict = compute_geodesics_poly_nn(z_in)
# function which compares local minimas. outputs sorted list of geodesics according to loss

geodesics_dict = sort_geodesics(geodesics_dict)

# returns a dictionary of results
# key = method
# value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value


for method in methods:
    [curves_in_latent_space_value, curves_in_sample_space_value, qq] = geodesics_dict[method]
    plot_geodesic(real_samples, curves_in_latent_space_value, curves_in_sample_space_value, method, suppl_dict)


