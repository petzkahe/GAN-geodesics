from GAN.mnist.Geodesic_Learning.config_geodesic_mnist import *
from GAN.mnist.utils.generate_data import *
from GAN.mnist.Geodesic_Learning.compute_geodesics_mnist import compute_geodesics
from GAN.mnist.utils.plotting import plot_geodesic
from GAN.mnist.utils.plotting import plot_geodesics_in_pca_space


import os

if os.path.exists( log_directory_geodesics ):
    # raise Exception("The directory ({}) exists and should not be overwritten".format(log_directory_geodesics))
    pass
else:
    os.makedirs( log_directory_geodesics )
    print( "Log directory for geodesics is set to {}".format(log_directory_geodesics))


def initialize_endpoints_of_curve(initialization_mode):
    if initialization_mode == "random":

        z_start_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[1, dim_latent, n_geodesics] ).astype(
            'float32' )
        z_end_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                         size=[1, dim_latent, n_geodesics] ).astype('float32')
    elif initialization_mode == "custom":

        z_start_value = np.repeat( np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[1, dim_latent, 1] ).astype(
            'float32' ),n_geodesics,axis=2)
        z_end_value = np.repeat(np.random.uniform( low=latent_min_value, high=latent_max_value, size=[1, dim_latent, 1] ).astype('float32'),n_geodesics, axis=2)

    else:
        raise Exception( "Initialization_mode {} not known".format( initialization_mode ) )

    return z_start_value, z_end_value


def sort_geodesics(_geodesics_dict):
    for method in methods:
        if method != 'linear':
            _curves_in_sample_space_value, _objective_values, _curves_in_pca_space_value = _geodesics_dict[method]
            sorted_indices = np.argsort( _objective_values )

            _curves_in_sample_space_value = _curves_in_sample_space_value[:, :, sorted_indices]
            _objective_values = _objective_values[sorted_indices]
            _curves_in_pca_space_value = _curves_in_pca_space_value[:, :, sorted_indices]
            _geodesics_dict[method] =_curves_in_sample_space_value, _objective_values, _curves_in_pca_space_value
    return _geodesics_dict


z_start_values, z_end_values = initialize_endpoints_of_curve( endpoint_initialization_mode )


geodesics_dict, geodesics_suppl_dict = compute_geodesics(z_start_values, z_end_values)


if endpoint_initialization_mode=="custom":
    geodesics_dict = sort_geodesics(geodesics_dict)




# returns a dictionary of results
# key = method
# value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value


reals,labels = geodesics_suppl_dict['reals']

for method in methods:
	print(method)
	curves_in_sample_space_value,cost,curves_in_pca_space_value = geodesics_dict[method]
	print(cost)
	plot_geodesic(curves_in_sample_space_value, method)

	plot_geodesics_in_pca_space(curves_in_pca_space_value,method,reals,labels)


