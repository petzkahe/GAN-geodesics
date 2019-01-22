from GAN.swiss_roll.Geodesic_Learning.Standard_Approach.config_geodesics import *
from GAN.swiss_roll.utils.generate_data import *
from GAN.swiss_roll.Geodesic_Learning.Standard_Approach.compute_geodesics import compute_geodesics
from GAN.swiss_roll.utils.plotting import plot_geodesic


import os

if os.path.exists( log_directory_geodesics ):
    # raise Exception("The directory ({}) exists and should not be overwritten".format(log_directory_geodesics))
    pass
else:
    os.makedirs( log_directory_geodesics )
    print( "Log directory for geodesics is set to {}".format( log_directory_geodesics ) )


def initialize_endpoints_of_curve(initialization_mode):
    if initialization_mode == "random":

        z_start_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[1, dim_latent, n_geodesics] ).astype(
            'float32' )
        z_end_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                         size=[1, dim_latent, n_geodesics] ).astype( 'float32' )

    elif initialization_mode == "custom_repeat":
        z_starts = np.transpose( np.array(
            [[-0.5, 0.5], [-0.8, 0.8], [-1.0, 0.5], [-0.5, 0.4], [-0.55, 0.55], [-0.7, 0.7], [-0.85, 0.55],
             [-0.35, 0.37], [-0.8, 0.0], [-0.4, 0.6]] ) ).reshape( 1, 2, 10 )
        z_ends = np.transpose( np.array(
            [[0.5, 0.5], [0.3, 0.8], [0.8, 0.8], [0.2, 0.8], [0.25, 0.85], [0.22, 0.82], [0.33, 0.82], [0.52, 0.47],
             [0.0, 1.0], [0.5, -0.1]] ) ).reshape( 1, 2, 10 )
        z_start_value = np.repeat( z_starts, n_repeat, axis=2 )
        z_end_value = np.repeat( z_ends, n_repeat, axis=2 )


    elif initialization_mode == "random_repeat":
        z_start_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[1, dim_latent, n_geodesic_endpoints] ).astype(
            'float32' )
        z_start_value = np.repeat(z_start_value,n_repeat,axis=2)
        z_end_value = np.random.uniform( low=latent_min_value, high=latent_max_value,
                                           size=[1, dim_latent, n_geodesic_endpoints] ).astype(
            'float32' )
        z_end_value = np.repeat( z_end_value, 5, axis=2)


    elif initialization_mode == "custom":
        z_start_value = np.repeat( np.reshape( np.array( z_start_center ), (1, dim_latent, 1) ),  n_geodesics, axis=2 )
        z_end_value = np.repeat( np.reshape( np.array( z_end_center ), (1, dim_latent, 1) ), n_geodesics, axis=2 )


    else:
        raise Exception( "Initialization_mode {} not known".format( initialization_mode ) )

    return z_start_value, z_end_value


def sort_geodesics(_geodesics_dict):
    for method in methods:
        if method != 'linear' :
            _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values = _geodesics_dict[method]
            sorted_indices = np.argsort( _objective_values )
            _curves_in_latent_space_value = _curves_in_latent_space_value[:, :, sorted_indices]
            _curves_in_sample_space_value = _curves_in_sample_space_value[:, :, sorted_indices]
            _objective_values = _objective_values[sorted_indices]
            _geodesics_dict[method] = _curves_in_latent_space_value, _curves_in_sample_space_value, _objective_values

    return _geodesics_dict


z_start_values, z_end_values = initialize_endpoints_of_curve( endpoint_initialization_mode )
generate_real_samples = generate_real_data()
real_samples = generate_real_samples.__next__()

geodesics_dict, suppl_dict = compute_geodesics( z_start_values, z_end_values )

# function which compares local minimas. outputs sorted list of geodesics according to loss


geodesics_dict = sort_geodesics( geodesics_dict )

# returns a dictionary of results
# key = method
# value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value


for method in methods:
    [curves_in_latent_space_value, curves_in_sample_space_value, objective_values] = geodesics_dict[method]
    print(objective_values)
    plot_geodesic( real_samples, curves_in_latent_space_value, curves_in_sample_space_value, method, suppl_dict )

curves_in_latent_space_value, curves_in_sample_space_value = geodesics_dict["before"]
plot_geodesic( real_samples, curves_in_latent_space_value, curves_in_sample_space_value, "before", suppl_dict )
