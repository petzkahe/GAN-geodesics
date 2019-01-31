import matplotlib
import numpy as np

matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.swiss_roll.GAN_Learning.config_GAN import *
from GAN.swiss_roll.Geodesic_Learning.Standard_Approach.config_geodesics import *



def plot_sample_space(batch_real_data, batch_generated_data, iteration_step):
	"""
	Generates and saves a plot of the true distribution, the generator, and the
	critic.
	"""
	plt.clf()  # clear current figure


	plt.scatter(batch_real_data[:, 0], batch_real_data[:, 1], c='orange', marker='+')
	plt.scatter(batch_generated_data[:, 0], batch_generated_data[:, 1], c='green', marker='+')

	plt.savefig('{}/frame_{}.pdf'.format(log_directory, iteration_step))
	plt.savefig( '{}/frame_{}.eps'.format( log_directory, iteration_step ) )

	return None



def plot_geodesic(samples_real, geodesics_in_latent, geodesics_in_sample_space, method, suppl_dict):


	disc_values_over_latent_grid = suppl_dict["disc_values_over_latent_grid"]

	plt.clf()

	c = plt.pcolormesh(np.linspace(latent_grid_minima[0], latent_grid_maxima[0], n_discriminator_grid_latent),
				   np.linspace(latent_grid_minima[1], latent_grid_maxima[1], n_discriminator_grid_latent),
				   np.transpose(disc_values_over_latent_grid), cmap='gray')
	plt.colorbar(c)


	if endpoint_initialization_mode == "custom":
		for k_geodesics in range(1,geodesics_in_latent.shape[2]):
			plt.scatter(geodesics_in_latent[:, 0, k_geodesics], geodesics_in_latent[:, 1, k_geodesics],
					color='green', marker='.', s=1)
		#plt.scatter(geodesics_in_latent[:, 0, 0], geodesics_in_latent[:, 1, 0],
		#                color='yellow', marker='.', s=1)
		plt.plot(geodesics_in_latent[:, 0, 0], geodesics_in_latent[:, 1, 0], 'y-')

	else:
		for k_geodesics in range(geodesics_in_latent.shape[2] ):
			plt.scatter( geodesics_in_latent[:, 0, k_geodesics], geodesics_in_latent[:, 1, k_geodesics],
						 color='green', marker='.', s=1 )




	plt.savefig('{}/geodesics_in_latent_space_{}.png'.format(log_directory_geodesics, method))
	#plt.savefig('{}/geodesics_in_latent_space_{}.eps'.format(log_directory_geodesics, method))
	#plt.savefig( '{}/geodesics_in_latent_space_{}.pdf'.format( log_directory_geodesics, method ) )


	# This plots the geodesics in sample space

	plt.clf()

	disc_values_over_sample_grid = suppl_dict["disc_values_over_sample_grid"]



	c = plt.pcolormesh(np.linspace(sample_grid_minima[0], sample_grid_maxima[0], n_discriminator_grid_sample),
				   np.linspace(sample_grid_minima[1], sample_grid_maxima[1], n_discriminator_grid_sample),
				   np.transpose(disc_values_over_sample_grid), cmap='gray')
	plt.colorbar(c)




	plt.scatter(samples_real[:, 0], samples_real[:, 1], c='orange', marker='+')

	if endpoint_initialization_mode == "custom":

		for k_geodesics in range(1,geodesics_in_latent.shape[2]):
			plt.scatter(geodesics_in_sample_space[:, 0, k_geodesics],
					geodesics_in_sample_space[:, 1, k_geodesics], color='green', marker='.', s=4)
		#plt.scatter(geodesics_in_sample_space[:, 0, 0], geodesics_in_sample_space[:, 1, 0], color='yellow', marker='.', s=4)
		plt.plot(geodesics_in_sample_space[:, 0, 0], geodesics_in_sample_space[:, 1, 0], 'y-')

	else:
		for k_geodesics in range(geodesics_in_latent.shape[2] ):
			plt.scatter( geodesics_in_sample_space[:, 0, k_geodesics],
						 geodesics_in_sample_space[:, 1, k_geodesics], color='green', marker='.', s=4 )

	plt.savefig('{}/geodesics_in_sample_space_{}.png'.format(log_directory_geodesics, method))
	plt.savefig('{}/geodesics_in_sample_space_{}.eps'.format(log_directory_geodesics, method))
	plt.savefig( '{}/geodesics_in_sample_space_{}.pdf'.format( log_directory_geodesics, method ) )

	return None

