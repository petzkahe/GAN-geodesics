import matplotlib
import numpy as np

matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.swiss_roll.config_GAN import *
from GAN.swiss_roll.config_geodesics import *



def plot_sample_space(batch_real_data, batch_generated_data, iteration_step):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    plt.clf()  # clear current figure


    plt.scatter(batch_real_data[:, 0], batch_real_data[:, 1], c='orange', marker='+')
    plt.scatter(batch_generated_data[:, 0], batch_generated_data[:, 1], c='green', marker='+')

    plt.savefig('{}/frame_{}.png'.format(log_directory, iteration_step))

    return None


def plot_loss_surface(_dict):

    loss_surface_proposed = _dict['loss_surface_proposed'] 
    loss_surface_Jacobian = _dict['loss_surface_Jacobian'] 

    xyaxis = np.linspace(coefficient_range[0], coefficient_range[1], n_loss_grid)

    plt.clf()

    c = plt.pcolormesh(xyaxis,xyaxis, np.transpose(np.log(loss_surface_proposed+1e-8)), cmap='gray')
    plt.colorbar(c)

    plt.savefig('{}/loss_surface_proposed.png'.format(log_directory_geodesics))

    
    plt.clf()

    c = plt.pcolormesh(xyaxis,xyaxis, np.transpose(np.log(loss_surface_Jacobian+1e-8)), cmap='gray')
    plt.colorbar(c)

    plt.savefig('{}/loss_surface_Jacobian.png'.format(log_directory_geodesics))



def plot_geodesic(samples_real, geodesics_in_latent, geodesics_in_sample_space, method, suppl_dict):



    # #  Creates fakes for 64^2 many uniformly distributed images
    # points = np.zeros((n_discriminator_grid, n_discriminator_grid, 2), dtype='float32')
    # points[:, :, 0] = np.linspace(min_value, max_value, n_discriminator_grid)[:,None]
    # # for zero: for any second entry linspace runs over first coordinate
    # points[:, :, 1] = np.linspace(min_value, max_value, n_discriminator_grid)[None,:]
    # # for one: for any first entry linspace runs over second coordinate
    # points = points.reshape((-1, 2))  # gives list of points of all combinations
    #
    #
    # samples_fake, critic_value_on_fake, = session.run(
    #     [fakes, critic_fake, ],
    #     feed_dict={noise: points}
    # )
    #
    #
    #
    #
    # ## Plot geodesics in sample space
    #
    # plt.clf()  # clear current figure
    #
    # #  Creates a grid in sample space
    # x_grid = np.zeros((n_discriminator_grid, n_discriminator_grid, 2), dtype='float32')
    # x_grid[:, :, 0] = np.linspace(-2., 2., n_discriminator_grid)[:,None]
    # # for zero: for any second entry linspace runs over first coordinate
    # x_grid[:, :, 1] = np.linspace(-2., 2., n_discriminator_grid)[None,:]
    # # for one: for any first entry linspace runs over second coordinate
    # x_grid = x_grid.reshape((-1, 2))  # gives list of points of all combinations
    #
    # # Pass grid through Discriminator
    # [disc_over_x] = session.run([critic_real], feed_dict={reals:x_grid})
    #
    # disc_over_x=disc_over_x.reshape((n_discriminator_grid, n_discriminator_grid))
    #
    # c = plt.pcolormesh(np.linspace(-2., 2., n_discriminator_grid),
    #                np.linspace(-2., 2., n_discriminator_grid),
    #                np.transpose(disc_over_x), cmap='gray')
    # plt.colorbar(c)
    #
    #
    # plt.scatter(samples_real[:, 0], samples_real[:, 1], c='orange', marker='+')
    #
    # for k_geodesics in range(no_geodesics):
    #     plt.scatter(geodesic_points_in_sample_space_value[:, 0, k_geodesics],
    #                 geodesic_points_in_sample_space_value[:, 1, k_geodesics], color='green', marker='.', s=4)
    #
    #
    # plt.savefig('{}/reproduction/geodesics_in_sample_space_{}.pdf'.format(log_directory,method))




    ## Plot geodesics in latent space

    # metric_measure_on_fakes = np.zeros(n_discriminator_grid ** 2)
    #
    # for i in range(0, n_discriminator_grid ** 2):
    #     _, critic_fake_out, metric_measure_on_fakes[i] = session.run(
    #         [fakes, critic_fake, determinant_metric],
    #         feed_dict={noise: points[i, :].reshape(1, 2)})
    #
    #     metric_measure_on_fakes[i] = np.log(metric_measure_on_fakes[i])




    # ax = fig.gca(projection='3d')
    # plt.pcolormesh([points[:, 0], points[:, 1]],critic_value_on_fake)
    # ax.plot_surface(points[:, 0], points[:, 1],critic_value_on_fake, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)
    # c = plt.pcolormesh(np.linspace(min_value, max_value, n_discriminator_grid), np.linspace(min_value, max_value, n_discriminator_grid),
    #                np.transpose(metric_measure_on_fakes.reshape((n_discriminator_grid, n_discriminator_grid))),cmap='gray')
    # plt.colorbar(c)
    # # plt.scatter(points[:, 0], points[:, 1],critic_value_on_fake, c='green', marker='+')


    # plt.scatter(samples_real[:, 0], samples_real[:, 1], c='orange', marker='+')


    # This plots the geodesics in latent space


    disc_values_over_latent_grid = suppl_dict["disc_values_over_latent_grid"]

    plt.clf()

    c = plt.pcolormesh(np.linspace(latent_grid_minima[0], latent_grid_maxima[0], n_discriminator_grid_latent),
                   np.linspace(latent_grid_minima[1], latent_grid_maxima[1], n_discriminator_grid_latent),
                   np.transpose(disc_values_over_latent_grid), cmap='gray')
    plt.colorbar(c)


    for k_geodesics in range(1,n_geodesics):
        plt.scatter(geodesics_in_latent[:, 0, k_geodesics], geodesics_in_latent[:, 1, k_geodesics],
                    color='green', marker='.', s=4)
    plt.scatter(geodesics_in_latent[:, 0, 0], geodesics_in_latent[:, 1, 0],
                    color='yellow', marker='.', s=4)
    

    #plt.xlim(latent_min_value, latent_max_value)
    #plt.ylim(latent_min_value, latent_max_value)


    plt.savefig('{}/geodesics_in_latent_space_{}.png'.format(log_directory_geodesics, method))



    # This plots the geodesics in sample space

    plt.clf()

    disc_values_over_sample_grid = suppl_dict["disc_values_over_sample_grid"]



    c = plt.pcolormesh(np.linspace(sample_grid_minima[0], sample_grid_maxima[0], n_discriminator_grid_sample),
                   np.linspace(sample_grid_minima[1], sample_grid_maxima[1], n_discriminator_grid_sample),
                   np.transpose(disc_values_over_sample_grid), cmap='gray')
    plt.colorbar(c)




    plt.scatter(samples_real[:, 0], samples_real[:, 1], c='orange', marker='+')

    for k_geodesics in range(1,n_geodesics):
        plt.scatter(geodesics_in_sample_space[:, 0, k_geodesics],
                    geodesics_in_sample_space[:, 1, k_geodesics], color='green', marker='.', s=4)
    plt.scatter(geodesics_in_sample_space[:, 0, 0], geodesics_in_sample_space[:, 1, 0], color='yellow', marker='.', s=4)


    plt.savefig('{}/geodesics_in_sample_space_{}.png'.format(log_directory_geodesics, method))

    return None














###### OLD PLOTTING CODE

    #
    # # Plot second picture for noise to discriminator value image
    #
    # fig = plt.figure()
    # #ax = fig.gca(projection='3d')
    # # plt.pcolormesh([points[:, 0], points[:, 1]],critic_value_on_fake)
    # #ax.plot_surface(points[:, 0], points[:, 1],critic_value_on_fake, cmap=cm.coolwarm,
    # #                       linewidth=0, antialiased=False)
    # c = plt.pcolor(np.linspace(latent_min_value, latent_max_value, n_latent_grid), np.linspace(latent_min_value, latent_max_value, n_latent_grid), critic_value_on_fake.reshape((n_latent_grid,n_latent_grid)) )
    # plt.colorbar(c)
    # #plt.scatter(points[:, 0], points[:, 1],critic_value_on_fake, c='green', marker='+')
    #
    # plt.savefig('{}/frame_{}_disc_values.png'.format(log_directory, iteration_step))
    #
    # plt.clf()  # clear current figure
    #
    # if iteration_step  % 5000 == 4999:
    #
    #     # Plot third picture for Jacobian
    #
    #     metric_measure_on_fakes = np.zeros(n_latent_grid**2)
    #     metric_measure_over_disc_on_fakes = np.zeros(n_latent_grid**2)
    #
    #     for i in range(0, n_latent_grid ** 2):
    #         _, disc_values_on_generated_out, metric_measure_on_fakes[i] = session.run(
    #             [data_generated, disc_values_on_generated, determinant_metric],
    #             feed_dict={data_latent: points[i, :].reshape(1, 2)})
    #
    #         metric_measure_on_fakes[i] = np.log(metric_measure_on_fakes[i])
    #
    #         metric_measure_over_disc_on_fakes[i]=metric_measure_on_fakes[i] - np.log(disc_values_on_generated_out[0])
    #
    #
    #
    #     fig = plt.figure()
    #     # ax = fig.gca(projection='3d')
    #     # plt.pcolormesh([points[:, 0], points[:, 1]],critic_value_on_fake)
    #     # ax.plot_surface(points[:, 0], points[:, 1],critic_value_on_fake, cmap=cm.coolwarm,
    #     #                       linewidth=0, antialiased=False)
    #     c = plt.pcolor(np.linspace(latent_min_value, latent_max_value, n_latent_grid), np.linspace(latent_min_value, latent_max_value, n_latent_grid),
    #                    metric_measure_on_fakes.reshape((n_latent_grid, n_latent_grid)))
    #     plt.colorbar(c)
    #     # plt.scatter(points[:, 0], points[:, 1],critic_value_on_fake, c='green', marker='+')
    #
    #     plt.savefig('{}/frame_{}_metric_norm.png'.format(log_directory, iteration_step))
    #
    #     plt.clf()  # clear current figure
    #
    #
    #
    #     # Plot fourth picture for Jacobian together with 1 over discriminator
    #
    #     fig = plt.figure()
    #     # ax = fig.gca(projection='3d')
    #     # plt.pcolormesh([points[:, 0], points[:, 1]],critic_value_on_fake)
    #     # ax.plot_surface(points[:, 0], points[:, 1],critic_value_on_fake, cmap=cm.coolwarm,
    #     #                       linewidth=0, antialiased=False)
    #     c = plt.pcolor(np.linspace(latent_min_value, latent_max_value, n_latent_grid), np.linspace(latent_min_value, latent_max_value, n_latent_grid),
    #                    metric_measure_over_disc_on_fakes.reshape((n_latent_grid, n_latent_grid)))
    #     plt.colorbar(c)
    #     # plt.scatter(points[:, 0], points[:, 1],critic_value_on_fake, c='green', marker='+')
    #
    #     plt.savefig('{}/frame_{}_metric_norm_over_disc.png'.format(log_directory, iteration_step))
    #
    #     plt.clf()  # clear current figure
