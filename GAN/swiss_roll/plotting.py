import matplotlib

matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt
from GAN.swiss_roll.config_GAN import *



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
