import matplotlib
import numpy as np

# sys.path = A list of strings that specifies the search path for modules.
# Initialized from the environment variable PYTHONPATH, plus an installation-dependent default.
# import random
matplotlib.use('pdf')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import sklearn.datasets  # sklearn for machine learning  in python

from GAN.swiss_roll.config_GAN import *



# generate real samples
def generate_real_data():

    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=n_batch_size,
            noise=0.25
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5  # stdev plus a little
        yield data


def generate_latent_data():

    if latent_distribution == "uniform":

        if dim_latent == 1:

            points = np.zeros((n_latent_grid, 1), dtype='float32')
            points[:, 0] = np.linspace(latent_min_value, latent_max_value, n_latent_grid**2)


        elif dim_latent == 2:

            #  Creates fakes for 64^2 many uniformly distributed images
            points = np.zeros((n_latent_grid, n_latent_grid, 2), dtype='float32')
            points[:, :, 0] = np.linspace(latent_min_value, latent_max_value, n_latent_grid)[:, None]
            # for zero: for any second entry linspace runs over first coordinate
            points[:, :, 1] = np.linspace(latent_min_value, latent_max_value, n_latent_grid)[None, :]
            # for one: for any first entry linspace runs over second coordinate
            points = points.reshape((-1, 2))  # gives list of points of all combinations

        elif dim_latent == 3:

            # n_latent_grid = 16
            points = np.zeros((n_latent_grid, n_latent_grid, n_latent_grid, 3))
            points[:, :, :, 0] = np.linspace(latent_min_value, latent_max_value, n_latent_grid)[:, None, None]
            points[:, :, :, 1] = np.linspace(latent_min_value, latent_max_value, n_latent_grid)[None, :, None]
            points[:, :, :, 2] = np.linspace(latent_min_value, latent_max_value, n_latent_grid)[None, None, :]
            points = points.reshape((-1, dim_latent))

        else:
            print("Code for higher input dimensions not yet implemented")

    if latent_distribution == "Gaussian":

        # n_latent_grid = 32

        r = np.array(
            [0.23, 0.33, 0.4, 0.47, 0.53, 0.58, 0.63, 0.68, 0.725, 0.77, 0.815, 0.655, 0.895, 0.935, 0.98, 1.025,
             1.07, 1.115, 1.15, 1.2, 1.245, 1.29, 1.325, 1.38, 1.43, 1.48, 1.535, 1.59, 1.645, 1.71, 1.77, 1.85,
             1.94, 2.04, 2.13, 2.25, 2.37, 2.69, 3.5, 4])

        points = np.empty([len(r), n_latent_grid, 2])

        for j in range(len(r)):

            for i in range(n_latent_grid):
                x = r[j] * np.cos(2 * np.pi * i / n_latent_grid)
                y = r[j] * np.sin(2 * np.pi * i / n_latent_grid)
                points[j, i, :] = x, y

        points = points.reshape([len(r) * n_latent_grid, 2])

    return points
