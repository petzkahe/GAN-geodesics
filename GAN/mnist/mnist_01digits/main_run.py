# Main file for training BIGAN and geodesics for a custom selection of mnist digits.

import os

## MODULES
from GAN.mnist.mnist_01digits.utils.load_mnist_custom import *
from GAN.mnist.mnist_01digits.utils.get_pca_subspace import *
from GAN.mnist.mnist_01digits.main_config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2', '3'}


################################################################################


## EXTRACT AND SAVE SELECTED DATA


dataset = load_mnist_data(selected_labels)
train_data, train_labels = dataset['train']
n_training_examples = train_data.shape[0]
print(results_directory + 'Data')
if os.path.exists(results_directory + 'Data') == False:
	os.makedirs(results_directory + 'Data')
	np.save(results_directory + 'Data/selected_train_data',train_data)
	np.save(results_directory + 'Data/selected_train_labels',train_labels)

print('Selected data loaded!')

## CALCULATE PCA

print('Checking for PCA...')

if os.path.exists(results_directory + 'PCA'):
	print('... PCA exists!')
else:
	os.makedirs(results_directory + 'PCA')
	print('not found, calculating...')
	subspace_map, mean_per_pixel = get_pca_subspace(train_data,dim_subspace,results_directory+'PCA/')
	print('... Done!')


## TRAIN BIGAN
from GAN.mnist.mnist_01digits.BIGAN_Learning.train_01 import *

if os.path.isfile(results_directory + 'BIGAN/trained_model/mnistBIGAN.meta'):
	# set directory for trained_model or do nothing
	print('BIGAN already trained!')
else:
	print('Training BIGAN...')
	train_BIGAN(n_epochs_BIGAN,n_training_examples,results_directory + 'BIGAN/')
	# train BIGAN with new train function and data and epochs and save model
	print('... Finished!')



## TRAIN/CALCULATE GEODESICS
from GAN.mnist.mnist_01digits.Geodesic_Learning.train_geodesics import *

train_geodesics()




# run new geodesics run file with plots and all





