
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load and arrange data
dataset = input_data.read_data_sets( "../MNIST_data" )

train_data = dataset.train.images
train_labels = dataset.train.labels

validation_data = dataset.validation.images
validation_labels = dataset.validation.labels

test_data = dataset.test.images
test_labels = dataset.test.labels

dim_data = train_data.shape[1]



def load_mnist_data(selected_labels = np.arange(10)):
	#filter, concatenate
	global train_data 
	global train_labels
	global test_data
	global test_labels
	global validation_data
	global validation_labels
	global dim_data

	train_data_selection = np.empty((0,dim_data),np.float32)
	train_labels_selection = np.empty((0),int)
	test_data_selection = np.empty((0,dim_data),np.float32)
	test_labels_selection = np.empty((0),int)
	validation_data_selection = np.empty((0,dim_data),np.float32)
	validation_labels_selection = np.empty((0),int)

	for selected_label in selected_labels:
		
		train_data_selection = np.append(train_data_selection, np.array([train_data[key] for (key, label) in enumerate(train_labels) if int(label) == selected_label]),axis=0)
		train_labels_selection = np.append(train_labels_selection,np.array([train_labels[key] for (key, label) in enumerate(train_labels) if int(label) == selected_label]),axis=0)
		
		test_data_selection = np.append(test_data_selection, np.array([test_data[key] for (key, label) in enumerate(test_labels) if int(label) == selected_label]),axis=0)
		test_labels_selection = np.append(test_labels_selection,np.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) == selected_label]),axis=0)
		
		validation_data_selection = np.append(validation_data_selection, np.array([validation_data[key] for (key, label) in enumerate(validation_labels) if int(label) == selected_label]),axis=0)
		validation_labels_selection = np.append(validation_labels_selection,np.array([validation_labels[key] for (key, label) in enumerate(validation_labels) if int(label) == selected_label]),axis=0)

	#shuffle
	rng_state = np.random.get_state()
	np.random.shuffle(train_data_selection)
	np.random.set_state(rng_state)
	np.random.shuffle(train_labels_selection)

	rng_state = np.random.get_state()
	np.random.shuffle(test_data_selection)
	np.random.set_state(rng_state)
	np.random.shuffle(test_labels_selection)

	rng_state = np.random.get_state()
	np.random.shuffle(validation_data_selection)
	np.random.set_state(rng_state)
	np.random.shuffle(validation_labels_selection)

	dict_data = {}
	dict_data['train'] = [train_data_selection,train_labels_selection]
	dict_data['test'] = [test_data_selection,test_labels_selection]
	dict_data['validation'] = [validation_data_selection,validation_labels_selection]


	return dict_data
		

