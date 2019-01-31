dim_subspace = 2
#selected_labels = [0,1] ## DONE!
selected_labels = [0,1,2]
#selected_labels = [3,7]

config_name = "".join([str(label) for label in selected_labels])
results_directory = "Results/" + config_name + "/"
n_epochs_BIGAN = 2000
