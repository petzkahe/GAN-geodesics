dim_subspace = 2
#selected_labels = [0,1]
#selected_labels = [0,1,2]
#selected_labels = [0,1,2,3]
#selected_labels = [3,7]
selected_labels = [0,7]
#selected_labels = [1,8] 

 
config_name = "".join([str(label) for label in selected_labels])
results_directory = "Results/" + config_name + "/"
n_epochs_BIGAN = 3000
