import os

# folders
output_data_folder              = 'data'
output_data_generated           = os.path.join(output_data_folder, 'generated')
output_datasets                 = os.path.join(output_data_folder, 'datasets')
output_zones_learned            = os.path.join(output_data_folder, 'learned_zones')
output_models                   = os.path.join(output_data_folder, 'saved_models')
output_results_folder           = os.path.join(output_data_folder, 'results')

## min_max_custom_folder           = 'custom_norm'
## correlation_indices_folder      = 'corr_indices'
normalization_choices           = ['svd', 'svdn']
models_list                     = ['svm_model', 'ensemble_model', 'ensemble_model_v2']

# variables
features_choices_labels         = ['svd_entropy']

## models_names_list               = ["svm_model","ensemble_model","ensemble_model_v2","deep_keras"]
## normalization_choices           = ['svd', 'svdn', 'svdne']

# parameters
keras_epochs                    = 30
keras_batch                     = 32
## val_dataset_size                = 0.2