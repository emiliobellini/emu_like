"""
This is the single place that contains all the default
values and names of the code.
"""

default_parameters = {
}

# File names
file_names = {
    'params': {
        'name': 'params.yaml',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n'),
    },
    'x_sample': {
        'name': 'x_sample.txt',
    },
    'y_sample': {
        'name': 'y_sample.txt',
    },
    'log': {
        'name': 'history_log.cvs',
        'folder': 'history_log',
    },
    'checkpoint': {
        'name': 'checkpoint_epoch{epoch:04d}.hdf5',
        'folder': 'checkpoints',
    },
    'model': {
        'name': 'model.keras',
        'folder': 'model',
    },
    'x_scaler': {
        'name': 'x_scaler.save',
        'folder': 'scalers',
    },
    'y_scaler': {
        'name': 'y_scaler.save',
        'folder': 'scalers',
    },
    'sample_details': {
        'name': 'sample_details.yaml',
        'folder': 'sample_details',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n\n'
            ),
    },
}

# Use None when you only want to check a key and not their nested keys
params_to_check = {
    'output': None,
    'emulator_type': None,
    'frac_train': None,
    'train_test_random_seed': None,
    'rescale_x': None,
    'rescale_y': None,
    'ffnn_model': (
      'activation_function',
      'neurons_hidden_layer',
      'batch_normalization',
      'dropout_rate',
      'optimizer',
      'loss_function',
      'batch_size',
      'want_output_layer',
    )
}
