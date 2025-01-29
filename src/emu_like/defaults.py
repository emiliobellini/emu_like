"""
This is the place that contains all the default
values and names of the code.
"""

# File names
file_names = {
    'params': {
        'name': 'params.yaml',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n'),
    },
    'x_data': {
        'name': 'x_data.txt',
    },
    'y_data': {
        'name': 'y_data{}.txt',
    },
    'log': {
        'name': 'history_log.cvs',
    },
    'checkpoint': {
        'name': 'checkpoint_epoch{epoch:04d}.weights.h5',
        'folder': 'checkpoints',
    },
    'model_last': {
        'name': 'model_last.keras',
    },
    'model_best': {
        'name': 'model_best.keras',
    },
    'x_scaler': {
        'name': 'x_scaler.save',
    },
    'y_scaler': {
        'name': 'y_scaler.save',
    },
    'dataset_details': {
        'name': 'dataset_details.yaml',
        'header': (
            '# This is an automatically generated file. Do not modify it!\n'
            '# It is used to resume training instead of the input one.\n\n'),
    },
    'chains': {
        'name': 'chain.txt',
    },
}
