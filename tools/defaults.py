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
        'folder': 'sample',
    },
    'y_sample': {
        'name': 'y_sample.txt',
        'folder': 'sample',
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
}
