"""
This is the place that contains all the default
values and names of the code.
"""

# File names
file_names = {
    'y_data': {
        'name': 'y_data_{}.txt',
    },
    'log': {
        'name': 'history_log.cvs',
    },
    'checkpoint': {
        'name': 'checkpoint_epoch{epoch:04d}.weights.h5',
        'folder': 'checkpoints',
    },
    'model': {
        'name': 'model.keras',
    },
    'x_scaler': {
        'name': 'x_scaler.save',
    },
    'y_scaler': {
        'name': 'y_scaler.save',
    },
    'x_pca': {
        'name': 'x_pca.save',
    },
    'y_pca': {
        'name': 'y_pca.save',
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
    'spectra_factor': {
        'name': 'spectra_factor.fits',
    },
}
