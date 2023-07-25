"""
This is the single place that contains all the default
values and names of the code.
"""

default_parameters = {
}

# File names
file_names = {
    'params_name': 'params.yaml',
    'params_header': (
        '# This is an automatically generated file. Do not modify it!\n'
        '# It is used to resume training instead of the input one.\n\n'),
}
# Info about the param file that will be copied in the
# output folder. Name of the file, and header to be used.
output_params_file = {
}

log_file = {
    'name': 'history_log.cvs'
}

checkpoint_file = {
    'name': 'checkpoint_epoch{epoch:04d}.hdf5'
}
