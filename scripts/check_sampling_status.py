import argparse
import os
import re
import emu_like.io as io

def get_head(fname, lines=1):
    """
    Imitates the bash head command
    """
    with open(fname, 'r') as f:
        f.seek(0, 2)     # go to end of file
        total_bytes = f.tell()
        lines_found, total_bytes_scanned = 0, 0
        while (lines+1 > lines_found and
                total_bytes > total_bytes_scanned):
            byte_block = min(1024, total_bytes-total_bytes_scanned)
            f.seek(total_bytes_scanned, 0)
            total_bytes_scanned += byte_block
            lines_found += f.read(byte_block).count('\n')
        f.seek(0, 0)
        line_list = list(f.readlines(total_bytes_scanned))
        line_list = [x.rstrip() for x in line_list[:lines]]
    return line_list

def get_tail(fname, lines=1):
    """
    Imitates the bash tail command
    """
    # Main body
    with open(fname, 'r') as f:
        f.seek(0, 2)     # go to end of file
        total_bytes = f.tell()
        lines_found, total_bytes_scanned = 0, 0
        while (lines+1 > lines_found and
                total_bytes > total_bytes_scanned):
            byte_block = min(1024, total_bytes-total_bytes_scanned)
            f.seek(total_bytes-total_bytes_scanned-byte_block, 0)
            try:
                lines_found += f.read(byte_block).count('\n')
            except UnicodeDecodeError:
                pass
            total_bytes_scanned += byte_block
        f.seek(total_bytes-total_bytes_scanned, 0)
        line_list = list(f.readlines())
        line_list = [x.rstrip() for x in line_list[-lines:]]
    return line_list


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', nargs='+')
    args = parser.parse_args()

    str_out = '\x1b[1;32m[info]\x1b[00m Resuming from '
    str_save = '\x1b[1;32m[info]\x1b[00m Saving every '

    for log_file in args.log_files:
        if log_file.startswith('logs/e'):
            continue
        else:
            err_file = log_file.replace('logs/o', 'logs/e')

        # Locate output file
        head = get_head(log_file, lines=40)
        for line in head:
            if line.startswith(str_out):
                output_file = line.replace(str_out, '')[:-1]
            elif line.startswith(str_save):
                save_interval = line.replace(str_save, '')
                save_interval = int(save_interval.replace(' steps', ''))

        # Get samples this run
        tail = get_tail(err_file, lines=1)[0]
        tail = re.search(r'[0-9]+/[0-9]+', tail).group()
        samples_this_run, remaining_this_run = [int(x) for x in tail.split('/')]

        # Get saved samples
        fits = io.FitsFile(output_file)
        hd = fits.get_header(0)
        sp_name = list(hd['y_model']['outputs'].keys())[0]
        samples_saved = fits.get_data(sp_name).shape[0]
        samples_tot = fits.get_data('x_data').shape[0]

        # Print stuff
        samples_run = samples_tot-remaining_this_run+samples_this_run
        io.info('Folder {}'.format(output_file))
        io.print_level(1, 'Total samples: {}'.format(samples_tot))
        io.print_level(1, 'Number of samples run: {}'.format(samples_run))
        io.print_level(1, 'Number of samples saved: {}'.format(samples_saved))
        io.print_level(1, 'Remaining samples to run: {}'.format(samples_tot-samples_run))
        io.print_level(1, 'Remaining samples to save: {}'.format(samples_tot-samples_saved))
        io.print_level(1, 'Remaining samples to next save: {}'.format(save_interval-samples_run+samples_saved))
        print()
