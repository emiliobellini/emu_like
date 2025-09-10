import argparse
import csv
import numpy as np
import os
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

    str_out = '\x1b[1;32m[info]\x1b[00m Writing output in '
    str_res = '\x1b[1;32m[info]\x1b[00m Resuming from '

    for log_file in args.log_files:
        if log_file.startswith('logs/e'):
            continue
        # Locate output folder
        head = get_head(log_file, lines=40)
        for line in head:
            if line.startswith(str_out):
                output_folder = line.replace(str_out, '')
            if line.startswith(str_res):
                output_folder = line.replace(str_res, '')

        # Load history
        # history = np.genfromtxt(
        #     os.path.join(output_folder, 'history_log.csv'),
        #     delimiter=',',
        #     skip_header=1)
        with open(os.path.join(output_folder, 'history_log.csv')) as csvfile:
            history = np.array(list(csv.reader(csvfile, delimiter=',')))[1:].astype(float)

        # Last epoch
        last_epoch = int(history[-1, 0])

        # Best epoch
        idx_best = np.where(history[:, 2] == np.min(history[:, 2]))[0][0]
        best_epoch, loss, val_loss = history[idx_best]
        best_epoch = int(best_epoch)

        # Print stuff
        io.info('Folder {}'.format(output_folder))
        io.print_level(1, 'Last epoch: {}'.format(last_epoch))
        io.print_level(1, 'Best epoch: {}'.format(best_epoch))
        io.print_level(1, 'Epochs without improvement: {}'.format(last_epoch-best_epoch))
        io.print_level(1, 'Loss: {:.2e}'.format(loss))
        io.print_level(1, 'Validation Loss: {:.2e}'.format(val_loss))
        print()
