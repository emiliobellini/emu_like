import argparse
import csv
import numpy as np
import os
import emu_like.io as io
from itertools import islice


# def get_head(fname, lines=1):
#     """
#     Imitates the bash head command
#     """
#     with open(fname, 'r') as f:
#         f.seek(0, 2)     # go to end of file
#         total_bytes = f.tell()
#         lines_found, total_bytes_scanned = 0, 0
#         while (lines+1 > lines_found and
#                 total_bytes > total_bytes_scanned):
#             byte_block = min(1024, total_bytes-total_bytes_scanned)
#             f.seek(total_bytes_scanned, 0)
#             total_bytes_scanned += byte_block
#             lines_found += f.read(byte_block).count('\n')
#         f.seek(0, 0)
#         line_list = list(f.readlines(total_bytes_scanned))
#         line_list = [x.rstrip() for x in line_list[:lines]]
#     return line_list

def get_head(fname, lines=1):
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in islice(f, lines)]


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
    # parser.add_argument('common_folder')
    parser.add_argument('--exclude', '-e', type=int, default=-1)
    args = parser.parse_args()

    str_out = '\x1b[1;32m[info]\x1b[00m Writing output in '
    str_res = '\x1b[1;32m[info]\x1b[00m Resuming from '

    # for output_folder in io.Folder(args.common_folder).list_subfolders():
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
        with open(os.path.join(output_folder, 'history_log.csv')) as csvfile:
            history = np.array(
                list(csv.reader(csvfile, delimiter=',')))[1:].astype(float)

        # Best epoch
        idx_best = np.where(history[:, 3] == np.min(history[:, 3]))[0][-1]
        best_epoch, learning_rate, loss, val_loss = history[idx_best]
        last_epoch, last_learning_rate, last_loss, last_val_loss = history[-1]
        best_epoch = int(best_epoch)
        last_epoch = int(last_epoch)

        if args.exclude > 0 and last_epoch-best_epoch >= args.exclude:
            continue

        # Print stuff
        io.info('Folder {}'.format(output_folder))
        io.print_level(1, 'Last epoch: {}'.format(last_epoch))
        io.print_level(1, 'Best epoch: {}'.format(best_epoch))
        io.print_level(1, 'Epochs without improvement: {}'.format(
            last_epoch-best_epoch))
        io.print_level(1, 'Learning rate: {:.2e} (best), {:.2e} (last)'.format(
            learning_rate, last_learning_rate))
        io.print_level(1, 'Loss: {:.2e} (best), {:.2e} (last)'.format(
            loss, last_loss))
        io.print_level(
            1, 'Validation Loss: {:.2e} (best), {:.2e} (last)'.format(
                val_loss, last_val_loss))
        print()
