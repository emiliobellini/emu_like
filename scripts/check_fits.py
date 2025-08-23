import argparse
import emu_like.io as io

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    args = parser.parse_args()

    folder = io.Folder(args.sample_folder)

    for fname in folder.list_files():
        if '.fits' in fname:
            safe_to_copy = True
            try:
                fits = io.FitsFile(fname)
                header = fits.get_header(0)
            except:
                safe_to_copy = False
            if safe_to_copy:
                io.info('Safe to copy {}'.format(fname))
                # print(header)
            else:
                io.warning('File {} corrupted!'.format(fname))
