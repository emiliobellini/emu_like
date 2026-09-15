import argparse
import numpy as np
import emu_like.io as io

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file_1', type=str)
    parser.add_argument('sample_file_2', type=str)
    args = parser.parse_args()

    fits_1 = io.FitsFile(args.sample_file_1)
    fits_2 = io.FitsFile(args.sample_file_2)

    for key in fits_1.get_keys():
        if (key.startswith('PK_') or key.startswith('CL_')
                or key.startswith('FK_') or key == 'X_DATA'):
            data_1 = fits_1.get_data(key)
            data_2 = fits_2.get_data(key)
            # Exclude NaN comparisons
            mask = ~np.isnan(data_2)
            if not (data_1[mask] == data_2[mask]).all():
                io.warning(f"Data mismatch in HDU: {key}")
                # Find indices where data differ
                diff_indices = np.where(data_1[mask] != data_2[mask])
                io.print_level(1, "Number of differing elements: {}".format(
                    len(diff_indices[0])))
                # for idx in zip(*diff_indices):
                #     io.print_level(1, f"  Mismatch at index {idx}: "
                #                       f"{data_1[idx]} (file 1) != "
                #                       f"{data_2[idx]} (file 2)")
            else:
                io.print_level(1, f"Data match in HDU: {key}")
    io.print_level(1, "Check completed.")
