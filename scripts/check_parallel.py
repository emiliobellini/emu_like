import emu_like.io as io

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    par = io.FitsFile('/ceph/hpc/data/s25r06-05-users/test_parallel.fits')
    ser = io.FitsFile('/ceph/hpc/data/s25r06-05-users/test_serial.fits')

    for key in par.get_keys():
        if key != 'PRIMARY':
            data_par = par.get_data(key)
            data_ser = ser.get_data(key)
            if not (data_par == data_ser).all():
                print(f"Data mismatch in HDU: {key}")
            else:
                print(f"Data match in HDU: {key}")
    print("Check completed.")