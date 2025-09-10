from emu_like.ffnn_emu import FFNNEmu
from emu_like.datasets import Dataset


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    spectrum = 'pk_m'
    emu = {}
    for nemu in range(4):
        path_emu = '/ceph/hpc/data/s25r06-05-users/lcdm/train/{}_x{}'.format(spectrum, nemu+1)
        emu[nemu] = FFNNEmu()
        emu[nemu].load(path_emu)
