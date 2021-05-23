import h5py, dask
from py4xs.hdf import lsh5, h5exp, h5sol_HPLC, h5sol_HT
from py4xs.data2d import Data2d
from dask import delayed
import dask.array as da
import numpy as np

from dask.distributed import Client, LocalCluster

# Dask correctly sets the number of workers and threads
# the user is responsible for defining the client???
#client = Client("tcp://127.0.0.1:41541") 
#client = Client(processes=True)

# two types of operations
# 1. load data from h5 file and create Data1d or MatrixWithCoords objects
#        require detector configuration and the 
#        if the target data is already saved in the h5 file, simply load them
# 2a. attribute extraction from these pre-processed data using a sequence of pre-defined operations
# 2b. further processing (merging/averaging) of data produced from 1
#
#
#
#
#
#
#
