import os
import glob 
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from x15_utils import *

#x15 system globals
n_control             = 6
n_wls                 = 15
n_pixels              = 99
wl_set                = [450 , 520 , 550 , 580 , 610 , 635 , 660 , 700 , 740 , 770 , 810 , 840 , 870 , 900 , 930] 
visible_end_index     = 11 
max_post_adapt_length = 32

colors                = ['b' , 'g' , 'y' , 'r'] 
four_wl_legend        = ['520' , '610' , '700' , '930']
x4_band_indices       = [1,4,7,14]
#x15 system globals

#lambdas
vis_read = lambda x: (int(x[1] + x[3] + x[5]) == int(48))
nir_read = lambda x: (int(x[0] + x[2] + x[4]) == int(48))
#lambdas

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

if __name__ == '__main__':
    _ = 0
    #plot spectrograms
    #plot 4wl plots
    #plot air-ref normalized 4wl plots
    #(Will have above data after start of protocol. 1208 should be right.)
    #plot the mean intensity of 15 bands/might want to store that as 
    #usable/transferrable data.
    #LED Histograms by case.
    #store as mat files.

    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'x15_rt')

    children    = glob.glob(os.path.join(data_path,'*x15_rt*'))
    n_children  = len(children)

    print('')
    for i in range(0,len(children)):
        print('Child : ' , children[i])
    print('')
    n_children      = len(children)
    targets_crossed = 0

