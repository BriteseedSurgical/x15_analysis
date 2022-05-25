import os
import csv
import json
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from x15_utils import *
from pathlib import Path

if __name__ == '__main__':

    root        = os.getcwd()
    parent      = Path(root)
    parent      = parent.parent.absolute()
    data_path   = os.path.join(parent,'x15_rt')

    children    = glob.glob(os.path.join(data_path,'*x15_rt*'))
    n_children  = len(children)

    print('')
    print('Children : ' , n_children)
    
    print('')
    targets_crossed = 0

    x15_rt_processed_path = os.path.join(parent,'x15_rt_processed')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    for i in range(0,n_children):
        print('Child : ' , children[i])
        #create if doesn't exist , folder in the top processed folder.
        
        current_folder_name = children[i].split('/')[-1]
        current_dump_path   = os.path.join(x15_rt_processed_path,current_folder_name)
        if not os.path.exists(current_dump_path):
            os.mkdir(current_dump_path)

        dump_x15_plots_at   = os.path.join(current_dump_path,'band_plots')
        if not os.path.exists(dump_x15_plots_at):
            os.mkdir(dump_x15_plots_at)

        dump_mat_at         = os.path.join(current_dump_path,'mat_files')
        if not os.path.exists(dump_mat_at):
            os.mkdir(dump_mat_at) 
        
        targets   = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets = len(targets)
        targets_crossed = targets_crossed + n_targets
        print('Targets in path : ' , n_targets)
        
        for j in range(0,n_targets):

            current_mat_name = targets[j].split('/')[-1].split('.csv')[0]

            plots_folder = os.path.join(dump_x15_plots_at,current_mat_name)
            if not os.path.exists(plots_folder):
                os.mkdir(plots_folder)

            current_mat_name = current_mat_name + '.mat' 
            meta             = process_x15_rt_file_name(targets[j])
            processed        = process_file_content_x15_rt_two_step(targets[j])
            sio.savemat(os.path.join(dump_mat_at,current_mat_name),make_rt_mat(processed,meta))
            
            if not processed['incomplete_data']:
                if not processed['too_few_rows']:
                    title = 'LEDs: ' + str(processed['l'])
                    plot_x15_raw(processed['x15_c'],plots_folder,int(meta['baseline']),title)

    print('')

    print('Targets crossed : ' , targets_crossed)

    print('')
    print('At main exit line.')

    #use the average values and look for variations in spectral response over space
    #Final method should make the patterns homogeneous for a homogeneous sample.