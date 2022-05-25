import os
import csv
import json
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from x15_utils import *
from pathlib import Path

#x15 system globals
n_control             = 6
n_wls                 = 15
n_pixels              = 99
wl_set                = [450 , 520 , 550 , 580 , 610 , 635 , 660 , 700 , 740 , 770 , 810 , 840 , 870 , 900 , 930] 
visible_end_index     = 11 
max_post_adapt_length = 32

four_wl_legend        = ['520' , '610' , '700' , '930']
x4_band_indices       = [1,4,7,14]
#x15 system globals

tissue_pick_config =  [['ureter','peritoneum'],['peritoneum','none']]
reference_keywords  = ['dailytest' , 'daily_test']

sensor_floor        = 425
watch_baseline      = 1800
vertical_lines      = 0

split_ratio         = 0.8
randomize_data      = False

if __name__ == '__main__':

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

    x15_dl_data_main = os.path.join(parent , 'x15_dl')
    if not os.path.exists(x15_dl_data_main):
        os.mkdir(x15_dl_data_main)

    pretraining_main = os.path.join(x15_dl_data_main, 'unsupervised')
    if not os.path.exists(pretraining_main):
        os.mkdir(pretraining_main)

    #save jaw angles in store.
    #Save as two lists , make two datasets compose batches in real time ? 
    #Save all pairs and labels as a dictionary ?
    #If 1000 of each , 1 million pairs -> too much data , non optimal to store.

    n_categories = len(tissue_pick_config)

    data_lists              = {}
    data_lists['n_classes'] = n_categories
    config_list             = []
    for i in range(0,n_categories):
        primary_tissue_target   = tissue_pick_config[i][0]
        secondary_tissue_target = tissue_pick_config[i][1]
        target_config           = primary_tissue_target + '_' + secondary_tissue_target
        config_list.append(target_config)
    data_lists['config_list']   = config_list

    for i in range(0,n_categories):

        #collect all relevant filenames , from all folders.

        primary_tissue_target   = tissue_pick_config[i][0]
        secondary_tissue_target = tissue_pick_config[i][1]
        target_config           = primary_tissue_target + '_' + secondary_tissue_target
        
        t = []
        for j in range(0,n_children):

            targets                 = glob.glob(os.path.join(children[j],'*.csv'))
            n_targets               = len(targets)

            meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
            meta_keys               = meta_keys['keys']

            for k in range(0,n_targets):

                current_meta      = process_x15_rt_file_name(targets[k])
                
                current_primary   = current_meta['vessel'] 
                current_secondary = current_meta['tissue']  
                current_baseline  = current_meta['baseline']

                if current_primary == primary_tissue_target and current_secondary == secondary_tissue_target and current_baseline == str(watch_baseline):
                    t.append(targets[k])

        print('Configuration : ' , target_config)
        print('Files found   : ' , len(t))

        data_lists[target_config] = t

    data = {}
    data['n_classes'] = data_lists['n_classes']

    for i in range(0,data_lists['n_classes']):

        leds         = []
        cmos         = []
        cmos_x4      = [] 
        truths       = []
        jaw_angles   = []
        primary_rois = []


        current_config    = data_lists['config_list'][i]
        n_targets_current = len(data_lists[current_config])

        x = {}
        x_tr = {}
        x_va = {}
        for k in range(0,n_targets_current):

            current      = data_lists[current_config][k]
            current_meta = process_x15_rt_file_name(data_lists[current_config][k])

            roi           = current_meta['position_1']
            roi_start     = int(roi.split('-')[0])
            roi_end       = int(roi.split('-')[-1])
            if roi_start < 0:
                roi_start = 0
            if roi_end > n_pixels:
                roi_end   = n_pixels
            current_roi   = str(roi_start) + '-' + str(roi_end)
            current_ja    = current_meta['ja_manual']
            current_prima = current_meta['vessel']
            current_truth = None
            if current_prima == 'ureter':
                current_truth = int(1)
            else:
                current_truth = int(0)

            current_content = process_file_content_x15_rt_two_step(data_lists[current_config][k])
            current_cmos    = current_content['x15_c']
            current_cmos_x4 = np.array([current_cmos[x4_index , :] for x4_index in x4_band_indices])
            current_cmos    = current_cmos[np.newaxis,:,:]
            current_cmos_x4 = current_cmos_x4[np.newaxis,:,:]
            current_leds    = current_content['l']

            leds.append(current_leds)
            cmos.append(current_cmos)
            cmos_x4.append(current_cmos_x4)
            truths.append(current_truth)
            jaw_angles.append(current_ja)
            primary_rois.append(current_roi)

        #A randomizer. Split 80-20 or 1-r:r by category - Retains ratios.
        #randomize each of the above.
        #save a training and testing config dict.

        n_samples       = len(leds)
        divide_point    = int(split_ratio * n_samples)
        if randomize_data:
            random_indices  = np.random.permutation(n_samples)
        else:
            random_indices  = []
            for m in range(0,n_samples):
                random_indices.append(int(m))

        leds_tr         = [leds[random_indices[q]]         for q in range(0,divide_point)]
        cmos_x4_tr      = [cmos_x4[random_indices[q]]      for q in range(0,divide_point)]
        cmos_tr         = [cmos[random_indices[q]]         for q in range(0,divide_point)]
        truths_tr       = [truths[random_indices[q]]       for q in range(0,divide_point)]
        jaw_angles_tr   = [jaw_angles[random_indices[q]]   for q in range(0,divide_point)]
        primary_rois_tr = [primary_rois[random_indices[q]] for q in range(0,divide_point)]

        leds_va         = [leds[random_indices[q]]         for q in range(divide_point,n_samples)]
        cmos_x4_va      = [cmos_x4[random_indices[q]]      for q in range(divide_point,n_samples)]
        cmos_va         = [cmos[random_indices[q]]         for q in range(divide_point,n_samples)]
        truths_va       = [truths[random_indices[q]]       for q in range(divide_point,n_samples)]
        jaw_angles_va   = [jaw_angles[random_indices[q]]   for q in range(divide_point,n_samples)]
        primary_rois_va = [primary_rois[random_indices[q]] for q in range(divide_point,n_samples)]

        x_tr['ls']           = leds_tr
        x_tr['x4_cs']        = cmos_x4_tr
        x_tr['x15_cs']       = cmos_tr
        x_tr['ts']           = truths_tr
        x_tr['jaw_angles']   = jaw_angles_tr
        x_tr['primary_rois'] = primary_rois_tr

        x_va['ls']           = leds_va
        x_va['x4_cs']        = cmos_x4_va
        x_va['x15_cs']       = cmos_va
        x_va['ts']           = truths_va
        x_va['jaw_angles']   = jaw_angles_va
        x_va['primary_rois'] = primary_rois_va

        data[current_config + '_tr'] = x_tr
        data[current_config + '_va'] = x_va
        
    #Save data as a dictionary.
    content_to_file = json.dumps(data, cls=NumpyEncoder)
    if randomize_data:
        store_at        = os.path.join(pretraining_main , 'x15_x4_ut.json')
    else:
        store_at        = os.path.join(pretraining_main , 'x15_x4_ut_unrand.json')
    with open(store_at, 'w') as file_pointer:
        json.dump(content_to_file, file_pointer)
    #Save data as a dictionary.
    
    #Verify keys and shapes.
    for key in data:

        if key not in ['n_classes']:

            print('')
            print('')
            print('Key     : ' , key)
            print('L2 keys : ' , data[key].keys())

            for l2_key in data[key]:
                
                print('')
                print('L2 Key   : ' , l2_key)
                print('Elements : ' , len(data[key][l2_key]))
                print(type(data[key][l2_key]) , np.shape(data[key][l2_key]))
    #Verify keys and shapes.
    
    #if not using LEDs , can  normalize using only 2 files.