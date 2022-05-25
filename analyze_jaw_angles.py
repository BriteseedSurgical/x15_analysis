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

colors                = ['b' , 'g' , 'y' , 'r'] 
four_wl_legend        = ['520' , '610' , '700' , '930']
x4_band_indices       = [1,4,7,14]
#x15 system globals

#calib-adapt pixels
pixels_vis_calib_adapt = [4,36,72]
pixels_nir_calib_adapt = [16,48,90]
#calib-adapt pixels

sensor_floor        = 425

watch_baseline      = 1600
vertical_lines      = 0

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

    x15_rt_processed_path    = os.path.join(parent,'x15_rt_processed')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    x15_rt_processed_path = os.path.join(x15_rt_processed_path , 'x15_jaw_angle_analysis')
    if not os.path.exists(x15_rt_processed_path):
        os.mkdir(x15_rt_processed_path)

    ure_plots_path = os.path.join(x15_rt_processed_path , 'ureters')
    if not os.path.exists(ure_plots_path):
        os.mkdir(ure_plots_path)

    peri_plots_path = os.path.join(x15_rt_processed_path,'peris')
    if not os.path.exists(peri_plots_path):
        os.mkdir(peri_plots_path)

    #(category_a , category b) : can do only ure and peri,
    #(sample A , sample B),(position A , position B)

    for i in range(0,n_children):
    
        child_name              = children[i].split('/')[-1] 
        print('')
        print('Current child : ' , child_name)

        targets                 = glob.glob(os.path.join(children[i],'*.csv'))
        n_targets               = len(targets)

        meta_keys               = process_x15_rt_file_name('_pos1_0-20_pos2_0-20')
        meta_keys               = meta_keys['keys']

        unique_ure_configurations   = []
        unique_peri_configurations  = []

        categorized_lists       = {}

        for j in range(0,n_targets):

            current_meta       = process_x15_rt_file_name(targets[j])
            
            current_tissue1    = current_meta['vessel']
            current_tissue2    = current_meta['tissue']

            current_baseline   = current_meta['baseline']
            current_jaw_angle  = current_meta['ja_manual'] #a string.
            
            current_pig1_id    = current_meta['pig1']      #a string.
            current_spec1_id   = current_meta['spec1']
            current_sample1_id = current_meta['sample1']
            current_s1         = current_pig1_id + '-' + current_spec1_id + '-' + current_sample1_id

            current_pig2_id    = current_meta['pig2']     
            current_spec2_id   = current_meta['spec2']
            current_sample2_id = current_meta['sample2']
            current_s2         = current_pig2_id + '-' + current_spec2_id + '-' + current_sample2_id

            current_position1  = current_meta['position_1']
            current_position2  = current_meta['position_2']

            current_configuration = 's1_' + current_s1 + '_pos1_' + current_position1 + '_s2_' + current_s2 + '_pos2_' + current_position2
            
            if current_tissue1 == 'ureter' and current_tissue2 == 'peritoneum' and current_baseline == str(1600):
                
                if current_configuration not in unique_ure_configurations:
                    unique_ure_configurations.append(current_configuration)

            if current_tissue1 == 'peritoneum' and current_tissue2 == 'none' and current_baseline == str(1600):

                if current_configuration not in unique_peri_configurations:
                    unique_peri_configurations.append(current_configuration)
        
        #print('Unique ureter configurations.')
        #for j in range(0,len(unique_ure_configurations)):
        #    print(unique_ure_configurations[j])
        #print('')

        #print('Unique peritoneum configurations.')
        #for j in range(0,len(unique_peri_configurations)):
        #    print(unique_peri_configurations[j])
        #print('')

        ure_configs = {}
        peri_configs= {}
        ure_configs['n_configs'] = 0
        peri_configs['n_configs'] = 0
        for config in unique_ure_configurations:

            config_list = []
            for j in range(0,n_targets):

                current_meta       = process_x15_rt_file_name(targets[j])
                
                current_tissue1    = current_meta['vessel']
                current_tissue2    = current_meta['tissue']

                current_baseline   = current_meta['baseline']
                current_jaw_angle  = current_meta['ja_manual'] #a string.
                
                current_pig1_id    = current_meta['pig1']      #a string.
                current_spec1_id   = current_meta['spec1']
                current_sample1_id = current_meta['sample1']
                current_s1         = current_pig1_id + '-' + current_spec1_id + '-' + current_sample1_id

                current_pig2_id    = current_meta['pig2']     
                current_spec2_id   = current_meta['spec2']
                current_sample2_id = current_meta['sample2']
                current_s2         = current_pig2_id + '-' + current_spec2_id + '-' + current_sample2_id

                current_position1  = current_meta['position_1']
                current_position2  = current_meta['position_2']

                current_configuration = 's1_' + current_s1 + '_pos1_' + current_position1 + '_s2_' + current_s2 + '_pos2_' + current_position2
                
                if current_tissue1 == 'ureter' and current_tissue2 == 'peritoneum' and current_baseline == str(1600):
                    
                    if current_configuration == config:
                        config_list.append(targets[j])

            ure_configs[config] = config_list
            #print('Length (ure) : ' , len(config_list))
            ure_configs['n_configs'] = len(unique_ure_configurations)

        for config in unique_peri_configurations:

            config_list = []
            for j in range(0,n_targets):

                current_meta       = process_x15_rt_file_name(targets[j])
                
                current_tissue1    = current_meta['vessel']
                current_tissue2    = current_meta['tissue']

                current_baseline   = current_meta['baseline'] 
                current_jaw_angle  = current_meta['ja_manual'] #a string.
                
                current_pig1_id    = current_meta['pig1']      #a string.
                current_spec1_id   = current_meta['spec1']
                current_sample1_id = current_meta['sample1']
                current_s1         = current_pig1_id + '-' + current_spec1_id + '-' + current_sample1_id

                current_pig2_id    = current_meta['pig2']     
                current_spec2_id   = current_meta['spec2']
                current_sample2_id = current_meta['sample2']
                current_s2         = current_pig2_id + '-' + current_spec2_id + '-' + current_sample2_id

                current_position1  = current_meta['position_1']
                current_position2  = current_meta['position_2']

                current_configuration = 's1_' + current_s1 + '_pos1_' + current_position1 + '_s2_' + current_s2 + '_pos2_' + current_position2
                
                if current_tissue1 == 'peritoneum' and current_tissue2 == 'none' and current_baseline == str(1600):
                    
                    if current_configuration == config:
                        config_list.append(targets[j])

            peri_configs[config] = config_list
            #print('Length (peri): ' , len(config_list))
            peri_configs['n_configs'] = len(unique_peri_configurations)

        #process if length is exactly 7.
        #Metric : Mean L1 distance (reference bands), Max L1 distance(reference bands)
        #Metric : Other bands ? 
        #Put it into a csv
        #Plot each block.
        #Only compare a pair of baselines ? 
        
        n_ure_configs  = ure_configs['n_configs']
        n_peri_configs = peri_configs['n_configs']
        
        print('N configs : ' , n_ure_configs , n_peri_configs)
        
        ure_plots_folder = os.path.join(ure_plots_path,child_name)
        if not os.path.exists(ure_plots_folder):
            os.mkdir(ure_plots_folder)
        peri_plots_folder = os.path.join(peri_plots_path,child_name)
        if not os.path.exists(peri_plots_folder):
            os.mkdir(peri_plots_folder)

        if n_ure_configs:

            for config in ure_configs:

                if config not in ['n_configs']:
                
                    data_list = ure_configs[config] 
                    print('Config : ' , config)
                    n_data    = len(data_list)
                    jaw_angles= [int(process_x15_rt_file_name(data_list[k])['ja_manual']) for k in range(0,n_data)]
                    #data      = np.array([process_file_content_x15_rt_two_step(data_list[k])['x15_c']  for k in range(0,n_data)])
                    jaw_angles = []
                    cmos       = []
                    roi        = process_x15_rt_file_name(data_list[0])['position_1']
                    roi_start  = int(roi.split('-')[0])
                    roi_end    = int(roi.split('-')[-1])
                    
                    for k in range(0,n_data):
                        jaw_angle = int(process_x15_rt_file_name(data_list[k])['ja_manual'])
                        data = process_file_content_x15_rt_two_step(data_list[k])
                        if not data['incomplete_data']:
                            data = data['x15_c']
                            jaw_angles.append(jaw_angle)
                            cmos.append(data)

                    jaw_angles.append('roi_start')
                    jaw_angles.append('roi_end')

                    plt.figure()
                    plt.title('610nm : ' + config)
                    plt.xlabel('pixel count')
                    plt.ylabel('cmos intensity')
                    plt.ylim([256,4096])
                    for k in range(0,len(cmos)):
                        plt.plot(cmos[k][4,:])
                    plt.axvline(roi_start)
                    plt.axvline(roi_end)
                    plt.legend(jaw_angles)
                    plt.savefig(os.path.join(ure_plots_folder , '610nm_' + config))
                    
                    plt.figure()
                    plt.title('930nm : ' + config)
                    plt.xlabel('pixel count')
                    plt.ylabel('cmos intensity')
                    plt.ylim([256,4096])
                    for k in range(0,len(cmos)):
                        plt.plot(cmos[k][14,:])
                    plt.axvline(roi_start)
                    plt.axvline(roi_end)
                    plt.legend(jaw_angles)
                    plt.savefig(os.path.join(ure_plots_folder , '930nm_' + config))
        
        if n_peri_configs:

            for config in peri_configs:

                if config not in ['n_configs']:
                
                    data_list = peri_configs[config] 
                    print('Config : ' , config)
                    n_data    = len(data_list)
                    jaw_angles = [int(process_x15_rt_file_name(data_list[k])['ja_manual']) for k in range(0,n_data)]
                    
                    roi        = process_x15_rt_file_name(data_list[0])['position_1']
                    roi_start  = int(roi.split('-')[0])
                    roi_end    = int(roi.split('-')[-1])
                    
                    jaw_angles = []
                    cmos       = []
                    for k in range(0,n_data):
                        jaw_angle = int(process_x15_rt_file_name(data_list[k])['ja_manual'])
                        data = process_file_content_x15_rt_two_step(data_list[k])
                        if not data['incomplete_data']:
                            data = data['x15_c']
                            jaw_angles.append(jaw_angle)
                            cmos.append(data)

                    jaw_angles.append('roi_start')
                    jaw_angles.append('roi_end')

                    plt.figure()
                    plt.title('610nm : ' + config)
                    plt.xlabel('pixel count')
                    plt.ylabel('cmos intensity')
                    plt.ylim([0,4096])
                    for k in range(0,len(cmos)):
                        plt.plot(cmos[k][4,:])
                    plt.axvline(roi_start)
                    plt.axvline(roi_end)
                    plt.legend(jaw_angles)
                    plt.savefig(os.path.join(peri_plots_folder , '610nm_' + config))


                    plt.figure()
                    plt.title('930nm : ' + config)
                    plt.xlabel('pixel count')
                    plt.ylabel('cmos intensity')
                    plt.ylim([256,4096])
                    for k in range(0,len(cmos)):
                        plt.plot(cmos[k][14,:])
                    plt.axvline(roi_start)
                    plt.axvline(roi_end)
                    plt.legend(jaw_angles)
                    plt.savefig(os.path.join(peri_plots_folder , '930nm_' + config))

        #compare 1600 and 2000.