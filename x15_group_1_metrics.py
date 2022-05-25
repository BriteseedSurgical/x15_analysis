import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt

from x15_utils import *

#how much of a loss if we had included the  the 10.20 dataset?
# what metrics are important here ? - my 3 LED and 2 spatial metrics 
# Hari's spectral slope metrics (ethough it was developed for filter paper
# normalized ddata). Do we enter in the group ID into the file name
if __name__ == '__main__':

    interest = '20211029_x15_rt_fp0b'

    root        = os.getcwd()
    parent      = os.path.join(root , 'x15_rt_data')
    parent_copy = parent[:]

    parent      = os.path.join(parent , interest)
    targets     = glob.glob(os.path.join(parent , '*.csv'))

    result_dump_path = os.path.join(root,'metrics_x15_rt')
    if not os.path.exists(result_dump_path):
        os.mkdir(result_dump_path)
    result_dump_path = os.path.join(result_dump_path,interest)
    if not os.path.exists(result_dump_path):
        os.mkdir(result_dump_path)

    relevant_ure_files  = []
    relevant_peri_files = []

    relevance = {}
    relevance['orientation'] = 'parallel'
    relevance['baseline']    = 1600

    for target in targets:
        meta  = process_x15_rt_file_name(target)
        if int(meta['baseline']) == relevance['baseline'] and meta['tissue'] == 'Ureter':
            orientation = target.split('/')[-1]
            orientation = orientation.split('_O_')[-1].split('_')[0]
            if orientation == relevance['orientation']:
                relevant_ure_files.append(target)

        if int(meta['baseline']) == relevance['baseline'] and meta['tissue'] == 'Peritoneum':
            orientation = target.split('/')[-1]
            orientation = orientation.split('_O_')[-1].split('_')[0]
            if orientation == relevance['orientation']:
                relevant_peri_files.append(target) 

    n_ure  = len(relevant_ure_files)
    n_peri = len(relevant_peri_files)

    print('Relevant ureters in path     : ' , n_ure)
    print('Relevant peritoneums in path : ' , n_peri)

    processed_data_file_name = 'metrics_group_1_' + interest + '.csv'
    processed_file_header    = ['tissue' , 'thickness', 'baseline',\
                                'orientation', 'Pig ID' , 'tool' , \
                                 'S1','S2','660/580','700/610','930/610','M1','M2','M3']
    file_pointer             = open(os.path.join(result_dump_path,processed_data_file_name),'w',encoding='UTF8')
    file_writer              = csv.writer(file_pointer)
    file_writer.writerow(processed_file_header)

    for i in range(0,n_ure):
        current_meta    = process_x15_rt_file_name(relevant_ure_files[i])
        current_sample  = process_file_content_x15_rt_two_step(relevant_ure_files[i] , plot_calib_bands = False)
        pig_ID          = current_meta['comment'].split('-')[0][-1] + '-' + current_meta['comment'].split('-')[-1].split('_')[0]
        metric_set      = generate_metrics_group_1(current_sample['l'],current_sample['x15_c'])
        content_to_file = [current_meta['tissue'],current_meta['thickness'],\
                           current_meta['baseline'],relevance['orientation'],\
                           pig_ID , current_meta['tool']]
        for j in range(0,len(metric_set)):
            content_to_file.append(metric_set[j])
        file_writer.writerow(content_to_file)

    for i in range(0,n_peri):
        current_meta    = process_x15_rt_file_name(relevant_peri_files[i])
        current_sample  = process_file_content_x15_rt_two_step(relevant_peri_files[i] , plot_calib_bands = False)
        pig_ID          = current_meta['comment'].split('-')[0][-1] + '-' + current_meta['comment'].split('-')[-1].split('_')[0]
        metric_set      = generate_metrics_group_1(current_sample['l'],current_sample['x15_c'])
        content_to_file = [current_meta['tissue'],current_meta['thickness'],\
                           current_meta['baseline'],relevance['orientation'],\
                           pig_ID , current_meta['tool']]
        for j in range(0,len(metric_set)):
            content_to_file.append(metric_set[j])
        file_writer.writerow(content_to_file)

    print('At main exit line.')