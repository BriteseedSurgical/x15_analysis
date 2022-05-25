import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd
from pathlib import Path
from scipy import optimize

n_control             = 6
n_wls                 = 15
n_pixels              = 99
wl_set                = [450 , 520 , 550 , 580 , 610 , 635 , 660 , 700 , 740 , 770 , 810 , 840 , 870 , 900 , 930] 
visible_end_index     = 11 
max_post_adapt_length = 32

gaussian_generator = lambda x:(1/(np.sqrt(2*np.pi*x[1]*x[1]))) * np.exp(-1 * (1/(2*x[1]*x[1])) * (x[2] - x[0]) * (x[2] - x[0]))

sensor_x = [400,500,550,600,650,700,800,900,950,1000]
sensor_y = [0.5,0.73,0.82,0.9,0.96,1.0,0.85,0.55,0.35,0.15]

nir_led_x = [400,800,850,875,900,925,950,962.5,975,1000]
nir_led_y = [0,0,0.02,0.075,0.2,0.5,1,0.6,0.2,0.02]

def generate_gaussian(start,stop,step,mu,sigma):
    
    global gaussian_generator

    n = int((stop - start)/step) + 1
    gaussian = np.zeros([n])
    for j in range(0,n):
        evaluate_at = start + j*step
        gaussian[j] = gaussian_generator([mu,sigma,evaluate_at])
    return gaussian

if __name__ == '__main__':


    y_vis = generate_gaussian(400,1000,10,560,44)
    y_vis = y_vis/np.max(y_vis)

    root           = os.getcwd()
    parent         = Path(root)
    parent         = parent.parent.absolute()
    data_path      = os.path.join(parent,'docs')
    vis_led_target = os.path.join(data_path , 'vis_led_characteristic.csv')

    vis_datagram   = pd.read_excel(vis_led_target).values
    n_vis          = np.shape(vis_datagram)[0]
    x_vis          = vis_datagram[15:,0]
    y_vis          = vis_datagram[15:,1]
    y_vis          = y_vis/np.max(y_vis)

    wl_axis        = np.linspace(400,1000,np.shape(y_vis)[0])
    f_sensor       = np.interp(wl_axis,sensor_x,sensor_y)
    f_sensor       = f_sensor/np.max(f_sensor)

    vis_cmos       = np.multiply(f_sensor,y_vis)
    vis_cmos       = vis_cmos/np.max(vis_cmos)

    f_nir_led      = np.interp(wl_axis,nir_led_x,nir_led_y)
    nir_cmos       = np.multiply(f_sensor,f_nir_led)
    nir_cmos       = nir_cmos/np.max(nir_cmos)

    overall = nir_cmos + vis_cmos
    overall = overall/np.max(overall)

    plt.figure()
    plt.title('LED - CMOS composite frequency responses')
    plt.ylabel('Relative Tx/Rx')
    plt.xlabel('wavelength (nm)')
    plt.plot(wl_axis,f_sensor)
    plt.plot(wl_axis,y_vis)
    plt.plot(wl_axis,vis_cmos)
    plt.plot(wl_axis,f_nir_led)
    plt.plot(wl_axis,nir_cmos)
    plt.plot(wl_axis,overall)
    plt.plot(wl_axis,np.ones([np.shape(nir_cmos)[0]])*0.1)
    plt.plot(wl_axis,np.ones([np.shape(nir_cmos)[0]])*0.2)
    plt.plot(wl_axis,np.ones([np.shape(nir_cmos)[0]])*0.5)
    plt.legend(['Sensor response','Vis LED Tx(relative)','vis_led - sensor combined response','NIR-LED Tx(relative)','nir_led - sensor response',' sensor-LED response','0.1','0.2','0.5'])
    for i in range(0,n_wls):
        plt.axvline(wl_set[i])
    plt.show()
    
