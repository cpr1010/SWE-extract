#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import pydicom.data
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.pixel_data_handlers import convert_color_space
import pydicom.encoders.gdcm
import pydicom.encoders.pylibjpeg
import numpy as np
from PIL import Image 
import tifffile as tiff
import scipy.cluster.vq as scv
from scipy import stats
import cv2 as cv
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import _jet_data
from textwrap import dedent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
mpl.use('TkAgg')



# Takes DICOM file and turns it into an object 

class tendon_elastogram:
    
    
    def __init__(self, path):
        
        self.dicom = pydicom.dcmread(path)
        self.img = self.dicom.pixel_array # the actual image 
        
        
    def clr_map(self): 
        
        # This function extracts the color map as well as the gray value colormap used in the image, combines them into one
        # in order to both get shear wave speeds from colored sections, and remove gray value pixes from analysis
        
        exp2 = self.dicom[0x7fe11001][0][0x7fe11010][0][0x7fe11018][10][0x7fe11055] # location of color bar data in DICOM


        ls = []
        for b in exp2:
            v = b
            ls.append(v)

        ls_unique = set(ls)

        chunked_list = list()
        chunk_size = 3

        for i in range(0, len(ls), chunk_size):
            chunked_list.append(ls[i:i+chunk_size])


        chunked_list

        chunked_list1 = list()

        chunk_size = 256

        for i in range(0, len(chunked_list), chunk_size):
            chunked_list1.append(chunked_list[i:i+chunk_size])




        cl1_arr = np.array(chunked_list1)
        

        pre_cm = cl1_arr[:,0]
        pre_cm_rgba = []
        for i in range(0, len(pre_cm)):
            a = np.insert(pre_cm[i], len(pre_cm[i]), 1)
            a = a[0:3] / 255
            pre_cm_rgba.append(a)


        my_clr_cmap = mcolors.ListedColormap(pre_cm_rgba, name='my_clr_colormap')

        cmap = plt.get_cmap(my_clr_cmap)

        tst = plt.cm.gray
        tst = tst(np.linspace(0,1, 50))

        clr = cmap(np.linspace(0,1,100))

        colors = np.vstack((tst, clr))
        clr_gs_cmap = mcolors.LinearSegmentedColormap.from_list('clr_gs_cmap', colors)

        #plt.register_cmap('clr_gs_cmap', clr_gs_cmap)

        clr_gs_cmap = plt.get_cmap(clr_gs_cmap)

        return clr_gs_cmap

    def ROI_trace(self):
        
        # Allows the user to select a region of interest for analysis instead of using the whole ultrasound image 
        # For some reason 
        
        img = self.img

        ROI = cv.selectROI('select',img) #input the color image of tendon (ds_sm)
        roi = img[int(ROI[1]):int(ROI[1]+ROI[3]),
                          int(ROI[0]):int(ROI[0]+ROI[2])]

        cv.destroyWindow('select')
        roi_rgba = []
        for a in roi:
            for b in a:
                c = b/255
                c = np.insert(c,len(c),1)
                roi_rgba.append(c)

        roi_rgba = np.array(roi_rgba)  

        shape = np.array(roi.shape)
        shp = np.array([0,0,1])
        shape = shape + shp

        roi_rgba = roi_rgba.reshape(shape)
        return roi_rgba

    def colormap2arr2(roi,cmap): 
        
        # This function basically takes the color observations of each pixel within the region of interest and performs a 
        # vector quantization to map it to the color map we extracted from the DICOM file. 
         

        arr = roi

        # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
        gradient=cmap(np.linspace(0.0,1.0,150))

        # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
        arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))

        # Use vector quantization to shift the values in arr2 to the nearest point in
        # the code book (gradient).
        code,dist=scv.vq(arr2,gradient)


        # code is an array of length arr2 (240*240), holding the code book index for
        # each observation. (arr2 are the "observations".)
        # Scale the values so they are from 0 to 1.
        values=code.astype('float')/100


        # Reshape values back to (240,240)
        values=values.reshape(arr.shape[0],arr.shape[1])
        values=values[::-1]
        
        values2 = values*10
        values2 = values2 - 5
        
        return values2

    def sws_calc(values2):
        
        # Removes the negative values from converted shear wave speed array as these are grayvalue pixels and shouldn't contain
        # any acual shear wave speed data
    

        values3 = values2[values2 >= 0]

        sws_min = 'minimum shear wave speed: ' + str(round(np.min(values3),2)) + " m/s"
        sws_max = 'maximum shear wave speed: ' + str(round(np.max(values3),2), ) + " m/s"
        sws_median = 'median shear wave speed: ' + str(round(np.median(values3),2)) + " m/s"
        sws_75 = '75th percentile shear wave speed: ' + str(round(np.percentile(values3, 75),2)) + " m/s"
        
        return sws_min, sws_max, sws_median, sws_75
            

    def sws_graph(values2, roi, cmap):
        
        # Produces a graph of the selected region of interest alongside the used color map (excluding the grayvalue portion)
        
        fig = plt.figure()
        im = plt.imshow(roi, cmap=plt.cm.jet, interpolation='nearest', vmin=0, vmax=10)
        plt.title('Achilles Tendon Shear Wave Speed Map')
        fig.colorbar(im, label='Meters/second')
        figg = plt.gcf()
        return figg
    
    def sws_graph_test(values2, roi, cmap):
        
        # Creates graph of original ROI, as well as a graph of the shear wave speed array against our created colormap to 
        # provide visual assesment that evrything worked accordingly
        
        fig = plt.figure()
        im = plt.imshow(roi, cmap=plt.cm.jet, interpolation='nearest', vmin=0, vmax=10)
        plt.title('Achilles Tendon Shear Wave Speed Map')
        fig.colorbar(im)
        figg = plt.gcf()
        
        fig = plt.figure()
        im2 = plt.imshow(values2, cmap=cmap, interpolation='nearest', vmin=-5, vmax=10)
        plt.title('Achilles Tendon Shear Wave Speed Map')
        fig.colorbar(im2)
        figg2 = plt.gcf()
        return figg, figg2
    
    #def save_data(graph, sws_min, sws_max, sws_median, sws_75, sub_id):
        
        # create dataframe for sws data with participant ID
        
        
        # save graph to same folder where data is stored, name accordingly
        


# In[2]:


import PySimpleGUI as sg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg    

def is_number(x):
    '''
        Takes a word and checks if Number (Integer or Float).
    '''
    try:
        # only integers and float converts safely
        num = float(x)
        return True
    except ValueError as e: # not convertable to float
        return False
    
def pull_float(string):
    
    '''
    Pulls the number value from string and converts to float.
    
    '''
    
    obj = string.split()
    for i in obj:
        if is_number(i) == True:
            return float(i)
        else:
            pass

def save_graph(graphs, ID, session_num):
    
    graph = graphs
    
    current_path = os.getcwd()
    folder = 'SWE_data'
    pth = os.path.join(current_path, folder)
    subject_id = ID

    if os.path.exists(pth) == True:
        # save graphs and data to existing directories/datasets
        im_pth = os.path.join(pth, 'sws_graphs')
        im_name = subject_id+'_session'+str(session_num)
        output_pth = os.path.join(im_pth, im_name+'.png')
        plt.savefig(output_pth)
        
        

    else: 

        os.mkdir(os.path.join(current_path, folder))
        fldr = os.path.join(current_path, folder)

        im_fldr = 'sws_graphs'
        os.mkdir(os.path.join(fldr, im_fldr))
        im_pth = os.path.join(pth, im_fldr)
        im_name = subject_id+'_session'+str(session_num)
        output_pth = os.path.join(im_pth, im_name+'.png')
        plt.savefig(output_pth)

        
# make save_sws_data function to save to dataframe (as excel or something)    

def save_sws_data(sws_min, sws_max, sws_median, sws_75, ID, session_num):
    
    sws_min = pull_float(sws_min)
    sws_max = pull_float(sws_max)
    sws_median = pull_float(sws_median)
    sws_75 = pull_float(sws_75)
    
    current_path = os.getcwd()
    folder = 'SWE_data'
    file_name = 'swe_numbers'
    pth = os.path.join(current_path, folder, file_name+'.csv')
    subject_id = ID
    
    
    if os.path.exists(pth) == True:
        
        data = pd.read_csv(pth, index_col=0)
    
        new_data = {'Subject ID': ID, 'session': session_num, 'SWS min': sws_min, 'sws max': sws_max, 
            'sws median': sws_median, 'sws 75 percentile': sws_75}
        data = data.append(new_data, ignore_index=True)
        data.to_csv(pth)
        
    else:
        
        new_data = {'Subject ID': [ID], 'session': [session_num], 'SWS min': [sws_min], 'sws max': [sws_max], 
            'sws median': [sws_median], 'sws 75 percentile': [sws_75]}
        
        data_df = pd.DataFrame(new_data)
        data_df.to_csv(pth)
    


# In[3]:


def make_win1():
    sg.theme("DarkTeal2")
    layout = [[sg.T("")], [sg.Text("Choose a DICOM file: "), sg.Input(), sg.FileBrowse(key="-IN-")],[sg.Button("SELECT ROI & ANALYZE")],[sg.Text("This program calculates the shear wave speeds for all pixels in the ultrasound elastogram.")]]
    return sg.Window('My File Browser', layout, size=(800,500), finalize=True, element_justification='c')   
    
    
def make_win2():
    sg.theme("DarkTeal4")          
    layout = [
            [sg.Text("* Pixels with grayvalues do NOT contain shear wave speed data")],
            [sg.Canvas(key="-CANVAS-")], 
            [sg.Text("", key='OUTPUT')],
            [sg.Text("", key='OUTPUT2')],
            [sg.Text("", key='OUTPUT3')],
            [sg.Text("", key='OUTPUT4')],
            [sg.Button("Exit")],
            [sg.Text('Subject ID', size =(15, 1)), sg.InputText(key='-ID-')],
            [sg.Text('Session Number', size =(15, 1)), sg.InputText(key='-sessNum-')],
            [sg.Button("Save Data")]
            ]
    
    return sg.Window('Tendon Shear Wave Analysis', layout, size=(800,800), finalize=True, element_justification='c')

    
    
window1, window2 = make_win1(), None 

while True:
    window, event, values = sg.read_all_windows()
    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == window2:       # if closing win 2, mark as closed
            window2 = None
        elif window == window1:     # if closing win 1, exit program
            break

    elif event == "SELECT ROI & ANALYZE" and not window2:
        window2 = make_win2()
        dicom = values['-IN-']
        test_elast = tendon_elastogram(repr(dicom)[1:-1])

        dicom = test_elast.dicom

        roi = test_elast.ROI_trace()

        cmap = test_elast.clr_map()

        values = tendon_elastogram.colormap2arr2(roi, cmap)

        sws_min, sws_max, sws_median, sws_75 = tendon_elastogram.sws_calc(values)

        graphs = tendon_elastogram.sws_graph(values, roi, cmap)

        fig_photo = draw_figure(window2['-CANVAS-'].TKCanvas, graphs)
        window2['OUTPUT'].update(sws_min)
        window2['OUTPUT2'].update(sws_max)
        window2['OUTPUT3'].update(sws_median)
        window2['OUTPUT4'].update(sws_75)
        
    elif event == "Save Data":
        
        ID = values['-ID-']
        session_num = values['-sessNum-']
        
        save_graph(graphs, ID, session_num)
        save_sws_data(sws_min, sws_max, sws_median, sws_75, ID, session_num)
        
        window.close()
        if window == window2:       # if closing win 2, mark as closed
            window2 = None
        elif window == window1:     # if closing win 1, exit program
            break
        
        
window.close()
                

