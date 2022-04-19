#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Evaluate SPHARM decomposition coefficients for all subjects

Repeat for training and testing datasets.
Repeat for regular and spherical segmentations.

BY THIERRY LEFEBVRE 
for MSc Thesis Project 2020/08
under the supervision of Prof Peter Savadjiev
Medical Physics Unit, McGill University

"""


import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, mkdir
import os as os
from os.path import isdir, join
import scipy.io as sio
import pandas as pd
import matlab.engine


mypathimg = 'IMG'
mypathseg = 'SEG'

mypathsave = 'SAVE'

R_max = 25
L_max = 25

dirsimg = listdir(mypathimg)
dirsimg.sort()
dirsseg = listdir(mypathseg)
dirsseg.sort()


fullpathsimg = []
fullpathsseg = []

for dir1 in dirsimg:
    fullpathsimg.append(join(mypathimg,dir1[0:-4]+'.nii'))
    fullpathsseg.append(join(mypathseg,dir1[0:-4]+'.nii'))


eng = matlab.engine.start_matlab()

# Has to be adjusted to split based on outcomes in clinical outcomes
for i in range(len(dirsimg)):
    
    flmr_in = eng.fun_spharm(fullpathsimg[i],fullpathsseg[i],R_max,L_max)
    flmr_in = np.asarray(flmr_in)

    sio.savemat((join(mypathsave,dirsimg[i][0:-4])+'.mat'),{'flmr_in':flmr_in})

 