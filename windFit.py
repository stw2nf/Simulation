# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:30:18 2021

@author: stw2nf
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import signal

# %% Functions
# Python3 program to find element 
# closet to given target. 

# Returns element closest to target in arr[] 
def findClosest(arr, n, target): 
    # Corner cases 
    if (target <= arr[0]): 
        return arr[0] 
    if (target >= arr[n - 1]): 
        return arr[n - 1] 
    # Doing binary search 
    i = 0; j = n; mid = 0
    while (i < j):  
        mid = int((i + j) / 2)
        if (arr[mid] == target): 
            return arr[mid] 
        # If target is less than array  
        # element, then search in left 
        if (target < arr[mid]) : 
            # If target is greater than previous 
            # to mid, return closest of two 
            if (mid > 0 and target > arr[mid - 1]): 
                return getClosest(arr[mid - 1], arr[mid], target) 
            # Repeat for left half  
            j = mid 
        # If target is greater than mid 
        else : 
            if (mid < n - 1 and target < arr[mid + 1]): 
                return getClosest(arr[mid], arr[mid + 1], target) 
            # update i 
            i = mid + 1  
    # Only single element left after search 
    return arr[mid] 
# Method to compare which one is the more close. 
# We find the closest by taking the difference 
# between the target and both values. It assumes 
# that val2 is greater than val1 and target lies 
# between these two. 
def getClosest(val1, val2, target): 
    if (target - val1 >= val2 - target): 
        return val2 
    else: 
        return val1 
# This code is contributed by Smitha Dinesh Semwal

# %% Main
def characterizeWind():
    m2ft = 3.28 #meter to foot conversion
    
    startAlt = 14500
    endAlt = 4500
    windFiles = 14
    address = r'C:\Users\stw2nf\Box\Scripts\Renamed Balloon Files\wind'
    
    windFits = np.zeros((windFiles, windFiles))
    type1 = np.zeros((1,3))
    type2 = np.zeros((1,3))
    type3 = np.zeros((1,3))
    
    for i in range(windFiles):
        
        locI =  address + str(i) + '.csv'
        real_wind = pd.read_csv(locI)
        real_wind = real_wind.to_numpy()
        wind_alt = real_wind[:,0]*m2ft #MSL altitude (ft)
        windI = (real_wind[:,1]**2 + real_wind[:,2]**2)**(1/2)*m2ft #ft/s
        startI = np.where(wind_alt == findClosest(wind_alt, len(wind_alt), startAlt))
        endI = np.where(wind_alt == findClosest(wind_alt, len(wind_alt), endAlt))
        startI = startI[0][0]
        endI = endI[0][0]
        if startI > endI:
            temp = endI
            endI = startI
            startI = temp
                
        windI = windI[startI:endI]
        for j in range(windFiles):
                
            locJ =  address + str(j) + '.csv'
            pred_wind = pd.read_csv(locJ)
            pred_wind = pred_wind.to_numpy()
            pwind_alt = pred_wind[:,0]*m2ft #MSL altitude (ft)
            windJ = (pred_wind[:,1]**2 + pred_wind[:,2]**2)**(1/2)*m2ft #ft/s
            startJ = np.where(pwind_alt == findClosest(pwind_alt, len(pwind_alt), startAlt))
            endJ = np.where(pwind_alt == findClosest(pwind_alt, len(pwind_alt), endAlt))
            startJ = startJ[0][0]
            endJ = endJ[0][0]
            if startJ > endJ:
                temp = endJ
                endJ = startJ
                startJ = temp
            
            windJ = windJ[startJ:endJ]
            
            if len(windI) > len(windJ):
                windJ = signal.resample(windJ, len(windI))
            else:
                windI = signal.resample(windI, len(windJ))
            
            windFits[i,j] = mean_squared_error(windI, windJ, squared=False)
            
            if (i != j) and (i < j):
                if (windFits[i,j]<=6.66):
                    type1 = np.append(type1, [[i,j,windFits[i,j]]],axis=0)
                if (windFits[i,j]>6.66) and (windFits[i,j]<=13.334):
                   type2 = np.append(type2, [[i,j,windFits[i,j]]],axis=0)       
                if (windFits[i,j]>13.334):
                   type3 = np.append(type3, [[i,j,windFits[i,j]]],axis=0)
    maxError = np.amax(windFits)
    type1 = np.delete(type1, 0, axis=0)
    type2 = np.delete(type2, 0, axis=0)
    type3 = np.delete(type3, 0, axis=0)
    
    allTypes = [type1, type2, type3]
    return allTypes, type1, type2, type3