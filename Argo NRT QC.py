# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:33:19 2024

@author: JethandH
"""
"""
Near real-time QC checks for Argo data
"""
import datetime
import math
import numpy as np
import gsw
#import pandas as pd

####################################################
############## To do: 03 Dec 2024 ##################
# Regional range test (test 7) Get range from Tammy
# Tests 15, 16, 18, 19, 25
# Test functions on seatrec float data

def Platform_Identification_test(float_id):
    
    """"
    Test 1:
        Input:
            Argo floats ID number (Intager, float or string like)
        Output:
            True if test is passed
            False if test is failed
    
    """
    
    float_id = str(float_id)
    known_floats = ['1465']
    
    return float_id in known_floats

def Impossible_date_test(Argo_date):
    
    """ 
    Test 2:
        Input:
            Datetime (datetime like)
        Output:
            True if test is passed
            False if test is failed
    
    """
    
    test_date = datetime.datetime.strptime('1997-01-01','%Y-%m-%d')
    return Argo_date >= test_date
    
def Impossible_Location_test(lon,lat):
    
    """ Test 3:
        Input: 
            Longitude (float like)
            Latitude (float like)
        Outhput: 
            True if test is passed
            False if test is failed
        
        """
    
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return False
    else:
        return True
    
def Position_on_Land_test(argo_lon,argo_lat,path_to_bathy):
    
    """ 
    Test 4:
        Input:
            Longitude (float like)
            Latitude (float like)
            File path to bathymetry data (string like)
        Output:
            True if test is passed
            False if test is failed
        
        """
    
    import xarray as xr
    
    bathy = xr.open_dataset(path_to_bathy)
    
    elevation = bathy.interp(lat=argo_lat, lon=argo_lon, method='linear').elevation.item()
    
    return elevation > 0



def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers. Use 3956 for miles. Determines return value units.
    r = 6371.0
    
    # Calculate the result
    return c * r

def Impossible_Speed_test(lon1,lat1,lon2,lat2,time1,time2):
    
    """ 
    Test 5:
        Input:
            Longitude (original position) (Float like)
            Latitude (original position) (Float like)
            Longitude (new position) (Float like)
            latitude (new poisition) (Float like)
            Datetime of original position (Datetime like)
            Datetime of new position (Datetime like)
        Output:
            True if test is passed
            False if test is failed
        """
    
    dist = haversine(lat1, lon1, lat2, lon2)*1000
    elapsed_time = (datetime.datetime.strptime(time2,'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(time1,'%Y-%m-%d %H:%M:%S')).seconds
    
    speed = dist/elapsed_time
    
    if abs(speed) <= 3:
        return True
    else:
        return False
    
def Global_Range_test(pressure,temperature,salinity):
    
    """ 
    Test 6:
        Input: 
            Pressue profile (Array like)
            Temperature profile (Array like)
            Salinity profile (Array like)
            
        Output:
            Pressure flag (Array like)
            Temperature flag (Array like)
            Salinity flag (Array like)
        
        """
    pressure_flag = np.empty(np.shape(pressure))
    temperature_flag = np.empty(np.shape(temperature))
    salinity_flag = np.empty(np.shape(salinity))    
    
    for pp in range(len(pressure)): 
        if 5 < pressure[pp]:
            pressure_flag[pp] = 4
            temperature_flag[pp] = 4
            salinity_flag[pp] = 4
        elif 2.4 <= pressure[pp]  <=5: # 2.5 <= value <= 5
            pressure_flag[pp] = 3
            temperature_flag[pp] = 3
            salinity_flag[pp] = 3
        else:
            pressure_flag[pp] = 1
            temperature_flag[pp] = 1
            salinity_flag[pp] = 1
    for tt in range(len(temperature)):
        if temperature[tt] < -2.5 or temperature[tt] > 40:
            temperature_flag[tt] = 4
    
    for ss in range(len(salinity)):    
        if salinity[ss] < 2 or salinity[ss] > 41:
            salinity_flag[ss] = 4
        
    return pressure_flag, temperature_flag, salinity_flag

def regional_range_test():
    """ Test 7 """
    return

def Pressure_Increasing_test(pressure):
    
    """ 
    Test 8:
        Input: 
            Pressue profile (Array like)
        Output:
            Pressure flag (Array like)
    """
    
    pressure_flag = []
    for ii in range(len(pressure)):
        
        p1 = pressure[ii-1]
        p2 = pressure[ii-2]
        
        p_dif = p2-p1
        
        if p_dif >= 20:
            pressure_flag.append(4)
        else:
            pressure_flag.append(1)
            
        pressure_flag = pressure_flag[-1::]
    
    return pressure_flag


def spike_test(temperature, salinity, pressure):
    
    """ 
    Test 9:
        
        Input: 
            Temperaure profile (Array like)
            Salinity profile (Array like)
            Pressue profile (Array like)
            
        Output: 
            Temperaure flag (Array like)
            Salinity flag (Array like)
    
    """
    
    
    for ii in range(len(temperature)-3):
        
        pres = pressure[ii]
        t1 = temperature[ii]
        t2 = temperature[ii+1]
        t3 = temperature[ii+2]
       
        temp_test_val = (t2-(t3+t2)/2) - ((t3-t1)/2)
        
        s1 = salinity[ii]
        s2 = salinity[ii+1]
        s3 = salinity[ii+2]
        sal_test_val = (s2-(s3+s2)/2) - ((s3-s1)/2)
       
        if pres < 500 & temp_test_val > 6:
            temp_val_flag = 4
        elif pres >= 500 & temp_test_val > 2:
            temp_val_flag = 4
        else:
            temp_val_flag = 1
            
        if pres < 500 & sal_test_val > 0.9:
            sal_val_flag = 4
        elif pres >= 500 & sal_test_val > 0.3:
            sal_val_flag = 4
        else:
            sal_val_flag = 1
   
    return temp_val_flag, sal_val_flag


def top_bottom_spike_test():
    """ Test 10 Obsolete """
    return

def gradient_test():
    """ Test 11 Obsolete """
    return



def digit_rollover_test(temperature, salinity, pressure, temp_threshold=10, sal_threshold=5):
    """
    Test 12:
        
    Perform the digit rollover test on Argo float temperature and salinity profiles.

    Parameters:
        temperature (array-like): Temperature values.
        salinity (array-like): Salinity values.
        pressure (array-like): Pressure levels corresponding to the values.
        temp_threshold (float): Maximum allowed difference in temperature between adjacent levels.
        sal_threshold (float): Maximum allowed difference in salinity between adjacent levels.

    Returns:
        qc flags for temperature and salinity
    """
    # Initialize QC flags (default: 1 = good data)
    qc_temp = np.ones_like(temperature, dtype=int)
    qc_sal = np.ones_like(salinity, dtype=int)

    # Compute differences between adjacent pressure levels
    temp_diff = np.abs(np.diff(temperature))
    sal_diff = np.abs(np.diff(salinity))

    # Identify indices where differences exceed thresholds
    temp_fail = np.where(temp_diff > temp_threshold)[0]
    sal_fail = np.where(sal_diff > sal_threshold)[0]

    # Flag the failing values and their corresponding pair
    for idx in temp_fail:
        qc_temp[idx] = 4
        qc_temp[idx + 1] = 4

    for idx in sal_fail:
        qc_sal[idx] = 4
        qc_sal[idx + 1] = 4

    # If both temp and salinity fail, ensure both are flagged
    for idx in set(temp_fail) & set(sal_fail):
        qc_temp[idx] = 4
        qc_sal[idx] = 4

    
    return qc_temp, qc_sal

def Stuck_value_test(temperature,salinity):
    """ 
    Test 13:
        Input:
            Temperature profile (Array like)
            Salinity profile (Array like)
        Output:
            Temperature flag (Array like)
            Salinity flag (Array like)
            
    """
    # Initialize QC flags (default: 1 = good data)
    qc_temp = np.ones_like(temperature, dtype=int)
    qc_sal = np.ones_like(salinity, dtype=int)

    # Check if all temperature values are identical
    if np.all(temperature == temperature[0]):
        qc_temp[:] = 4  # Flag all temperature values as bad

    # Check if all salinity values are identical
    if np.all(salinity == salinity[0]):
        qc_sal[:] = 4  # Flag all salinity values as bad

    # If both temperature and salinity are stuck, flag the entire profile
    if np.all(qc_temp == 4) and np.all(qc_sal == 4):
        qc_temp[:] = 4
        qc_sal[:] = 4
    
    return qc_temp, qc_sal


def Density_inversion_test(temperature, salinity, pressure,longitude,latitude):
    """ 
    Test 14:
        Input:
            Temperature (array like)
            Salinity (array like)
            Pressure (array like)
        Output:
            Potential Density (array like)
            Temperature flag (array like)
            Salinity flag (array like)
    
    """
    
    temp_flag = np.empty(np.shape(temperature))
    sal_flag = np.empty(np.shape(salinity))
    absolute_salinity = gsw.SA_from_SP(salinity,pressure,longitude,latitude)
    potential_density = gsw.pot_rho_t_exact(absolute_salinity, temperature, pressure, max(pressure))
    
    for dd in range(len(potential_density)-1):
        
        dens_diff = potential_density[dd] - potential_density[dd+1]
        dens_diff_2 = potential_density[dd+1] - potential_density[dd]
        
        if dens_diff > 0.03 or dens_diff_2 > 0.03:
            temp_flag[dd] = 4
            sal_flag[dd] = 4
            
        else:
            temp_flag[dd] = 1
            sal_flag[dd] = 1
        
    return temp_flag, sal_flag, potential_density

def Grey_list_test():
    """ 
    Test 15:
        Wont implement at this stage as we only have one float (and we dont have the list)
    """
    return

def Gross_sensor_drift_test():
    """Test 16"""
    return

def visual_profile_test():
    """
    Test 17: 
        Only required for DMQC
    """
    return

def Frozen_profile_test():
    """Test 18"""
    return

def Deepest_pressure_test():
    """Test 19"""
    return

def MEDD_test():
    """Test 25"""
    return

# Example profiles
pressure = [0, 10, 20, 30, 40]  # Depth in dbar
temperature = [20, 22, 35, -5, 5]  # Example temp with rollover
salinity = [35, 36, 45, 1, 37]  # Example salinity with rollover

# Run the test
results = digit_rollover_test(temperature, salinity, pressure)
print(results)



Argo_dates = datetime.datetime.strptime('2024-05-10  06:16:00', '%Y-%m-%d %H:%M:%S')
Impossible_date_test(Argo_dates)

Impossible_Location_test(23.4, -34.5)


lon1 = 26.2233
lat1 = -34.5062
lon2 = 26.0708 
lat2 = -34.6218
lon3 = 23.1308
lat3 = -38.0697	

t1 = '2024-05-08  23:27:30'
t2 = '2024-05-09  09:35:20'
t3 = '2024-05-23  11:54:00'

Impossible_Speed_test(lon1, lat1, lon2, lat2, t1, t2)


pres = -100
temp = 35
salt = 44

p_flag, t_flag, s_flag = Global_Range_test(pres,temp,salt)

import numpy as np

pres_t = np.linspace(0,100,100)

pref_flag = Pressure_Increasing_test(pres_t)
