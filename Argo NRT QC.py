# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:33:19 2024

@author: Jethan d'Hotman
Email: jethandh@gmail.com
"""
"""
Near real-time QC checks for Argo data
"""
import datetime
import math
import numpy as np
import gsw
import pandas as pd
import xarray as xr

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
    if Argo_date >= test_date:
        return 1
    else:
        return 4
    
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
        return 4
    else:
        return 1
    
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
    
    bathy = xr.open_dataset(path_to_bathy)
    
    elevation = bathy.interp(lat=argo_lat, lon=argo_lon, method='linear').elevation.item()
    
    if elevation < 0:
        return 1
    else:
        return 4



def haversine(lat1, lon1, lat2, lon2):
    
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude.

    Parameters:
    lat1, lon1: Latitude and Longitude of point 1 (degrees)
    lat2, lon2: Latitude and Longitude of point 2 (degrees)

    Returns:
    Distance in meters
    """
    
    R = 6371000  # Earth radius in meters

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


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
            1 if test is passed
            4 if test is failed
        """
    
    dist = haversine(lat1, lon1, lat2, lon2)
    elapsed_time = (datetime.datetime.strptime(time2,'%Y-%m-%d %H:%M:%S') - 
                    datetime.datetime.strptime(time1,'%Y-%m-%d %H:%M:%S')).total_seconds()
    
    speed = dist/elapsed_time
    
    if abs(speed) <= 3:
        flag = 1
    else:
        flag = 4
        
    return flag

    
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
            pressure_flag[pp] = int(4)
            temperature_flag[pp] = int(4)
            salinity_flag[pp] = int(4)
        elif 2.4 <= pressure[pp]  <=5: # 2.5 <= value <= 5
            pressure_flag[pp] = int(3)
            temperature_flag[pp] = int(3)
            salinity_flag[pp] = int(3)
        else:
            pressure_flag[pp] = int(1)
            temperature_flag[pp] = int(1)
            salinity_flag[pp] = int(1)
    for tt in range(len(temperature)):
        if temperature[tt] < -2.5 or temperature[tt] > 40:
            temperature_flag[tt] = int(4)
    
    for ss in range(len(salinity)):    
        if salinity[ss] < 2 or salinity[ss] > 41:
            salinity_flag[ss] = int(4)
        
    return pressure_flag, temperature_flag, salinity_flag

def Regional_range_test():
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
    for ii in range(len(pressure)-1):
        
        p1 = pressure[ii]
        p2 = pressure[ii+1]
        
        p_dif = p2-p1
        
        if p_dif >= 20:
            pressure_flag.append(4)
        else:
            pressure_flag.append(1)
            
        #pressure_flag = pressure_flag[-1::]
    
    return pressure_flag


def Spike_test(temperature, salinity, pressure):
    
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



def Digit_rollover_test(temperature, salinity, pressure, temp_threshold=10, sal_threshold=5):
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
            Temperature flag (array like)
            Salinity flag (array like)
    
    """
    
    temp_flag = np.empty(np.shape(temperature))
    sal_flag = np.empty(np.shape(salinity))
    absolute_salinity = gsw.SA_from_SP(salinity,pressure,longitude,latitude)
    pot_temp = gsw.pt0_from_t(absolute_salinity,temperature,pressure)
    potential_density = gsw.pot_rho_t_exact(absolute_salinity, pot_temp , pressure, max(pressure))
    
    for dd in range(len(potential_density)-1):
        
        dens_diff = potential_density[dd] - potential_density[dd+1]
        dens_diff_2 = potential_density[dd+1] - potential_density[dd]
        
        if dens_diff > 0.05 or dens_diff_2 > 0.05:
            temp_flag[dd] = int(4)
            sal_flag[dd] = int(4)
            
        else:
            temp_flag[dd] = int(1)
            sal_flag[dd] = int(1)
        
    return temp_flag, sal_flag

def Grey_list_test():
    """ 
    Test 15:
        Wont implement at this stage as we only have one float (and we dont have the list)
    """
    return

def Gross_sensor_drift_test(
    salinity_current, salinity_previous, temperature_current, temperature_previous,
    pressure_current, pressure_previous, salinity_qc, temperature_qc,
    salinity_threshold=0.5, temperature_threshold=1.0
):
    """
    Test 16: Implements the Gross Salinity or Temperature Sensor Drift Test for Argo float data.

    Parameters:
        salinity_current (array): Array of salinity values for the current profile.
        salinity_previous (float or None): Average salinity value from the previous profile.
        temperature_current (array): Array of temperature values for the current profile.
        temperature_previous (float or None): Average temperature value from the previous profile.
        pressure_current (array): Array of pressure values for the current profile.
        pressure_previous (array or None): Array of pressure values for the previous profile.
        salinity_qc (array): Array of salinity QC flags for the current profile.
        temperature_qc (array): Array of temperature QC flags for the current profile.
        salinity_threshold (float): Threshold for salinity drift (default: 0.5 PSU).
        temperature_threshold (float): Threshold for temperature drift (default: 1.0°C).

    Returns:
        tuple: Arrays of salinity and temperature flags.
    """
    # Initialize flags
    sal_flag = np.ones_like(salinity_current, dtype=int)  # Default to 'good' (1)
    temp_flag = np.ones_like(temperature_current, dtype=int)  # Default to 'good' (1)

    # Extract data from the deepest 100 dbar for the current profile
    deep_mask_current = pressure_current >= (np.max(pressure_current) - 100)
    deep_salinity_current = salinity_current[deep_mask_current & (salinity_qc == 1)]
    deep_temperature_current = temperature_current[deep_mask_current & (temperature_qc == 1)]

    if len(deep_salinity_current) > 0 and len(deep_temperature_current) > 0:
        # Calculate averages for the current profile
        avg_salinity_current = np.mean(deep_salinity_current)
        avg_temperature_current = np.mean(deep_temperature_current)

        if salinity_previous is not None and temperature_previous is not None and pressure_previous is not None:
            # Extract data from the deepest 100 dbar for the previous profile
            deep_mask_previous = pressure_previous >= (np.max(pressure_previous) - 100)

            if np.any(deep_mask_previous):
                # Compute differences
                sal_diff = abs(avg_salinity_current - salinity_previous)
                temp_diff = abs(avg_temperature_current - temperature_previous)

                # Flag the current profile if thresholds are exceeded
                if sal_diff > salinity_threshold:
                    sal_flag[:] = 3  # Probably bad
                if temp_diff > temperature_threshold:
                    temp_flag[:] = 3  # Probably bad

    return sal_flag, temp_flag

def visual_profile_test():
    """
    Test 17: 
        Only required for DMQC
    """
    return

def Frozen_profile_test(pressure_p1,temperature_p1,salinity_p1,pressure_p2,temperature_p2,salinity_p2):
    """
    Test 18:
        Input:
            Pressure from profile 1 (array like)
            Temperature from profile 1 (array like)
            Salinity from profile 1 (array like)
            Pressure from profile 2 (array like)
            Temperature from profile 2 (array like)
            Salinity from profile 2 (array like)
        Output:
            Quality control flag (Intager)
    
    """
    
    import pandas as pd
    
    # convert to dataframs to take advantage of pandas functions
    profile_1 = pd.DataFrame({'Pressure_db':pressure_p1,'Temp_DegC':temperature_p1, 'Salinity':salinity_p1})
    profile_2 = pd.DataFrame({'Pressure_db':pressure_p2,'Temp_DegC':temperature_p2, 'Salinity':salinity_p2})
    
    # Itentify depth bins
    profile_1_bins = range(0, int(profile_1['Pressure_db'].max()) + 50, 50)
    profile_2_bins = range(0, int(profile_2['Pressure_db'].max()) + 50, 50)

    # Sort the data into the bins
    profile_1['Depth_bin'] = pd.cut(profile_1['Pressure_db'],profile_1_bins,right=False)
    profile_2['Depth_bin'] = pd.cut(profile_2['Pressure_db'],profile_2_bins,right=False)
    
    # Average data within the bins
    p1_depth_aves = profile_1.groupby('Depth_bin')[['Temp_DegC', 'Salinity']].mean().reset_index()
    p2_depth_aves = profile_2.groupby('Depth_bin')[['Temp_DegC', 'Salinity']].mean().reset_index()

    # Extract binned temperture and salinity profiles
    p1_temp = p1_depth_aves.Temp_DegC
    p2_temp = p2_depth_aves.Temp_DegC
    p1_salt = p1_depth_aves.Salinity
    p2_salt = p2_depth_aves.Salinity
    
    # Compare profiles
    deltaT = abs(p1_temp - p2_temp)
    deltaS = abs(p1_salt - p2_salt)

    # Calculate QC flag
    if max(deltaT) < 0.3 and min(deltaT) < 0.001 and deltaT.mean() < 0.02 or max(deltaS) < 0.3 and min(deltaS) < 0.001 and deltaS.mean() < 0.004:
        qc_flag = 4
    else:
        qc_flag = 1
    
    return qc_flag

def Deepest_pressure_test(pressure_data, thresholdFile):
    """
    Test 19
    Flags the pressure values based on their corresponding thresholds.
    Returns an array of flags: 1 for good data, 3 for data exceeding the threshold.
    
    Parameters:
        pressure_data (pd.Series or np.ndarray): Array or Series of pressure values.
        thresholdFile (str): Path to the CSV file containing pressure thresholds.
        
    Returns:
        np.ndarray: An array of flags (1 or 3) corresponding to the input pressure data.
    """
    

    # Load the pressure thresholds from the CSV file
    thresholds = pd.read_csv(thresholdFile,sep=';')
    thresholds['Config_ProfilePressure_dbar'] = pd.to_numeric(thresholds['Config_ProfilePressure_dbar'], errors='coerce')
    thresholds['Pressure_Threshold (dbar)'] = pd.to_numeric(thresholds['Pressure_Threshold (dbar)'], errors='coerce')
    thresholds = thresholds.dropna(subset=['Config_ProfilePressure_dbar', 'Pressure_Threshold (dbar)'])

    # Convert pressure data to a NumPy array for vectorized operations
    pressure_data = pd.to_numeric(pressure_data, errors='coerce')
    pressure_data = np.asarray(pressure_data)
    
    # Initialize an array for flags, defaulting to 3 (bad data)
    flags = np.full_like(pressure_data, fill_value=3, dtype=int)
    
    # Apply the thresholds
    for _, row in thresholds.iterrows():
        config = row['Config_ProfilePressure_dbar']
        max_threshold = row['Pressure_Threshold (dbar)']
        flags = np.where((pressure_data <= config) & (pressure_data <= max_threshold), 1, flags)
    
    return flags

    
### MEDD test

from scipy.signal import medfilt

# def relative_2D_distance(median, d):
#     # Placeholder function for relative_2D_distance
#     # Implement the actual logic based on the MATLAB code
#     upper_limit = median + d
#     lower_limit = median - d
#     return upper_limit, lower_limit

def QTRT_spike_check_MEDD(PRES, PARAM, DENS, LAT, d, medd_width, medd_width_ext, medd_width_ext2, medd_width_ext3):
    """
    Detects spikes in oceanographic data using the MEDD algorithm.

    Parameters:
    PRES : numpy.ndarray
        Pressure values.
    PARAM : numpy.ndarray
        Parameter values (e.g., temperature or salinity).
    DENS : numpy.ndarray
        Potential density values.
    LAT : float
        Latitude of the profiles in degrees.
    d : float
        Relative 2-dimensional distance parameter.
    medd_width : int
        Width of the sliding median window.
    medd_width_ext : int
        Extended width for the sliding median.
    medd_width_ext2 : int
        Second extended width for the sliding median.
    medd_width_ext3 : int
        Third extended width for the sliding median.

    Returns:
    numpy.ndarray
        Array indicating the presence of spikes (1 for spike, 0 for no spike).
    """

    # Ensure input arrays are numpy arrays
    PRES = np.asarray(PRES)
    PARAM = np.asarray(PARAM)
    DENS = np.asarray(DENS)

    # Initialize the spike array
    spikes = np.zeros_like(PRES, dtype=int)

    # Compute the sliding median
    median_param = medfilt(PARAM, kernel_size=medd_width)
    

    # Compute the relative 2D distance limits
    upper_limit, lower_limit = relative_2D_distance(median_param, d, LAT)

    # Identify potential spikes
    for i in range(len(PRES)):
        if PARAM[i] > upper_limit[i] or PARAM[i] < lower_limit[i]:
            if DENS[i] > upper_limit[i] or DENS[i] < lower_limit[i]:
                spikes[i] = 1

    return spikes


def relative_2D_distance(xa, ya, d):
    """
    Computes two lines (xbmoins and xbplus) that are at distance d from the
    points defined by (xa, ya). The resulting lines have the same size as ya.

    Parameters:
    xa : numpy.ndarray
        X-coordinates of the original points.
    ya : numpy.ndarray
        Y-coordinates of the original points (can be a single value).
    d : float
        Distance parameter.

    Returns:
    xbmoins : numpy.ndarray
        Y-coordinates of the lower line at distance d.
    xbplus : numpy.ndarray
        Y-coordinates of the upper line at distance d.
    """
    infinite = 9e15
    eps = 1e-10

    # Ensure xa and ya are numpy arrays
    xa = np.asarray(xa).flatten()
    ya = np.asarray(ya).flatten()

    # If ya is a single value, broadcast it to match xa
    if ya.size == 1:
        ya = np.full_like(xa, ya)

    # Ensure at least two points for interpolation
    if xa.size < 2 or ya.size < 2:
        raise ValueError("xa and ya must have at least two points for interpolation.")

    # STEP 1: Curvilinear Interpolation
    # Save original arrays before interpolation
    xa_ori = xa.copy()
    ya_ori = ya.copy()

    x_to_add = []
    y_to_add = []

    # Compute the slope and intercept of the line segments
    dxa = np.diff(xa)
    dya = np.diff(ya)

    # Avoid division by zero for vertical segments
    vertical = np.abs(dxa) < eps
    a = np.where(vertical, infinite, dya / dxa)
    b = np.where(vertical, ya[:-1], ya[:-1] - a * xa[:-1])

    # Compute the distance between consecutive points
    dist = np.sqrt(dxa**2 + dya**2)

    # Interpolate points where distance is greater than d
    for i in range(len(dist)):
        if dist[i] > d:
            num_new_points = int(np.ceil(dist[i] / d)) - 1
            for j in range(1, num_new_points + 1):
                x_new = xa[i] + j * dxa[i] / (num_new_points + 1)
                y_new = ya[i] + j * dya[i] / (num_new_points + 1)
                x_to_add.append(x_new)
                y_to_add.append(y_new)

    # Append new points to the original arrays
    if x_to_add:
        xa = np.hstack((xa, x_to_add))
        ya = np.hstack((ya, y_to_add))

        # Sort the arrays based on xa
        sort_indices = np.argsort(xa)
        xa = xa[sort_indices]
        ya = ya[sort_indices]

    # STEP 2: Compute xbmoins and xbplus
    # Initialize output arrays
    xbmoins = np.zeros_like(ya)
    xbplus = np.zeros_like(ya)

    # Compute the perpendicular distance
    for i in range(len(xa) - 1):
        if vertical[i]:
            xbmoins[i] = ya[i] - d
            xbplus[i] = ya[i] + d
        else:
            perp_slope = -1 / a[i]
            delta_x = d / np.sqrt(1 + perp_slope**2)
            delta_y = perp_slope * delta_x
            xbmoins[i] = ya[i] - delta_y
            xbplus[i] = ya[i] + delta_y

    # Handle the last point
    xbmoins[-1] = xbmoins[-2]
    xbplus[-1] = xbplus[-2]

    return xbmoins, xbplus



def QTRT_spike_check_MEDD_main(PRES, TEMP, PSAL, DENS, LAT):
    """
    Main function for MEDD spike detection algorithm.

    Parameters:
    PRES : numpy.ndarray
        Pressure values.
    TEMP : numpy.ndarray
        Temperature values.
    PSAL : numpy.ndarray
        Salinity values. Set to np.nan if not available.
    DENS : numpy.ndarray
        Potential density values referenced to 0 dbar. Set to np.nan if not available.
    LAT : float
        Latitude of the profiles in degrees.

    Returns:
    SPIKE_T : numpy.ndarray
        Array indicating temperature spikes (1 for spike, 0 for no spike). Set to np.nan if computation could not be made.
    SPIKE_S : numpy.ndarray
        Array indicating salinity spikes (1 for spike, 0 for no spike). Set to np.nan if computation could not be made.
    """

    # Configuration parameters
    d = 0.8  #0.1 Relative 2-dimensional distance parameter
    medd_width = 5  # Width of the sliding median window
    medd_width_ext = 7  # Extended width for the sliding median
    medd_width_ext2 = 9  # Second extended width for the sliding median
    medd_width_ext3 = 11  # Third extended width for the sliding median

    # Initialize output arrays
    SPIKE_T = np.full_like(PRES, np.nan)
    SPIKE_S = np.full_like(PRES, np.nan)

    # Detect spikes in temperature
    if not np.all(np.isnan(TEMP)):
        SPIKE_T = QTRT_spike_check_MEDD(PRES, TEMP, DENS, LAT, d, medd_width, medd_width_ext, medd_width_ext2, medd_width_ext3)

    # Detect spikes in salinity
    if not np.all(np.isnan(PSAL)):
        SPIKE_S = QTRT_spike_check_MEDD(PRES, PSAL, DENS, LAT, d, medd_width, medd_width_ext, medd_width_ext2, medd_width_ext3)

    return SPIKE_T, SPIKE_S


def sort_back(data, ind, axis=0):
    """
    Restores data to its original order after sorting.

    Parameters:
    data : numpy.ndarray or list
        The sorted data array or list.
    ind : numpy.ndarray
        The indices obtained from the sorting process.
    axis : int, optional
        The dimension along which the data was sorted (default is 0).

    Returns:
    out : numpy.ndarray or list
        The data restored to its original order.
    """
    data = np.asarray(data)
    ind = np.asarray(ind)

    if data.shape != ind.shape:
        raise ValueError('Different size of indexes and input data')

    # Initialize an array to hold the original order
    out = np.empty_like(data)

    # Generate an index grid
    idx = np.ogrid[tuple(map(slice, data.shape))]

    # Place the sorted data back to its original positions
    idx[axis] = ind
    out[tuple(idx)] = data

    return out
