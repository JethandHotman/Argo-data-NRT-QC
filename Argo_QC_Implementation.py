# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:57:25 2025

@author: Jethan
"""

# Script to QC Argo data

import argo_nrt_qc as aqc
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import gsw

# Order of tests:
    # 1 - Platform Identification test (NA)
    # 2 - Impossible Date test
    # 3 - Impossible location test
    # 4 - Position on land test
    # 5 - Impossible speed test
    # 6 - Grey lest test (NA)
    # 7 - Deepest Pressure test
    # 8 - Global range test
    # 9 - Regional range test (NA)
    # 10 - Pressure increasing test
    # 11 - Spike test (NA)
    # 12 - MEDD test
    # 13 - Digit rollover test
    # 14 - Stuck value test
    # 15 - Density Inversion test
    # 16 - Gross salinity or temperature sensor drift test
    # 17 - Frozen profile test


data_path = r"C:\Users\Jethan\Documents\SeaTrec Float\Data\*_Converted.csv"

data_file_list_1 = glob(data_path)

# Remove profiles 0 and 174 from the list, they contain no data
a = data_file_list_1[1:174]
b = data_file_list_1[175:]

data_file_list = a+b 


for tt in range(len(data_file_list)-1):
    
    print('Prcessing file:')
    print(data_file_list[tt+1])
    print('  ')
    
    data = pd.read_csv(data_file_list[tt+1],
                       usecols=['Timestamp', 
                                'Pressure_db',
                                'Temp_DegC',
                                'Salinity',
                                'Longitude',
                                'Latitude'],
                       parse_dates=True)
    
    # Test 2
    
    t1 = pd.to_datetime(data.Timestamp[0])
    t2 = aqc.Impossible_date_test(t1)
    test2 = np.repeat(t2, len(data.Timestamp))
    
    
    # Test 3
    lon = data.Longitude[0]
    lat = data.Latitude[0]
    
    t3 = aqc.Impossible_Location_test(lon, lat)
    test3 = np.repeat(t3, len(data.Longitude))
    
    # Test 4
    t4 = aqc.Position_on_Land_test(lon, lat, r"C:\Users\Jethan\Documents\SeaTrec Float\Argo_QC\GEBCO_21_Nov_2024\gebco_2024_n0.0_s-70.0_w0.0_e120.0.nc")
    test4 = np.repeat(t4,len(data.Longitude))
    
    # Test 5
    
    data2 = pd.read_csv(data_file_list[tt],
                       usecols=['Timestamp', 
                                'Pressure_db',
                                'Temp_DegC',
                                'Salinity',
                                'Longitude',
                                'Latitude'],
                       parse_dates=True)
    
    lon0 = data2.Longitude[0]
    lat0 = data2.Latitude[0]
    t0 = pd.to_datetime(data2.Timestamp[0])
    t5 = aqc.Impossible_Speed_test(lon, lat, lon0, lat0, str(t1), str(t0))
    
    test5 = np.repeat(t5,len(data.Longitude))
    
    #del t1, t2, t3, t4, t5
    
    # Test 7
    
    pres = data.Pressure_db
    test7 = aqc.Deepest_pressure_test(pres,r"C:\Users\Jethan\Documents\SeaTrec Float\Argo_QC\Test19_pressureThresholdValues.csv")
    
    
    # Test 8
    temp = data.Temp_DegC
    salt = data.Salinity
    test8_pres,test8_temp,test8_salt = aqc.Global_Range_test(-pres,temp,salt)
    
    # Test 10
    
    test10 = aqc.Pressure_Increasing_test(-pres)
    t10 = test10[-1]
    test10.append(t10) # copy last flag to ensure array length is the same
    
    
    # Test 12
    dens = gsw.density.rho(salt,temp,pres)
    # Apply the QTRT spike check main function
    test12_tflag, test12_sflag = aqc.QTRT_spike_check_MEDD_main(
        -pres, temp, salt, dens, lat)
    
    # Use the sort_back function to restore original order
    sorted_indices = np.argsort(pres)  # Sort by pressure
    original_order_SPIKE_T = aqc.sort_back(test12_tflag, sorted_indices)
    original_order_SPIKE_S = aqc.sort_back(test12_sflag, sorted_indices)
    
    test12_temp_flag = []
    test12_salt_flag = []
    for qq in range(len(original_order_SPIKE_T)):
        if original_order_SPIKE_T[qq] == 1:
            test12_temp_flag.append(4)
        else: 
            test12_temp_flag.append(1)
            
        if original_order_SPIKE_S[qq] == 1:
            test12_salt_flag.append(4)
        else:
            test12_salt_flag.append(1)
        
    
    # Test 13
    
    test13_tflag,test13_sflag = aqc.Digit_rollover_test(temp, salt, pres)
    
    # Test 14
    
    test14_tflag,test14_sflag = aqc.Stuck_value_test(temp, salt)
    
    # Test 15
    
    test15_tflag,test15_sflag = aqc.Density_inversion_test(temp, salt, pres, lon, lat)
    test15_tflag[-1] = test15_tflag[-2]
    test15_sflag[-1] = test15_sflag[-2]
    
    
    # Combine flags thus far for next tests
    
    #del t1, t2, t3, t4, t5, t10 # Delete previous single value test results to reuse variable names in loop
    
    combined_flags_temp = []
    combined_flags_salt = []
    combined_flags_pres = []
    
    for ii in range(len(test2)):
        
        f2 = int(test2[ii])
        f3 = int(test3[ii])
        f4 = int(test4[ii])
        f5 = int(test5[ii])
        f7 = int(test7[ii])
        f10 = int(test10[ii])
        
        f8_temp = int(test8_temp[ii])
        f8_salt = int(test8_salt[ii])
        f8_pres = int(test8_pres[ii])
        
        f12_temp = int(test12_temp_flag[ii])
        f12_salt = int(test12_salt_flag[ii])
        
        f13_temp = int(test13_tflag[ii])
        f13_salt = int(test13_sflag[ii])
        
        f14_temp = int(test14_tflag[ii])
        f14_salt = int(test14_sflag[ii])
        
        f15_temp = int(test15_tflag[ii])
        f15_salt = int(test15_sflag[ii])
        
        # Combine all flags into a list for temperature and salinity
        flags_temp = [f2, f3, f4, f5, f7, f10, f8_temp, f12_temp, f13_temp, f14_temp, f15_temp]
        flags_salt = [f2, f3, f4, f5, f7, f10, f8_salt, f12_salt, f13_salt, f14_salt, f15_salt]
        flags_pres = [f2, f3, f4, f5, f7, f10, f8_pres]

        # Check if any flag equals 4
        if any(flag == 4 for flag in flags_temp):
            combined_flags_temp.append(4)
        else:
            combined_flags_temp.append(1)
            
        if any(flag == 4 for flag in flags_salt):
            combined_flags_salt.append(4)
        else:
            combined_flags_salt.append(1)
            
        if any(flag == 4 for flag in flags_pres):
            combined_flags_pres.append(4)
        else: 
            combined_flags_pres.append(1)

        
        
    # Test 16
    
    temp2 = data2['Temp_DegC']
    salt2 = data2['Salinity']
    pres2 = data2['Pressure_db']
    
    test16_tflag,test16_sflag = aqc.Gross_sensor_drift_test(salt, salt2, 
                                                            temp, temp2, 
                                                            pres, pres2,
                                                            combined_flags_salt,
                                                            combined_flags_temp) # Need to edit function to work with profiles in seperate csv files
    
    # Test 17
    
    test17a = aqc.Frozen_profile_test(pres, temp, salt, pres2, temp2, salt2)
    test17 = np.repeat(test17a,len(data.Longitude))
    
    final_tflag = []
    final_sflag = []
    final_pflag = []
    temp_qced = []
    salt_qced = []
    pres_qced = []
    
    for ii in range(len(test2)):
        
        cf_t = int(combined_flags_temp[ii])
        cf_s = int(combined_flags_salt[ii])
        cf_p = int(combined_flags_pres[ii])
        
        t16_temp = int(test16_tflag[ii])
        t16_salt = int(test16_sflag[ii])
        
        t17 = int(test17[ii])
        
        new_t_flags = [cf_t, t16_temp, t17]
        new_s_flag = [cf_s, t16_salt, t17]
        new_p_flag = [cf_p, t17]
        
        if any(flag == 4 for flag in new_t_flags):
            final_tflag.append(4)
            temp_qced.append(np.nan)
        else:
            final_tflag.append(1)
            temp_qced.append(temp[ii])
            
        if any(flag == 4 for flag in new_s_flag):
            final_sflag.append(4)
            salt_qced.append(np.nan)
        else:
            final_sflag.append(1)
            salt_qced.append(salt[ii])
            
        if any(flag == 4 for flag in new_p_flag):
            final_pflag.append(4)
            pres_qced.append(np.nan)
        else:
            final_pflag.append(1)
            pres_qced.append(pres[ii])
            
    pn = data_file_list[tt+1][-17:-14]
    new_data_dict = {'Profile': pn,
                     'Timestamp':data.Timestamp,
                     'Pressure_db_raw':-pres,
                     'Temp_DegC_raw': temp,
                     'Salinity_raw': salt,
                     'Pressure_db_QC': pres_qced,
                     'Temp_DegC_QC':temp_qced,
                     'Salinity_QC':salt_qced,
                     'Longitude': lon,
                     'Latitude': lat,
                     'Pressure_flag':final_pflag,
                     'Temp_flag': final_tflag,
                     'Salinity_flag':final_sflag}
    
    qcd_data = pd.DataFrame(new_data_dict)
    
    new_fname = r'C:\Users\Jethan\Documents\SeaTrec Float\Argo_QC\QCed_data\\' + data_file_list[tt+1][-22:-4] + '_QCed.csv'
    
    qcd_data.to_csv(new_fname)
    
    flag_dict = {'Test 2': test2,
                 'Test 3': test3,
                 'Test 4': test4,
                 'Test 5': test5,
                 'Test 7': test7,
                 #'Tets 8 Pres': test8_pres,
                 #'Test 8 Temp': test8_temp,
                 'Test 8 Salt': test8_salt,
                 'Test 10': test10,
                 #'Test 12 Temp': test12_temp_flag,
                 'Test 12 Salt': test12_salt_flag,
                 #'Test 13 Temp': test13_tflag,
                 'Test 13 Salt': test13_sflag,
                 #'Test 14 Temp': test14_tflag,
                 'Test 14 Salt': test14_sflag,
                 #'Test 15 Temp': test15_tflag,
                 'Test 15 Salt': test15_sflag,
                 #'Combined flag Pres': combined_flags_pres,
                 #'Combined flag Temp': combined_flags_temp,
                 'Combined flag Salt': combined_flags_salt,
                 #'Test 16 Temp': test16_tflag,
                 'Test 16 Salt': test16_sflag,
                 'Test 17': test17,
                 #'Final flag Pres': final_pflag,
                 #'Final flag Temp': final_tflag,
                 'Final flag Salt': final_sflag
        }

    temp_flags = pd.DataFrame(flag_dict)
    flag_fname = r'C:\Users\Jethan\Documents\SeaTrec Float\Argo_QC\QCed_data\QCFlags\\' + data_file_list[tt+1][-22:-4] + '_flags.csv'
    temp_flags.to_csv(flag_fname)



# lon1 = 23.6651
# lat1 = -37.2846
# lon2 = 23.5015
# lat2 = -37.0396
# time1 = pd.to_datetime('2024/06/27 17:28')
# time2 = pd.to_datetime('2024/06/28 16:58')

# speed = aqc.Impossible_Speed_test(lon1, lat1, lon2, lat2, time1, time2)