# -*- coding: utf-8 -*-
"""
generate air basin-scale emission rate and lifetime, calling functions in rrnes.py

Created on Tue Nov 24 13:41:45 2020

@author: kangsun
"""
#%%# use this section to generate a control txt file
'''
import yaml
control = {}

control['rrnes directory'] = r'C:\research\RRNES'
control['save path'] = r'C:\research\RRNES\test_level4.npz'
control['which air basin'] = 'po'
control['which molecule'] = 'NO2'

control['universal f'] = 1
control['min wind speed'] = 3
control['max wind speed'] = 8

control['sensor1'] = 'OMI'
control['sensor1 directory'] = r'C:\research\RRNES\omno2_era5'
control['sensor1 start year'] = 2004
control['sensor1 start month'] = 10
control['sensor1 end year'] = 2020
control['sensor1 end month'] = 10
control['sensor1 bound Q'] = 70
control['sensor1 bound tau'] = 5
control['sensor1 bound Q3'] = 70
control['sensor1 bound tau3'] = 1.5

control['if sensor2'] = True
control['sensor2'] = 'TROPOMI'
control['sensor2 directory'] = r'C:\research\RRNES\s5pno2_era5'
control['sensor2 start year'] = 2018
control['sensor2 start month'] = 5
control['sensor2 end year'] = 2020
control['sensor2 end month'] = 10
control['sensor2 bound Q'] = 70
control['sensor2 bound tau'] = 5
control['sensor2 bound Q3'] = 70
control['sensor2 bound tau3'] = 1.5

control['if bootstrap'] = False
control['nB'] = 100

control['tau climatology corr month'] = 1.5
control['Q climatology corr month'] = 1.5
control['monthly corr month'] = 1.5
control['monthly corr year'] = 1.5
control['jpl annual emission path'] = r'C:\research\RRNES\po_nox_monthly_anth.csv'
control['sensor1 Q all prior'] = 300
control['sensor2 Q all prior'] = 200
control['sensor1 climatology ridgeLambda'] = 1.33e-9
control['sensor2 climatology ridgeLambda'] = 3.16e-10
control['sensor1 all month ridgeLambda'] = 6e-9
control['sensor2 all month ridgeLambda'] = 2e-9

control['if test ridgeLambda'] = True
with open('rrnes_level4_control.txt', 'w') as stream:
    yaml.dump(control, stream,sort_keys=False)
'''
#%%
import sys
#control_path = r'C:\research\RRNES\rrnes_level4_control.txt'
if __name__ == "__main__":
    control_path = str(sys.argv[1])
import yaml
with open(control_path,'r') as stream:
    control = yaml.full_load(stream)
import numpy as np
import os
import logging
if 'if verbose' not in control.keys(): control['if verbose']=False
if control['if verbose']:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(control['rrnes directory'])
from rrnes import RRNES
universal_f = control['universal f']
universal_wsRange = (control['min wind speed'],control['max wind speed'])
if control['which molecule'] == 'NO2':
    nox_per_no2 =1.32
else:
    nox_per_no2 = 1.
#%% load sensor1
startYear = control['sensor1 start year']
startMonth = control['sensor1 start month']

endYear = control['sensor1 end year']
endMonth = control['sensor1 end month']

dateArray = np.array([])
for year in range(startYear,endYear+1):
    for month in range(1,13):
        if year == startYear and month < startMonth:
            continue
        if year == endYear and month > endMonth:
            continue
        dateArray = np.append(dateArray,dt.date(year,month,1))

r = RRNES(whichBasin=control['which air basin'],
          whichSatellite=control['sensor1'],
          dateArray=dateArray,
          dataDir=control['sensor1 directory'],
          moleculeList=[control['which molecule']])
monthlyDictArray = r.F_load_monthly_h5()
#% sensor1 monthly climatology fit
Q_month = np.full((12),np.nan,dtype=np.float)
tau_month = np.full((12),np.nan,dtype=np.float)
Q_e_month = np.full((12),np.nan,dtype=np.float)
tau_e_month = np.full((12),np.nan,dtype=np.float)
for month in range(1,13):
    monthInterval = np.array([month-1,month,month+1])
    monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
    monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
    mask = np.array([(d.month in monthInterval) for d in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
    Q_month[month-1] = ime.popt1[0]
    tau_month[month-1] = ime.popt1[1]
    Q_e_month[month-1] = np.sqrt(ime.pcov1[0,0])
    tau_e_month[month-1] = np.sqrt(ime.pcov1[1,1])
#% sensor1 monthly fit bootstrap
if control['if bootstrap']:
    nB = control['nB']
    all_month_b = np.full((12,nB),np.nan)
    Q_month_b = np.full((12,nB),np.nan)
    tau_month_b = np.full((12,nB),np.nan)
    rho_month_b = np.full((12,nB),np.nan)
    rms_month_b = np.full((12,nB),np.nan)
    btMonthNumber = np.zeros((12))
    for month in range(1,13):
        monthInterval = np.array([month-1,month,month+1])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        mask = np.array([(d.month in monthInterval) for d in dateArray])
        chooseIndex = np.where(mask==True)[0]
        nChoose = len(chooseIndex)
        btMonthNumber[month-1] = nChoose
        for iB in range(nB):
            randomChooseIndex = np.random.choice(chooseIndex,nChoose)
            randomMonth = np.array([d.month for d in dateArray[randomChooseIndex]])
            if month == 1:
                randomMonth[randomMonth==12] = 0
            if month == 12:
                randomMonth[randomMonth==1] = 13
            mergedDict = r.F_merge_monthly_data(monthlyDictArray[randomChooseIndex])
            ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
            all_month_b[month-1,iB] = np.mean(randomMonth)
            Q_month_b[month-1,iB] = ime.popt1[0]
            tau_month_b[month-1,iB] = ime.popt1[1]
            rho_month_b[month-1,iB] = ime.pcov1[0,1]/np.sqrt(ime.pcov1[0,0])/np.sqrt(ime.pcov1[1,1])
            rms_month_b[month-1,iB] = np.nanstd(ime.residual1)
    mask = (rho_month_b <-1) | (rho_month_b >1)
    Q_month_b[mask] = np.nan
    tau_month_b[mask] = np.nan
#% sensor1 black month list
dQ = np.full((len(dateArray)),np.nan)
dtau = np.full((len(dateArray)),np.nan)
for (i,d) in enumerate(dateArray):
    month = d.month
    monthInterval = np.array([month])
    monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
    monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
    mask = np.array([(m.month in monthInterval) for m in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])
    
    mask = np.array([(m.month in monthInterval) and ((m.year!=d.year) or (m.month!=d.month)) for m in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime2 = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])
    dQ[i] = ime2.popt1[0]-ime.popt1[0]
    dtau[i] = ime2.popt1[1]-ime.popt1[1]

dQ3 = np.full((len(dateArray)),np.nan)
dtau3 = np.full((len(dateArray)),np.nan)
for (i,d) in enumerate(dateArray):
    month = d.month
    monthInterval = np.array([month-1,month,month+1])
    monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
    monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
    mask = np.array([(m.month in monthInterval)  for m in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])
    
    mask = np.array([(m.month in monthInterval) and ((m.year!=d.year) or (m.month!=d.month)) for m in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime2 = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])
    dQ3[i] = ime2.popt1[0]-ime.popt1[0]
    dtau3[i] = ime2.popt1[1]-ime.popt1[1]
bound_tau = control['sensor1 bound tau']
bound_Q = control['sensor1 bound Q']/nox_per_no2
bound_tau3 = control['sensor1 bound tau3']
bound_Q3 = control['sensor1 bound Q3']/nox_per_no2
mask_omiblack = (np.abs(dtau) > bound_tau*3600) | (np.abs(dQ) > bound_Q) | (np.abs(dtau3) > bound_tau3*3600) | (np.abs(dQ3) > bound_Q3)
blackListDateArray = dateArray[mask_omiblack]
#% sensor1 monthly climatology clean fit
Q_month_clean = np.full((12),np.nan,dtype=np.float)
tau_month_clean = np.full((12),np.nan,dtype=np.float)
for month in range(1,13):
    monthInterval = np.array([month-1,month,month+1])
    monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
    monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
    mask = np.array([(d.month in monthInterval) and (d not in blackListDateArray) for d in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
    Q_month_clean[month-1] = ime.popt1[0]
    tau_month_clean[month-1] = ime.popt1[1]
#% sensor1 monthly fit clean bootstrap
if control['if bootstrap']:
    nB = control['nB']
    all_month_clean_b = np.full((12,nB),np.nan)
    Q_month_clean_b = np.full((12,nB),np.nan)
    tau_month_clean_b = np.full((12,nB),np.nan)
    rho_month_clean_b = np.full((12,nB),np.nan)
    btMonthNumber_clean = np.zeros((12))
    whiteDateArray = dateArray[np.array([(d not in blackListDateArray) for d in dateArray])]
    whiteMonthlyDictArray = monthlyDictArray[np.array([(d not in blackListDateArray) for d in dateArray])]
    for month in range(1,13):
        monthInterval = np.array([month-1,month,month+1])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        
        mask = np.array([(d.month in monthInterval) for d in whiteDateArray])
        chooseIndex = np.where(mask==True)[0]
        nChoose = len(chooseIndex)
        btMonthNumber_clean[month-1] = nChoose
        for iB in range(nB):
            randomChooseIndex = np.random.choice(chooseIndex,nChoose)
            randomMonth = np.array([d.month for d in whiteDateArray[randomChooseIndex]])
            if month == 1:
                randomMonth[randomMonth==12] = 0
            if month == 12:
                randomMonth[randomMonth==1] = 13
            mergedDict = r.F_merge_monthly_data(whiteMonthlyDictArray[randomChooseIndex])
            ime = r.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
            all_month_clean_b[month-1,iB] = np.mean(randomMonth)
            Q_month_clean_b[month-1,iB] = ime.popt1[0]
            tau_month_clean_b[month-1,iB] = ime.popt1[1]
            rho_month_clean_b[month-1,iB] = ime.pcov1[0,1]/np.sqrt(ime.pcov1[0,0])/np.sqrt(ime.pcov1[1,1])
    mask = (rho_month_clean_b <-1) | (rho_month_clean_b >1)
    Q_month_clean_b[mask] = np.nan
    tau_month_clean_b[mask] = np.nan
#%% sensor2 - tropomi for now
if control['if sensor2']:
    startYear = control['sensor2 start year']
    startMonth = control['sensor2 start month']
    
    endYear = control['sensor2 end year']
    endMonth = control['sensor2 end month']
    
    dateArray_s5pno2 = np.array([])
    for year in range(startYear,endYear+1):
        for month in range(1,13):
            if year == startYear and month < startMonth:
                continue
            if year == endYear and month > endMonth:
                continue
            dateArray_s5pno2 = np.append(dateArray_s5pno2,dt.date(year,month,1))
    
    r_s5pno2 = RRNES(whichBasin=control['which air basin'],
              whichSatellite=control['sensor2'],
              dateArray=dateArray_s5pno2,
              dataDir=control['sensor2 directory'],
              moleculeList=[control['which molecule']])
    monthlyDictArray_s5pno2 = r_s5pno2.F_load_monthly_h5()
    
    #% monthly climatology tropomi fit
    Q_month_s5p = np.full((12),np.nan,dtype=np.float)
    tau_month_s5p = np.full((12),np.nan,dtype=np.float)
    Q_e_month_s5p = np.full((12),np.nan,dtype=np.float)
    tau_e_month_s5p = np.full((12),np.nan,dtype=np.float)
    for month in range(1,13):
        monthInterval = np.array([month-1,month,month+1])
        #    monthInterval = np.array([month])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        mask = np.array([(d.month in monthInterval) for d in dateArray_s5pno2])
        mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[mask])
        ime = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
        Q_month_s5p[month-1] = ime.popt1[0]
        tau_month_s5p[month-1] = ime.popt1[1]
        Q_e_month_s5p[month-1] = np.sqrt(ime.pcov1[0,0])
        tau_e_month_s5p[month-1] = np.sqrt(ime.pcov1[1,1])
    
    #% tropomi monthly fit bootstrap
    if control['if bootstrap']:
        nB = control['nB']
        all_month_b_s5p = np.full((12,nB),np.nan)
        Q_month_b_s5p = np.full((12,nB),np.nan)
        tau_month_b_s5p = np.full((12,nB),np.nan)
        rho_month_b_s5p = np.full((12,nB),np.nan)
        rms_month_b_s5p = np.full((12,nB),np.nan)
        btMonthNumber_s5p = np.zeros((12))
        for month in range(1,13):
            monthInterval = np.array([month-1,month,month+1])
            monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
            monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
            mask = np.array([(d.month in monthInterval) for d in dateArray_s5pno2])
            chooseIndex = np.where(mask==True)[0]
            nChoose = len(chooseIndex)
            btMonthNumber_s5p[month-1] = nChoose
            for iB in range(nB):
                randomChooseIndex = np.random.choice(chooseIndex,nChoose)
                randomMonth = np.array([d.month for d in dateArray_s5pno2[randomChooseIndex]])
                if month == 1:
                    randomMonth[randomMonth==12] = 0
                if month == 12:
                    randomMonth[randomMonth==1] = 13
                mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[randomChooseIndex])
                ime = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
                all_month_b_s5p[month-1,iB] = np.mean(randomMonth)
                Q_month_b_s5p[month-1,iB] = ime.popt1[0]
                tau_month_b_s5p[month-1,iB] = ime.popt1[1]
                rho_month_b_s5p[month-1,iB] = ime.pcov1[0,1]/np.sqrt(ime.pcov1[0,0])/np.sqrt(ime.pcov1[1,1])
                rms_month_b_s5p[month-1,iB] = np.nanstd(ime.residual1)
    #% sensor2 black month list
    dQ3_s5p = np.full((len(dateArray_s5pno2)),np.nan)
    dtau3_s5p = np.full((len(dateArray_s5pno2)),np.nan)
    for (i,d) in enumerate(dateArray_s5pno2):
        month = d.month
        monthInterval = np.array([month-1,month,month+1])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        mask = np.array([(m.month in monthInterval) and (m.year < 2021) for m in dateArray_s5pno2])
        mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[mask])
        ime = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])    
        mask = np.array([(m.month in monthInterval) and (m.year < 2021) and ((m.year!=d.year) or (m.month!=d.month)) for m in dateArray_s5pno2])
        mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[mask])
        ime2 = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=2e10,ime_f=[universal_f])
        dQ3_s5p[i] = ime2.popt1[0]-ime.popt1[0]
        dtau3_s5p[i] = ime2.popt1[1]-ime.popt1[1]
    bound_tau3_s5p = control['sensor2 bound tau3']
    bound_Q3_s5p = control['sensor2 bound Q3']/nox_per_no2
    mask_s5pblack = (np.abs(dtau3_s5p) > bound_tau3_s5p*3600) | (np.abs(dQ3_s5p) > bound_Q3_s5p)
    blackListDateArray_s5p = dateArray_s5pno2[mask_s5pblack]
    #% sensor2 monthly climatology clean fit
    Q_month_clean_s5p = np.full((12),np.nan,dtype=np.float)
    tau_month_clean_s5p = np.full((12),np.nan,dtype=np.float)
    for month in range(1,13):
        monthInterval = np.array([month-1,month,month+1])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        mask = np.array([(d.month in monthInterval) and (d not in blackListDateArray_s5p) for d in dateArray_s5pno2])
        mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[mask])
        if not mergedDict:continue
        ime = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
        Q_month_clean_s5p[month-1] = ime.popt1[0]
        tau_month_clean_s5p[month-1] = ime.popt1[1]
    #% sensor2 monthly clean fit bootstrap
    if control['if bootstrap']:
        nB = control['nB']
        all_month_b_clean_s5p = np.full((12,nB),np.nan)
        Q_month_b_clean_s5p = np.full((12,nB),np.nan)
        tau_month_b_clean_s5p = np.full((12,nB),np.nan)
        rho_month_b_clean_s5p = np.full((12,nB),np.nan)
        btMonthNumber_clean_s5p = np.zeros((12))
        whiteDateArray_s5p = dateArray_s5pno2[np.array([(d not in blackListDateArray_s5p) for d in dateArray_s5pno2])]
        whiteMonthlyDictArray_s5p = monthlyDictArray_s5pno2[np.array([(d not in blackListDateArray_s5p) for d in dateArray_s5pno2])]
        for month in range(1,13):
            monthInterval = np.array([month-1,month,month+1])
            monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
            monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
            mask = np.array([(d.month in monthInterval) for d in whiteDateArray_s5p])
            chooseIndex = np.where(mask==True)[0]
            nChoose = len(chooseIndex)
            btMonthNumber_clean_s5p[month-1] = nChoose
            for iB in range(nB):
                randomChooseIndex = np.random.choice(chooseIndex,nChoose)
                randomMonth = np.array([d.month for d in whiteDateArray_s5p[randomChooseIndex]])
                if month == 1:
                    randomMonth[randomMonth==12] = 0
                if month == 12:
                    randomMonth[randomMonth==1] = 13
                mergedDict = r_s5pno2.F_merge_monthly_data(whiteMonthlyDictArray_s5p[randomChooseIndex])
                if not mergedDict:continue
                ime = r_s5pno2.F_fit_ime_C(mergedDict,wsRange=universal_wsRange,softResidualThreshold=0.2,ime_f=[universal_f])
                all_month_b_clean_s5p[month-1,iB] = np.mean(randomMonth)
                Q_month_b_clean_s5p[month-1,iB] = ime.popt1[0]
                tau_month_b_clean_s5p[month-1,iB] = ime.popt1[1]
                rho_month_b_clean_s5p[month-1,iB] = ime.pcov1[0,1]/np.sqrt(ime.pcov1[0,0])/np.sqrt(ime.pcov1[1,1])
#%% average clean tau climatology if there are two sensors
from scipy.signal import savgol_filter
nRepeat = 3
if control['if sensor2']:
    tau_month_both = np.nanmean(np.vstack((tau_month_clean,tau_month_clean_s5p)),axis=0)
else:
    tau_month_both = tau_month_clean
smooth_repeat_tau_month = savgol_filter(np.tile(tau_month_both,nRepeat),window_length=3,polyorder=1)
tau_month_prior_both = smooth_repeat_tau_month[24:36]
#%% omi monthly climatology oe
omi_lambda = control['sensor1 climatology ridgeLambda']
month_array = np.arange(1,13)
nmonth = len(month_array)
def F_month_corr(m1,m2,dm):
    # error correlation between two months
    return np.exp(-np.min([np.abs(m1-m2),12-np.abs(m1-m2)])/dm)
Q_ap = 200*np.ones((nmonth))
tau_ap = tau_month_prior_both#10*np.ones((nmonth))*3600

Q_ap_e = 200*np.ones((nmonth))
tau_ap_e = tau_ap*1.5

correlation_month_tau = control['tau climatology corr month']
correlation_month_Q = control['Q climatology corr month']
Sa_Q = np.diag(Q_ap_e**2)
Sa_tau = np.diag(tau_ap_e**2)
for i in range(nmonth):
    for j in range(nmonth):
        Sa_Q[i,j] = Q_ap_e[i]*Q_ap_e[j]*F_month_corr(i+1,j+1,correlation_month_Q)
        Sa_tau[i,j] = tau_ap_e[i]*tau_ap_e[j]*F_month_corr(i+1,j+1,correlation_month_tau)
Sa = np.zeros((nmonth*2,nmonth*2))
Sa[0:nmonth,0:nmonth] = Sa_Q
Sa[nmonth:,nmonth:] = Sa_tau

Sa_cli = Sa.copy()
Q_ap_cli = Q_ap.copy()
tau_ap_cli = tau_ap.copy()
#plt.clf();plt.imshow(np.log(Sa))

md12 = np.array([])
for month in range(1,13):
    monthInterval = np.array([month-1,month,month+1])
#    monthInterval = np.array([month])
    monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
    monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
    mask = np.array([(d.month in monthInterval) and (d not in blackListDateArray) for d in dateArray])
    mergedDict = r.F_merge_monthly_data(monthlyDictArray[mask])
    md12 = np.append(md12,mergedDict)
ime = r.F_multimonth_OE(monthlyDictArray=md12,Q_ap=Q_ap,tau_ap=tau_ap,Sa=Sa,universal_f=universal_f,ridgeLambda=omi_lambda,wsRange=universal_wsRange)
tau_month_post = ime.tau_post
Q_month_post = ime.Q_post
avk_Q_month_post = np.diag(ime.avk)[0:len(Q_month_post)]
avk_tau_month_post = np.diag(ime.avk)[len(Q_month_post):]
#% sensor1 monthly climatology oe bootstrap
if control['if bootstrap']:
    nB = control['nB']
    all_month_post_b = np.full((12,nB),np.nan)
    Q_month_post_b = np.full((12,nB),np.nan)
    tau_month_post_b = np.full((12,nB),np.nan)
    whiteDateArray = dateArray[np.array([(d not in blackListDateArray) for d in dateArray])]
    whiteMonthlyDictArray = monthlyDictArray[np.array([(d not in blackListDateArray) for d in dateArray])]
    for iB in range(nB):
        md12_b = np.array([])
        for month in range(1,13):
            monthInterval = np.array([month-1,month,month+1])
            #    monthInterval = np.array([month])
            monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
            monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
            mask = np.array([(d.month in monthInterval) for d in whiteDateArray])
            chooseIndex = np.where(mask==True)[0]
            nChoose = len(chooseIndex)
            randomChooseIndex = np.random.choice(chooseIndex,nChoose)
            randomMonth = np.array([d.month for d in whiteDateArray[randomChooseIndex]]) 
            if month == 1:
                randomMonth[randomMonth==12] = 0
            if month == 12:
                randomMonth[randomMonth==1] = 13
            all_month_post_b[month-1,iB] = np.mean(randomMonth)
            mergedDict = r.F_merge_monthly_data(whiteMonthlyDictArray[randomChooseIndex])
            md12_b = np.append(md12_b,mergedDict)
        ime_omi_cli_bt = r.F_multimonth_OE(monthlyDictArray=md12_b,Q_ap=Q_ap,tau_ap=tau_ap,Sa=Sa,universal_f=universal_f,ridgeLambda=omi_lambda,wsRange=universal_wsRange)
        tau_month_post_b[:,iB] = ime_omi_cli_bt.tau_post
        Q_month_post_b[:,iB] = ime_omi_cli_bt.Q_post
#% sensor2 climatology oe
if control['if sensor2']:
    s5p_lambda = control['sensor2 climatology ridgeLambda']
    month_array = np.arange(1,13)
    nmonth = len(month_array)
    Q_ap = 200*np.ones((nmonth))
    tau_ap = tau_month_prior_both#10*np.ones((nmonth))*3600
    
    Q_ap_e = 200*np.ones((nmonth))
    tau_ap_e = tau_ap*1.5
    
    correlation_month_tau = control['tau climatology corr month']
    correlation_month_Q = control['Q climatology corr month']
    Sa_Q = np.diag(Q_ap_e**2)
    Sa_tau = np.diag(tau_ap_e**2)
    for i in range(nmonth):
        for j in range(nmonth):
            Sa_Q[i,j] = Q_ap_e[i]*Q_ap_e[j]*F_month_corr(i+1,j+1,correlation_month_Q)
            Sa_tau[i,j] = tau_ap_e[i]*tau_ap_e[j]*F_month_corr(i+1,j+1,correlation_month_tau)
    Sa = np.zeros((nmonth*2,nmonth*2))
    Sa[0:nmonth,0:nmonth] = Sa_Q
    Sa[nmonth:,nmonth:] = Sa_tau
    Sa_cli_s5p = Sa.copy()
    Q_ap_cli_s5p = Q_ap.copy()
    tau_ap_cli_s5p = tau_ap.copy()
    md12_s5p = np.array([])
    for month in range(1,13):
        monthInterval = np.array([month-1,month,month+1])
        #    monthInterval = np.array([month])
        monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
        monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
        mask = np.array([(d.month in monthInterval) and (d not in blackListDateArray_s5p) for d in dateArray_s5pno2])
        mergedDict = r_s5pno2.F_merge_monthly_data(monthlyDictArray_s5pno2[mask])
        md12_s5p = np.append(md12_s5p,mergedDict)
    ime_s5p = r_s5pno2.F_multimonth_OE(monthlyDictArray=md12_s5p,Q_ap=Q_ap,tau_ap=tau_ap,Sa=Sa,universal_f=universal_f,ridgeLambda=s5p_lambda,wsRange=universal_wsRange)
    tau_month_post_s5p = ime_s5p.tau_post
    Q_month_post_s5p = ime_s5p.Q_post
    avk_Q_month_post_s5p = np.diag(ime.avk)[0:len(Q_month_post_s5p)]
    avk_tau_month_post_s5p = np.diag(ime.avk)[len(Q_month_post_s5p):]
    #% bootstrap sensor2 monthly climatology oe
    if control['if bootstrap']:
        nB = control['nB']
        all_month_post_b_s5p = np.full((12,nB),np.nan)
        Q_month_post_b_s5p = np.full((12,nB),np.nan)
        tau_month_post_b_s5p = np.full((12,nB),np.nan)
        whiteDateArray_s5p = dateArray_s5pno2[np.array([(d not in blackListDateArray_s5p) for d in dateArray_s5pno2])]
        whiteMonthlyDictArray_s5p = monthlyDictArray_s5pno2[np.array([(d not in blackListDateArray_s5p) for d in dateArray_s5pno2])]
        for iB in range(nB):
            md12_s5p_b = np.array([])
            for month in range(1,13):
                monthInterval = np.array([month-1,month,month+1])
                #    monthInterval = np.array([month])
                monthInterval[monthInterval<=0] = monthInterval[monthInterval<=0]+12
                monthInterval[monthInterval>12] = monthInterval[monthInterval>12]-12
                mask = np.array([(d.month in monthInterval) for d in whiteDateArray_s5p])
                chooseIndex = np.where(mask==True)[0]
                nChoose = len(chooseIndex)
                randomChooseIndex = np.random.choice(chooseIndex,nChoose)
                randomMonth = np.array([d.month for d in whiteDateArray_s5p[randomChooseIndex]]) 
                if month == 1:
                    randomMonth[randomMonth==12] = 0
                if month == 12:
                    randomMonth[randomMonth==1] = 13
                all_month_post_b_s5p[month-1,iB] = np.mean(randomMonth)
                mergedDict = r_s5pno2.F_merge_monthly_data(whiteMonthlyDictArray_s5p[randomChooseIndex])
                md12_s5p_b = np.append(md12_s5p_b,mergedDict)
            ime_s5p_cli_bt = r_s5pno2.F_multimonth_OE(monthlyDictArray=md12_s5p_b,Q_ap=Q_ap,tau_ap=tau_ap,Sa=Sa,universal_f=universal_f,ridgeLambda=s5p_lambda,wsRange=universal_wsRange)
            tau_month_post_b_s5p[:,iB] = ime_s5p_cli_bt.tau_post
            Q_month_post_b_s5p[:,iB] = ime_s5p_cli_bt.Q_post
    
#%% all month sensor1 oe
nmonth_omi = len(dateArray)
if os.path.exists(control['jpl annual emission path']):
    T = pd.read_csv(control['jpl annual emission path'],index_col=0)
    jpl_nox = np.full((len(dateArray)),np.nan)
    for (i,d) in enumerate(dateArray):
        try:
            jpl_nox[i] = T.loc[d.year].iloc[d.month-1]
            if jpl_nox[i] == 0:
                jpl_nox[i] = np.nan
        except:
            print(d.strftime('%Y%m')+' does not exist')
    jpl_year = np.arange(2005,2019)
    jpl_annual = np.full(jpl_year.shape,np.nan)
    for (j,jy) in enumerate(jpl_year):
        jpl_annual[j] = np.nanmean(jpl_nox[np.array([d.year == jy for d in dateArray])])
    ols_jpl = np.polyfit(jpl_year+0.5,np.log(jpl_annual),1)
    def F_predict_priorQ(dateArray,ols_jpl):
        xx = np.array([d.year+(d.month-0.5)/12 for d in dateArray])
        yy = np.exp(np.polyval(ols_jpl,xx))
        return yy
    Q_ap_omi = F_predict_priorQ(dateArray,ols_jpl)
else:
    Q_ap_omi = control['sensor1 Q all prior']/nox_per_no2*np.ones((nmonth_omi))
Q_ap_e_omi = Q_ap_omi.copy()
tau_ap_omi = np.array([tau_month_post[d.month-1] for d in dateArray])
tau_ap_e_omi = 0.3*tau_ap_omi
corr_m_omi = control['monthly corr month']
corr_y_omi = control['monthly corr year']
corrcoef_Q_tau = 0.
Sa_Q_omi = np.diag(Q_ap_e_omi**2)
Sa_tau_omi = np.diag(tau_ap_e_omi**2)
for (i,d1) in enumerate(dateArray):
    for (j,d2) in enumerate(dateArray):
        if i == j:
            continue
        m1 = d1.month
        m2 = d2.month
        dmonth = np.min([np.abs(m1-m2),12-np.abs(m1-m2)])
        dyear = np.abs((d1-d2).days/365)
        Sa_Q_omi[i,j] = Q_ap_e_omi[i]*Q_ap_e_omi[j]*np.exp(-dmonth/corr_m_omi)*np.exp(-dyear/corr_y_omi)
        Sa_tau_omi[i,j] = tau_ap_e_omi[i]*tau_ap_e_omi[j]*np.exp(-dmonth/corr_m_omi)*np.exp(-dyear/corr_y_omi)
Sa_omi = np.zeros((nmonth_omi*2,nmonth_omi*2))
Sa_omi[0:nmonth_omi,0:nmonth_omi] = Sa_Q_omi
Sa_omi[nmonth_omi:,nmonth_omi:] = Sa_tau_omi
for (i,d1) in enumerate(dateArray):
    for (j,d2) in enumerate(dateArray):
        Sa_omi[i,nmonth_omi+j] = np.sqrt(Sa_Q_omi[i,i])*np.sqrt(Sa_tau_omi[j,j])*corrcoef_Q_tau
        Sa_omi[nmonth_omi+j,i] = np.sqrt(Sa_Q_omi[i,i])*np.sqrt(Sa_tau_omi[j,j])*corrcoef_Q_tau

ime_omi_allmonth = r.F_multimonth_OE(monthlyDictArray=monthlyDictArray,
                                     Q_ap=Q_ap_omi,
                                     tau_ap=tau_ap_omi,
                                     Sa=Sa_omi,
                                     universal_f=universal_f,
                                     ridgeLambda=control['sensor1 all month ridgeLambda'],
                                     wsRange=universal_wsRange,
                                     tol=1e-5)#convergenceThreshold=0.1)
Q_all_post = ime_omi_allmonth.Q_post
tau_all_post = ime_omi_allmonth.tau_post
avk_Q_all = np.diag(ime_omi_allmonth.avk)[0:len(ime_omi_allmonth.Q_post)]
avk_tau_all = np.diag(ime_omi_allmonth.avk)[len(ime_omi_allmonth.Q_post):]

#% sensor2 all month oe
if control['if sensor2']:
    nmonth_s5p = len(dateArray_s5pno2)
    if os.path.exists(control['jpl annual emission path']):
        Q_ap_s5p = F_predict_priorQ(dateArray_s5pno2,ols_jpl)
    else:
        Q_ap_s5p = control['sensor2 Q all prior']/nox_per_no2*np.ones((nmonth_s5p))
    Q_ap_e_s5p = Q_ap_s5p.copy()
    tau_ap_s5p= np.array([tau_month_post[d.month-1] for d in dateArray_s5pno2])
    tau_ap_e_s5p = 0.3*tau_ap_s5p
    corr_m_s5p = control['monthly corr month']
    corr_y_s5p = control['monthly corr year']
    corrcoef_Q_tau = 0.
    Sa_Q_s5p = np.diag(Q_ap_e_s5p**2)
    Sa_tau_s5p = np.diag(tau_ap_e_s5p**2)
    for (i,d1) in enumerate(dateArray_s5pno2):
        for (j,d2) in enumerate(dateArray_s5pno2):
            if i == j:
                continue
            m1 = d1.month
            m2 = d2.month
            dmonth = np.min([np.abs(m1-m2),12-np.abs(m1-m2)])
            dyear = np.abs((d1-d2).days/365)
            Sa_Q_s5p[i,j] = Q_ap_e_s5p[i]*Q_ap_e_s5p[j]*np.exp(-dmonth/corr_m_s5p)*np.exp(-dyear/corr_y_s5p)
            Sa_tau_s5p[i,j] = tau_ap_e_s5p[i]*tau_ap_e_s5p[j]*np.exp(-dmonth/corr_m_s5p)*np.exp(-dyear/corr_y_s5p)
    Sa_s5p = np.zeros((nmonth_s5p*2,nmonth_s5p*2))
    Sa_s5p[0:nmonth_s5p,0:nmonth_s5p] = Sa_Q_s5p
    Sa_s5p[nmonth_s5p:,nmonth_s5p:] = Sa_tau_s5p
    for (i,d1) in enumerate(dateArray_s5pno2):
        for (j,d2) in enumerate(dateArray_s5pno2):
            Sa_s5p[i,nmonth_s5p+j] = np.sqrt(Sa_Q_s5p[i,i])*np.sqrt(Sa_tau_s5p[j,j])*corrcoef_Q_tau
            Sa_s5p[nmonth_s5p+j,i] = np.sqrt(Sa_Q_s5p[i,i])*np.sqrt(Sa_tau_s5p[j,j])*corrcoef_Q_tau
    ime_s5p_allmonth = r_s5pno2.F_multimonth_OE(monthlyDictArray=monthlyDictArray_s5pno2,
                                                Q_ap=Q_ap_s5p,
                                                tau_ap=tau_ap_s5p,
                                                Sa=Sa_s5p,
                                                universal_f=universal_f,
                                                ridgeLambda=control['sensor2 all month ridgeLambda'],
                                                wsRange=universal_wsRange,
                                                tol=1e-5)#convergenceThreshold=0.1)
    Q_all_post_s5p = ime_s5p_allmonth.Q_post
    tau_all_post_s5p = ime_s5p_allmonth.tau_post
    avk_Q_all_s5p = np.diag(ime_s5p_allmonth.avk)[0:len(ime_s5p_allmonth.Q_post)]
    avk_tau_all_s5p = np.diag(ime_s5p_allmonth.avk)[len(ime_s5p_allmonth.Q_post):]

if control['if test ridgeLambda']:
    l1 = -10
    l2 = -7
    ln = (l2-l1)*8+1
    l_vec = np.logspace(l1,l2,ln)
    l_vec = l_vec[3:]
    Ja_l_omi_allmonth = np.full((len(l_vec)),np.nan)
    Jo_l_omi_allmonth = np.full((len(l_vec)),np.nan)
    for (j,l) in enumerate(l_vec):
        ime = r.F_multimonth_OE(monthlyDictArray=monthlyDictArray,
                                         Q_ap=Q_ap_omi,
                                         tau_ap=tau_ap_omi,
                                         Sa=Sa_omi,
                                         universal_f=universal_f,
                                         ridgeLambda=l,
                                         wsRange=universal_wsRange,
                                         maxIteration=300,
                                         tol=1e-5)
        Ja_l_omi_allmonth[j] = ime.Jprior
        Jo_l_omi_allmonth[j] = ime.Jobs
    if control['if sensor2']:
        Ja_l_s5p_allmonth = np.full((len(l_vec)),np.nan)
        Jo_l_s5p_allmonth = np.full((len(l_vec)),np.nan)
        for (j,l) in enumerate(l_vec):
            ime = r_s5pno2.F_multimonth_OE(monthlyDictArray=monthlyDictArray_s5pno2,
                                           Q_ap=Q_ap_s5p,
                                           tau_ap=tau_ap_s5p,
                                           Sa=Sa_s5p,
                                           universal_f=universal_f,
                                           ridgeLambda=l,
                                           wsRange=universal_wsRange,
                                           maxIteration=300,
                                           tol=1e-5)
            Ja_l_s5p_allmonth[j] = ime.Jprior
            Jo_l_s5p_allmonth[j] = ime.Jobs
    # climatology lambda sweep
    l1 = -11
    l2 = -8
    ln = (l2-l1)*8+1
    l_vec_cli = np.logspace(l1,l2,ln)
    if control['if sensor2']:
        Ja_l_s5p = np.full((len(l_vec_cli)),np.nan)
        Jo_l_s5p = np.full((len(l_vec_cli)),np.nan)
        for (j,l) in enumerate(l_vec_cli):
            ime = r_s5pno2.F_multimonth_OE(monthlyDictArray=md12_s5p,Q_ap=Q_ap_cli_s5p,tau_ap=tau_ap_cli_s5p,Sa=Sa_cli_s5p,universal_f=universal_f,ridgeLambda=l,wsRange=universal_wsRange)
            Ja_l_s5p[j] = ime.Jprior
            Jo_l_s5p[j] = ime.Jobs
    Ja_l = np.full((len(l_vec_cli)),np.nan)
    Jo_l = np.full((len(l_vec_cli)),np.nan)
    for (j,l) in enumerate(l_vec_cli):
        ime = r.F_multimonth_OE(monthlyDictArray=md12,Q_ap=Q_ap_cli,tau_ap=tau_ap_cli,Sa=Sa_cli,universal_f=universal_f,ridgeLambda=l,wsRange=universal_wsRange)
        Ja_l[j] = ime.Jprior
        Jo_l[j] = ime.Jobs
# saving
var_list_full = ['Q_month','tau_month','all_month_b','btMonthNumber','Q_month_b','tau_month_b','dQ','dtau','dQ3','dtau3',
            'bound_Q','bound_tau','bound_Q3','bound_tau3','dateArray','blackListDateArray',
            'Q_month_clean','tau_month_clean','all_month_clean_b','btMonthNumber_clean','Q_month_clean_b','tau_month_clean_b',
            'tau_month_prior_both','l_vec','l_vec_cli',
            'Q_ap_cli','tau_ap_cli','Sa_cli','tau_month_post','Q_month_post','avk_Q_month_post','avk_tau_month_post',
            'all_month_post_b','Q_month_post_b','tau_month_post_b',
            'Q_ap_omi','tau_ap_omi','Q_all_post','tau_all_post','avk_Q_all','avk_tau_all',
            'Ja_l_omi_allmonth','Jo_l_omi_allmonth','Ja_l','Jo_l',
            'Q_month_s5p','tau_month_s5p','all_month_b_s5p','btMonthNumber_s5p','Q_month_b_s5p','tau_month_b_s5p','dQ_s5p','dtau_s5p','dQ3_s5p','dtau3_s5p',
            'bound_Q_s5p','bound_tau_s5p','bound_Q3_s5p','bound_tau3_s5p','dateArray_s5pno2','blackListDateArray_s5p',
            'Q_month_clean_s5p','tau_month_clean_s5p','all_month_b_clean_s5p','btMonthNumber_clean_s5p','Q_month_b_clean_s5p','tau_month_b_clean_s5p',
            'Q_ap_cli_s5p','tau_ap_cli_s5p','Sa_cli_s5p','tau_month_post_s5p','Q_month_post_s5p','avk_Q_month_post_s5p','avk_tau_month_post_s5p',
            'all_month_post_b_s5p','Q_month_post_b_s5p','tau_month_post_b_s5p',
            'Q_ap_s5p','tau_ap_s5p','Q_all_post_s5p','tau_all_post_s5p','avk_Q_all_s5p','avk_tau_all_s5p',
            'Ja_l_s5p_allmonth','Jo_l_s5p_allmonth','Ja_l_s5p','Jo_l_s5p']
dump_dict = {}
for v in var_list_full:
    if v in locals():
        dump_dict[v] = locals()[v]
np.savez(control['save path'],**dump_dict)
