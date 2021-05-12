# python script to subset SAO ozone profile L2 data saved in xdr format
# Written by Kang Sun on 2017/12/30
# yes! I do use python

import numpy as np
from scipy.io import readsav
from scipy.io import savemat 
import os

L2dir = 'C:\\data_ks\\PROFOZ\L2\\'
L2gdir = 'C:\\data_ks\\PROFOZ\L2g\\'

L2dir = '/data/tempo2/xliu/PROFOZL2XDR/'
L2gdir = '/data/tempo1/Shared/kangsun/PROFOZ/L2g/'
# os.chdir(L2gdir)

MaxSZA = 75
MaxCF = 0.3
MinLat = 25
MaxLat = 50
MinLon = -130
MaxLon = -63

usextrack = np.arange(1,31)

runyear = [2005,2006,2007]
runmonth = np.arange(1,13)
runday = np.arange(1,32)

tmpfn = L2gdir+'tmp.sav'

for iyear in runyear:
    line = np.ndarray([0],np.int32)
    ift = np.ndarray([0],np.int32)
    lat_r = np.ndarray([0,4],np.float32)
    lon_r = np.ndarray([0,4],np.float32)
    lat_c = np.ndarray([0],np.float32)
    lon_c = np.ndarray([0],np.float32)

    toc_du = np.ndarray([0],np.float32)
    toc_e = np.ndarray([0],np.float32)
    toc_dfs = np.ndarray([0],np.float32)
    cloudfrac = np.ndarray([0],np.float32)
    sfc_du = np.ndarray([0],np.float32)
    sfc_e = np.ndarray([0],np.float32)
    toc_vmr = np.ndarray([0],np.float32)
    sfc_vmr = np.ndarray([0],np.float32)
    day = np.ndarray([0],np.int16)
    year = np.ndarray([0],np.int16)
    month = np.ndarray([0],np.int16)
    hour = np.ndarray([0],np.float32)
    sza = np.ndarray([0],np.float32)
    alb_0 = np.ndarray([0],np.float32)
    alb_1 = np.ndarray([0],np.float32)
    bro = np.ndarray([0],np.float32)
    for imonth in runmonth:
        for iday in runday:
            fn = L2dir+'%04d'%iyear+'/OMI-Aura_L2-PROFOZ_'+'%04d'%iyear+'m'+'%02d'%imonth+'%02d'%iday+'.xdr'
            
            if os.path.isfile(fn):
                data = readsav(fn,python_dict=True,uncompressed_file_name=tmpfn,verbose=True)
                os.remove(tmpfn)
                f1 = data['omisza'] <= MaxSZA
                f2 = data['omicfrac'] <= MaxCF
                f3 = data['omidescend'] == 0;
                f4 = data['omicol'][3,2,] >= 0.5;
                f5 = (data['omilon'][4,:] >= MinLon) & (data['omilon'][4,:] <= MaxLon) & (data['omilat'][4,:] >= MinLat) & (data['omilat'][4,:] <= MaxLat)
                f6 = np.in1d(data['omipix'],usextrack)

                validmask = f1 & f2 & f3 & f4 & f5 & f6
                print( 'You have '+'%s'%np.sum(validmask)+' valid L2 pixels on '+'%04d'%iyear+'m'+'%02d'%imonth+'%02d'%iday)
                line = np.concatenate((line,data['omiline'][validmask]))
                ift = np.concatenate((ift,data['omipix'][validmask]))
                lat_r = np.concatenate((lat_r,data['omilat'][0:4,validmask].T))
                lon_r = np.concatenate((lon_r,data['omilon'][0:4,validmask].T))
                lat_c = np.concatenate((lat_c,data['omilat'][4,validmask]))
                lon_c = np.concatenate((lon_c,data['omilon'][4,validmask]))
                local_toc_du = data['omicol'][0,2,validmask]
                toc_du = np.concatenate((toc_du,local_toc_du))
                toc_e = np.concatenate((toc_e,data['omicol'][1,2,validmask]))
                toc_dfs = np.concatenate((toc_dfs,data['omicol'][3,2,validmask]))
                cloudfrac = np.concatenate((cloudfrac,data['omicfrac'][validmask]))
                nl = data['nl']
                local_sfc_du = data['ozprofs'][nl-1,2,validmask]
                sfc_du = np.concatenate((sfc_du,local_sfc_du))
                sfc_e = np.concatenate((sfc_e,data['ozprofs'][nl-1,3,validmask]))
                ntp = data['omintp'][validmask]
                atmos_P = data['atmos'][:,0,validmask]
                toc_dp = np.zeros(ntp.size,dtype=np.float32)
                sfc_dp = np.zeros(ntp.size,dtype=np.float32)

                for i in range(ntp.size):
                    toc_dp[i] = atmos_P[nl,i]-atmos_P[ntp[i],i]
                    sfc_dp[i] = atmos_P[nl,i]-atmos_P[nl-1,i]

                toc_vmr = np.concatenate((toc_vmr,local_toc_du[:]/toc_dp[:]*1267)) # in ppb
                sfc_vmr = np.concatenate((sfc_vmr,local_sfc_du[:]/sfc_dp[:]*1267))

                day = np.concatenate((day,data['omiday'][validmask]))
                year = np.concatenate((year,data['omiyear'][validmask]))
                month = np.concatenate((month,data['omimon'][validmask]))
                hour = np.concatenate((hour,data['omiutc'][validmask]))
				
                sza = np.concatenate((sza,data['omisza'][validmask]))
                alb_0 = np.concatenate((alb_0,data['omiofitvar'][2,0,validmask]))
                alb_1 = np.concatenate((alb_1,data['omiofitvar'][2,2,validmask]))
                bro = np.concatenate((bro,data['omiofitvar'][0,2,validmask]))
            else:
                print( 'L2 file '+fn+' does not exist...')
                continue
    output_subset = {}
    output_subset['line'] = line
    output_subset['ift'] = ift
    output_subset['lon_r'] = lon_r
    output_subset['lat_r'] = lat_r
    output_subset['lon_c'] = lon_c
    output_subset['lat_c'] = lat_c
    output_subset['toc_du'] = toc_du
    output_subset['toc_e'] = toc_e
    output_subset['toc_dfs'] = toc_dfs
    output_subset['toc_du'] = toc_du
    output_subset['cloudfrac'] = cloudfrac
    output_subset['sfc_du'] = sfc_du
    output_subset['sfc_e'] = sfc_e
    output_subset['toc_vmr'] = toc_vmr
    output_subset['sfc_vmr'] = sfc_vmr
    
    output_subset['day'] = day
    output_subset['hour'] = hour
    output_subset['year'] = year
    output_subset['month'] = month
	
    output_subset['sza'] = sza
    output_subset['alb_0'] = alb_0
    output_subset['alb_1'] = alb_1
    output_subset['bro'] = bro
	
    matfn = L2gdir+'CONUS_'+'%04d'%iyear+'.mat'
    savemat(matfn,output_subset)
