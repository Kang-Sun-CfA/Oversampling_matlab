# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:50:30 2019

@author: Kang Sun

2019/03/09: match measures l3 format
2019/03/25: use control.txt
2019/04/22: include S5PNO2
2019/05/26: sample met data
2019/07/13: implement optimized regridding from chris chan miller
2019/07/19: fix fwhm -> w bug (pixel size corrected, by 2)
2019/10/23: add CrISNH3 subsetting function
2020/03/14: standardize met sampling functions
2020/05/19: add subsetting fields option as input
2020/07/20: parallel regrid function done
"""

import numpy as np
# conda install -c conda-forge opencv 
import datetime
import os
import logging

def datedev_py(matlab_datenum):
    """
    convert matlab datenum double to python datetime object
    """
    python_datetime = datetime.datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return python_datetime

def datetime2datenum(python_datetime):
    '''
    convert python datetime to matlab datenum
    '''
    matlab_datenum = python_datetime.toordinal()\
                                    +python_datetime.hour/24.\
                                    +python_datetime.minute/1440.\
                                    +python_datetime.second/86400.+366.
    return matlab_datenum

def F_collocate_l2g(l2g_data1,l2g_data2,hour_difference=0.5,
                    field_to_average='column_amount'):
    '''
    collocate two l2g dictionaries
    l2g_data1:
        the one with bigger pixels
    hour_difference:
        max difference between pixels in hour
    field_to_average:
        the l2g field in l2g_data2 to be averaged to l2g_data1 pixels
    updated on 2020/08/23
    updated on 2020/10/13
    '''
    from shapely.geometry import Polygon
    l2g_2_west = np.min(l2g_data2['lonr'],axis=1)
    l2g_2_east = np.max(l2g_data2['lonr'],axis=1)
    l2g_2_south = np.min(l2g_data2['latr'],axis=1)
    l2g_2_north = np.max(l2g_data2['latr'],axis=1)
    
    l2g_1_west = np.min(l2g_data1['lonr'],axis=1)
    l2g_1_east = np.max(l2g_data1['lonr'],axis=1)
    l2g_1_south = np.min(l2g_data1['latr'],axis=1)
    l2g_1_north = np.max(l2g_data1['latr'],axis=1)
    
    l2g_2_utc = l2g_data2['UTC_matlab_datenum']
    l2g_1_utc = l2g_data1['UTC_matlab_datenum']
    
    l2g_2_lonr = l2g_data2['lonr']
    l2g_1_lonr = l2g_data1['lonr']
    l2g_2_latr = l2g_data2['latr']
    l2g_1_latr = l2g_data1['latr']
    
    l2g_2_C = l2g_data2[field_to_average]
    
    mask_list = [np.where((l2g_2_utc >= l2g_1_utc[i]-hour_difference/24)\
        & (l2g_2_utc <= l2g_1_utc[i]+hour_difference/24)\
        & (l2g_2_south <= l2g_1_north[i])\
        & (l2g_2_north >= l2g_1_south[i])\
        & (l2g_2_east >= l2g_1_west[i])\
        & (l2g_2_west <= l2g_1_east[i])) for i in range(len(l2g_data1['latc']))]
    
    def F_poly_intersect(x1,y1,X2,Y2,l2g_2_C):
        '''
        x1, y1 defines a bigger polygon
        each row of X2 Y2 defines a smaller polygon
        '''
        if len(X2) == 0:
            return np.array([np.nan, np.nan, np.nan])
        poly1 = Polygon(np.vstack((x1,y1)).T)
        area1 = poly1.area
        n = X2.shape[0]
        poly2_list = [Polygon(np.vstack((X2[j,],Y2[j,])).T) for j in range(n)]
        area_list = np.array([np.array([poly1.intersection(poly2).area,poly2.area]) for poly2 in poly2_list])
        npix = np.sum(area_list[:,0]/area_list[:,1])
        weighted_mean_l2g_2_C = np.sum(area_list[:,0]*l2g_2_C)/np.sum(area_list[:,0])
        relative_overlap = np.sum(area_list[:,0])/area1
        return np.array([weighted_mean_l2g_2_C,relative_overlap,npix])
    
    result_array = np.array([F_poly_intersect(l2g_1_lonr[i,],
                                              l2g_1_latr[i,],
                                              l2g_2_lonr[mask_list[i][0],],
                                              l2g_2_latr[mask_list[i][0],],
                                              l2g_2_C[mask_list[i][0]]) for i in range(len(l2g_data1['latc']))])
    l2g_data1[field_to_average+'2'] = result_array[:,0]
    l2g_data1['relative_overlap2'] = result_array[:,1]
    l2g_data1['npix2'] = result_array[:,2]
    overlap_mask = (~np.isnan(result_array[:,0])) & (result_array[:,2] > 0)
        
    l2g_data1_has2 = {k:v[overlap_mask,] for (k,v) in l2g_data1.items()}
    l2g_data1_hasnot2 = {k:v[~overlap_mask,] for (k,v) in l2g_data1.items()}
    return l2g_data1_has2, l2g_data1_hasnot2
    
def F_interp_gcrs(sounding_lon,sounding_lat,sounding_datenum,sounding_ps,
                  gcrs_dir='/mnt/Data2/GEOS-Chem_Silvern/',
                  product='NO2',if_monthly=False):
    """
    sample a field from GEOS-Chem data by Rachel Silvern (gcrs) in .nc format. 
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    sounding_ps:
        surface pressure for each sounding
    gcrs_dir:
        directory where geos chem data are saved
    if_monthly:
        if use monthly profile, instead of daily profile
    created on 2020/03/09
    """
    from netCDF4 import Dataset
    from scipy.interpolate import RegularGridInterpolator
    from calendar import isleap
    # hybrid Ap parameter in Pa
    Ap = np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
                   3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
                   7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
                   1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
                   2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
                   2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
                   7.851231e+01, 5.638791e+01, 4.017541e+01, 2.836781e+01, 1.979160e+01, 9.292942e+00,
                   4.076571e+00, 1.650790e+00, 6.167791e-01, 2.113490e-01, 6.600001e-02, 1.000000e-02],dtype=np.float32)*1e2
    # hybrid Bp parameter
    Bp = np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                   8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                   7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                   5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                   2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                   6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],dtype=np.float32)
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_datetime = datedev_py(start_datenum)
    end_datetime = datedev_py(end_datenum)
    nl2 = len(sounding_datenum)
    nlayer = 47 # layer number in geos chem
    # claim space for interopolated profiles, at all level 2 pixels
    sounding_profile = np.zeros((nl2,nlayer),dtype=np.float32)
    sounding_pEdge = sounding_ps[:,np.newaxis]*Bp+Ap
    lat_interp = np.tile(sounding_lat,(nlayer,1)).T.astype(np.float32)
    lon_interp = np.tile(sounding_lon,(nlayer,1)).T.astype(np.float32)
    layer_interp = np.tile(np.arange(nlayer),(nl2,1)).astype(np.float32)
    for year in range(start_datetime.year,end_datetime.year+1): 
        if product == 'NO2' and if_monthly == False:
            year_sounding_doy = np.floor(sounding_datenum)-(datetime.datetime(year=year,month=1,day=1).toordinal()+365.)
            year_sounding_doy = year_sounding_doy.astype(int)
            if isleap(year):
                nday = 366
            else:
                nday = 365
            loop_sounding_doy = np.unique(year_sounding_doy)
            f1 = (loop_sounding_doy>=1)
            f2 = (loop_sounding_doy<=nday)
            loop_sounding_doy = loop_sounding_doy[f1&f2]
            gc_fn = os.path.join(gcrs_dir,'NO2_PROF.05x0625_NA.%0d.nc'%year)
            print('loading '+gc_fn)
            gc_id = Dataset(gc_fn)
            gc_gas = gc_id['NO2_ppb'][:].astype(np.float32)
            gc_lon = gc_id['longitude'][:]
            gc_lat = gc_id['latitude'][:]
            for doy in loop_sounding_doy:
                # remember python is 0-based
                gc_gas_doy = gc_gas[doy-1,...].squeeze()
                rowIndex = np.nonzero(year_sounding_doy==doy)
                f = RegularGridInterpolator((np.arange(nlayer),gc_lat,gc_lon),\
                                            gc_gas_doy,bounds_error=False,fill_value=np.nan)
                sounding_profile[rowIndex,:] = f((layer_interp[rowIndex,:],\
                                lat_interp[rowIndex,:],lon_interp[rowIndex,:]))
        elif product in {'NH3','HCHO'} or if_monthly == True:
            sounding_dt = [datedev_py(sounding_datenum[il2]) for il2 in range(nl2)]
            sounding_year = np.array([dt.year for dt in sounding_dt])
            sounding_month = np.array([dt.month for dt in sounding_dt])
            loop_month = np.unique(sounding_month[sounding_year==year])
            gc_fn = os.path.join(gcrs_dir,'NH3_HCHO_PROF.05x0625_NA.%0d.nc'%year)
            print('loading '+gc_fn)
            gc_id = Dataset(gc_fn)
            gc_gas = gc_id[product+'_ppb'][:].astype(np.float32)
            gc_lon = gc_id['longitude'][:]
            gc_lat = gc_id['latitude'][:]
            for month in loop_month:
                # remember python is 0-based
                gc_gas_doy = gc_gas[month-1,...].squeeze()
                rowIndex = np.nonzero((sounding_year==year)&(sounding_month==month))
                f = RegularGridInterpolator((np.arange(nlayer),gc_lat,gc_lon),\
                                            gc_gas_doy,bounds_error=False,fill_value=np.nan)
                sounding_profile[rowIndex,:] = f((layer_interp[rowIndex,:],\
                                lat_interp[rowIndex,:],lon_interp[rowIndex,:]))
    return sounding_profile, sounding_pEdge
        
def F_interp_merra2(sounding_lon,sounding_lat,sounding_datenum,\
                  merra2_dir='/mnt/Data2/MERRA/',\
                  interp_fields=['PBLTOP','PS','TROPPT'],\
                  fn_header='MERRA2_300.tavg1_2d_slv_Nx'):
    """
    sample a field from merra2 data in .nc format. 
    see download_merra2.py for downloading
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    merra2_dir:
        directory where merra2 data are saved
    interp_fields:
        variables to interpolate from merra2, only 2d fields are supported
    fn_header:
        following nasa ges disc naming
    created on 2020/03/09
    """
    import glob
    from scipy.interpolate import RegularGridInterpolator
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_date = datedev_py(start_datenum).date()
    
    end_date = datedev_py(end_datenum).date()
    days = (end_date-start_date).days+1
    DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
    merra2_data = {}
    iday = 0
    for DATE in DATES:
        merra_filedir = os.path.join(merra2_dir,DATE.strftime('Y%Y'),\
                                      DATE.strftime('M%m'),DATE.strftime('D%d'))
        merra_flist = glob.glob(merra_filedir+'/*.nc')
        if len(merra_flist) > 1:
            print('Careful! More than one nc file in MERRA daily folder!')
        elif len(merra_flist) == 0:
            print('No merra file')
            continue
        fn = merra_flist[0]
        if not merra2_data:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['lat','lon','time'])))
            merra2_data['lon'] = nc_out['lon']
            merra2_data['lat'] = nc_out['lat']
            # how many hours are there in each daily file? have to be the same 
            nhour = len(nc_out['time'])
            merra2_data['datenum'] = np.zeros((nhour*(days)),dtype=np.float64)
            # merra2 time is defined as minutes since 00:30:00 on that day
            merra2_data['datenum'][iday*nhour:((iday+1)*nhour)] = DATE.toordinal()+366.+(nc_out['time']+30)/1440
            for field in interp_fields:
                merra2_data[field] = np.zeros((len(merra2_data['lon']),len(merra2_data['lat']),nhour*(days)))
                # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                merra2_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))
        else:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['time'])))
            # merra2 time is defined as minutes since 00:30:00 on that day
            merra2_data['datenum'][iday*nhour:((iday+1)*nhour)] = DATE.toordinal()+366.+(nc_out['time']+30)/1440
            for field in interp_fields:
                # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                merra2_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))
        # forgot to increment iday
        iday = iday+1
    
    sounding_interp = {}
    if not merra2_data:
        for fn in interp_fields:
            sounding_interp[fn] = sounding_lon*np.nan
        return sounding_interp
    # interpolate
    for fn in interp_fields:
        my_interpolating_function = \
        RegularGridInterpolator((merra2_data['lon'],merra2_data['lat'],merra2_data['datenum']),\
                                merra2_data[fn],bounds_error=False,fill_value=np.nan)
        sounding_interp[fn] = my_interpolating_function((sounding_lon,sounding_lat,sounding_datenum))
    return sounding_interp


def F_interp_era5_3D(sounding_lon,sounding_lat,sounding_datenum,
                     sounding_p0,sounding_p1,nlevel=10,\
                     era5_dir='/mnt/Data2/ERA5/',\
                     interp_fields=['v','u'],\
                     fn_header='CONUS'):
    """
    sample 3D field from era5 data in .nc format. 
    see era5.py for era5 downloading/subsetting
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    sounding_p0:
        bottom bound of pressure
    sounding_p1:
        top bound of pressure
    nlevel:
        how many pressure-linear levels between sounding_p0 and sounding_p1
    era5_dir:
        directory where subset era5 data in .nc are saved
    interp_fields:
        variables to interpolate from era5, u and v
    fn_header:
        in general should denote domain location of era5 data
    created on 2020/09/20
    """
    from scipy.interpolate import RegularGridInterpolator
#    nl2 = len(sounding_datenum)
    p_interp = np.linspace(sounding_p0,sounding_p1,nlevel).T
    lat_interp = np.tile(sounding_lat,(nlevel,1)).T
    lon_interp = np.tile(sounding_lon,(nlevel,1)).T
    time_interp = np.tile(sounding_datenum,(nlevel,1)).T
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_date = datedev_py(start_datenum).date()
    
    end_date = datedev_py(end_datenum).date()
    days = (end_date-start_date).days+1
    DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
    era5_data = {}
    iday = 0
    for DATE in DATES:
        fn = os.path.join(era5_dir,DATE.strftime('Y%Y'),\
                                   DATE.strftime('M%m'),\
                                   DATE.strftime('D%d'),\
                                   fn_header+'_3D_'+DATE.strftime('%Y%m%d')+'.nc')
        if not era5_data:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['latitude','longitude','time','level'])))
            era5_data['lon'] = nc_out['longitude']
            era5_data['level'] = nc_out['level']*100 # hPa to Pa
            era5_data['lat'] = nc_out['latitude'][::-1]
            # how many hours are there in each daily file? have to be the same 
            nhour = len(nc_out['time'])
            era5_data['datenum'] = np.zeros((nhour*(days)),dtype=np.float64)
            # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
            era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
            for field in interp_fields:
                era5_data[field] = np.zeros((len(era5_data['lon']),len(era5_data['lat']),len(era5_data['level']),nhour*(days)))
                if len(nc_out[field].shape) != 4:
                    print('Warning!!! Anomaly in the dimension of ERA5 fields.')
                    print('Tentatively taking only the first element of the second dimension')
                    nc_out[field] = nc_out[field][:,0,...].squeeze()
                # was read in as 4-d array in time, level, lat, lon; transpose to lon, lat, level, time
                era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((3,2,1,0))[:,::-1,:,:]
        else:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['time'])))
            # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
            era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
            for field in interp_fields:
                # was read in as 4-d array in time, level, lat, lon; transpose to lon, lat, level, time
                era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((3,2,1,0))[:,::-1,:,:]
        # forgot to increment iday
        iday = iday+1
    
    sounding_interp = {}
    if not era5_data:
        for fn in interp_fields:
            sounding_interp[fn] = lon_interp*np.nan
        return sounding_interp
    # interpolate
    for fn in interp_fields:
        my_interpolating_function = \
        RegularGridInterpolator((era5_data['lon'],era5_data['lat'],era5_data['level'],era5_data['datenum']),\
                                era5_data[fn],bounds_error=False,fill_value=np.nan)
        sounding_interp[fn] = my_interpolating_function((lon_interp,lat_interp,p_interp,time_interp))
    return sounding_interp


def F_interp_era5(sounding_lon,sounding_lat,sounding_datenum,\
                  era5_dir='/mnt/Data2/ERA5/',\
                  interp_fields=['blh','u10','v10','u100','v100','sp'],\
                  fn_header='CONUS'):
    """
    sample a field from era5 data in .nc format. 
    see era5.py for era5 downloading/subsetting
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    era5_dir:
        directory where subset era5 data in .nc are saved
    interp_fields:
        variables to interpolate from era5, only 2d fields are supported
    fn_header:
        in general should denote domain location of era5 data
    created on 2019/09/18
    """
    from scipy.interpolate import RegularGridInterpolator
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_date = datedev_py(start_datenum).date()
    
    end_date = datedev_py(end_datenum).date()
    days = (end_date-start_date).days+1
    DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
    era5_data = {}
    iday = 0
    for DATE in DATES:
        fn = os.path.join(era5_dir,DATE.strftime('Y%Y'),\
                                   DATE.strftime('M%m'),\
                                   DATE.strftime('D%d'),\
                                   fn_header+'_2D_'+DATE.strftime('%Y%m%d')+'.nc')
        if not era5_data:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['latitude','longitude','time'])))
            era5_data['lon'] = nc_out['longitude']
            era5_data['lat'] = nc_out['latitude'][::-1]
            # how many hours are there in each daily file? have to be the same 
            nhour = len(nc_out['time'])
            era5_data['datenum'] = np.zeros((nhour*(days)),dtype=np.float64)
            # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
            era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
            for field in interp_fields:
                era5_data[field] = np.zeros((len(era5_data['lon']),len(era5_data['lat']),nhour*(days)))
                if len(nc_out[field].shape) != 3:
                    print('Warning!!! Anomaly in the dimension of ERA5 fields.')
                    print('Tentatively taking only the first element of the second dimension')
                    nc_out[field] = nc_out[field][:,0,...].squeeze()
                # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))[:,::-1,:]
        else:
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (interp_fields,['time'])))
            # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
            era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
            for field in interp_fields:
                # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))[:,::-1,:]
        # forgot to increment iday
        iday = iday+1
    
    sounding_interp = {}
    if not era5_data:
        for fn in interp_fields:
            sounding_interp[fn] = sounding_lon*np.nan
        return sounding_interp
    # interpolate
    for fn in interp_fields:
        my_interpolating_function = \
        RegularGridInterpolator((era5_data['lon'],era5_data['lat'],era5_data['datenum']),\
                                era5_data[fn],bounds_error=False,fill_value=np.nan)
        sounding_interp[fn] = my_interpolating_function((sounding_lon,sounding_lat,sounding_datenum))
    return sounding_interp
    
def F_interp_geos_mat(sounding_lon,sounding_lat,sounding_datenum,\
                  geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                  interp_fields=['TROPPT'],\
                  time_collection='inst3',\
                  fn_header='subset'):
    """
    sample a field from subset geos fp data in .mat format. 
    see geos.py for geos downloading/subsetting
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    geos_dir:
        directory where subset geos data in .mat are saved
    interp_fields:
        variables to interpolate from geos fp, only 2d fields are supported
    time_collection:
            choose from inst3, tavg1, tavg3
    created on 2019/05/26
    updated on 2019/07/01 to be compatible with different file collections and non continues time steps
    """
    from scipy.io import loadmat
    from scipy.interpolate import RegularGridInterpolator
    
    if time_collection == 'inst3' or time_collection == '':
        step_hour = 3
        daily_start_time = datetime.time(hour=0,minute=0)
    elif time_collection == 'tavg1':
        step_hour = 1
        daily_start_time = datetime.time(hour=0,minute=30)
    elif time_collection == 'tavg3':
        step_hour = 3
        daily_start_time = datetime.time(hour=1,minute=30)
    
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_datetime = datedev_py(start_datenum)
    start_year = start_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day
    
    end_datetime = datedev_py(end_datenum)
    end_year = end_datetime.year
    end_month = end_datetime.month
    end_day = end_datetime.day
    
    # extend the start/end datetime to the closest step_hour intervals
    t_array0 = datetime.datetime.combine(datetime.date(start_year,start_month,start_day),\
    daily_start_time)-datetime.timedelta(hours=step_hour)
    t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
    tn_array = np.array([(start_datetime-dt).total_seconds() for dt in t_array])
    geos_start_datetime = t_array[tn_array >= 0.][-1]
    
    t_array0 = datetime.datetime.combine(datetime.date(end_year,end_month,end_day),\
    daily_start_time)-datetime.timedelta(hours=step_hour)
    t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
    tn_array = np.array([(end_datetime-dt).total_seconds() for dt in t_array])
    geos_end_datetime = t_array[tn_array <= 0.][0]
    
    nstep = (geos_end_datetime-geos_start_datetime).total_seconds()/3600/step_hour+1
    nstep = int(nstep)
    
    geos_data = {}
    # load narr data
    for istep in range(nstep):
        file_datetime = geos_start_datetime+datetime.timedelta(hours=step_hour*istep)
        file_dir = os.path.join(geos_dir,file_datetime.strftime('Y%Y'),\
                                   file_datetime.strftime('M%m'),\
                                   file_datetime.strftime('D%d'))
        file_path = os.path.join(file_dir,fn_header+'_'+file_datetime.strftime('%Y%m%d_%H%M')+'.mat')
        if not os.path.exists(file_path):
            continue
        if not geos_data:
            mat_data = loadmat(file_path,variable_names=np.concatenate((['lat','lon'],interp_fields)))
            geos_data['lon'] = mat_data['lon'].flatten()
            geos_data['lat'] = mat_data['lat'].flatten()
            geos_data['datenum'] = np.zeros((nstep),dtype=np.float64)
            for fn in interp_fields:
                geos_data[fn] = np.zeros((len(geos_data['lon']),len(geos_data['lat']),nstep))
                # geos fp uses 9.9999999E14 as missing value
                mat_data[fn][mat_data[fn]>9e14] = np.nan
                geos_data[fn][...,istep] = mat_data[fn]
        else:
            mat_data = loadmat(file_path,variable_names=interp_fields)
            for fn in interp_fields:
                geos_data[fn][...,istep] = mat_data[fn]
        
        geos_data['datenum'][istep] = (file_datetime.toordinal()\
                                    +file_datetime.hour/24.\
                                    +file_datetime.minute/1440.\
                                    +file_datetime.second/86400.+366.)
    sounding_interp = {}
    if not geos_data:
        for fn in interp_fields:
            sounding_interp[fn] = sounding_lon*np.nan
        return sounding_interp
    f1 = geos_data['datenum'] != 0
    if np.sum(f1) == 0:
        for fn in interp_fields:
            sounding_interp[fn] = sounding_lon*np.nan
        return sounding_interp
    
    geos_data['datenum'] = geos_data['datenum'][f1]
    for fn in interp_fields:
        geos_data[fn] = geos_data[fn][...,f1]
    # interpolate
    
    for fn in interp_fields:
        my_interpolating_function = \
        RegularGridInterpolator((geos_data['lon'],geos_data['lat'],geos_data['datenum']),\
                                geos_data[fn],bounds_error=False,fill_value=np.nan)
        sounding_interp[fn] = my_interpolating_function((sounding_lon,sounding_lat,sounding_datenum))
    return sounding_interp

def F_interp_narr_mat(sounding_lon,sounding_lat,sounding_datenum,\
                  narr_dir='/mnt/Data2/NARR/acmap_narr/',\
                  interp_fields=['GPH_tropopause','P_tropopause',
                                 'PBLH','P_surf','T_surf',
                                 'U_10m','V_10m','U_30m','V_30m'],\
                  fn_header='subset'):
    """
    sample a field from presaved narr data
    sounding_lon:
        longitude for interpolation
    sounding_lat:
        latitude for interpolation
    sounding_datenum:
        time for interpolation in matlab datenum double format
    narr_dir:
        directory where narr is saved
    interp_fields:
        variables to interpolate, only 2d fields are supported
    created on 2019/05/25
    updated on 2019/09/19 to enable linear interpolation in a projection
    """
    from scipy.io import loadmat
    from scipy.interpolate import RegularGridInterpolator
    from pyproj import Proj
    #p1 = Proj(proj='latlong',datum='WGS84')
    # really don't know why y_0=-6245.456824468616 has to be here
    p2 = Proj(proj='lcc',R=6367.470, lat_1=50, lat_2=50,lon_0=360-107,lat_0=50)#, ellps='clrk66')#the ellps option doesn't matter
    sounding_x,sounding_y = p2(sounding_lon,sounding_lat)
    start_datenum = np.amin(sounding_datenum)
    end_datenum = np.amax(sounding_datenum)
    start_datetime = datedev_py(start_datenum)
    start_year = start_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day
    start_hour = start_datetime.hour
    end_datetime = datedev_py(end_datenum)
    end_year = end_datetime.year
    end_month = end_datetime.month
    end_day = end_datetime.day
    end_hour = end_datetime.hour
    step_hour = 3 # narr data are 3-hourly
    narr_start_hour = start_hour-start_hour%step_hour
    narr_start_datetime = datetime.datetime(year=start_year,month=start_month,day=start_day,hour=narr_start_hour)
    if end_hour >= 24-step_hour:
        narr_end_hour = 0
        narr_end_datetime = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=narr_end_hour)\
        +datetime.timedelta(days=1)
    else:
        narr_end_hour = (step_hour-(end_hour+1)%step_hour)%step_hour+end_hour+1
        narr_end_datetime = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=narr_end_hour)
    nstep = (narr_end_datetime-narr_start_datetime).total_seconds()/3600/step_hour+1
    nstep = int(nstep)
    
    narr_data = {}
    # load narr data
    for istep in range(nstep):
        file_datetime = narr_start_datetime+datetime.timedelta(hours=step_hour*istep)
        file_name = fn_header+'_'+file_datetime.strftime('%Y%m%d_%H%M')+'.mat'
        file_path = os.path.join(narr_dir,file_datetime.strftime('Y%Y'),\
                                 file_datetime.strftime('M%m'),\
                                 file_datetime.strftime('D%d'),file_name)
        if not narr_data:
            mat_data = loadmat(file_path,variable_names=np.concatenate((['x','y'],interp_fields)))
            narr_data['x'] = mat_data['x'].squeeze()
            narr_data['y'] = mat_data['y'].squeeze()
            for fn in interp_fields:
                narr_data[fn] = np.zeros((len(narr_data['x']),len(narr_data['y']),nstep))
                narr_data[fn][...,istep] = mat_data[fn].T
        else:
            mat_data = loadmat(file_path,variable_names=interp_fields)
            for fn in interp_fields:
                narr_data[fn][...,istep] = mat_data[fn].T
    # construct time axis
    narr_data['datenum'] = np.zeros((nstep),dtype=np.float64)
    for istep in range(nstep):
        file_datetime = narr_start_datetime+datetime.timedelta(hours=step_hour*istep)
        narr_data['datenum'][istep] = (file_datetime.toordinal()\
                                    +file_datetime.hour/24.\
                                    +file_datetime.minute/1440.\
                                    +file_datetime.second/86400.+366.)
    # interpolate
    sounding_interp = {}
    for fn in interp_fields:
        my_interpolating_function = \
        RegularGridInterpolator((narr_data['x'],narr_data['y'],narr_data['datenum']),\
                                narr_data[fn],bounds_error=False,fill_value=np.nan)
        sounding_interp[fn] = my_interpolating_function((sounding_x,sounding_y,sounding_datenum))
    return sounding_interp

def F_ncread_selective(fn,varnames):
    """
    very basic netcdf reader, similar to F_ncread_selective.m
    created on 2019/08/13
    """
    from netCDF4 import Dataset
    ncid = Dataset(fn,'r')
    outp = {}
    for varname in varnames:
        outp[varname] = ncid.variables[varname][:]
    ncid.close()
    return outp

# Utilities for F_block_regrid_ccm
def bound_arr(i1,i2,mx,ncols):
    arr = np.arange(i1,i2,dtype=int)
    arr[arr<0] += mx
    arr[arr>=mx] -= mx
    return arr[arr<ncols]

def bound_lat(i1,i2,mx):
    arr = np.arange(i1,i2,dtype=int)
    return arr[ np.logical_and( arr>=0, arr < mx ) ]

def F_lon_distance(lon1,lon2):
    distance = lon2 - lon1
    distance[lon2<lon1] += 360.0
    return distance

def F_ellipse(a,b,alpha,npoint,xcenter=0,ycenter=0):
    t = np.linspace(0.,np.pi*2,npoint)[::-1]
    Q = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
    X = Q.dot(np.vstack((a * np.cos(t),b * np.sin(t))))
    X[0,] = X[0,]+xcenter
    X[1,] = X[1,]+ycenter
    minlon_e = X[0,].min()
    minlat_e = X[1,].min()
    return X, minlon_e, minlat_e

def F_block_regrid_wrapper(args):
    '''
    repackage F_block_regrid_ccm following example of pysplat.hitran_absco
    '''
    return F_block_regrid_ccm(*args)

def F_block_regrid_ccm(l2g_data,xmesh,ymesh,
                       oversampling_list,instrum,error_model,
                       k1,k2,k3,xmargin,ymargin,
                       iblock=1,verbose=False):
    '''
    a more compact version of F_regrid_ccm designed for parallel regridding
    l2g_data:
        a l2g_data dictionary compatible with popy
    xmesh:
        lon mesh grid
    ymesh:
        lat mesh grid
    grid_size:
        in degree
    oversampling_list:
        a list of l2(g) variables to be oversampled
    instrum:
        instrument name in popy
    error_model:
        error model in popy
    k1, k2, k3:
        2d super gaussian shape parameter
    xmargin, ymargin:
        factors extending beyond pixel boundary
    iblock:
        indicate block in parallel regridding
    created on 2020/07/19
    '''
    if len(l2g_data['latc']) == 0:
        l3_data = {}
        l3_data['xmesh'] = xmesh
        l3_data['ymesh'] = ymesh
        l3_data['total_sample_weight'] = xmesh*0.
        l3_data['num_samples'] = xmesh*0.
        for ikey in range(len(oversampling_list)):
            l3_data[oversampling_list[ikey]] = xmesh*np.nan
        if 'cloud_fraction' in oversampling_list:
            l3_data['pres_total_sample_weight'] = xmesh*0.
            l3_data['pres_num_samples'] = xmesh*0.
        return l3_data
    import cv2
    from shapely.geometry import Polygon
    sg_kfacx = 2*(np.log(2)**(1/k1/k3))
    sg_kfacy = 2*(np.log(2)**(1/k2/k3))
    nvar_oversampling = len(oversampling_list)
    nl2 = len(l2g_data['latc'])
    xgrid = xmesh[0,:]
    ygrid = ymesh[:,0]
    nrows = len(ygrid)
    ncols = len(xgrid)
    grid_size = np.median(np.abs(np.diff(xgrid)))
    max_ncol = np.array(np.round(360/grid_size),dtype=int)
    # Allocate memory for regrid fields
    total_sample_weight = np.zeros(xmesh.shape)
    num_samples = np.zeros(xmesh.shape)
    sum_aboves = []
    for n in range(nvar_oversampling):
        sum_aboves.append(np.zeros(xmesh.shape))
    # To only average cloud pressure using pixels where cloud fraction > 0.0
    pres_total_sample_weight = np.zeros(xmesh.shape)
    pres_num_samples = np.zeros(xmesh.shape)
    pres_sum_aboves = np.zeros(xmesh.shape)
    
    # Move as much as possible outside loop
    if instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B",\
                            "SCIAMACHY","TROPOMI","OMPS-N20"}:
        # Set 
        latc = l2g_data['latc']
        lonc = l2g_data['lonc']
        latr = l2g_data['latr']
        lonr = l2g_data['lonr']
        # Get lonc/latc center indices
        lonc_index = [np.argmin(np.abs(xgrid-lonc[i])) for i in range(nl2)]
        latc_index = [np.argmin(np.abs(ygrid-latc[i])) for i in range(nl2)]
        # Get East/West indices
        tmp = np.array([F_lon_distance(lonr[:,0].squeeze(),lonc),F_lon_distance(lonr[:,1].squeeze(),lonc)]).T
        west_extent = np.round( np.max(tmp,axis=1)/grid_size*xmargin )
        tmp = np.array([F_lon_distance(lonc,lonr[:,2].squeeze()),F_lon_distance(lonc,lonr[:,3].squeeze())]).T
        east_extent = np.round( np.max(tmp,axis=1)/grid_size*xmargin )
        # Get lists of indices
        lon_index = [bound_arr(lonc_index[i]-west_extent[i],lonc_index[i]+east_extent[i]+1,max_ncol,ncols)  for i in range(nl2)]
        # The western most longitude
        patch_west = [xgrid[lon_index[i][0]] for i in range(nl2)]
        # Get north/south indices
        north_extent = np.ceil( (latr.max(axis=1)-latr.min(axis=1))/2/grid_size*ymargin)
        south_extent = north_extent
        # List of latitude indices
        lat_index = [bound_lat(latc_index[i]-south_extent[i],latc_index[i]+north_extent[i]+1,nrows) for i in range(nl2)]
        # This might be faster
        patch_lonr = np.array([lonr[i,:] - patch_west[i] for i in range(nl2)]) ; #patch_lonr[patch_lonr<0.0] += 360.0
        patch_lonc = lonc - patch_west ; #patch_lonc[patch_lonc<0.0] += 360.0
        area_weight = [Polygon(np.column_stack([patch_lonr[i,:],latr[i,:].squeeze()])).area for i in range(nl2)]
        # Compute transforms for SG outside loop
        vlist = np.zeros((nl2,4,2),dtype=np.float32)
        for n in range(4):
            vlist[:,n,0] = patch_lonr[:,n] - patch_lonc[:]
            vlist[:,n,1] = latr[:,n] - latc[:]
        xvector  = np.mean(vlist[:,2:4,:],axis=1) - np.mean(vlist[:,0:2,:],axis=1)
        yvector = np.mean(vlist[:,1:3,:],axis=1) - np.mean(vlist[:,[0,3],:],axis=1)
        fwhmx = np.linalg.norm(xvector,axis=1)
        fwhmy = np.linalg.norm(yvector,axis=1)
        fixedPoints = np.array([[-fwhmx,-fwhmy],[-fwhmx,fwhmy],[fwhmx,fwhmy],[fwhmx,-fwhmy]],dtype=np.float32).transpose((2,0,1))/2.0
        tform = [cv2.getPerspectiveTransform(vlist[i,:,:].squeeze(),fixedPoints[i,:,:].squeeze()) for i in range(nl2)]
        
    elif instrum in {"IASI","CrIS"}:
        # Set 
        latc = l2g_data['latc']
        lonc = l2g_data['lonc']
        u = l2g_data['u']
        v = l2g_data['v']
        t = l2g_data['t']
        lonc_index = [np.argmin(np.abs(xgrid-lonc[i])) for i in range(nl2)]
        latc_index = [np.argmin(np.abs(ygrid-latc[i])) for i in range(nl2)]
        # Get East/West indices
        minlon_e = np.zeros((nl2))
        minlat_e = np.zeros((nl2))
        for i in range(nl2):
            X, minlon_e[i], minlat_e[i] = F_ellipse(v[i],u[i],t[i],10)
        west_extent = np.round(np.abs(minlon_e)/grid_size*xmargin)
        east_extent = west_extent
        # Get lists of indices
        lon_index = [bound_arr(lonc_index[i]-west_extent[i],lonc_index[i]+east_extent[i]+1,max_ncol,ncols)  for i in range(nl2)]
        # The western most longitude
        patch_west = [xgrid[lon_index[i][0]] for i in range(nl2)]
        # Get north/south indices
        north_extent = np.ceil(np.abs(minlat_e)/grid_size*xmargin)
        south_extent = north_extent
        # List of latitude indices
        lat_index = [bound_lat(latc_index[i]-south_extent[i],latc_index[i]+north_extent[i]+1,nrows) for i in range(nl2)]
        # This might be faster
        patch_lonc = lonc - patch_west ; #patch_lonc[patch_lonc<0.0] += 360.0
        area_weight = u*v
        fwhmx = 2*v
        fwhmy = 2*u
        
    else:
        print(instrum+' is not supported for regridding yet!')
        return
    # Compute uncertainty weights
    if error_model == "square":
        uncertainty_weight = l2g_data['column_uncertainty']**2
    elif error_model == "log":
        uncertainty_weight = np.log10(l2g_data['column_uncertainty'])
    else:
        uncertainty_weight = l2g_data['column_uncertainty']
    # Cloud Fraction
    if 'cloud_fraction' in oversampling_list:
        cloud_fraction = l2g_data['cloud_fraction']
    # Pull out grid variables from dictionary as it is slow to access
    grid_flds = np.zeros((nl2,nvar_oversampling)) ; pcld_idx = -1
    for n in range(nvar_oversampling):
        grid_flds[:,n] = l2g_data[oversampling_list[n]]
        if oversampling_list[n] == 'cloud_pressure':
            pcld_idx = n
        # Apply log to variable if error model is log
        if(error_model == 'log') and (oversampling_list[n] == 'column_amount'):
            grid_flds[:,n] = np.log10(grid_flds[:,n])
        #t1 = time.time()
    sg_wx = fwhmx/sg_kfacx
    sg_wy = fwhmy/sg_kfacy
    # Init point counter for logger
    count = 0
    for il2 in range(nl2):
        ijmsh = np.ix_(lat_index[il2],lon_index[il2])
        patch_xmesh = xmesh[ijmsh] - patch_west[il2]
        #patch_xmesh[patch_xmesh<0.0] += 360.0
        patch_ymesh = ymesh[ijmsh] - latc[il2]
        if instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B",\
                                "SCIAMACHY","TROPOMI","OMPS-N20"}:
            xym1 = np.column_stack((patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten()))
            xym2 = np.hstack((xym1,np.ones((patch_xmesh.size,1)))).dot(tform[il2].T)[:,0:2]
        elif instrum in {"IASI","CrIS"}:
            rotation_matrix = np.array([[np.cos(-t[il2]), -np.sin(-t[il2])],[np.sin(-t[il2]),  np.cos(-t[il2])]])
            xym1 = np.array([patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten()])#np.column_stack((patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten())).T
            xym2 = rotation_matrix.dot(xym1).T
            
        SG = np.exp(-(np.power( np.power(np.abs(xym2[:,0]/sg_wx[il2]),k1)           \
                                  +np.power(np.abs(xym2[:,1]/sg_wy[il2]),k2),k3)) )
        SG = SG.reshape(patch_xmesh.shape)
        # Update Number of samples
        num_samples[ijmsh] += SG
        # Only bother doing this if regridding cloud pressure
        if 'cloud_fraction' in oversampling_list:
            if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                pres_num_samples[ijmsh] += SG
        # The weights
        tmp_wt = SG/area_weight[il2]/uncertainty_weight[il2]
        # Update total weights
        total_sample_weight[ijmsh] += tmp_wt
        # This only needs to be done if we are gridding pressure
        if 'cloud_fraction' in oversampling_list:
            if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                pres_total_sample_weight[ijmsh] += tmp_wt
        # Update the desired grid variables
        for ivar in range(nvar_oversampling):
            sum_aboves[ivar][ijmsh] += tmp_wt[:,:]*grid_flds[il2,ivar]
        if 'cloud_fraction' in oversampling_list:
            if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                pres_sum_aboves[ijmsh] += tmp_wt[:,:]*grid_flds[il2,pcld_idx]
        if il2 == count*np.round(nl2/10.):
            if verbose:print('block %d'%iblock+' %d%% finished' %(count*10))
            count = count + 1
        
    if verbose:print('block %d'%iblock+' completed at '+datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    l3_data = {}
    np.seterr(divide='ignore', invalid='ignore')
    for ikey in range(len(oversampling_list)):
        l3_data[oversampling_list[ikey]] = sum_aboves[ikey][:,:].squeeze()\
        /total_sample_weight
        # Special case for cloud pressure (only considere pixels with
        # cloud fraction > 0.0
        if oversampling_list[ikey] == 'cloud_pressure':
            l3_data[oversampling_list[ikey]] = pres_sum_aboves[:,:]\
            /pres_total_sample_weight
    # Make cloud pressure = 0 where cloud fraction = 0
    if 'cloud_fraction' in oversampling_list and 'cloud_pressure' in oversampling_list:
        f1 = (l3_data['cloud_fraction'] == 0.0)
        l3_data['cloud_pressure'][f1] = 0.0
    
    # Set quality flag based on the number of samples
    # It has already being initialized to fill value
    # of 2
    quality_flag = np.full((nrows,ncols),2,dtype=np.int8)
    quality_flag[num_samples >= 0.1] = 0
    quality_flag[(num_samples > 1.e-6) & (num_samples < 0.1)] = 1
    
    l3_data['xmesh'] = xmesh
    l3_data['ymesh'] = ymesh
    l3_data['total_sample_weight'] = total_sample_weight
    l3_data['num_samples'] = num_samples
    if 'cloud_fraction' in oversampling_list:
        l3_data['pres_total_sample_weight'] = pres_total_sample_weight
        l3_data['pres_num_samples'] = pres_num_samples
    return l3_data

class popy(object):
    
    def __init__(self,instrum,product,\
                 grid_size=0.1,west=-180,east=180,south=-90,north=90,\
                 start_year=1995,start_month=1,start_day=1,\
                 start_hour=0,start_minute=0,start_second=0,\
                 end_year=2025,end_month=12,end_day=31,\
                 end_hour=23,end_minute=59,end_second=59,verbose=False):
        
        self.instrum = instrum
        self.product = product
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of popy')
        self.verbose = verbose
        if(instrum == "OMI"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 2
            maxsza = 70
            maxcf = 0.3
            self.maxMDQF = 0
            self.maxEXTQF = 0
            if product == 'H2O':
                maxcf = 0.15
        elif(instrum == "GOME-1"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
        elif(instrum == "SCIAMACHY"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
        elif(instrum == "GOME-2A"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
        elif(instrum == "GOME-2B"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
        elif(instrum == "OMPS-NM"):
            k1 = 6
            k2 = 2
            k3 = 3
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
        elif(instrum == "OMPS-N20"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'amf','cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
            maxcf = 0.3
            self.max_qa_value = 0
        elif(instrum == "TROPOMI"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            if product in ['AI']:
                oversampling_list = ['AI']
            elif product in ['CH4']:
                oversampling_list = ['xch4']
            else:
                oversampling_list = ['column_amount','albedo',\
                                     'cloud_fraction']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 70
            maxcf = 0.3
            self.min_qa_value = 0.5
        elif(instrum == "IASI"):
            k1 = 2
            k2 = 2
            k3 = 9
            error_model = "square"
            oversampling_list = ['column_amount']
            xmargin = 2
            ymargin = 2
            maxsza = 60
            maxcf = 0.25
        elif(instrum == "CrIS"):
            k1 = 2
            k2 = 2
            k3 = 4
            error_model = "log"
            oversampling_list = ['column_amount']
            xmargin = 2
            ymargin = 2
            maxsza = 60
            maxcf = 0.25
            self.mindofs = 0.0
            self.min_Quality_Flag = 3
        elif(instrum == "TES"):
            k1 = 4
            k2 = 4
            k3 = 1
            error_model = "log"
            oversampling_list = ['column_amount']
            xmargin = 2
            ymargin = 2
            maxsza = 60
            maxcf = 0.25
            self.mindofs = 0.1
        else:
            k1 = 2
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount']
            xmargin = 2
            ymargin = 2
            maxsza = 60
            maxcf = 0.3
        
        self.xmargin = xmargin
        self.ymargin = ymargin
        self.maxsza = maxsza
        self.maxcf = maxcf
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.sg_kfacx = 2*(np.log(2)**(1/k1/k3))
        self.sg_kfacy = 2*(np.log(2)**(1/k2/k3))
        self.error_model = error_model
        self.oversampling_list = oversampling_list
        self.grid_size = grid_size
        
        if east < west:
            east = east+360
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        
        xgrid = np.arange(west,east,grid_size,dtype=np.float64)+grid_size/2
        ygrid = np.arange(south,north,grid_size,dtype=np.float64)+grid_size/2
        [xmesh,ymesh] = np.meshgrid(xgrid,ygrid)
        
        xgridr = np.hstack((np.arange(west,east,grid_size),east))
        ygridr = np.hstack((np.arange(south,north,grid_size),north))
        [xmeshr,ymeshr] = np.meshgrid(xgridr,ygridr)
        
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.xgridr = xgridr
        self.ygridr = ygridr
        self.xmeshr = xmeshr
        self.ymeshr = ymeshr
        
        self.nrows = len(ygrid)
        self.ncols = len(xgrid)
        
        start_python_datetime = datetime.datetime(start_year,start_month,start_day,\
                                                  start_hour,start_minute,start_second)
        end_python_datetime = datetime.datetime(end_year,end_month,end_day,\
                                                end_hour,end_minute,end_second)
        
        self.start_python_datetime = start_python_datetime
        self.end_python_datetime = end_python_datetime
        # python iso string is stupid, why no Z?
        self.tstart = start_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.tend = end_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        # most of my data are saved in matlab format, where time is defined as UTC days since 0000, Jan 0
        start_matlab_datenum = (start_python_datetime.toordinal()\
                                +start_python_datetime.hour/24.\
                                +start_python_datetime.minute/1440.\
                                +start_python_datetime.second/86400.+366.)
        
        end_matlab_datenum = (end_python_datetime.toordinal()\
                                +end_python_datetime.hour/24.\
                                +end_python_datetime.minute/1440.\
                                +end_python_datetime.second/86400.+366.)
        self.start_matlab_datenum = start_matlab_datenum
        self.end_matlab_datenum = end_matlab_datenum
        self.show_progress = True
    
    def F_mat_reader(self,mat_filename):
        import scipy.io
        
        mat_data = scipy.io.loadmat(mat_filename)
        
        l2g_data = {}
        for key_name in mat_data['output_subset'].dtype.names:
            if key_name == 'lat':
                l2g_data['latc'] = mat_data['output_subset']['lat'][0][0].flatten()
            elif key_name == 'lon':
                l2g_data['lonc'] = mat_data['output_subset']['lon'][0][0].flatten()
            elif key_name == 'lonr':
                l2g_data['lonr'] = mat_data['output_subset']['lonr'][0][0]
            elif key_name == 'latr':
                l2g_data['latr'] = mat_data['output_subset']['latr'][0][0]
            elif key_name in {'colnh3','colno2','colhcho','colchocho','colco'}:
                l2g_data['column_amount'] = mat_data['output_subset'][key_name][0][0].flatten()
            elif key_name in {'colnh3error','colno2error','colhchoerror','colchochoerror','colcoerror','xch4error'}:
                l2g_data['column_uncertainty'] = mat_data['output_subset'][key_name][0][0].flatten()
            elif key_name in {'ift','ifov'}:
                l2g_data['across_track_position'] = mat_data['output_subset'][key_name][0][0].flatten()
            elif key_name == 'cloudfrac':
                l2g_data['cloud_fraction'] = mat_data['output_subset']['cloudfrac'][0][0].flatten()
            elif key_name == 'utc':
                l2g_data['UTC_matlab_datenum'] = mat_data['output_subset']['utc'][0][0].flatten()
            else:
                l2g_data[key_name] = mat_data['output_subset'][key_name][0][0].flatten()
                #exec(key_name + " =  mat_data['output_subset'][key_name][0][0].flatten()")
                #exec('l2g_data[key_name]=' + key_name)
        
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        
        start_matlab_datenum = self.start_matlab_datenum
        end_matlab_datenum = self.end_matlab_datenum
                
        tmplon = l2g_data['lonc']-west
        tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
        # filter data within the lat/lon box and time interval
        validmask = (tmplon >= 0) & (tmplon <= east-west) &\
        (l2g_data['latc'] >= south) & (l2g_data['latc'] <= north) &\
        (l2g_data['UTC_matlab_datenum'] >= start_matlab_datenum) &\
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum)
        
        nl20 = len(l2g_data['latc'])
        min_time = datedev_py(
                l2g_data['UTC_matlab_datenum'].min()).strftime(
                        "%d-%b-%Y %H:%M:%S")
        max_time = datedev_py(
                l2g_data['UTC_matlab_datenum'].max()).strftime(
                        "%d-%b-%Y %H:%M:%S")
        l2g_data = {k:v[validmask,] for (k,v) in l2g_data.items()}
        nl2 = len(l2g_data['latc'])
        
        self.logger.info('Loading and subsetting file '+mat_filename+'...')
        self.logger.info('containing %d pixels...' %nl20)
        self.logger.info('min observation time at '+min_time)
        self.logger.info('max observation time at '+max_time)
        self.logger.info('%d pixels fall in the spatiotemporal window...' %nl2)
        
        del mat_data    
        self.l2g_data = l2g_data
        self.nl2 = nl2
    
    def F_merge_l2g_data(self,l2g_data0,l2g_data1):
        if not l2g_data0:
            return l2g_data1
        common_keys = set(l2g_data0).intersection(set(l2g_data1))
        for key in common_keys:
            l2g_data0[key] = np.concatenate((l2g_data0[key],l2g_data1[key]),0)
        return l2g_data0
    
    def F_merge_l3_data(self,l3_data0,l3_data1):
        if not l3_data0:
            l3_data = l3_data1
            return l3_data
        common_keys = set(l3_data0).intersection(set(l3_data1))
        l3_data = {}
        for key in common_keys:
            l3_data0[key][np.isnan(l3_data0[key])] = 0.
            l3_data1[key][np.isnan(l3_data1[key])] = 0.
            if key in ['total_sample_weight','pres_total_sample_weight','num_samples','pres_num_samples']:
                l3_data[key] = l3_data0[key]+l3_data1[key]
            elif key in ['xmesh','ymesh']:
                l3_data[key] = l3_data0[key]
            elif key == 'cloud_pressure':
                l3_data[key] = (l3_data0[key]*l3_data0['pres_total_sample_weight']
                +l3_data1[key]*l3_data1['pres_total_sample_weight'])\
                /(l3_data0['pres_total_sample_weight']
                +l3_data1['pres_total_sample_weight'])
            else:
                l3_data[key] = (l3_data0[key]*l3_data0['total_sample_weight']
                +l3_data1[key]*l3_data1['total_sample_weight'])\
                /(l3_data0['total_sample_weight']
                +l3_data1['total_sample_weight'])
        return l3_data
    def F_read_S5P_nc(self,fn,data_fields,data_fields_l2g=[]):
        """ 
        function to read tropomi's level 2 netcdf file to a dictionary
        fn: file name
        data_fields: a list of string containing absolution path of variables to extract
        data_fields_l2g: what do you want to call the variables in the output
        updated on 2019/04/22
        updated on 2019/11/20 to handle SUB.nc
        additional packages:
            netCDF4, conda install -c anaconda netcdf4
        """
        from netCDF4 import Dataset
        ncid = Dataset(fn,'r')
        outp = {}
        for i in range(len(data_fields)):
            tmp = ncid[data_fields[i]]
            tmpdtype = tmp.dtype
            if not data_fields_l2g:
                varname = tmp.name
            else:
                varname = data_fields_l2g[i]
            if tmpdtype is "str":
                outp[varname] = tmp[:]
            else:
                outp[varname] = np.squeeze(tmp[:],axis=0)
        if 'time_utc' in outp.keys():
            UTC_matlab_datenum = np.zeros((len(outp['time_utc']),1),dtype=np.float64)
            for i in range(len(outp['time_utc'])):
                if outp['time_utc'][i]:
                    tmp = datetime.datetime.strptime(outp['time_utc'][i],'%Y-%m-%dT%H:%M:%S.%fZ')
                    UTC_matlab_datenum[i] = (tmp.toordinal()\
                                      +tmp.hour/24.\
                                      +tmp.minute/1440.\
                                      +tmp.second/86400.\
                                      +tmp.microsecond/86400/1000000+366.)
                else:
                    UTC_matlab_datenum[i] = 0;self.logger.warning('empty time stamp!')
            outp['UTC_matlab_datenum'] = np.tile(UTC_matlab_datenum,(1,outp['latc'].shape[1]))
        else: # hcho l2 does not have time_utc
            # the delta_time field of hcho fills all across track position, but ch4 is one per scanline
            if len(outp['delta_time'].shape) == 1:
                outp['delta_time'] = np.tile(outp['delta_time'][...,None],(1,outp['latc'].shape[1]))
            outp['UTC_matlab_datenum'] = (outp['time']+outp['delta_time']/1000.)/86400.+734139.
        
        outp['across_track_position'] = np.tile(np.arange(1.,outp['latc'].shape[1]+1),\
            (outp['latc'].shape[0],1)).astype(np.int16)
        return outp
    
    def F_read_MEaSUREs_nc(self,fn,data_fields,data_fields_l2g=[]):
        """ 
        function to read MEaSURE's level 2 netcdf file to a dictionary
        fn: file name
        data_fields: a list of string containing absolution path of variables to extract
        data_fields_l2g: what do you want to call the variables in the output
        created on 2020/03/03 based on F_read_S5P_nc
        additional packages:
            netCDF4, conda install -c anaconda netcdf4
        """
        from netCDF4 import Dataset
        ncid = Dataset(fn,'r')
        outp = {}
        for i in range(len(data_fields)):
            tmp = ncid[data_fields[i]]
            tmpdtype = tmp.dtype
            if not data_fields_l2g:
                varname = tmp.name
            else:
                varname = data_fields_l2g[i]
            outp[varname] = tmp[:]
        
        if 'time' in outp.keys():
            UTC_matlab_datenum = np.zeros((len(outp['time']),1),dtype=np.float64)
            ref_dt = datetime.datetime.strptime('1993-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
            for i in range(len(outp['time'])):
                if outp['time'][i]:
                    tmp = ref_dt+datetime.timedelta(seconds=outp['time'][i])
                    UTC_matlab_datenum[i] = (tmp.toordinal()\
                                      +tmp.hour/24.\
                                      +tmp.minute/1440.\
                                      +tmp.second/86400.\
                                      +tmp.microsecond/86400/1000000+366.)
                else:
                    UTC_matlab_datenum[i] = 0;self.logger.warning('empty time stamp!')
            outp['UTC_matlab_datenum'] = np.tile(UTC_matlab_datenum,(1,outp['latc'].shape[1]))
        else: 
            # just report error
            if len(outp['delta_time'].shape) == 1:
                outp['delta_time'] = np.tile(outp['delta_time'][...,None],(1,outp['latc'].shape[1]))
            outp['UTC_matlab_datenum'] = (outp['time']+outp['delta_time']/1000.)/86400.+734139.
        
        outp['across_track_position'] = np.tile(np.arange(1.,outp['latc'].shape[1]+1),\
            (outp['latc'].shape[0],1)).astype(np.int16)
        return outp
    
    def F_read_BEHR_h5(self,fn,data_fields,data_fields_l2g=[]):
        
        import h5py
        outp = {}
        swath_count = 0
        f = h5py.File(fn,mode='r')
        for swath in f['Data'].keys():
            try:
                for (i,field) in enumerate(data_fields):
                    if swath_count == 0:
                        outp[data_fields_l2g[i]] = f['Data'][swath][field][:]
                    else:
                        outp[data_fields_l2g[i]] = np.concatenate((outp[data_fields_l2g[i]],f['Data'][swath][field][:]),axis=1)                
                swath_count = swath_count+1
            except:# Exception as e:
                self.logger.warning('BEHR '+swath+' cannot be read!')
        outp['Time'] = np.tile(outp['Time'],(outp['latc'].shape[0],1)).astype(np.float64)#BEHR Time is float32, not accurate
        outp['UTC_matlab_datenum'] = outp['Time']/86400.+727930.
        outp['across_track_position'] = np.tile(np.arange\
                        (1.,outp['latc'].shape[1]+1),\
                        (outp['latc'].shape[0],1)).astype(np.int16)
        f.close()
        return outp        
    
    def F_read_he5(self,fn,swathname,data_fields,geo_fields,data_fields_l2g=[],geo_fields_l2g=[]):
        import h5py
        outp_he5 = {}
        if not data_fields_l2g:
            data_fields_l2g = data_fields
        if not geo_fields_l2g:
            geo_fields_l2g = geo_fields
        with h5py.File(fn,mode='r') as f:
            for i in range(len(data_fields)):
                DATAFIELD_NAME = '/HDFEOS/SWATHS/'+swathname+'/Data Fields/'+data_fields[i]
                data = f[DATAFIELD_NAME]
                try:
                    ScaleFactor = data.attrs['ScaleFactor']
                    Offset = data.attrs['Offset']
                except:
                    ScaleFactor = 1.
                    Offset = 0.
                data = data[:]*ScaleFactor+Offset
                outp_he5[data_fields_l2g[i]] = data
                    
            for i in range(len(geo_fields)):
                DATAFIELD_NAME = '/HDFEOS/SWATHS/'+swathname+'/Geolocation Fields/'+geo_fields[i]
                try:
                    data = f[DATAFIELD_NAME]
                except:
                    self.logger.warning(DATAFIELD_NAME+' does not exist');continue
                data = data[:]
                outp_he5[geo_fields_l2g[i]] = data
            
            
            if 'TimeUTC' in outp_he5.keys():
                TimeUTC = outp_he5['TimeUTC'].astype(np.int)
                # python datetime does not allow vectorization
                UTC_matlab_datenum = np.zeros((TimeUTC.shape[0],1),dtype=np.float64)
                for i in range(TimeUTC.shape[0]):
                    tmp = datetime.datetime(year=TimeUTC[i,0],month=TimeUTC[i,1],day=TimeUTC[i,2],\
                                            hour=TimeUTC[i,3],minute=TimeUTC[i,4],second=TimeUTC[i,5])
                    UTC_matlab_datenum[i] = (tmp.toordinal()\
                                      +tmp.hour/24.\
                                      +tmp.minute/1440.\
                                      +tmp.second/86400.+366.)
                    outp_he5['UTC_matlab_datenum'] = np.tile(UTC_matlab_datenum,(1,outp_he5['latc'].shape[1]))
            else: # omno2 only have "Time", seconds after tai93, per scanline
                outp_he5['Time'] = np.tile(outp_he5['Time'][...,None],(1,outp_he5['latc'].shape[1]))
                outp_he5['UTC_matlab_datenum'] = outp_he5['Time']/86400.+727930.
            
            outp_he5['across_track_position'] = np.tile(np.arange\
                    (1.,outp_he5['latc'].shape[1]+1),\
                    (outp_he5['latc'].shape[0],1)).astype(np.int16)
        return outp_he5
    
    def F_update_popy_with_control_file(self,control_path):
        """ 
        function to update self with parameters found in control.txt
        control_path: absolution path to control.txt
        updated on 2019/04/22
        additional packages: 
            yaml, conda install -c anaconda yaml
        """
        import yaml
        with open(control_path,'r') as stream:
            control = yaml.load(stream)
        l2_list = control['Input Files']['OMHCHO']
        l2_dir = control['Runtime Parameters']['Lv2Dir']
        
        maxsza = float(control['Runtime Parameters']['maxSZA'])
        maxcf = float(control['Runtime Parameters']['maxCfr'])
        west = float(control['Runtime Parameters']['minLon'])
        east = float(control['Runtime Parameters']['maxLon'])
        south = float(control['Runtime Parameters']['minLat'])
        north = float(control['Runtime Parameters']['maxLat'])
        grid_size = float(control['Runtime Parameters']['res'])
        maxMDQF= int(control['Runtime Parameters']['maxMDQF'])
        maxEXTQF= int(control['Runtime Parameters']['maxEXTQF'])
        self.maxsza = maxsza
        self.maxcf = maxcf
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.grid_size = grid_size
        self.maxMDQF = maxMDQF
        self.maxEXTQF = maxEXTQF
        xgrid = np.arange(west,east,grid_size,dtype=np.float64)+grid_size/2
        ygrid = np.arange(south,north,grid_size,dtype=np.float64)+grid_size/2
        [xmesh,ymesh] = np.meshgrid(xgrid,ygrid)

        xgridr = np.hstack((np.arange(west,east,grid_size),east))
        ygridr = np.hstack((np.arange(south,north,grid_size),north))
        [xmeshr,ymeshr] = np.meshgrid(xgridr,ygridr)

        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.xgridr = xgridr
        self.ygridr = ygridr
        self.xmeshr = xmeshr
        self.ymeshr = ymeshr
        self.nrows = len(ygrid)
        self.ncols = len(xgrid)
        
        start_python_datetime = datetime.datetime.strptime(
                control['Runtime Parameters']['StartTime'],'%Y-%m-%dT%H:%M:%Sz')
        end_python_datetime = datetime.datetime.strptime(
                control['Runtime Parameters']['EndTime'],'%Y-%m-%dT%H:%M:%Sz')
        self.start_python_datetime = start_python_datetime
        self.end_python_datetime = end_python_datetime
        
        self.tstart = start_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.tend = end_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        # most of my data are saved in matlab format, where time is defined as UTC days since 0000, Jan 0
        start_matlab_datenum = (start_python_datetime.toordinal()\
                                +start_python_datetime.hour/24.\
                                +start_python_datetime.minute/1440.\
                                +start_python_datetime.second/86400.+366.)
        
        end_matlab_datenum = (end_python_datetime.toordinal()\
                                +end_python_datetime.hour/24.\
                                +end_python_datetime.minute/1440.\
                                +end_python_datetime.second/86400.+366.)
        self.start_matlab_datenum = start_matlab_datenum
        self.end_matlab_datenum = end_matlab_datenum
        self.l2_list = l2_list
        self.l2_dir = l2_dir
        self.logger.info('The following parameters from control.txt will overwrite intital popy values:')
        self.logger.info('maxsza = '+'%s'%maxsza)
        self.logger.info('maxcf  = '+'%s'%maxcf)
        self.logger.info('west   = '+'%s'%west)
        self.logger.info('east   = '+'%s'%east)
        self.logger.info('south  = '+'%s'%south)
        self.logger.info('north  = '+'%s'%north)
        self.logger.info('tstart = '+self.tstart)
        self.logger.info('tend   = '+self.tend)
        self.logger.info('res    = '+'%s'%self.grid_size)
            
    def F_subset_OMHCHO(self,path):
        """ 
        function to subset omi hcho level 2 data, calling self.F_read_he5
        path: directory containing omhcho level 2 files, OR path to control.txt
        updated on 2019/04/23
        updated on 2019/12/17 to handle sub.he5 (but pixel corners are not subset by ges disc)
        updated on 2020/07/09 to adapt full orbit (non-subset) he5
        """
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('OMI-Aura_L2-OMHCHO_'+DATE.strftime("%Ym%m%d")+'t*.he5')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        maxMDQF = self.maxMDQF
        maxEXTQF = self.maxEXTQF
        
        data_fields = ['AMFCloudFraction','AMFCloudPressure','AirMassFactor','Albedo',\
                       'ReferenceSectorCorrectedVerticalColumn','ColumnUncertainty','MainDataQualityFlag',\
                       'PixelCornerLatitudes','PixelCornerLongitudes','FittingRMS']
        data_fields_l2g = ['cloud_fraction','cloud_pressure','amf','albedo',\
                           'column_amount','column_uncertainty','MainDataQualityFlag',\
                           'PixelCornerLatitudes','PixelCornerLongitudes','FittingRMS']
        geo_fields = ['Latitude','Longitude','TimeUTC','SolarZenithAngle',\
                      'TerrainHeight','XtrackQualityFlagsExpanded',\
                      'nTimes_idx','nXtrack_idx']
        geo_fields_l2g = ['latc','lonc','TimeUTC','SolarZenithAngle',\
                          'terrain_height','XtrackQualityFlagsExpanded',\
                      'nTimes_idx','nXtrack_idx']
        swathname = 'OMI Total Column Amount HCHO'
                
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading'+fn_dir)
            outp_he5 = self.F_read_he5(fn_dir,swathname,data_fields,geo_fields,data_fields_l2g,geo_fields_l2g)
            if 'nTimes_idx' in outp_he5.keys():
                along_track_idx = np.concatenate((outp_he5['nTimes_idx'],np.array([outp_he5['nTimes_idx'][-1]+1])))
                across_track_idx = np.concatenate((outp_he5['nXtrack_idx'],np.array([outp_he5['nXtrack_idx'][-1]+1])))
                outp_he5['PixelCornerLatitudes'] = outp_he5['PixelCornerLatitudes'][np.ix_(along_track_idx,across_track_idx)]
                outp_he5['PixelCornerLongitudes'] = outp_he5['PixelCornerLongitudes'][np.ix_(along_track_idx,across_track_idx)]
            f1 = outp_he5['SolarZenithAngle'] <= maxsza
            f2 = outp_he5['cloud_fraction'] <= maxcf
            f3 = outp_he5['MainDataQualityFlag'] <= maxMDQF              
            f4 = outp_he5['latc'] >= south
            f5 = outp_he5['latc'] <= north
            tmplon = outp_he5['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_he5['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_he5['UTC_matlab_datenum'] <= self.end_matlab_datenum
            f10 = outp_he5['XtrackQualityFlagsExpanded'] <= maxEXTQF
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            
            l2g_data0 = {}
            
            Lat_lowerleft = outp_he5['PixelCornerLatitudes'][0:-1,0:-1][validmask]
            Lat_upperleft = outp_he5['PixelCornerLatitudes'][1:,0:-1][validmask]
            Lat_lowerright = outp_he5['PixelCornerLatitudes'][0:-1,1:][validmask]
            Lat_upperright = outp_he5['PixelCornerLatitudes'][1:,1:][validmask]               
            Lon_lowerleft = outp_he5['PixelCornerLongitudes'][0:-1,0:-1][validmask]
            Lon_upperleft = outp_he5['PixelCornerLongitudes'][1:,0:-1][validmask]
            Lon_lowerright = outp_he5['PixelCornerLongitudes'][0:-1,1:][validmask]
            Lon_upperright = outp_he5['PixelCornerLongitudes'][1:,1:][validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_he5.keys():
                if key not in {'MainDataQualityFlag','PixelCornerLatitudes',\
                               'PixelCornerLongitudes','TimeUTC','XtrackQualityFlagsExpanded',\
                               'nTimes_idx','nXtrack_idx'}:
                    l2g_data0[key] = outp_he5[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_OMPSN20HCHO(self,path):
        """ 
        function to subset OMPS-N20 hcho level 2 data, calling self.F_read_S5P_nc
        path:
            l2 data directory, or path to control file
        created on 2020/03/03
        """      
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('OMPS-N20_NMHCHO-L2_v1.0_'+DATE.strftime("%Ym%m%d")+'t*.nc')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        max_qa_value = self.max_qa_value
        
        # absolute path of useful variables in the nc file
        # not sure about cloud fraction
        # the time_utc string is empty?! why are you doing this to the user!
        data_fields = ['/support_data/cloud_fraction',\
                       '/support_data/cloud_pressure',\
               '/geolocation/latitude_bounds',\
               '/geolocation/longitude_bounds',\
               '/geolocation/solar_zenith_angle',\
               '/geolocation/viewing_zenith_angle',\
               '/support_data/albedo',\
               '/geolocation/latitude',\
               '/geolocation/longitude',\
               '/key_science_data/main_data_quality_flag',\
               '/geolocation/time',\
               '/key_science_data/column_amount',\
               '/key_science_data/column_uncertainty']    
        # standardized variable names in l2g file. should map one-on-one to data_fields
        data_fields_l2g = ['cloud_fraction','cloud_pressure','latitude_bounds','longitude_bounds','SolarZenithAngle',\
                           'vza','albedo','latc','lonc','qa_value','time',\
                           'column_amount','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_MEaSUREs_nc(fn_dir,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            f1 = outp_nc['SolarZenithAngle'] <= maxsza
            f2 = outp_nc['cloud_fraction'] <= maxcf
            f3 = outp_nc['qa_value'] <= max_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_S5PAI(self,path,data_fields=[],data_fields_l2g=[],
                       s5p_product='*',whichAI='aerosol_index_340_380'):
        """ 
        function to subset tropomi aerosol level 2 data, calling self.F_read_S5P_nc
        path:
            l2 data directory, or path to control file
        s5p_product:
            choose from RPRO and OFFL, default '*' means all
        data_fields:
            a list of strings indicating which fields in the l2 file to keep
        data_fields_l2g:
            shortened data_fields used in the output dictionary l2g_data
        whichAI:
            'aerosol_index_340_380' or 'aerosol_index_354_380'
        updated on 2020/07/09
        """      
        
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('S5P_'+s5p_product+'_L2__AER_AI_'+DATE.strftime("%Y%m%d")+'T*.nc')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
#        maxsza = self.maxsza
#        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        min_qa_value = self.min_qa_value
        
        if not data_fields:
            # default, absolute path of useful variables in the nc file
            data_fields = ['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
                           '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
                           '/PRODUCT/latitude',\
                           '/PRODUCT/longitude',\
                           '/PRODUCT/qa_value',\
                           '/PRODUCT/time_utc',\
                           '/PRODUCT/'+whichAI,\
                           '/PRODUCT/'+whichAI+'_precision']    
            # standardized variable names in l2g file. should map one-on-one to data_fields
            data_fields_l2g = ['latitude_bounds','longitude_bounds','latc','lonc','qa_value','time_utc',\
                               'AI','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_S5P_nc(fn_dir,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            f3 = outp_nc['qa_value'] >= min_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
            
    def F_subset_S5PNO2(self,path,data_fields=[],data_fields_l2g=[],
                        s5p_product='*',
                        geos_interp_variables=[],geos_time_collection=''):
        """ 
        function to subset tropomi no2 level 2 data, calling self.F_read_S5P_nc
        path:
            l2 data directory, or path to control file
        s5p_product:
            choose from RPRO and OFFL, default '*' means all
        data_fields:
            a list of strings indicating which fields in the l2 file to keep
        data_fields_l2g:
            shortened data_fields used in the output dictionary l2g_data
        geos_interp_variables:
            a list of variables (only 2d fields are supported now) to be 
            resampled from geos fp (has to be subsetted/resaved into .mat). see
            the geos class for geos fp data handling
        geos_time_collection:
            choose from inst3, tavg1, tavg3
        updated on 2019/04/24
        updated on 2019/06/20 to add s5p_product/geos_interp_variables option
        """      
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('S5P_'+s5p_product+'_L2__NO2____'+DATE.strftime("%Y%m%d")+'T*.nc')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        min_qa_value = self.min_qa_value
        
        if not data_fields:
            # default, absolute path of useful variables in the nc file
            data_fields = ['/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_crb_nitrogendioxide_window',\
                           '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
                           '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
                           '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
                           '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
                           '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_nitrogendioxide_window',\
                           '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure',\
                           '/PRODUCT/latitude',\
                           '/PRODUCT/longitude',\
                           '/PRODUCT/qa_value',\
                           '/PRODUCT/time_utc',\
                           '/PRODUCT/nitrogendioxide_tropospheric_column',\
                           '/PRODUCT/nitrogendioxide_tropospheric_column_precision']    
            # standardized variable names in l2g file. should map one-on-one to data_fields
            data_fields_l2g = ['cloud_fraction','latitude_bounds','longitude_bounds','SolarZenithAngle',\
                               'vza','albedo','surface_pressure','latc','lonc','qa_value','time_utc',\
                               'column_amount','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_S5P_nc(fn_dir,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            if geos_interp_variables != []:
                sounding_interp = F_interp_geos_mat(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                                                interp_fields=geos_interp_variables,\
                                                time_collection=geos_time_collection)
                for var in geos_interp_variables:
                    outp_nc[var] = sounding_interp[var]
            f1 = outp_nc['SolarZenithAngle'] <= maxsza
            f2 = outp_nc['cloud_fraction'] <= maxcf
            # ridiculously, qa_value has a scale_factor of 0.01. so error-prone
            f3 = outp_nc['qa_value'] >= min_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
            
    def F_subset_S5PHCHO(self,path,s5p_product='*',geos_interp_variables=[],
                         geos_time_collection=''):
        """ 
        function to subset tropomi hcho level 2 data, calling self.F_read_S5P_nc
        path:
            l2 data directory, or path to control file
        s5p_product:
            choose from RPRO and OFFL, default '*' means all
        geos_interp_variables:
            a list of variables (only 2d fields are supported now) to be 
            resampled from geos fp (has to be subsetted/resaved into .mat). see
            the geos class for geos fp data handling
        geos_time_collection:
            choose from inst3, tavg1, tavg3
        updated on 2019/04/30
        updated on 2019/06/20 to add s5p_product/geos_interp_variables option
        """      
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('S5P_'+s5p_product+'_L2__HCHO___'+DATE.strftime("%Y%m%d")+'T*.nc')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        min_qa_value = self.min_qa_value
        
        # absolute path of useful variables in the nc file
        # not sure about cloud fraction
        # the time_utc string is empty?! why are you doing this to the user!
        data_fields = ['/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_crb',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo',\
               '/PRODUCT/latitude',\
               '/PRODUCT/longitude',\
               '/PRODUCT/qa_value',\
               '/PRODUCT/time',\
               '/PRODUCT/delta_time',\
               '/PRODUCT/formaldehyde_tropospheric_vertical_column',\
               '/PRODUCT/formaldehyde_tropospheric_vertical_column_precision']    
        # standardized variable names in l2g file. should map one-on-one to data_fields
        data_fields_l2g = ['cloud_fraction','latitude_bounds','longitude_bounds','SolarZenithAngle',\
                           'vza','albedo','latc','lonc','qa_value','time','delta_time',\
                           'column_amount','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_S5P_nc(fn_dir,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            if geos_interp_variables != []:
                sounding_interp = F_interp_geos_mat(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                                                interp_fields=geos_interp_variables,
                                                time_collection=geos_time_collection)
                for var in geos_interp_variables:
                    outp_nc[var] = sounding_interp[var]
            f1 = outp_nc['SolarZenithAngle'] <= maxsza
            f2 = outp_nc['cloud_fraction'] <= maxcf
            # ridiculously, qa_value has a scale_factor of 0.01. so error-prone
            f3 = outp_nc['qa_value'] >= min_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
        
    def F_subset_S5PCH4(self,path,if_trop_xch4=False,s5p_product='*',
                        merra2_interp_variables=['TROPPT','PS','U50M','V50M'],
                        merra2_dir='./',
                        geos_interp_variables=[],geos_time_collection=''):
        """ 
        function to subset tropomi ch4 level 2 data, calling self.F_read_S5P_nc
        path: directory containing S5PCH4 level 2 files, OR path to control.txt
        for methane, many of auxiliary data are not saved as I trust qa_value
        path:
            l2 data directory, or path to control file
        if_trop_xch4:
            if calculate tropospheric xch4
        s5p_product:
            choose from RPRO and OFFL, '*' means all
        merra2_interp_fields:
            variables to interpolate from merra2, only 2d fields are supported
        merra2_dir:
            directory where merra2 data are saved
        geos_interp_variables (the geos fp option is obsolete):
            a list of variables (only 2d fields are supported now) to be 
            resampled from geos fp (has to be subsetted/resaved into .mat). see
            the geos class for geos fp data handling
        geos_time_collection:
            choose from inst3, tavg1, tavg3
        updated on 2019/05/08
        updated from 2019/05/24 to add tropospheric xch4
        updated on 2019/06/20 to include more interpolation options from geos fp
        """      
        from scipy.interpolate import interp1d
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('S5P_'+s5p_product+'_L2__CH4____'+DATE.strftime("%Y%m%d")+'T*.nc')
                l2_list = l2_list+flist
            
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        #maxsza = self.maxsza 
        #maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        min_qa_value = self.min_qa_value
        
        # absolute path of useful variables in the nc file
        data_fields = ['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
               '/PRODUCT/latitude',\
               '/PRODUCT/longitude',\
               '/PRODUCT/qa_value',\
               '/PRODUCT/time',\
               '/PRODUCT/delta_time',\
               '/PRODUCT/methane_mixing_ratio',\
               '/PRODUCT/methane_mixing_ratio_bias_corrected',\
               '/PRODUCT/methane_mixing_ratio_precision']    
        # standardized variable names in l2g file. should map one-on-one to data_fields
        data_fields_l2g = ['latitude_bounds','longitude_bounds','SolarZenithAngle',\
                           'vza','latc','lonc','qa_value','time','delta_time',\
                           'column_amount_no_bias_correction','column_amount','column_uncertainty']
        if if_trop_xch4:
             # absolute path of useful variables in the nc file
             data_fields = ['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
                            '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
                            '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
                            '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
                            '/PRODUCT/SUPPORT_DATA/INPUT_DATA/dry_air_subcolumns',\
                            '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure',\
                            '/PRODUCT/SUPPORT_DATA/INPUT_DATA/pressure_interval',\
                            '/PRODUCT/SUPPORT_DATA/INPUT_DATA/methane_profile_apriori',\
                            '/PRODUCT/latitude',\
                            '/PRODUCT/longitude',\
                            '/PRODUCT/qa_value',\
                            '/PRODUCT/time',\
                            '/PRODUCT/delta_time',\
                            '/PRODUCT/methane_mixing_ratio',\
                            '/PRODUCT/methane_mixing_ratio_bias_corrected',\
                            '/PRODUCT/methane_mixing_ratio_precision',\
                            '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_SWIR',\
                            '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude']    
             # standardized variable names in l2g file. should map one-on-one to data_fields
             data_fields_l2g = ['latitude_bounds','longitude_bounds','SolarZenithAngle',\
                                'vza','dry_air_subcolumns','surface_pressure','pressure_interval',
                                'methane_profile_apriori','latc','lonc','qa_value','time','delta_time',\
                                'column_amount_no_bias_correction','column_amount','column_uncertainty',\
                                'albedo','surface_altitude']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_path = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_S5P_nc(fn_path,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            
#            if if_trop_xch4:
#                
#                if 'TROPPT' not in geos_interp_variables:
#                    self.logger.warning('tropopause has to be resampled from geos fp to calculate tropospheric xch4!')
#                    geos_interp_variables = np.concatenate((geos_interp_variables,['TROPPT']),0)
            
            if merra2_interp_variables != []:
                sounding_interp = F_interp_merra2(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                merra2_dir=merra2_dir,\
                                                interp_fields=merra2_interp_variables,\
                                                fn_header='MERRA2_300.tavg1_2d_slv_Nx')
                for var in merra2_interp_variables:
                    outp_nc['merra2_'+var] = sounding_interp[var]
            if geos_interp_variables != []:
                sounding_interp = F_interp_geos_mat(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                                                interp_fields=geos_interp_variables,\
                                                time_collection=geos_time_collection)
                for var in geos_interp_variables:
                    outp_nc[var] = sounding_interp[var]
                outp_nc['merra2_TROPPT'] = outp_nc['TROPPT']
            #f1 = outp_nc['SolarZenithAngle'] <= maxsza
            #f2 = outp_nc['cloud_fraction'] <= maxcf
            # ridiculously, qa_value has a scale_factor of 0.01. so error-prone
            f3 = outp_nc['qa_value'] >= min_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            if if_trop_xch4:
                # calculate trop xch4 using l2g_data0
                l2g_data0['air_column_strat'] = np.zeros(l2g_data0['latc'].shape)
                l2g_data0['air_column_total'] = np.zeros(l2g_data0['latc'].shape)
                l2g_data0['methane_ap_column_strat'] = np.zeros(l2g_data0['latc'].shape)
                for il2 in range(len(l2g_data0['latc'])):
                    cum_air = np.concatenate(([0.],np.cumsum(l2g_data0['dry_air_subcolumns'][il2,].squeeze())))
                    cum_methane = np.concatenate(([0.],np.cumsum(l2g_data0['methane_profile_apriori'][il2,].squeeze())))
                    # model top is 10 Pa, 12 layers, 13 levels
                    plevel = 10.+np.arange(0,13)*l2g_data0['pressure_interval'][il2]
                    tropp = l2g_data0['merra2_TROPPT'][il2]
                    l2g_data0['air_column_total'][il2] = np.sum(l2g_data0['dry_air_subcolumns'][il2,])
                    f = interp1d(plevel,cum_air)
                    l2g_data0['air_column_strat'][il2] = f(tropp)
                    f = interp1d(plevel,cum_methane)
                    l2g_data0['methane_ap_column_strat'][il2] = f(tropp)
                del l2g_data0['dry_air_subcolumns']
                del l2g_data0['methane_profile_apriori']                
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_S5PCO(self,path,s5p_product='*',geos_interp_variables=[],
                        geos_time_collection=''):
        """ 
        function to subset tropomi co level 2 data, calling self.F_read_S5P_nc
        path:
            l2 data directory, or path to control file
        s5p_product:
            choose from RPRO and OFFL, default is combining all (RPRO, OFFL, Near real time)
        geos_interp_variables:
            a list of variables (only 2d fields are supported now) to be 
            resampled from geos fp (has to be subsetted/resaved into .mat). see
            the geos class for geos fp data handling
        geos_time_collection:
            choose from inst3, tavg1, tavg3
        created on 2019/08/12 based on F_subset_S5PNO2
        """      
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                flist = glob.glob('S5P_'+s5p_product+'_L2__CO_____'+DATE.strftime("%Y%m%d")+'T*.nc')
                l2_list = l2_list+flist
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        maxsza = self.maxsza
        #maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        min_qa_value = self.min_qa_value
        
        # absolute path of useful variables in the nc file
        data_fields = ['/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/scattering_optical_thickness_SWIR',\
                       '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column',\
                       '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/height_scattering_layer',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure',\
               '/PRODUCT/latitude',\
               '/PRODUCT/longitude',\
               '/PRODUCT/qa_value',\
               '/PRODUCT/time_utc',\
               '/PRODUCT/carbonmonoxide_total_column',\
               '/PRODUCT/carbonmonoxide_total_column_precision']    
        # standardized variable names in l2g file. should map one-on-one to data_fields
        data_fields_l2g = ['scattering_OD','colh2o','scattering_height','latitude_bounds','longitude_bounds','SolarZenithAngle',\
                           'vza','surface_pressure','latc','lonc','qa_value','time_utc',\
                           'column_amount','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = os.path.join(l2_dir,fn)
            self.logger.info('Loading '+fn)
            try:
                outp_nc = self.F_read_S5P_nc(fn_dir,data_fields,data_fields_l2g)
            except Exception as e:
                self.logger.warning(fn+' gives error:');
                print(e)
                input("Press Enter to continue...")
                continue
            if geos_interp_variables != []:
                sounding_interp = F_interp_geos_mat(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                                                interp_fields=geos_interp_variables,\
                                                time_collection=geos_time_collection)
                for var in geos_interp_variables:
                    outp_nc[var] = sounding_interp[var]
            f1 = outp_nc['SolarZenithAngle'] <= maxsza
            #f2 = outp_nc['cloud_fraction'] <= maxcf
            # ridiculously, qa_value has a scale_factor of 0.01. so error-prone
            f3 = outp_nc['qa_value'] >= min_qa_value              
            f4 = outp_nc['latc'] >= south
            f5 = outp_nc['latc'] <= north
            tmplon = outp_nc['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_nc['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_nc['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            # yep it's indeed messed up
            Lat_lowerleft = np.squeeze(outp_nc['latitude_bounds'][:,:,0])[validmask]
            Lat_upperleft = np.squeeze(outp_nc['latitude_bounds'][:,:,3])[validmask]
            Lat_lowerright = np.squeeze(outp_nc['latitude_bounds'][:,:,1])[validmask]
            Lat_upperright = np.squeeze(outp_nc['latitude_bounds'][:,:,2])[validmask]
            Lon_lowerleft = np.squeeze(outp_nc['longitude_bounds'][:,:,0])[validmask]
            Lon_upperleft = np.squeeze(outp_nc['longitude_bounds'][:,:,3])[validmask]
            Lon_lowerright = np.squeeze(outp_nc['longitude_bounds'][:,:,1])[validmask]
            Lon_upperright = np.squeeze(outp_nc['longitude_bounds'][:,:,2])[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_nc.keys():
                if key not in {'latitude_bounds','longitude_bounds','time_utc','time','delta_time'}:
                    l2g_data0[key] = outp_nc[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_BEHR(self,path,l2_path_structure='OMI_BEHR-DAILY_US_v3-0B_%Y%m/',
                       data_fields=[],data_fields_l2g=[]):
        '''
        subsetting behr no2 level 2 product
        written on 2021/01/14
        '''
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                if l2_path_structure == None:
                    flist = glob.glob('OMI_BEHR-DAILY_US_*'+DATE.strftime("%Y%m%d")+'*.hdf')
                else:
                    flist = glob.glob(DATE.strftime(l2_path_structure)+\
                                      'OMI_BEHR-DAILY_US_*'+DATE.strftime("%Y%m%d")+'*.hdf')
                l2_list = l2_list+flist
            
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        if not data_fields:
            data_fields = ['CloudFraction','CloudPressure','TerrainReflectivity',\
                           'BEHRColumnAmountNO2Trop','ColumnAmountNO2TropStd',
                           'BEHRNO2apriori','BEHRAvgKernels','BEHRPressureLevels',
                           'BEHRQualityFlags','VcdQualityFlags',\
                           'XTrackQualityFlags','BEHRSurfacePressure','BEHRTropopausePressure',
                           'Latitude','Longitude','Time','SolarZenithAngle','FoV75CornerLatitude','FoV75CornerLongitude']
            data_fields_l2g = ['cloud_fraction','cloud_pressure','albedo',\
                               'column_amount','column_uncertainty',
                               'BEHRNO2apriori','BEHRAvgKernels','BEHRPressureLevels',
                               'BEHRQualityFlags','VcdQualityFlags',\
                               'XTrackQualityFlags','surface_pressure','tropopause_pressure',
                               'latc','lonc','Time','SolarZenithAngle','FoV75CornerLatitude','FoV75CornerLongitude']
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        start_date = self.start_python_datetime.date()
        end_date = self.end_python_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        l2g_data = {}
        for fn in l2_list:
            file_path = os.path.join(l2_dir,fn)
            self.logger.info('loading '+fn)
            try:
                outp_h5 = self.F_read_BEHR_h5(file_path,data_fields,data_fields_l2g)
            except:
                self.logger.warning(fn+' cannot be read!')
                continue
            f1 = outp_h5['SolarZenithAngle'] <= maxsza
            f2 = outp_h5['cloud_fraction'] <= maxcf
            f3 = (outp_h5['VcdQualityFlags'] == 0) & \
            ((outp_h5['XTrackQualityFlags'] == 0) | (outp_h5['XTrackQualityFlags'] == 255)) & \
            (outp_h5['BEHRQualityFlags']%2 == 0)
            f4 = outp_h5['latc'] >= south
            f5 = outp_h5['latc'] <= north
            tmplon = outp_h5['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_h5['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_h5['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            # some omno2 fov is not consistently defined
            pixcor_dim = outp_h5['FoV75CornerLatitude'].shape.index(4)
            Lat_lowerleft = np.take(outp_h5['FoV75CornerLatitude'],0,axis=pixcor_dim)[validmask]
            Lat_upperleft = np.take(outp_h5['FoV75CornerLatitude'],3,axis=pixcor_dim)[validmask]
            Lat_lowerright = np.take(outp_h5['FoV75CornerLatitude'],1,axis=pixcor_dim)[validmask]
            Lat_upperright = np.take(outp_h5['FoV75CornerLatitude'],2,axis=pixcor_dim)[validmask]
            Lon_lowerleft = np.take(outp_h5['FoV75CornerLongitude'],0,axis=pixcor_dim)[validmask]
            Lon_upperleft = np.take(outp_h5['FoV75CornerLongitude'],3,axis=pixcor_dim)[validmask]
            Lon_lowerright = np.take(outp_h5['FoV75CornerLongitude'],1,axis=pixcor_dim)[validmask]
            Lon_upperright = np.take(outp_h5['FoV75CornerLongitude'],2,axis=pixcor_dim)[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_h5.keys():
                if key not in {'VcdQualityFlags','XTrackQualityFlags','FoV75CornerLatitude','FoV75CornerLongitude','TimeUTC','BEHRQualityFlags'}:
                    l2g_data0[key] = outp_h5[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
        
    def F_subset_OMNO2(self,path,l2_path_structure=None,
                       data_fields=[],data_fields_l2g=[]):
        """ 
        function to subset omno2, nasa sp level 2 data, calling self.F_read_he5
        path:
            l2 data root directory, or path to control file
        l2_path_structure:
            None by default, indicating individual files are directly under path
            '%Y/' if files are like l2_dir/2019/*.he5
            '%Y/%m/%d/' if files are like l2_dir/2019/05/01/*.he5
        data_fields:
            a list of strings indicating which fields in the l2 file to keep
        data_fields_l2g:
            shortened data_fields used in the output dictionary l2g_data
        updated on 2019/07/17
        modified on 2020/05/19 to include data_fields as input
        """      
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                if l2_path_structure == None:
                    flist = glob.glob('OMI-Aura_L2-OMNO2_'+DATE.strftime("%Ym%m%d")+'*.he5')
                else:
                    flist = glob.glob(DATE.strftime(l2_path_structure)+\
                                      'OMI-Aura_L2-OMNO2_'+DATE.strftime("%Ym%m%d")+'*.he5')
                l2_list = l2_list+flist
            
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        if not data_fields:
            data_fields = ['CloudFraction','CloudPressure','TerrainReflectivity',\
                           'ColumnAmountNO2Trop','ColumnAmountNO2TropStd','VcdQualityFlags',\
                           'XTrackQualityFlags','TerrainPressure','TropopausePressure']
            data_fields_l2g = ['cloud_fraction','cloud_pressure','albedo',\
                               'column_amount','column_uncertainty','VcdQualityFlags',\
                               'XTrackQualityFlags','surface_pressure','tropopause_pressure']
        geo_fields = ['Latitude','Longitude','Time','SolarZenithAngle','FoV75CornerLatitude','FoV75CornerLongitude']
        geo_fields_l2g = ['latc','lonc','Time','SolarZenithAngle','FoV75CornerLatitude','FoV75CornerLongitude']
        swathname = 'ColumnAmountNO2'
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        start_date = self.start_python_datetime.date()
        end_date = self.end_python_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        l2g_data = {}
        for fn in l2_list:
            file_path = os.path.join(l2_dir,fn)
            self.logger.info('loading '+fn)
            try:
                outp_he5 = self.F_read_he5(file_path,swathname,data_fields,geo_fields,data_fields_l2g,geo_fields_l2g)
            except:
                self.logger.warning(fn+' cannot be read!')
                continue
            f1 = outp_he5['SolarZenithAngle'] <= maxsza
            f2 = outp_he5['cloud_fraction'] <= maxcf
            f3 = (outp_he5['VcdQualityFlags'] == 0) & \
            ((outp_he5['XTrackQualityFlags'] == 0) | (outp_he5['XTrackQualityFlags'] == 255))
            f4 = outp_he5['latc'] >= south
            f5 = outp_he5['latc'] <= north
            tmplon = outp_he5['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_he5['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_he5['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            # some omno2 fov is not consistently defined
            pixcor_dim = outp_he5['FoV75CornerLatitude'].shape.index(4)
            Lat_lowerleft = np.take(outp_he5['FoV75CornerLatitude'],0,axis=pixcor_dim)[validmask]
            Lat_upperleft = np.take(outp_he5['FoV75CornerLatitude'],3,axis=pixcor_dim)[validmask]
            Lat_lowerright = np.take(outp_he5['FoV75CornerLatitude'],1,axis=pixcor_dim)[validmask]
            Lat_upperright = np.take(outp_he5['FoV75CornerLatitude'],2,axis=pixcor_dim)[validmask]
            Lon_lowerleft = np.take(outp_he5['FoV75CornerLongitude'],0,axis=pixcor_dim)[validmask]
            Lon_upperleft = np.take(outp_he5['FoV75CornerLongitude'],3,axis=pixcor_dim)[validmask]
            Lon_lowerright = np.take(outp_he5['FoV75CornerLongitude'],1,axis=pixcor_dim)[validmask]
            Lon_upperright = np.take(outp_he5['FoV75CornerLongitude'],2,axis=pixcor_dim)[validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_he5.keys():
                if key not in {'VcdQualityFlags','XTrackQualityFlags','FoV75CornerLatitude','FoV75CornerLongitude','TimeUTC'}:
                    l2g_data0[key] = outp_he5[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_OMH2O(self,path,l2_path_structure=None):
        """ 
        function to subset omi h2o level 2 data, calling self.F_read_he5
        path:
            l2 data root directory, or path to control file
        l2_path_structure:
            None by default, indicating individual files are directly under path
            '%Y/' if files are like l2_dir/2019/*.he5
            '%Y/%m/%d/' if files are like l2_dir/2019/05/01/*.he5
        updated on 2019/06/10
        """
        # find out list of l2 files to subset
        if os.path.isfile(path):
            self.F_update_popy_with_control_file(path)
            l2_list = self.l2_list
            l2_dir = self.l2_dir
        else:
            import glob
            l2_dir = path
            l2_list = []
            cwd = os.getcwd()
            os.chdir(l2_dir)
            start_date = self.start_python_datetime.date()
            end_date = self.end_python_datetime.date()
            days = (end_date-start_date).days+1
            DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
            for DATE in DATES:
                if l2_path_structure == None:
                    flist = glob.glob('OMI-Aura_L2-OMH2O_'+DATE.strftime("%Ym%m%d")+'*.he5')
                else:
                    flist = glob.glob(DATE.strftime(l2_path_structure)+\
                                      'OMI-Aura_L2-OMH2O_'+DATE.strftime("%Ym%m%d")+'*.he5')
                l2_list = l2_list+flist
            
            os.chdir(cwd)
            self.l2_dir = l2_dir
            self.l2_list = l2_list
        
        data_fields = ['AMFCloudFraction','AMFCloudPressure','AirMassFactor','Albedo',\
                       'ColumnAmountDestriped','ColumnUncertainty','MainDataQualityFlag',\
                       'PixelCornerLatitudes','PixelCornerLongitudes','FittingRMS']
        # omh2o on aura avdc have no ColumnAmountDestriped field?!
        data_fields = ['AMFCloudFraction','AMFCloudPressure','AirMassFactor','Albedo',\
                       'ColumnAmount','ColumnUncertainty','MainDataQualityFlag',\
                       'PixelCornerLatitudes','PixelCornerLongitudes','FittingRMS']
        data_fields_l2g = ['cloud_fraction','cloud_pressure','amf','albedo',\
                           'column_amount','column_uncertainty','MainDataQualityFlag',\
                           'PixelCornerLatitudes','PixelCornerLongitudes','FittingRMS']
        geo_fields = ['Latitude','Longitude','TimeUTC','SolarZenithAngle','TerrainHeight',\
                      'nTimes_idx','nXtrack_idx']
        geo_fields_l2g = ['latc','lonc','TimeUTC','SolarZenithAngle','terrain_height',\
                      'nTimes_idx','nXtrack_idx']
        swathname = 'OMI Total Column Amount H2O'
        maxsza = self.maxsza
        maxcf = self.maxcf
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        start_date = self.start_python_datetime.date()
        end_date = self.end_python_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        l2g_data = {}
        for fn in l2_list:
            file_path = os.path.join(l2_dir,fn)
            self.logger.info('loading '+fn)
            outp_he5 = self.F_read_he5(file_path,swathname,data_fields,geo_fields,data_fields_l2g,geo_fields_l2g)
            f1 = outp_he5['SolarZenithAngle'] <= maxsza
            f2 = outp_he5['cloud_fraction'] <= maxcf
            f3 = outp_he5['MainDataQualityFlag'] == 0              
            f4 = outp_he5['latc'] >= south
            f5 = outp_he5['latc'] <= north
            tmplon = outp_he5['lonc']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp_he5['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp_he5['UTC_matlab_datenum'] <= self.end_matlab_datenum
            f10 = outp_he5['FittingRMS'] < 0.005
            f11 = outp_he5['cloud_pressure'] > 750.
            validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9 & f10 & f11
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            # python vs. matlab orders are messed up.
            Lat_lowerleft = outp_he5['PixelCornerLatitudes'][0:-1,0:-1][validmask]
            Lat_upperleft = outp_he5['PixelCornerLatitudes'][1:,0:-1][validmask]
            Lat_lowerright = outp_he5['PixelCornerLatitudes'][0:-1,1:][validmask]
            Lat_upperright = outp_he5['PixelCornerLatitudes'][1:,1:][validmask]               
            Lon_lowerleft = outp_he5['PixelCornerLongitudes'][0:-1,0:-1][validmask]
            Lon_upperleft = outp_he5['PixelCornerLongitudes'][1:,0:-1][validmask]
            Lon_lowerright = outp_he5['PixelCornerLongitudes'][0:-1,1:][validmask]
            Lon_upperright = outp_he5['PixelCornerLongitudes'][1:,1:][validmask]
            l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
            l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
            for key in outp_he5.keys():
                if key not in {'MainDataQualityFlag','PixelCornerLatitudes','PixelCornerLongitudes','TimeUTC'}:
                    l2g_data0[key] = outp_he5[key][validmask]
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_OMCHOCHO(self,l2_dir):
       import glob
       data_fields = ['AMFCloudFraction','AMFCloudPressure','AirMassFactor','Albedo',\
                      'ColumnAmountDestriped','ColumnUncertainty','MainDataQualityFlag',\
                      'PixelCornerLatitudes','PixelCornerLongitudes']
       data_fields_l2g = ['cloud_fraction','cloud_pressure','amf','albedo',\
                          'column_amount','column_uncertainty','MainDataQualityFlag',\
                          'PixelCornerLatitudes','PixelCornerLongitudes']
       geo_fields = ['Latitude','Longitude','TimeUTC','SolarZenithAngle','TerrainHeight']
       geo_fields_l2g = ['latc','lonc','TimeUTC','SolarZenithAngle','terrain_height']
       swathname = 'OMI Total Column Amount CHOCHO'
       maxsza = self.maxsza
       maxcf = self.maxcf
       west = self.west
       east = self.east
       south = self.south
       north = self.north
       start_date = self.start_python_datetime.date()
       end_date = self.end_python_datetime.date()
       days = (end_date-start_date).days+1
       DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
       l2g_data = {}
       for DATE in DATES:
           date_dir = l2_dir+DATE.strftime("%Y/%m/%d/")
           flist = glob.glob(date_dir+'*.he5')
           for fn in flist:
               if self.show_progress:
                   print('Loading '+fn)
               outp_he5 = self.F_read_he5(fn,swathname,data_fields,geo_fields,data_fields_l2g,geo_fields_l2g)
               f1 = outp_he5['SolarZenithAngle'] <= maxsza
               f2 = outp_he5['cloud_fraction'] <= maxcf
               f3 = outp_he5['MainDataQualityFlag'] == 0              
               f4 = outp_he5['latc'] >= south
               f5 = outp_he5['latc'] <= north
               tmplon = outp_he5['lonc']-west
               tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
               f6 = tmplon >= 0
               f7 = tmplon <= east-west
               f8 = outp_he5['UTC_matlab_datenum'] >= self.start_matlab_datenum
               f9 = outp_he5['UTC_matlab_datenum'] <= self.end_matlab_datenum
               validmask = f1 & f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
               if self.show_progress:
                   print('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
               l2g_data0 = {}
               # python vs. matlab orders are messed up. 
               Lat_lowerleft = outp_he5['PixelCornerLatitudes'][0:-1,0:-1][validmask]
               Lat_upperleft = outp_he5['PixelCornerLatitudes'][1:,0:-1][validmask]
               Lat_lowerright = outp_he5['PixelCornerLatitudes'][0:-1,1:][validmask]
               Lat_upperright = outp_he5['PixelCornerLatitudes'][1:,1:][validmask]               
               Lon_lowerleft = outp_he5['PixelCornerLongitudes'][0:-1,0:-1][validmask]
               Lon_upperleft = outp_he5['PixelCornerLongitudes'][1:,0:-1][validmask]
               Lon_lowerright = outp_he5['PixelCornerLongitudes'][0:-1,1:][validmask]
               Lon_upperright = outp_he5['PixelCornerLongitudes'][1:,1:][validmask]
               l2g_data0['latr'] = np.column_stack((Lat_lowerleft,Lat_upperleft,Lat_upperright,Lat_lowerright))
               l2g_data0['lonr'] = np.column_stack((Lon_lowerleft,Lon_upperleft,Lon_upperright,Lon_lowerright))
               for key in outp_he5.keys():
                   if key not in {'MainDataQualityFlag','PixelCornerLatitudes','PixelCornerLongitudes','TimeUTC'}:
                       l2g_data0[key] = outp_he5[key][validmask]
               l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
       self.l2g_data = l2g_data
       if not l2g_data:
           self.nl2 = 0
       else:
           self.nl2 = len(l2g_data['latc'])
    
    def F_subset_TESNH3(self,path):
        """ 
        function to subset TES NH3 lite files, foreshadowing future work on CrIS
        latr/lonr are not support as they are not available from lite file
        path:
            l2 data root directory, only flat l2 file structure is supported
        created on 2019/08/13
        """      
        # find out list of l2 files to subset
        import glob
        l2_dir = path
        l2_list = []
        cwd = os.getcwd()
        os.chdir(l2_dir)
        start_date = self.start_python_datetime.date()
        end_date = self.end_python_datetime.date()
        start_year = start_date.year
        start_month = start_date.month
        end_year = end_date.year
        end_month = end_date.month
        for iyear in range(start_year,end_year+1):
            for imonth in range(13):
                if iyear == start_year and imonth < start_month:
                    continue
                if iyear == end_year and imonth > end_month:
                    continue
                flist = glob.glob('TES-Aura_L2-NH3-Nadir_%04d'%iyear+'-%02d'%imonth+'*.nc')
                l2_list = l2_list+flist
            
        os.chdir(cwd)
        self.l2_dir = l2_dir
        self.l2_list = l2_list
        
        varnames = ['AveragingKernel','DOFs','DayNightFlag','LandFlag','Latitude',\
                    'Longitude','ObservationErrorCovariance','Pressure','Quality',\
                    'Species','Time']
        
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        l2g_data = {}
        for fn in l2_list:
            file_path = os.path.join(l2_dir,fn)
            self.logger.info('loading '+fn)
            try:
                outp = F_ncread_selective(file_path,varnames)
            except:
                self.logger.warning(fn+' cannot be read!')
                continue
            outp['UTC_matlab_datenum'] = outp['Time']/86400.+727930.
            f2 = outp['DOFs'] >= 0.1#self.mindofs
            f3 = (outp['Quality'] == 1) & \
            (outp['LandFlag'] == 1) & (outp['DayNightFlag'] == 1)
            f4 = outp['Latitude'] >= south
            f5 = outp['Latitude'] <= north
            tmplon = outp['Longitude']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            
            nobs = np.sum(validmask)
            pressure0 = outp['Pressure'][validmask,]
            xretv0 = outp['Species'][validmask,]
            noise_error0 = outp['ObservationErrorCovariance'][validmask,]
            ak0 = outp['AveragingKernel'][validmask,]
            ak_colm = 0*xretv0;
            tot_col_test = np.zeros((nobs))
            sfcvmr = np.zeros((nobs))
            ps = np.zeros((nobs))
            noise_error_colm = np.zeros((nobs))
            latc = outp['Latitude'][validmask]
            lonc = outp['Longitude'][validmask]
            
            # loop over observations
            for io in range(nobs):
                index = (pressure0[io,] > 0)
                pressure = pressure0[io,index]
                nlev = len(pressure)
                dp = np.zeros((nlev))
                dp[0] = (pressure[0]-pressure[1])/2
                for ip in range(1,nlev-1):
                    dp[ip] = (pressure[ip-1]-pressure[ip])/2+(pressure[ip]-pressure[ip+1])/2
                dp[nlev-1] = pressure[nlev-2]-pressure[nlev-1]
                trans = 2.12e22*dp
                # calculate column AK
                xretv = xretv0[io,index]
                ak = ak0[io,][np.ix_(index,index)]
                noise_error = noise_error0[io,][np.ix_(index,index)]
                ak_colm[io,index] = (trans*xretv).transpose().dot(ak)
                # calculate errors
                xarr = np.diag(xretv)
                sx = (xarr.dot(noise_error)).dot(xarr)
                noise_error_colm[io] = np.sqrt((trans.transpose().dot(sx)).dot(trans))
                tot_col_test[io] = np.sum(trans*xretv)
                sfcvmr[io] = xretv[0]
                ps[io] = pressure[0]
            # some omno2 fov is not consistently defined
            l2g_data0['latc'] = latc
            l2g_data0['lonc'] = lonc
            l2g_data0['colnh3'] = tot_col_test
            l2g_data0['colnh3error'] = noise_error_colm
            l2g_data0['surface_pressure'] = ps
            l2g_data0['sfcvmr'] = sfcvmr
            l2g_data0['utc'] = outp['UTC_matlab_datenum'][validmask]
            l2g_data0['dofs'] = outp['DOFs'][validmask]
            
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_subset_CrISNH3(self,path,l2_path_structure='%Y/%m/%d/',ellipse_lut_path='CrIS_footprint.mat'):
        """ 
        function to subset CrIS NH3 level2 files
        path:
            l2 data root directory, only flat l2 file structure is supported
        l2_path_structure:
            None indicates that individual files are directly under path;
            '%Y/' if files are like l2_dir/2017/*.nc;
            '%Y/%m/%d/' if files are like l2_dir/2017/05/01/*.nc
        ellipse_lut_path:
            path to a look up table storing u, v, and t data to reconstruct CrIS pixel ellipsis
        created on 2019/10/22
        """      
        # find out list of l2 files to subset
        import glob
        from scipy.io import loadmat
        from scipy.interpolate import RegularGridInterpolator
        l2_dir = path
        l2_list = []
        cwd = os.getcwd()
        os.chdir(l2_dir)
        start_date = self.start_python_datetime.date()
        end_date = self.end_python_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        for DATE in DATES:
            if l2_path_structure == None:
                flist = glob.glob('Combined_NH3_*'+DATE.strftime("%Y%m%d")+'.nc')
            else:
                flist = glob.glob(DATE.strftime(l2_path_structure)+\
                                  'Combined_NH3_*'+DATE.strftime("%Y%m%d")+'.nc')
            l2_list = l2_list+flist
        
        os.chdir(cwd)
        self.l2_dir = l2_dir
        self.l2_list = l2_list
        
        varnames = ['DOF','Day_Night_Flag','LandFraction','Latitude','Longitude',
                    'Quality_Flag','Run_ID','mdate','rvmr','rvmr_error','tot_col','xretv',
                    'total_covariance_error','noise_error_covariance','pressure',
                    'xa','avg_kernel']#,'xa_Type']
        
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        l2g_data = {}
        pixel_lut = loadmat(ellipse_lut_path)
        f_uuu = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['uuu4']) 
        f_vvv = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['vvv4']) 
        f_ttt = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['ttt4']) 
        for fn in l2_list:
            file_path = os.path.join(l2_dir,fn)
            self.logger.info('loading '+fn)
            try:
                outp = F_ncread_selective(file_path,varnames)
            except:
                self.logger.warning(fn+' cannot be read!')
                continue
            outp['UTC_matlab_datenum'] = outp['mdate']+366.
            f2 = outp['DOF'] >= self.mindofs
            f3 = (outp['Quality_Flag'] >= self.min_Quality_Flag) & \
            (outp['Day_Night_Flag'] == 1)
            f4 = outp['Latitude'] >= south
            f5 = outp['Latitude'] <= north
            tmplon = outp['Longitude']-west
            tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
            f6 = tmplon >= 0
            f7 = tmplon <= east-west
            f8 = outp['UTC_matlab_datenum'] >= self.start_matlab_datenum
            f9 = outp['UTC_matlab_datenum'] <= self.end_matlab_datenum
            validmask = f2 & f3 & f4 & f5 & f6 & f7 & f8 & f9
            self.logger.info('You have '+'%s'%np.sum(validmask)+' valid L2 pixels')
            l2g_data0 = {}
            if np.sum(validmask) == 0:
                continue
            
            nobs = np.sum(validmask)
            # work out footprint number
            tmprunID = outp['Run_ID'][validmask]
            tmpfov = np.asarray([np.float(tmprunID[i][-3:]) for i in range(nobs)])
            tmpfor = np.asarray([np.float(tmprunID[i][-8:-4]) for i in range(nobs)])
            pressure0 = outp['pressure'][validmask,]
            xretv0 = outp['xretv'][validmask,]
            noise_error0 = outp['total_covariance_error'][validmask,]
            ak0 = outp['avg_kernel'][validmask,]
            ak_colm = 0*xretv0;
            tot_col_test = np.zeros((nobs))
            sfcvmr = np.zeros((nobs))
            ps = np.zeros((nobs))
            noise_error_colm = np.zeros((nobs))
            latc = outp['Latitude'][validmask]
            lonc = outp['Longitude'][validmask]
            
            # loop over observations
            for io in range(nobs):
                index = (pressure0[io,] > 0)
                pressure = pressure0[io,index]
                nlev = len(pressure)
                dp = np.zeros((nlev))
                dp[0] = (pressure[0]-pressure[1])/2
                for ip in range(1,nlev-1):
                    dp[ip] = (pressure[ip-1]-pressure[ip])/2+(pressure[ip]-pressure[ip+1])/2
                dp[nlev-1] = pressure[nlev-2]-pressure[nlev-1]
                trans = 2.12e16*dp
                # calculate column AK
                xretv = xretv0[io,index]
                ak = ak0[io,][np.ix_(index,index)]
                noise_error = noise_error0[io,][np.ix_(index,index)]
                ak_colm[io,index] = (trans*xretv).transpose().dot(ak)
                # calculate errors
                xarr = np.diag(xretv)
                sx = (xarr.dot(noise_error)).dot(xarr)
                noise_error_colm[io] = np.sqrt((trans.transpose().dot(sx)).dot(trans))
                tot_col_test[io] = np.sum(trans*xretv)
                sfcvmr[io] = xretv[0]
                ps[io] = pressure[0]
            
            l2g_data0['ifov'] = (tmpfor-1)*9+tmpfov
            l2g_data0['latc'] = latc
            l2g_data0['lonc'] = lonc
            # find out elliptical parameters using lookup table            
            l2g_data0['u'] = f_uuu((latc,l2g_data0['ifov']))
            l2g_data0['v'] = f_vvv((latc,l2g_data0['ifov']))
            l2g_data0['t'] = f_ttt((latc,l2g_data0['ifov']))
            l2g_data0['colnh3_simple'] = tot_col_test
            l2g_data0['column_amount'] = outp['tot_col'][validmask]
            l2g_data0['column_uncertainty'] = noise_error_colm
            l2g_data0['surface_pressure'] = ps
            l2g_data0['sfcvmr'] = sfcvmr
            l2g_data0['utc'] = outp['UTC_matlab_datenum'][validmask]
            l2g_data0['dofs'] = outp['DOF'][validmask]
            # a priori type in string format might slow down the whole thing
            #l2g_data0['xa_type'] = outp['xa_Type'][validmask]
            
            l2g_data = self.F_merge_l2g_data(l2g_data,l2g_data0)
        self.l2g_data = l2g_data
        if not l2g_data:
            self.nl2 = 0
        else:
            self.nl2 = len(l2g_data['latc'])
    
    def F_plot_l3(self,plot_field='column_amount',l3_data=None,
                  vmin=None,vmax=None):
        '''
        l3 data plotting utility updated from F_plot_oversampled_variable
        '''
        if l3_data == None:
            l3_data = self.C
        try:
            from mpl_toolkits.basemap import Basemap
            if_map = True
        except:
            self.logger.warning('Basemap cannot be imported! plot without map')
            if_map = False
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        if self.error_model == 'log' and plot_field == 'column_amount':
            plotdata = np.power(10,l3_data[plot_field])
        else:
            plotdata = l3_data[plot_field]
        if if_map:
            m = Basemap(projection= 'cyl',llcrnrlat=self.south,urcrnrlat=self.north,
                        llcrnrlon=self.west,urcrnrlon=self.east,resolution='l')
            m.drawstates(linewidth=0.5)
            m.drawcoastlines(linewidth=0.5)
            
            pc = m.pcolormesh(self.xgrid,self.ygrid,plotdata,latlon=True,cmap='rainbow')
            cb = fig.colorbar(pc,ax=ax,label=plot_field)
        else:
            pc = plt.pcolormesh(self.xgrid,self.ygrid,plotdata)
            cb = fig.colorbar(pc,ax=ax,label=plot_field)
            plt.xlim((self.west,self.east))
            plt.ylim((self.south,self.north))
            m = None
        if vmin != None:
            plt.clim(vmin=vmin)
        if vmax != None:
            plt.clim(vmax=vmax)
        return pc,fig,ax,m,cb
    
    def F_plot_l2g(self,ax=None,plot_field='column_amount',max_day=1,l2g_data=None,
                   alpha=0.7,vmin=None,vmax=None,
                   x_wind_field='era5_u100',y_wind_field='era5_v100',
                   wind_arrow_width=0.01,wind_arrow_scale=20):
        '''
        plot l2g pixels as polygons
        plot_field:
            which field in l2g_data to plot
        max_day:
            only plot limited number of days~layers
        l2g_data:
            l2g data dictionary can be supplied externally
        alpha:
            1 is opaque
        vmin, vmax:
            color limits
        '''
        if l2g_data == None:
            l2g_data = self.l2g_data
        if x_wind_field not in l2g_data.keys():
            self.logger.warning('wind field not in l2g_data! will not plot wind')
            x_wind_field=None
        try:
            from mpl_toolkits.basemap import Basemap
            if_map = True
        except:
            self.logger.warning('Basemap cannot be imported! plot without map')
            if_map = False
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        ax = plt.gca() if ax is None else ax
        fig = plt.gcf()
        plot_index = np.where(l2g_data['UTC_matlab_datenum']<=l2g_data['UTC_matlab_datenum'].min()+max_day)
        if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B","SCIAMACHY","TROPOMI"}:
            verts = [np.array([l2g_data['lonr'][i,:],l2g_data['latr'][i,:]]).T for i in plot_index[0]]
        elif self.instrum in {"IASI","CrIS"}:
            verts = [F_ellipse(l2g_data['v'][i],l2g_data['u'][i],l2g_data['t'][i],20,
                               l2g_data['lonc'][i],l2g_data['latc'][i])[0].T for i in plot_index[0]]
        collection = PolyCollection(verts,
                             array=l2g_data[plot_field],cmap='rainbow',edgecolors='none')
        collection.set_alpha(alpha)
#        fig,ax = plt.subplots()
        if if_map:
            m = Basemap(projection= 'cyl',llcrnrlat=self.south,urcrnrlat=self.north,
                        llcrnrlon=self.west,urcrnrlon=self.east,resolution='l')
            m.drawstates(linewidth=0.5)
            m.drawcoastlines(linewidth=0.5)
            ax.add_collection(collection)
            cb = fig.colorbar(collection,ax=ax,label=plot_field)
        else:
            ax.add_collection(collection)
            cb = fig.colorbar(collection,ax=ax,label=plot_field)
            plt.xlim((self.west,self.east))
            plt.ylim((self.south,self.north))
            m = None
        if vmin != None:
            collection.set_clim(vmin=vmin)
        if vmax != None:
            collection.set_clim(vmax=vmax)
        if x_wind_field != None:
            quiver = plt.quiver(l2g_data['lonc'][plot_index[0]],l2g_data['latc'][plot_index[0]],
                                l2g_data[x_wind_field][plot_index[0]],l2g_data[y_wind_field][plot_index[0]],
                                units='x',width=wind_arrow_width,scale=wind_arrow_scale)
        else:
            quiver=None
        return collection,fig,ax,m,cb,quiver
            

        
    def F_save_l2g_to_mat(self,file_path,data_fields=[],data_fields_l2g=[]):
        """ 
        save l2g dictionary to .mat file
        file_path: 
            absolute path to the .mat file to save
        data_fields and data_fields_l2g: 
            two one-on-one lists of variable names;
            field in data_fields will be saved as field in data_fields_l2g
        updated on 2019/05/26
        """
        if not self.l2g_data:
            self.logger.warning('l2g_data is empty. Nothing to save.')
            return
        
        import scipy.io
        l2g_data = self.l2g_data.copy()
        for i in range(len(data_fields)):
            if data_fields[i] in l2g_data.keys():
                l2g_data[data_fields_l2g[i]] = l2g_data.pop(data_fields[i])
        # reshape 1d arrays to (nl2, 1)
        for key in l2g_data.keys():
            if key not in {'UTC_matlab_datenum','utc','ift','across_track_position','xa_type'}:
                l2g_data[key] = np.float32(l2g_data[key])
            if key not in {'latr','lonr'}:
                l2g_data[key] = l2g_data[key].reshape(len(l2g_data[key]),1)
            else:# otherwise, the order of 2d array is COMPLETELY screwed
                l2g_data[key] = np.asfortranarray(l2g_data[key])
        scipy.io.savemat(file_path,{'output_subset':l2g_data})
        
        
    def F_generalized_SG(self,x,y,fwhmx,fwhmy):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        wx = fwhmx/2/(np.log(2)**(1/k1/k3))
        wy = fwhmy/2/(np.log(2)**(1/k2/k3))
        sg = np.exp(-(np.abs(x/wx)**k1+np.abs(y/wy)**k2)**k3)
        return sg
    
    def F_2D_SG_rotate(self,xmesh,ymesh,x_c,y_c,fwhmx,fwhmy,angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]])
        xym1 = np.array([xmesh.flatten()-x_c,ymesh.flatten()-y_c])
        xym2 = rotation_matrix.dot(xym1)
        sg0 = self.F_generalized_SG(xym2[0,:],xym2[1,:],fwhmx,fwhmy)
        sg = sg0.reshape(xmesh.shape)
        return sg
    
    def F_2D_SG_transform(self,xmesh,ymesh,x_r,y_r,x_c,y_c):
        import cv2
        vList = np.column_stack((x_r-x_c,y_r-y_c))
        leftpoint = np.mean(vList[0:2,:],axis=0)
        rightpoint = np.mean(vList[2:4,:],axis=0)
        uppoint = np.mean(vList[1:3,:],axis=0)
        lowpoint = np.mean(vList[[0,3],:],axis=0)
        xvector = rightpoint-leftpoint
        yvector = uppoint-lowpoint
        
        fwhmx = np.linalg.norm(xvector)
        fwhmy = np.linalg.norm(yvector)
        
        fixedPoints = np.array([[-fwhmx,-fwhmy],[-fwhmx,fwhmy],[fwhmx,fwhmy],
                                [fwhmx,-fwhmy]],dtype=vList.dtype)/2
        tform = cv2.getPerspectiveTransform(vList,fixedPoints)
        
        xym1 = np.column_stack((xmesh.flatten()-x_c,ymesh.flatten()-y_c))
        xym2 = np.hstack((xym1,np.ones((xmesh.size,1)))).dot(tform.T)[:,0:2]
        
        sg0 = self.F_generalized_SG(xym2[:,0],xym2[:,1],fwhmx,fwhmy)
        sg = sg0.reshape(xmesh.shape)
        return sg
    
    def F_construct_ellipse(self,a,b,alpha,npoint):
        t = np.linspace(0.,np.pi*2,npoint)[::-1]
        Q = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
        X = Q.dot(np.vstack((a * np.cos(t),b * np.sin(t))))
        minlon_e = X[0,].min()
        minlat_e = X[1,].min()
        return X, minlon_e, minlat_e
    
    def F_derive_surface_vmr(self,
                             omega_field='column_amount',
                             pblp_field=None,
                             pblh_field='era5_blh',
                             surface_pressure_field='surface_pressure',
                             surface_vmr_field='surface_vmr',
                             scale_height=7500.,
                             gamma=1.,
                             l2g_data=None):
        '''
        acmap surface vmr
        omega_field:
            column amount, has to be in mol/m2 to make sense
        pblp_field:
            thickness of pbl in Pa. will supersede the pblh stuff if provided
        pblh_field:
            pblh in m
        surface_pressure_field:
            surface pressure
        surface_vmr_field:
            surface mixing ratio field name to be saved
        gamma:
            non-dimensional shape number
        l2g_data:
            l2g_data in popy-compatible dict format. by default use self.l2g_data
        '''
        gravity = 9.8 # m/s2
        MA = 0.029 # kg/mol
        if l2g_data == None:
            do_output = False
            l2g_data = self.l2g_data
        else:
            do_output = True
        if 'UTC_matlab_datenum' not in l2g_data.keys():
            l2g_data['UTC_matlab_datenum'] = l2g_data.pop('utc')
        if pblp_field is not None:
            self.logger.info('pblp is provided and will be used')
            pblp = l2g_data[pblp_field]
        else:
            pblp = l2g_data[surface_pressure_field]*(1-np.exp(-l2g_data[pblh_field]/scale_height))
        
        l2g_data[surface_vmr_field] = gravity*MA*l2g_data[omega_field]/gamma/pblp
        if do_output:
            return l2g_data
        else:
            self.l2g_data = l2g_data
        
    def F_regrid_divergence(self,omega_field='column_amount',
                            x_wind_field='era5_u100',y_wind_field='era5_v100',
                            x_surface_wind_field='era5_u10',
                            y_surface_wind_field='era5_v10',
                            surface_pressure_field='surface_pressure',
                            surface_vmr_field='surface_vmr',
                            l2g_data=None,block_length=200,ncores=0,
                            simplify_oversampling_list=True,if_daily=True,
                            do_terrain=False):
        '''
        call F_parallel_regrid to oversample x/y-flux daily and calculate d(x-flux)/dx
        and d(y-flux)dy to form daily divergence map. Average daily divergence to
        get oversampled divergence map over the entire period defined by l2g_data
        omega_field:
            which scalar to calculate divergence
        x/y_wind_field:
            pbl representative horizontal wind
        x/y_surface_wind_field:
            near surface wind
        surface_pressure_field:
            surface pressure
        surface_vmr_field:
            surface mixing ratio
        l2g_data:
            l2g_data in popy-compatible dict format. by default use self.l2g_data
        block_length:
            l3 mesh grid will be cut to square blocks with this length
        ncores:
            number of cores, 0 calls non parallel F_regrid_ccm
        simplify_oversampling_list:
            if True, only oversampling omega_field and its divergence
        if_daily:
            if calculate spatial gradient every day. if False, calculate spatial gradient on the oversampled field
        do_terrain:
            calculate the terrain correction term or not
        created on 2020/08/16
        added terrain on 2020/09/28
        '''
        if l2g_data == None:
            l2g_data = self.l2g_data
        if 'UTC_matlab_datenum' not in l2g_data.keys():
            l2g_data['UTC_matlab_datenum'] = l2g_data.pop('utc')
        oversampling_list_full = self.oversampling_list.copy()
        
        if do_terrain:
            l2g_data['surface_x_flux'] = l2g_data[surface_vmr_field]*l2g_data[x_surface_wind_field]
            l2g_data['surface_y_flux'] = l2g_data[surface_vmr_field]*l2g_data[y_surface_wind_field]
            self.logger.info('calculate the terrain correction term')
            gravity = 9.8 # m/s2
            MA = 0.029 # kg/mol
            if not if_daily:
                self.logger.error('not compatible with non-daily divergence calculation!')
        
        if simplify_oversampling_list:
            self.oversampling_list = [omega_field,'x_flux','y_flux']
            if surface_pressure_field not in self.oversampling_list and do_terrain:
                self.oversampling_list.append(surface_pressure_field)
            if do_terrain:
                self.oversampling_list.append('surface_x_flux')
                self.oversampling_list.append('surface_y_flux')
        else:
            self.oversampling_list.append('x_flux')
            self.oversampling_list.append('y_flux')
            if surface_pressure_field not in self.oversampling_list and do_terrain:
                self.oversampling_list.append(surface_pressure_field)
            if do_terrain:
                self.oversampling_list.append('surface_x_flux')
                self.oversampling_list.append('surface_y_flux')
        
        l2g_data['x_flux'] = l2g_data[omega_field]*l2g_data[x_wind_field]
        l2g_data['y_flux'] = l2g_data[omega_field]*l2g_data[y_wind_field]
        
        
        day_list = np.arange(np.floor(l2g_data['UTC_matlab_datenum'].min()),
                             np.floor(l2g_data['UTC_matlab_datenum'].max())+1)
        # x-grid size in m
        dx_vec = np.cos(self.ygrid/180*np.pi)*111e3*self.grid_size
        # y-grid size in m
        dy = 111e3*self.grid_size
        l3_data = {}
        if not if_daily:
            l3_data = self.F_parallel_regrid(l2g_data=l2g_data,
                                             block_length=block_length,
                                             ncores=ncores)
            # d(x_flux)/dx
            xdiv = np.full(l3_data['x_flux'].shape,np.nan,dtype=np.float64)
            for irow in range(self.nrows):
                for icol in range(2,self.ncols-2):
                    xdiv[irow,icol] = (l3_data['x_flux'][irow,icol-2]
                    -8*l3_data['x_flux'][irow,icol-1]
                    +8*l3_data['x_flux'][irow,icol+1]
                    -l3_data['x_flux'][irow,icol+2])/(12*dx_vec[irow])
            # d(y_flux)/dy
            ydiv = np.full(l3_data['y_flux'].shape,np.nan,dtype=np.float64)
            for icol in range(self.ncols):
                for irow in range(2,self.nrows-2):
                    ydiv[irow,icol] = (l3_data['y_flux'][irow-2,icol]
                    -8*l3_data['y_flux'][irow-1,icol]
                    +8*l3_data['y_flux'][irow+1,icol]
                    -l3_data['y_flux'][irow+2,icol])/(12*dy)
            l3_data['div'] = xdiv+ydiv
            self.oversampling_list = oversampling_list_full
            return l3_data
        for day in day_list:
            mask = np.floor(l2g_data['UTC_matlab_datenum']) == day
            if np.sum(mask) == 0:
                continue
            self.logger.info('regridding daily fluxes on '+datedev_py(day).strftime('%Y%m%d'))
            self.logger.info('there are %d pixels'%(np.sum(mask)))
            daily_l2g_data = {k:v[mask,] for (k,v) in l2g_data.items()}
            daily_l3_data = self.F_parallel_regrid(l2g_data=daily_l2g_data,
                                                   block_length=block_length,
                                                   ncores=ncores)
            # d(x_flux)/dx
            xdiv = np.full(daily_l3_data['x_flux'].shape,np.nan,dtype=np.float64)
            for irow in range(self.nrows):
                for icol in range(2,self.ncols-2):
                    xdiv[irow,icol] = (daily_l3_data['x_flux'][irow,icol-2]
                    -8*daily_l3_data['x_flux'][irow,icol-1]
                    +8*daily_l3_data['x_flux'][irow,icol+1]
                    -daily_l3_data['x_flux'][irow,icol+2])/(12*dx_vec[irow])
            # d(y_flux)/dy
            ydiv = np.full(daily_l3_data['y_flux'].shape,np.nan,dtype=np.float64)
            for icol in range(self.ncols):
                for irow in range(2,self.nrows-2):
                    ydiv[irow,icol] = (daily_l3_data['y_flux'][irow-2,icol]
                    -8*daily_l3_data['y_flux'][irow-1,icol]
                    +8*daily_l3_data['y_flux'][irow+1,icol]
                    -daily_l3_data['y_flux'][irow+2,icol])/(12*dy)
            daily_l3_data['div'] = xdiv+ydiv
            
            if do_terrain:
                # d(p0)/dx
                xdp = np.full(daily_l3_data['x_flux'].shape,np.nan,dtype=np.float64)
                for irow in range(self.nrows):
                    for icol in range(2,self.ncols-2):
                        xdp[irow,icol] = (daily_l3_data[surface_pressure_field][irow,icol-2]
                        -8*daily_l3_data[surface_pressure_field][irow,icol-1]
                        +8*daily_l3_data[surface_pressure_field][irow,icol+1]
                        -daily_l3_data[surface_pressure_field][irow,icol+2])/(12*dx_vec[irow])
                # d(p0)/dy
                ydp = np.full(daily_l3_data['y_flux'].shape,np.nan,dtype=np.float64)
                for icol in range(self.ncols):
                    for irow in range(2,self.nrows-2):
                        ydp[irow,icol] = (daily_l3_data[surface_pressure_field][irow-2,icol]
                        -8*daily_l3_data[surface_pressure_field][irow-1,icol]
                        +8*daily_l3_data[surface_pressure_field][irow+1,icol]
                        -daily_l3_data[surface_pressure_field][irow+2,icol])/(12*dy)
                daily_l3_data['terrain_correction'] = (daily_l3_data['surface_x_flux']*xdp
                             +daily_l3_data['surface_y_flux']*ydp)/gravity/MA
                daily_l3_data.pop('surface_x_flux');
                daily_l3_data.pop('surface_y_flux');
            daily_l3_data.pop('x_flux');
            daily_l3_data.pop('y_flux');
            l3_data = self.F_merge_l3_data(l3_data,daily_l3_data)
        
        self.oversampling_list = oversampling_list_full
        return l3_data
        
    def F_parallel_regrid(self,l2g_data=None,block_length=200,ncores=None):
        '''
        regrid from l2g to l3 in parallel by cutting the l3 mesh into blocks
        l2g_data:
            l2g_data in popy-compatible dict format. by default use self.l2g_data
        block_length:
            l3 mesh grid will be cut to square blocks with this length
        ncores:
            number of cores
        created on 2020/07/19
        fix on 2020/08/17 so multiprocess does not consume all the memory
        '''
        west = self.west ; east = self.east ; south = self.south ; north = self.north
        nrows = self.nrows; ncols = self.ncols
        xmesh = self.xmesh ; ymesh = self.ymesh
#        grid_size = self.grid_size ; 
        xmargin = self.xmargin ; ymargin = self.ymargin
        start_matlab_datenum = self.start_matlab_datenum
        end_matlab_datenum = self.end_matlab_datenum
        oversampling_list = self.oversampling_list.copy()
        if l2g_data == None:
            l2g_data = self.l2g_data
        if 'UTC_matlab_datenum' not in l2g_data.keys():
            l2g_data['UTC_matlab_datenum'] = l2g_data.pop('utc')
        for key in self.oversampling_list:
            if key not in l2g_data.keys():
                oversampling_list.remove(key)
                self.logger.warning('You asked to oversample '+key+', but I cannot find it in your data!')
        self.oversampling_list_final = oversampling_list
#        error_model = self.error_model
        
        if ncores == 0:
            self.logger.info('ncores = 0 means no parallel and calling F_block_regridd_ccm using the entire domain as a block')
            l3_data = F_block_regrid_ccm(l2g_data,xmesh,ymesh,
                       oversampling_list,self.instrum,self.error_model,
                       self.k1,self.k2,self.k3,xmargin,ymargin,
                       iblock=1)
            return l3_data
        
        import multiprocessing
        
        tmplon = l2g_data['lonc']-west
        tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
        # filter data within the lat/lon box and time interval
        validmask = (tmplon >= 0) & (tmplon <= east-west) &\
        (l2g_data['latc'] >= south) & (l2g_data['latc'] <= north) &\
        (l2g_data['UTC_matlab_datenum'] >= start_matlab_datenum) &\
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum)
        nl20 = len(l2g_data['latc'])
        l2g_data = {k:v[validmask,] for (k,v) in l2g_data.items()}
        nl2 = len(l2g_data['latc'])
        self.nl2 = nl2
        self.l2g_data = l2g_data
        self.logger.info('%d pixels in the L2g data' %nl20)
        if nl2 > 0:
            self.logger.info('%d pixels to be regridded...' %nl2)
        else:
            self.logger.info('No pixel to be regridded, returning...')
            return None
        nblock_row = np.max([np.floor(nrows/block_length),1]).astype(np.int)
        nblock_col = np.max([np.floor(ncols/block_length),1]).astype(np.int)
        self.nblock_row = nblock_row
        self.nblock_col = nblock_col
        
        tmp = [np.array_split(a,nblock_col,axis=1) for a in np.array_split(xmesh,nblock_row,axis=0)]
        
        block_xmesh = [arr for sublist in tmp for arr in sublist]
        tmp = [np.array_split(a,nblock_col,axis=1) for a in np.array_split(ymesh,nblock_row,axis=0)]
        block_ymesh = [arr for sublist in tmp for arr in sublist]
        nblock = len(block_xmesh)
        self.nblock = nblock
        self.logger.info('l3 mesh grid will be cut into %d blocks'%nblock)
        lonc = l2g_data['lonc']
        latc = l2g_data['latc']
        if 'lonr' in l2g_data.keys():
            lonr = l2g_data['lonr']
            latr = l2g_data['latr']
            pixel_width = np.max([F_lon_distance(lonr[:,0],lonr[:,2]),F_lon_distance(lonr[:,1],lonr[:,3])],axis=0)
            pixel_height = np.max([np.abs(latr[:,2]-latr[:,0]),np.abs(latr[:,1]-latr[:,3])],axis=0)
        else:
            pixel_width = np.max([l2g_data['u'],l2g_data['v']],axis=0)*3
            pixel_height = pixel_width
        pixel_west = lonc-pixel_width/2*xmargin
        pixel_east = lonc+pixel_width/2*xmargin
        
        pixel_south = latc-pixel_height/2*ymargin
        pixel_north = latc+pixel_height/2*ymargin
        
        block_l2g_data = []
        for iblock in range(nblock):
            mask = (pixel_west <= block_xmesh[iblock][0,-1]) &\
            (pixel_east >= block_xmesh[iblock][0,0]) &\
            (pixel_south <= block_ymesh[iblock][-1,0]) &\
            (pixel_north >= block_ymesh[iblock][0,0])
            self.logger.info('block %d'%(iblock+1)+' contains %d pixels'%np.sum(mask))
            block_l2g_data.append({k:v[mask,] for (k,v) in l2g_data.items()})
        # parallel stuff
        ncores_max = multiprocessing.cpu_count()
        if(ncores is None):
            self.logger.info('no cpu number specified, use half of them')
            ncores = int( np.ceil(ncores_max/2) )
        else:
            if ncores > ncores_max:
                self.logger.warning('You asked for more cores than you have! Use max number %d'%ncores_max)
                ncores = ncores_max
        self.logger.info('Start parallel computing on '+str(ncores)+' cores...')
        with multiprocessing.Pool(ncores) as pp:
            l3_data_list = pp.map( F_block_regrid_wrapper, \
                        ((block_l2g_data[iblock],block_xmesh[iblock],\
                          block_ymesh[iblock],oversampling_list,\
                          self.instrum,self.error_model, \
                          self.k1,self.k2,self.k3,
                          xmargin,ymargin,iblock,self.verbose) for iblock in range(nblock) ) )
#        pp = multiprocessing.Pool(ncores)
#        l3_data_list = pp.map( F_block_regrid_wrapper, \
#                        ((block_l2g_data[iblock],block_xmesh[iblock],\
#                          block_ymesh[iblock],oversampling_list,\
#                          self.instrum,self.error_model, \
#                          self.k1,self.k2,self.k3,
#                          xmargin,ymargin,iblock) for iblock in range(nblock) ) )
        self.logger.info('Reassemble blocks back to l3 grid')
        dict_of_lists = {}
        for iblock in range(nblock):
            l3_data0 = l3_data_list[iblock]
            if iblock == 0:
                for key in l3_data0.keys():
                    dict_of_lists[key] = []
            for key in l3_data0.keys():
                dict_of_lists[key].append(l3_data0[key])
        l3_data = {}
        for key in l3_data0.keys():
            l3_data[key] = np.block([dict_of_lists[key][i:i+nblock_col] for i in range(0,nblock,nblock_col)])
        
        return l3_data
        
    def F_regrid_ccm(self):
        """
        written from F_regrid on 2019/07/13 to honor chris chan miller
        who optimitized the code
        oversampled fields are copied from dictionary l2g_data as np array
        operations are vectorized when possible
        """
        import cv2
        # conda install -c scitools/label/archive shapely
        from shapely.geometry import Polygon
        west = self.west ; east = self.east ; south = self.south ; north = self.north
        nrows = self.nrows; ncols = self.ncols
        xgrid = self.xgrid ; ygrid = self.ygrid ; xmesh = self.xmesh ; ymesh = self.ymesh
        grid_size = self.grid_size ; xmargin = self.xmargin ; ymargin = self.ymargin
        start_matlab_datenum = self.start_matlab_datenum
        end_matlab_datenum = self.end_matlab_datenum
        oversampling_list = self.oversampling_list.copy()
        l2g_data = self.l2g_data
        if 'UTC_matlab_datenum' not in l2g_data.keys():
            l2g_data['UTC_matlab_datenum'] = l2g_data.pop('utc')
        for key in self.oversampling_list:
            if key not in l2g_data.keys():
                oversampling_list.remove(key)
                self.logger.warning('You asked to oversample '+key+', but I cannot find it in your data!')
        self.oversampling_list_final = oversampling_list
        nvar_oversampling = len(oversampling_list)
        error_model = self.error_model
        
        max_ncol = np.array(np.round(360/grid_size),dtype=int)
        tmplon = l2g_data['lonc']-west
        tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
        # filter data within the lat/lon box and time interval
        validmask = (tmplon >= 0) & (tmplon <= east-west) &\
        (l2g_data['latc'] >= south) & (l2g_data['latc'] <= north) &\
        (l2g_data['UTC_matlab_datenum'] >= start_matlab_datenum) &\
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum)
        nl20 = len(l2g_data['latc'])
        l2g_data = {k:v[validmask,] for (k,v) in l2g_data.items()}
        nl2 = len(l2g_data['latc'])
        self.nl2 = nl2
        self.l2g_data = l2g_data
        self.logger.info('%d pixels in the L2g data' %nl20)
        if nl2 > 0:
            self.logger.info('%d pixels to be regridded...' %nl2)
        else:
            self.logger.info('No pixel to be regridded, returning...')
            return
        
        # Allocate memory for regrid fields
        total_sample_weight = np.zeros((nrows,ncols))
        num_samples = np.zeros((nrows,ncols))
        sum_aboves = []
        for n in range(nvar_oversampling):
            sum_aboves.append(np.zeros((nrows,ncols)))
        # To only average cloud pressure using pixels where cloud fraction > 0.0
        pres_total_sample_weight = np.zeros((nrows,ncols))
        pres_num_samples = np.zeros((nrows,ncols))
        pres_sum_aboves = np.zeros((nrows,ncols))
        
        # Utilities for x/y indice list comprehensions
        def bound_arr(i1,i2,mx,ncols):
            arr = np.arange(i1,i2,dtype=int)
            arr[arr<0] += mx
            arr[arr>=mx] -= mx
            return arr[arr<ncols]
        def bound_lat(i1,i2,mx):
            arr = np.arange(i1,i2,dtype=int)
            return arr[ np.logical_and( arr>=0, arr < mx ) ]
        def F_lon_distance(lon1,lon2):
            distance = lon2 - lon1
            distance[lon2<lon1] += 360.0
            return distance
        
        # Move as much as possible outside loop
        if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B",\
                            "SCIAMACHY","TROPOMI","OMPS-N20"}:
            # Set 
            latc = l2g_data['latc']
            lonc = l2g_data['lonc']
            latr = l2g_data['latr']
            lonr = l2g_data['lonr']
            # Get lonc/latc center indices
            lonc_index = [np.argmin(np.abs(xgrid-lonc[i])) for i in range(nl2)]
            latc_index = [np.argmin(np.abs(ygrid-latc[i])) for i in range(nl2)]
            # Get East/West indices
            tmp = np.array([F_lon_distance(lonr[:,0].squeeze(),lonc),F_lon_distance(lonr[:,1].squeeze(),lonc)]).T
            west_extent = np.round( np.max(tmp,axis=1)/grid_size*xmargin )
            self.west_dist_extent = tmp.copy()
            tmp = np.array([F_lon_distance(lonc,lonr[:,2].squeeze()),F_lon_distance(lonc,lonr[:,3].squeeze())]).T
            self.east_dist_extent = tmp.copy()
            east_extent = np.round( np.max(tmp,axis=1)/grid_size*xmargin )
            # Get lists of indices
            lon_index = [bound_arr(lonc_index[i]-west_extent[i],lonc_index[i]+east_extent[i]+1,max_ncol,ncols)  for i in range(nl2)]
            self.lon_index = lon_index
            self.lonc_index = lonc_index
            # The western most longitude
            patch_west = [xgrid[lon_index[i][0]] for i in range(nl2)]
            # Get north/south indices
            north_extent = np.ceil( (latr.max(axis=1)-latr.min(axis=1))/2/grid_size*ymargin)
            south_extent = north_extent
            # List of latitude indices
            lat_index = [bound_lat(latc_index[i]-south_extent[i],latc_index[i]+north_extent[i]+1,nrows) for i in range(nl2)]
            # This might be faster
            patch_lonr = np.array([lonr[i,:] - patch_west[i] for i in range(nl2)]) ; patch_lonr[patch_lonr<0.0] += 360.0
            patch_lonc = lonc - patch_west ; patch_lonc[patch_lonc<0.0] += 360.0
            area_weight = [Polygon(np.column_stack([patch_lonr[i,:],latr[i,:].squeeze()])).area for i in range(nl2)]
            # Compute transforms for SG outside loop
            vlist = np.zeros((nl2,4,2),dtype=np.float32)
            for n in range(4):
                vlist[:,n,0] = patch_lonr[:,n] - patch_lonc[:]
                vlist[:,n,1] = latr[:,n] - latc[:]
            xvector  = np.mean(vlist[:,2:4,:],axis=1) - np.mean(vlist[:,0:2,:],axis=1)
            yvector = np.mean(vlist[:,1:3,:],axis=1) - np.mean(vlist[:,[0,3],:],axis=1)
            fwhmx = np.linalg.norm(xvector,axis=1)
            fwhmy = np.linalg.norm(yvector,axis=1)
            fixedPoints = np.array([[-fwhmx,-fwhmy],[-fwhmx,fwhmy],[fwhmx,fwhmy],[fwhmx,-fwhmy]],dtype=np.float32).transpose((2,0,1))/2.0
            tform = [cv2.getPerspectiveTransform(vlist[i,:,:].squeeze(),fixedPoints[i,:,:].squeeze()) for i in range(nl2)]
        
        elif self.instrum in {"IASI","CrIS"}:
            # Set 
            latc = l2g_data['latc']
            lonc = l2g_data['lonc']
            u = l2g_data['u']
            v = l2g_data['v']
            t = l2g_data['t']
            lonc_index = [np.argmin(np.abs(xgrid-lonc[i])) for i in range(nl2)]
            latc_index = [np.argmin(np.abs(ygrid-latc[i])) for i in range(nl2)]
            # Get East/West indices
            minlon_e = np.zeros((nl2))
            minlat_e = np.zeros((nl2))
            for i in range(nl2):
                X, minlon_e[i], minlat_e[i] = self.F_construct_ellipse(v[i],u[i],t[i],10)
            west_extent = np.round(np.abs(minlon_e)/grid_size*xmargin)
            east_extent = west_extent
            # Get lists of indices
            lon_index = [bound_arr(lonc_index[i]-west_extent[i],lonc_index[i]+east_extent[i]+1,max_ncol,ncols)  for i in range(nl2)]
            self.lon_index = lon_index
            self.lonc_index = lonc_index
            # The western most longitude
            patch_west = [xgrid[lon_index[i][0]] for i in range(nl2)]
            # Get north/south indices
            north_extent = np.ceil(np.abs(minlat_e)/grid_size*xmargin)
            south_extent = north_extent
            # List of latitude indices
            lat_index = [bound_lat(latc_index[i]-south_extent[i],latc_index[i]+north_extent[i]+1,nrows) for i in range(nl2)]
            # This might be faster
            patch_lonc = lonc - patch_west ; patch_lonc[patch_lonc<0.0] += 360.0
            area_weight = u*v
            fwhmx = 2*v
            fwhmy = 2*u
        
        else:
            self.logger.error(self.instrum+' is not supported for regridding yet!')
            return
        # Compute uncertainty weights
        if error_model == "square":
            uncertainty_weight = l2g_data['column_uncertainty']**2
        elif error_model == "log":
            uncertainty_weight = np.log10(l2g_data['column_uncertainty'])
        else:
            uncertainty_weight = l2g_data['column_uncertainty']
        # Cloud Fraction
        if 'cloud_fraction' in oversampling_list:
            cloud_fraction = l2g_data['cloud_fraction']
        # Pull out grid variables from dictionary as it is slow to access
        grid_flds = np.zeros((nl2,nvar_oversampling)) ; pcld_idx = -1
        for n in range(nvar_oversampling):
            grid_flds[:,n] = l2g_data[oversampling_list[n]]
            if oversampling_list[n] == 'cloud_pressure':
                pcld_idx = n
            # Apply log to variable if error model is log
            if(error_model == 'log') and (oversampling_list[n] == 'column_amount'):
                grid_flds[:,n] = np.log10(grid_flds[:,n])
        #t1 = time.time()
        sg_wx = fwhmx/self.sg_kfacx
        sg_wy = fwhmy/self.sg_kfacy
        # Init point counter for logger
        count = 0
        for il2 in range(nl2):
            ijmsh = np.ix_(lat_index[il2],lon_index[il2])
            patch_xmesh = xmesh[ijmsh] - patch_west[il2]
            patch_xmesh[patch_xmesh<0.0] += 360.0
            patch_ymesh = ymesh[ijmsh] - latc[il2]
            if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B",\
                                "SCIAMACHY","TROPOMI","OMPS-N20"}:
                xym1 = np.column_stack((patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten()))
                xym2 = np.hstack((xym1,np.ones((patch_xmesh.size,1)))).dot(tform[il2].T)[:,0:2]
            elif self.instrum in {"IASI","CrIS"}:
                rotation_matrix = np.array([[np.cos(-t[il2]), -np.sin(-t[il2])],[np.sin(-t[il2]),  np.cos(-t[il2])]])
                xym1 = np.array([patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten()])#np.column_stack((patch_xmesh.flatten()-patch_lonc[il2],patch_ymesh.flatten())).T
                xym2 = rotation_matrix.dot(xym1).T
                
            SG = np.exp(-(np.power( np.power(np.abs(xym2[:,0]/sg_wx[il2]),self.k1)           \
                                   +np.power(np.abs(xym2[:,1]/sg_wy[il2]),self.k2),self.k3)) )
            SG = SG.reshape(patch_xmesh.shape)
            # Update Number of samples
            num_samples[ijmsh] += SG
            # Only bother doing this if regridding cloud pressure
            if 'cloud_fraction' in oversampling_list:
                if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                    pres_num_samples[ijmsh] += SG
            # The weights
            tmp_wt = SG/area_weight[il2]/uncertainty_weight[il2]
            # Update total weights
            total_sample_weight[ijmsh] += tmp_wt
            # This only needs to be done if we are gridding pressure
            if 'cloud_fraction' in oversampling_list:
                if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                    pres_total_sample_weight[ijmsh] += tmp_wt
            # Update the desired grid variables
            for ivar in range(nvar_oversampling):
                sum_aboves[ivar][ijmsh] += tmp_wt[:,:]*grid_flds[il2,ivar]
            if 'cloud_fraction' in oversampling_list:
                if(pcld_idx > 0 and cloud_fraction[il2] > 0.0):
                    pres_sum_aboves[ijmsh] += tmp_wt[:,:]*grid_flds[il2,pcld_idx]
            if il2 == count*np.round(nl2/10.):
                self.logger.info('%d%% finished' %(count*10))
                count = count + 1
        
        self.logger.info('Completed regridding!')
        self.C = {}
        np.seterr(divide='ignore', invalid='ignore')
        for ikey in range(len(oversampling_list)):
            self.C[oversampling_list[ikey]] = sum_aboves[ikey][:,:].squeeze()\
            /total_sample_weight
            # Special case for cloud pressure (only considere pixels with
            # cloud fraction > 0.0
            if oversampling_list[ikey] == 'cloud_pressure':
                self.C[oversampling_list[ikey]] = pres_sum_aboves[:,:]\
                    /pres_total_sample_weight
        # Make cloud pressure = 0 where cloud fraction = 0
        if 'cloud_fraction' in oversampling_list and 'cloud_pressure' in oversampling_list:
            f1 = (self.C['cloud_fraction'] == 0.0)
            self.C['cloud_pressure'][f1] = 0.0

        # Set 
        self.total_sample_weight = total_sample_weight
        self.num_samples = num_samples
        self.pres_num_samples = pres_num_samples
        self.pres_total_sample_weight = pres_total_sample_weight
        
        # Set quality flag based on the number of samples
        # It has already being initialized to fill value
        # of 2
        self.quality_flag = np.full((nrows,ncols),2,dtype=np.int8)
        self.quality_flag[num_samples >= 0.1] = 0
        self.quality_flag[(num_samples > 1.e-6) & (num_samples < 0.1)] = 1
    
    def F_regrid(self,do_standard_error=False):
        # conda install -c scitools/label/archive shapely
        from shapely.geometry import Polygon
        def F_reference2west(west,data):
            if data.size > 1:
                data = data-west
                data[data < 0.] = data[data < 0.]+360.
            else:
                data = data-west
                if data < 0:
                    data = data+360
            return data
        
        def F_lon_distance(lon1,lon2):
            if lon2 < lon1:
                lon2 = lon2+360.
            distance = lon2-lon1
            return distance
        
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        nrows = self.nrows
        ncols = self.ncols
        xgrid = self.xgrid
        ygrid = self.ygrid
        xmesh = self.xmesh
        ymesh = self.ymesh
        grid_size = self.grid_size
        oversampling_list = self.oversampling_list[:]
        l2g_data = self.l2g_data
        if 'UTC_matlab_datenum' not in l2g_data.keys():
            l2g_data['UTC_matlab_datenum'] = l2g_data.pop('utc')
        for key in self.oversampling_list:
            if key not in l2g_data.keys():
                oversampling_list.remove(key)
                self.logger.warning('You asked to oversample '+key+', but I cannot find it in your data!')
        self.oversampling_list_final = oversampling_list
        nvar_oversampling = len(oversampling_list)
        error_model = self.error_model
        
        start_matlab_datenum = self.start_matlab_datenum
        end_matlab_datenum = self.end_matlab_datenum
                
        max_ncol = np.round(360/grid_size)
        
        tmplon = l2g_data['lonc']-west
        tmplon[tmplon < 0] = tmplon[tmplon < 0]+360
        # filter data within the lat/lon box and time interval
        validmask = (tmplon >= 0) & (tmplon <= east-west) &\
        (l2g_data['latc'] >= south) & (l2g_data['latc'] <= north) &\
        (l2g_data['UTC_matlab_datenum'] >= start_matlab_datenum) &\
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum)
        #(not np.isnan(l2g_data['column_amount'])) &\
        #(not np.isnan(l2g_data['column_uncertainty']))
        
        nl20 = len(l2g_data['latc'])
        l2g_data = {k:v[validmask,] for (k,v) in l2g_data.items()}
        nl2 = len(l2g_data['latc'])
        self.nl2 = nl2
        self.l2g_data = l2g_data
        self.logger.info('%d pixels in the L2g data' %nl20)
        self.logger.info('%d pixels to be regridded...' %nl2)
        
        #construct a rectangle envelopes the orginal pixel
        xmargin = self.xmargin  #how many times to extend zonally
        ymargin = self.ymargin #how many times to extend meridonally
        
        total_sample_weight = np.zeros((nrows,ncols))
        num_samples = np.zeros((nrows,ncols))
        sum_aboves = np.zeros((nrows,ncols,nvar_oversampling))
        quality_flag = np.full((nrows,ncols),2,dtype=np.int8)
        # To only average cloud pressure using pixels where cloud fraction > 0.0
        pres_total_sample_weight = np.zeros((nrows,ncols))
        pres_num_samples = np.zeros((nrows,ncols))
        pres_sum_aboves = np.zeros((nrows,ncols))
        
        count = 0
        for il2 in range(nl2):
            local_l2g_data = {k:v[il2,] for (k,v) in l2g_data.items()}
            if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B",\
                                "SCIAMACHY","TROPOMI","OMPS-N20"}:
                latc = local_l2g_data['latc']
                latr = local_l2g_data['latr']
                lonc = local_l2g_data['lonc']-west
                lonr = local_l2g_data['lonr']-west
                if lonc < 0:
                    lonc = lonc+360
                lonr[lonr < 0] = lonr[lonr < 0]+360
                lonc = lonc+west
                lonr = lonr+west
                
                lonc_index = np.argmin(np.abs(xgrid-lonc))
                latc_index = np.argmin(np.abs(ygrid-latc))
                
                west_extent = np.round(
                        np.max([F_lon_distance(lonr[0],lonc),F_lon_distance(lonr[1],lonc)])
                        /grid_size*xmargin)
                east_extent = np.round(
                        np.max([F_lon_distance(lonc,lonr[2]),F_lon_distance(lonc,lonr[3])])
                        /grid_size*xmargin)
                
                lon_index = np.arange(lonc_index-west_extent,lonc_index+east_extent+1,dtype = int)
                lon_index[lon_index < 0] = lon_index[lon_index < 0]+max_ncol
                lon_index[lon_index >= max_ncol] = lon_index[lon_index >= max_ncol]-max_ncol
                lon_index = lon_index[lon_index < ncols]
                
                patch_west = xgrid[lon_index[0]]
                
                north_extent = np.ceil((latr.max()-latr.min())/2/grid_size*ymargin)
                south_extent = north_extent
                
                lat_index = np.arange(latc_index-south_extent,latc_index+north_extent+1,dtype = int)
                lat_index = lat_index[(lat_index >= 0) & (lat_index < nrows)]
                #xmesh[lat_index,:][:,lon_index]
                patch_xmesh = F_reference2west(patch_west,xmesh[np.ix_(lat_index,lon_index)])
                patch_ymesh = ymesh[np.ix_(lat_index,lon_index)]
                patch_lonr = F_reference2west(patch_west,lonr)
                patch_lonc = F_reference2west(patch_west,lonc)
                # this is not exactly accurate, may try sum(SG[:])
                area_weight = Polygon(np.column_stack([patch_lonr[:],latr[:]])).area
                
                SG = self.F_2D_SG_transform(patch_xmesh,patch_ymesh,patch_lonr,latr,
                                            patch_lonc,latc)
                #if il2==100:self.sg = SG;return
            elif self.instrum in {"IASI","CrIS"}:
                latc = local_l2g_data['latc']
                lonc = local_l2g_data['lonc']-west
                if lonc < 0:
                    lonc = lonc+360
                lonc = lonc+west
                u = local_l2g_data['u']
                v = local_l2g_data['v']
                t = local_l2g_data['t']
                
                lonc_index = np.argmin(np.abs(xgrid-lonc))
                latc_index = np.argmin(np.abs(ygrid-latc))
                
                X, minlon_e, minlat_e = self.F_construct_ellipse(v,u,t,10)
                west_extent = np.round(np.abs(minlon_e)/grid_size*xmargin)
                east_extent = west_extent
                
                lon_index = np.arange(lonc_index-west_extent,lonc_index+east_extent+1,dtype = int)
                lon_index[lon_index < 0] = lon_index[lon_index < 0]+max_ncol
                lon_index[lon_index >= max_ncol] = lon_index[lon_index >= max_ncol]-max_ncol
                lon_index = lon_index[lon_index < ncols]
                
                patch_west = xgrid[lon_index[0]]
                
                north_extent = np.ceil(np.abs(minlat_e)/grid_size*xmargin)
                south_extent = north_extent
                
                lat_index = np.arange(latc_index-south_extent,latc_index+north_extent+1,dtype = int)
                lat_index = lat_index[(lat_index >= 0) & (lat_index < nrows)]
                
                patch_xmesh = F_reference2west(patch_west,xmesh[np.ix_(lat_index,lon_index)])
                patch_ymesh = ymesh[np.ix_(lat_index,lon_index)]
                patch_lonc = F_reference2west(patch_west,lonc)
                
                area_weight = u*v
                
                SG = self.F_2D_SG_rotate(patch_xmesh,patch_ymesh,patch_lonc,latc,\
                                         2*v,2*u,-t)
            
            num_samples[np.ix_(lat_index,lon_index)] =\
            num_samples[np.ix_(lat_index,lon_index)]+SG
            if 'cloud_fraction' in local_l2g_data.keys():
                if local_l2g_data['cloud_fraction'] > 0.0:
                    pres_num_samples[np.ix_(lat_index,lon_index)] =\
                        pres_num_samples[np.ix_(lat_index,lon_index)]+SG
                    
            if error_model == "square":
                uncertainty_weight = local_l2g_data['column_uncertainty']**2
            elif error_model == "log":
                uncertainty_weight = np.log10(local_l2g_data['column_uncertainty'])
            else:
                uncertainty_weight = local_l2g_data['column_uncertainty']
            
            total_sample_weight[np.ix_(lat_index,lon_index)] =\
            total_sample_weight[np.ix_(lat_index,lon_index)]+\
            SG/area_weight/uncertainty_weight
            
            if 'cloud_fraction' in local_l2g_data.keys():
                if local_l2g_data['cloud_fraction'] > 0.0:
                    pres_total_sample_weight[np.ix_(lat_index,lon_index)] =\
                        pres_total_sample_weight[np.ix_(lat_index,lon_index)]+\
                        SG/area_weight/uncertainty_weight
            
            for ivar in range(nvar_oversampling):
                local_var = local_l2g_data[oversampling_list[ivar]]
                if error_model == 'log':
                    if oversampling_list[ivar] == 'column_amount':
                        local_var = np.log10(local_var)
                tmp_var = SG/area_weight/uncertainty_weight*local_var
                tmp_var = tmp_var[:,:,np.newaxis]
                sum_aboves[np.ix_(lat_index,lon_index,[ivar])] =\
                sum_aboves[np.ix_(lat_index,lon_index,[ivar])]+tmp_var
                
                if 'cloud_fraction' in local_l2g_data.keys():
                    if local_l2g_data['cloud_fraction'] > 0.0 and\
                            oversampling_list[ivar] == 'cloud_pressure':
                        tmp_var = SG/area_weight/uncertainty_weight*local_var
                        tmp_var = tmp_var[:,:]
                        pres_sum_aboves[np.ix_(lat_index,lon_index)] =\
                            pres_sum_aboves[np.ix_(lat_index,lon_index)]+tmp_var
            
            if il2 == count*np.round(nl2/10.):
                self.logger.info('%d%% finished' %(count*10))
                count = count + 1
        
        self.logger.info('Completed regridding!')
        C = {}
        np.seterr(divide='ignore', invalid='ignore')
        for ikey in range(len(oversampling_list)):
            C[oversampling_list[ikey]] = sum_aboves[:,:,ikey].squeeze()\
            /total_sample_weight
            # Special case for cloud pressure (only considere pixels with
            # cloud fraction > 0.0
            if oversampling_list[ikey] == 'cloud_pressure':
                C[oversampling_list[ikey]] = pres_sum_aboves[:,:]\
                    /pres_total_sample_weight
        
        # Make cloud pressure = 0 where cloud fraction = 0
        if 'cloud_pressure' in oversampling_list:
            f1 = (C['cloud_fraction'] == 0.0)
            C['cloud_pressure'][f1] = 0.0
        
        self.C = C 
        self.total_sample_weight = total_sample_weight
        self.num_samples = num_samples
        self.pres_num_samples = pres_num_samples
        self.pres_total_sample_weight = pres_total_sample_weight
        # Set quality flag based on the number of samples
        # It has already being initialized to fill value
        # of 2
        f1 = num_samples >= 0.1
        quality_flag[f1] = 0
        f1 = (num_samples > 1.e-6) & (num_samples < 0.1)
        quality_flag[f1] = 1
        self.quality_flag = quality_flag
        if not do_standard_error:
            return
        
        self.logger.info('OK, do standard error for weighted mean, looping through l2g_data, again...')
        
        #P_bar = self.total_sample_weight/nl2
        X_bar = self.C['column_amount']
        sum_above_SE = np.zeros((nrows,ncols))
        count = 0
        for il2 in range(nl2):
            local_l2g_data = {k:v[il2,] for (k,v) in l2g_data.items()}
            if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B","SCIAMACHY","TROPOMI"}:
                latc = local_l2g_data['latc']
                latr = local_l2g_data['latr']
                lonc = local_l2g_data['lonc']-west
                lonr = local_l2g_data['lonr']-west
                if lonc < 0:
                    lonc = lonc+360
                lonr[lonr < 0] = lonr[lonr < 0]+360
                lonc = lonc+west
                lonr = lonr+west
                
                lonc_index = np.argmin(np.abs(xgrid-lonc))
                latc_index = np.argmin(np.abs(ygrid-latc))
                
                west_extent = np.round(
                        np.max([F_lon_distance(lonr[0],lonc),F_lon_distance(lonr[1],lonc)])
                        /grid_size*xmargin)
                east_extent = np.round(
                        np.max([F_lon_distance(lonc,lonr[2]),F_lon_distance(lonc,lonr[3])])
                        /grid_size*xmargin)
                
                lon_index = np.arange(lonc_index-west_extent,lonc_index+east_extent+1,dtype = int)
                lon_index[lon_index < 0] = lon_index[lon_index < 0]+max_ncol
                lon_index[lon_index >= max_ncol] = lon_index[lon_index >= max_ncol]-max_ncol
                lon_index = lon_index[lon_index < ncols]
                
                patch_west = xgrid[lon_index[0]]
                
                north_extent = np.ceil((latr.max()-latr.min())/2/grid_size*ymargin)
                south_extent = north_extent
                
                lat_index = np.arange(latc_index-south_extent,latc_index+north_extent+1,dtype = int)
                lat_index = lat_index[(lat_index >= 0) & (lat_index < nrows)]
                #xmesh[lat_index,:][:,lon_index]
                patch_xmesh = F_reference2west(patch_west,xmesh[np.ix_(lat_index,lon_index)])
                patch_ymesh = ymesh[np.ix_(lat_index,lon_index)]
                patch_lonr = F_reference2west(patch_west,lonr)
                patch_lonc = F_reference2west(patch_west,lonc)
                # this is not exactly accurate, may try sum(SG[:])
                area_weight = Polygon(np.column_stack([patch_lonr[:],latr[:]])).area
                
                SG = self.F_2D_SG_transform(patch_xmesh,patch_ymesh,patch_lonr,latr,
                                            patch_lonc,latc)
            elif self.instrum in {"IASI","CrIS"}:
                latc = local_l2g_data['latc']
                lonc = local_l2g_data['lonc']-west
                if lonc < 0:
                    lonc = lonc+360
                lonc = lonc+west
                u = local_l2g_data['u']
                v = local_l2g_data['v']
                t = local_l2g_data['t']
                
                lonc_index = np.argmin(np.abs(xgrid-lonc))
                latc_index = np.argmin(np.abs(ygrid-latc))
                
                X, minlon_e, minlat_e = self.F_construct_ellipse(v,u,t,10)
                west_extent = np.round(np.abs(minlon_e)/grid_size*xmargin)
                east_extent = west_extent
                
                lon_index = np.arange(lonc_index-west_extent,lonc_index+east_extent+1,dtype = int)
                lon_index[lon_index < 0] = lon_index[lon_index < 0]+max_ncol
                lon_index[lon_index >= max_ncol] = lon_index[lon_index >= max_ncol]-max_ncol
                lon_index = lon_index[lon_index < ncols]
                
                patch_west = xgrid[lon_index[0]]
                
                north_extent = np.ceil(np.abs(minlat_e)/grid_size*xmargin)
                south_extent = north_extent
                
                lat_index = np.arange(latc_index-south_extent,latc_index+north_extent+1,dtype = int)
                lat_index = lat_index[(lat_index >= 0) & (lat_index < nrows)]
                
                patch_xmesh = F_reference2west(patch_west,xmesh[np.ix_(lat_index,lon_index)])
                patch_ymesh = ymesh[np.ix_(lat_index,lon_index)]
                patch_lonc = F_reference2west(patch_west,lonc)
                
                area_weight = u*v
                
                SG = self.F_2D_SG_rotate(patch_xmesh,patch_ymesh,patch_lonc,latc,\
                                         2*v,2*u,-t)
            if error_model == "square":
                uncertainty_weight = local_l2g_data['column_uncertainty']**2
            elif error_model == "log":
                uncertainty_weight = np.log10(local_l2g_data['column_uncertainty'])
            else:
                uncertainty_weight = local_l2g_data['column_uncertainty']
            
            # Cochran 1977 method, in https://doi.org/10.1016/1352-2310(94)00210-C, simplified by K. Sun
            P_i = SG/area_weight/uncertainty_weight
            local_var = local_l2g_data['column_amount']
            if error_model == 'log':
                local_var = np.log10(local_var)
            local_X_bar = X_bar[np.ix_(lat_index,lon_index)]
            #local_P_bar = P_bar[np.ix_(lat_index,lon_index)]
            #tmp_var = (P_i*local_X_bar-local_P_bar*local_X_bar)**2\
            #-2*local_X_bar*(P_i-local_P_bar)*(P_i*local_var-local_P_bar*local_X_bar)\
            #+local_X_bar**2*(P_i-local_P_bar)**2
            tmp_var = (P_i*(local_var-local_X_bar))**2
            sum_above_SE[np.ix_(lat_index,lon_index)] =\
            sum_above_SE[np.ix_(lat_index,lon_index)]+tmp_var
            
            if il2 == count*np.round(nl2/10.):
                self.logger.info('%d%% finished' %(count*10))
                count = count + 1
        
        variance_of_weighted_mean\
        = sum_above_SE/(self.total_sample_weight**2)
        
        # the following should be more accurate but I don't like it, as it makes trouble when merging. Plus in general nl2/(nl2-1) ~ 1
        if nl2 > 1:
            variance_of_weighted_mean\
            = variance_of_weighted_mean*nl2/(nl2-1)
        
        self.standard_error_of_weighted_mean = np.sqrt(variance_of_weighted_mean)
    
    def F_save_l3_to_mat(self,file_path,l3_data=None):
        """
        save regridded level 3 data, from F_regrid or F_regrid_ccm to .mat file
        file_path: 
            absolute path to the .mat file to save
        created on 2020/03/15
        updated on 2020/07/20 to be compatible with external l3_data dictionary
        """
        from scipy.io import savemat
        if l3_data is not None:
            l3_data['xgrid'] = self.xgrid
            l3_data['ygrid'] = self.ygrid
            l3_data['ncol'] = self.ncols
            l3_data['nrow'] = self.nrows
            if 'xmesh' in l3_data.keys():
                l3_data.pop('xmesh');
                l3_data.pop('ymesh');
            savemat(file_path,l3_data)    
            return
        if not self.C:
            self.logger.warning('l3_data is empty. Nothing to save.')
            return        
        
        C = self.C.copy()
        C['total_sample_weight'] = self.total_sample_weight
        C['num_samples'] = self.num_samples
        C['xgrid'] = self.xgrid
        C['ygrid'] = self.ygrid
        C['ncol'] = self.ncols
        C['nrow'] = self.nrows
        savemat(file_path,C)
    
    def F_vertically_weighted_wind(self,which_met,met_dir,
                                 fn_header='',nlevel=10):
        '''
        sample vertically weighted wind from 3D met data
        created on 2020/09/22
        '''
        sounding_lon = self.l2g_data['lonc']
        sounding_lat = self.l2g_data['latc']
        sounding_datenum = self.l2g_data['UTC_matlab_datenum']
        sounding_p0 = self.l2g_data['era5_sp']
        sounding_p1 = self.l2g_data['era5_sp']*np.exp(-self.l2g_data['era5_blh']/7500)
        if which_met in {'era','era5','ERA','ERA5'}:
            if not fn_header:
                fn_header_local = 'CONUS'
            else:
                fn_header_local = fn_header
            self.logger.info('sampling 3D u and v wind from ERA5...')
            sounding_interp = F_interp_era5_3D(sounding_lon,sounding_lat,sounding_datenum,
                                               sounding_p0,sounding_p1,nlevel,
                                               era5_dir=met_dir,interp_fields=['u','v'],
                                               fn_header=fn_header_local)
            self.logger.info('averaging 3D wind vertically...')
            self.l2g_data['era5_ubar'] = np.nanmean(sounding_interp['u'],axis=1)
            self.l2g_data['era5_vbar'] = np.nanmean(sounding_interp['v'],axis=1)
            
        
    def F_interp_profile(self,which_met,met_dir,if_monthly=False,
                         surface_pressure_field='merra2_PS'):
        """
        place holder for a more versatile function to sample vertical profiles
        from 3D met/CTM fields at l2g locations and times. currently only support
        RS's geos-chem
        which_met:
            gcrs for now
        met_dir:
            gcrs_dir='/mnt/Data2/GEOS-Chem_Silvern/'
        if_monthly:
            if use monthly profile, instead of daily profile
        surface_pressure_field:
            surface pressure field in l2g_data. suggest to use merra2 for gcrs
            because surface pressure determines the whole pressure levels
        created on 2020/03/14
        """
        sounding_lon = self.l2g_data['lonc']
        sounding_lat = self.l2g_data['latc']
        sounding_datenum = self.l2g_data['UTC_matlab_datenum']
        sounding_ps = self.l2g_data[surface_pressure_field]
        if which_met == 'gcrs':
            sounding_profiles,sounding_pEdge = \
            F_interp_gcrs(sounding_lon,sounding_lat,sounding_datenum,
                          sounding_ps,gcrs_dir=met_dir,
                          product=self.product,if_monthly=if_monthly)
            self.l2g_data['gcrs_'+self.product+'_profiles'] = sounding_profiles
            self.l2g_data['gcrs_plevel'] = sounding_pEdge
            self.logger.info('GEOS-Chem profiles sampled at level 2 g locations')
    
    def F_interp_met(self,which_met,met_dir,interp_fields,fn_header='',
                     time_collection='inst3'):
        """
        finally made the decision to integrate all meteorological interopolation
        to the same framework.
        which_met:
            a string, choosen from 'ERA5', 'NARR', 'GEOS-FP', 'MERRA-2'
        met_dir:
            directory containing those met data, data structure should be consistently
            Y%Y/M%M/D%D
        interp_fields:
            variables to interpolate from met data, only 2d fields are supported
        fn_header:
            in general should denote domain location of met data
        time_collection:
            only useful for geos fp. see F_interp_geos_mat
        created on 2020/03/04
        """
        sounding_lon = self.l2g_data['lonc']
        sounding_lat = self.l2g_data['latc']
        sounding_datenum = self.l2g_data['UTC_matlab_datenum']
        if which_met in {'era','era5','ERA','ERA5'}:
            if not fn_header:
                fn_header_local = 'CONUS'
            else:
                fn_header_local = fn_header
            sounding_interp = F_interp_era5(sounding_lon,sounding_lat,sounding_datenum,
                                            met_dir,interp_fields,fn_header_local)
            for key in sounding_interp.keys():
                self.logger.info(key+' from ERA5 is sampled to L2g coordinate/time')
                self.l2g_data['era5_'+key] = np.float32(sounding_interp[key])
                self.sounding_interp = sounding_interp
        elif which_met in {'geos','GEOS','GEOS-FP','geos-fp'}:
            if not fn_header:
                fn_header_local = 'subset'
            else:
                fn_header_local = fn_header
            sounding_interp = F_interp_geos_mat(sounding_lon,sounding_lat,sounding_datenum,
                                            met_dir,interp_fields,time_collection,fn_header_local)
            for key in sounding_interp.keys():
                self.logger.info(key+' from GEOS-FP is sampled to L2g coordinate/time')
                self.l2g_data['geosfp_'+key] = np.float32(sounding_interp[key])
        elif which_met in {'narr','NARR'}:
            if not fn_header:
                fn_header_local = 'subset'
            else:
                fn_header_local = fn_header
            sounding_interp = F_interp_narr_mat(sounding_lon,sounding_lat,sounding_datenum,
                                            met_dir,interp_fields,fn_header_local)
            for key in sounding_interp.keys():
                self.logger.info(key+' from NARR is sampled to L2g coordinate/time')
                self.l2g_data['narr_'+key] = np.float32(sounding_interp[key])
        elif which_met in {'merra-2','merra2','merra','MERRA-2','MERRA2','MERRA'}:
            if not fn_header:
                fn_header_local = 'MERRA2_300.tavg1_2d_slv_Nx'
            else:
                fn_header_local = fn_header
            sounding_interp = F_interp_merra2(sounding_lon,sounding_lat,sounding_datenum,
                                            met_dir,interp_fields,fn_header_local)
            for key in sounding_interp.keys():
                self.logger.info(key+' from MERRA2 is sampled to L2g coordinate/time')
                self.l2g_data['merra2_'+key] = np.float32(sounding_interp[key])
    
    def F_derive_model_subcolumn(self,pressure_boundaries=['ps','pbl',600,'tropopause',0],
                                 pbl_multiplier=[2.5],
                                 min_pbltop_pressure=600.,
                                 surface_pressure_field='merra2_PS',
                                 tropopause_field='merra2_TROPPT',
                                 pbltop_field='merra2_PBLTOP'):
        """
        derive subcolumns using interpolated model profiles and stored the results
        in l2g_data
        pressure_boundaries:
            boundaries between which to calculate subcolumn. for pressure boundaries,
            the unit is ***hPa***. use 'ps' for surface pressure and 'tropopause' 
            for tropopause pressure. 'pbl' marks a boundary at pbl_multiplier*pbl height in pressure
        pbl_multiplier:
            a list the same size as the number of appearances of 'pbl' in pressure_boundaries
        min_pbltop_pressure:
            the min pressure (highest altitude) that is allowed for the boundaries related to the pbl
            unit is ***hPa***
        surface_pressure_field:
            surface pressure in l2g_data dictionary
        tropopause_field:
            tropopause pressure in l2g_data dictionary
        pbltop_field:
            pbl top pressure in l2g_data dictionary
        created on 2020/03/14
        """
        from scipy.interpolate import interp1d
        if surface_pressure_field not in self.l2g_data.keys():
            self.logger.warning(surface_pressure_field+' is not in l2g_data!')
            return
        if tropopause_field not in self.l2g_data.keys() and 'tropopause' in pressure_boundaries:
            self.logger.warning(tropopause_field+' is not in l2g_data!')
            return
        if 'gcrs_'+self.product+'_profiles' not in self.l2g_data.keys():
            self.logger.warning('Please run popy.F_interp_profiles first!')
            return
        sounding_profile = self.l2g_data['gcrs_'+self.product+'_profiles']
        sounding_pEdge = self.l2g_data['gcrs_plevel']
        sfc_pressure = self.l2g_data[surface_pressure_field]
        pbltop_pressure = self.l2g_data[pbltop_field]
        tropopause_pressure = self.l2g_data[tropopause_field]
        nsubcol = len(pressure_boundaries)-1
        subcolumns = np.full((self.nl2,nsubcol),np.nan)
        pressure_boundaries = np.array(pressure_boundaries)
        ps_idx = np.nonzero(pressure_boundaries=='ps')
        pt_idx = np.nonzero(pressure_boundaries=='tropopause')
        pbltop_idxs = np.nonzero(pressure_boundaries=='pbl')
        num_pressure_boundaries = np.zeros((self.nl2,len(pressure_boundaries)),dtype=np.float32)
        vmr_pressure_boundaries = np.zeros((self.nl2,len(pressure_boundaries)),dtype=np.float32)
        msg_str = 'calculating subcolumns between'
        count_pbl = 0
        for ip in range(len(pressure_boundaries)):
            if ip == ps_idx[0]:
                num_pressure_boundaries[:,ip] = sfc_pressure
                msg_str = msg_str+' surface pressure ([%.1f'%(np.min(sfc_pressure)/1e2)+',%.1f] hPa)'%(np.max(sfc_pressure)/1e2)
            elif ip == pt_idx[0]:
                num_pressure_boundaries[:,ip] = tropopause_pressure
                msg_str = msg_str+' tropopause pressure ([%.1f'%(np.min(tropopause_pressure)/1e2)+',%.1f] hPa)'%(np.max(tropopause_pressure)/1e2)
            elif ip in pbltop_idxs:
                tmp = sfc_pressure-(sfc_pressure-pbltop_pressure)*pbl_multiplier[count_pbl]
                tmp[tmp < min_pbltop_pressure*100] = min_pbltop_pressure*100
                num_pressure_boundaries[:,ip] = tmp
                msg_str = msg_str+' %.1f'%(pbl_multiplier[count_pbl])+' x pbl thickness ([%.1f'%(np.min(tmp)/1e2)+',%.1f] hPa)'%(np.max(tmp)/1e2)
                count_pbl = count_pbl+1
            else:# hPa to Pa
                num_pressure_boundaries[:,ip] = np.float(pressure_boundaries[ip])*1e2
                msg_str = msg_str+' %.1f hPa'%(np.float(pressure_boundaries[ip]))
        self.logger.info(msg_str)
        self.num_pressure_boundaries = num_pressure_boundaries
        nl2 = self.nl2
        count = 0
        self.logger.info('Looping through l2g pixels to calculate subcolumns. could be slow...')
        for il2 in range(self.nl2):
            local_pressure_boundaries = num_pressure_boundaries[il2,]
            local_plevel = sounding_pEdge[il2,:]
            # subcolum of each layer, in mol/m2
            local_gas = sounding_profile[il2,:]*1e-9*np.abs(np.diff(local_plevel))/9.8/0.029
            cum_gas = np.concatenate(([0.],np.cumsum(local_gas)))
            # 1d interpolation function, cumulated mass from ps
            f = interp1d(local_plevel,cum_gas,fill_value='extrapolate')
            sfc2p_subcol = np.array([f(pb) for pb in local_pressure_boundaries])
            subcolumns[il2,] = np.diff(sfc2p_subcol)
            # interpolating vmr at pressure boundaries
            fvmr = interp1d(local_plevel[0:-1],sounding_profile[il2,:],fill_value='extrapolate')
            vmr_pressure_boundaries[il2,] = np.array([fvmr(pb) for pb in local_pressure_boundaries])
            if il2 == count*np.round(nl2/10.):
                self.logger.info('%d%% finished' %(count*10))
                count = count + 1
        self.l2g_data['sub_columns'] = subcolumns
        self.pressure_boundaries = pressure_boundaries
        self.vmr_pressure_boundaries = vmr_pressure_boundaries
            
    def F_remove_l2g_fields(self,fields_to_remove):
        """
        sometimes we don't want some fields in the l2g data anymore, e.g., the 
        interpolated fields. This function cleans the fields listed as input
        fields_to_remove:
            a list of field names to be removed from the l2g_data dictionary, 
            for example fields_to_remove=['U2M','V2M','U10M','V10M','U850','V850','U50M','V50M']
        created on 2020/03/04
        """
        if not hasattr(self,'l2g_data'):
            self.logger.warning('l2g_data is not there!')
            return
        for key in fields_to_remove:
            try:
                del self.l2g_data[key]
                self.logger.info(key+' has been removed from l2g_data...')
            except KeyError:
                self.logger.warning(key+' is not there!')
    
    def F_unload_l2g_data(self):
        if hasattr(self,'l2g_data'):
            self.logger.warning('l2g_data is not there!')
        else:
            self.logger.warning('Unloading l2g_data from the popy object...')
            del self.l2g_data
            if hasattr(self,'nl2'):
                del self.nl2
    
    def F_plot_oversampled_variable(self,plot_variable,save_fig_path='',\
                                    vmin=np.nan,vmax=np.nan,dpi=200):
        import matplotlib.pyplot as plt
        # conda install -c anaconda basemap
        from mpl_toolkits.basemap import Basemap
        # otherwise won't work at ssh
        #plt.switch_backend('agg')
        fig1 = plt.gcf()
        # Draw an equidistant cylindrical projection using the low resolution
        # coastline database.
        m = Basemap(projection='cyl', resolution='l',
                    llcrnrlat=self.south, urcrnrlat = self.north,
                    llcrnrlon=self.west, urcrnrlon = self.east)
        m.drawcoastlines(linewidth=0.5)
        m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180, 180., 45.), labels=[0, 0, 0, 1])
        if plot_variable == 'standard_error_of_weighted_mean':
            data = self.standard_error_of_weighted_mean
        else:
            data = self.C[plot_variable]
        m.pcolormesh(self.xgrid,self.ygrid,data,latlon=True,cmap='jet')
        vmin0,vmax0 = plt.gci().get_clim()
        if np.isnan(vmin):
            vmin = vmin0
        if np.isnan(vmax):
            vmax = vmax0
        plt.clim(vmin,vmax)
        cb = m.colorbar()
        cb.set_label(r'molc cm$^{-2}$')
        plt.title(self.tstart+'-'+self.tend,fontsize=8)
        if save_fig_path:
            fig1.savefig(save_fig_path,dpi=dpi)
        plt.close()
        

