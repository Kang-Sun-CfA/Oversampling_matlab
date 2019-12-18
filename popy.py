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
        directory where subset era5 data in .mat are saved
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
                  time_collection='inst3'):
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
        file_path = os.path.join(file_dir,'subset_'+file_datetime.strftime('%Y%m%d_%H%M')+'.mat')
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
                                 'PBLH','P_surf','T_surf',\
                                 'U_10m','V_10m','U_30m','V_30m']):
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
        file_name = 'subset_'+file_datetime.strftime('%Y%m%d_%H%M')+'.mat'
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

class popy(object):
    
    def __init__(self,instrum,product,\
                 grid_size=0.1,west=-180,east=180,south=-90,north=90,\
                 start_year=1995,start_month=1,start_day=1,\
                 start_hour=0,start_minute=0,start_second=0,\
                 end_year=2025,end_month=12,end_day=31,\
                 end_hour=0,end_minute=0,end_second=0):
        
        self.instrum = instrum
        self.product = product
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of popy')
        
        if(instrum == "OMI"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'cloud_fraction','cloud_pressure','terrain_height']
            xmargin = 1.5
            ymargin = 2
            maxsza = 60
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
        elif(instrum == "TROPOMI"):
            k1 = 4
            k2 = 2
            k3 = 1
            error_model = "linear"
            oversampling_list = ['column_amount','albedo',\
                                 'cloud_fraction']
            xmargin = 1.5
            ymargin = 1.5
            maxsza = 60
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
            elif key_name in {'colnh3','colno2','colhcho','colchocho'}:
                l2g_data['column_amount'] = mat_data['output_subset'][key_name][0][0].flatten()
            elif key_name in {'colnh3error','colno2error','colhchoerror','colchochoerror'}:
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
                data = f[DATAFIELD_NAME]
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
            fn_dir = l2_dir+fn
            self.logger.info('Loading'+fn_dir)
            outp_he5 = self.F_read_he5(fn_dir,swathname,data_fields,geo_fields,data_fields_l2g,geo_fields_l2g)
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
    
    def F_subset_S5PNO2(self,path,s5p_product='*',geos_interp_variables=[],
                        geos_time_collection=''):
        """ 
        function to subset tropomi no2 level 2 data, calling self.F_read_S5P_nc
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
        
        # absolute path of useful variables in the nc file
        data_fields = ['/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_crb_nitrogendioxide_window',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle',\
               '/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_nitrogendioxide_window',\
               '/PRODUCT/latitude',\
               '/PRODUCT/longitude',\
               '/PRODUCT/qa_value',\
               '/PRODUCT/time_utc',\
               '/PRODUCT/nitrogendioxide_tropospheric_column',\
               '/PRODUCT/nitrogendioxide_tropospheric_column_precision']    
        # standardized variable names in l2g file. should map one-on-one to data_fields
        data_fields_l2g = ['cloud_fraction','latitude_bounds','longitude_bounds','SolarZenithAngle',\
                           'vza','albedo','latc','lonc','qa_value','time_utc',\
                           'column_amount','column_uncertainty']
        self.logger.info('Read, subset, and store level 2 data to l2g_data')
        self.logger.info('Level 2 data are located at '+l2_dir)
        l2g_data = {}
        for fn in l2_list:
            fn_dir = l2_dir+fn
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
            fn_dir = l2_dir+fn
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
        geos_interp_variables:
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
            
            if if_trop_xch4:
                
                if 'TROPPT' not in geos_interp_variables:
                    self.logger.warning('tropopause has to be resampled from geos fp to calculate tropospheric xch4!')
                    geos_interp_variables = np.concatenate((geos_interp_variables,['TROPPT']),0)
            
            if geos_interp_variables != []:
                sounding_interp = F_interp_geos_mat(outp_nc['lonc'],outp_nc['latc'],outp_nc['UTC_matlab_datenum'],\
                                                geos_dir='/mnt/Data2/GEOS/s5p_interp/',\
                                                interp_fields=geos_interp_variables,\
                                                time_collection=geos_time_collection)
                for var in geos_interp_variables:
                    outp_nc[var] = sounding_interp[var]
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
                    tropp = l2g_data0['TROPPT'][il2]
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
            fn_dir = l2_dir+fn
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
    
    def F_subset_OMNO2(self,path,l2_path_structure=None):
        """ 
        function to subset omno2, nasa sp level 2 data, calling self.F_read_he5
        path:
            l2 data root directory, or path to control file
        l2_path_structure:
            None by default, indicating individual files are directly under path
            '%Y/' if files are like l2_dir/2019/*.he5
            '%Y/%m/%d/' if files are like l2_dir/2019/05/01/*.he5
        updated on 2019/07/17
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
        
        data_fields = ['CloudFraction','CloudPressure','TerrainReflectivity',\
                       'ColumnAmountNO2Trop','ColumnAmountNO2TropStd','VcdQualityFlags',\
                       'XTrackQualityFlags']
        data_fields_l2g = ['cloud_fraction','cloud_pressure','albedo',\
                           'column_amount','column_uncertainty','VcdQualityFlags',\
                       'XTrackQualityFlags']
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
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum) &\
        (l2g_data['column_amount'] > -1e25)
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
        if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B","SCIAMACHY","TROPOMI"}:
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
            if self.instrum in {"OMI","OMPS-NM","GOME-1","GOME-2A","GOME-2B","SCIAMACHY","TROPOMI"}:
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
        (l2g_data['UTC_matlab_datenum'] <= end_matlab_datenum) &\
        (l2g_data['column_amount'] > -1e25)
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
        

