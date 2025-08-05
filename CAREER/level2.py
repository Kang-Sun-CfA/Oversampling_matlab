'''
updated classes for level 2 data of grading spectrometers,
samplers of met/ctm data on level 2 pixels,
and processor of L2->4
'''
import numpy as np
import datetime as dt
import pandas as pd
import logging
from scipy.io import loadmat
import os,sys,glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from popy import datedev_py, datetime2datenum


class L2ToL4():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_DD(self,ds,us=None,vs=None,fs=None):
        '''get directiona derivative
        ds:
            the dataset, e.g., TEMPOL2 or S5PL2 classes, but not popy.l2g_data dict
        us/vs:
            lists of u/v fields, default [era5_u300,era5_u10] and [era5_v300,era5_v10]
        fs:
            list of fields to get dd, default ['column_amount','surface_altitude']
        '''
        us = us or ['era5_u300','era5_u10']
        vs = vs or ['era5_v300','era5_v10']
        fs = fs or ['column_amount','surface_altitude']
        # possible names of u dot grad(z0)
        wind_topo_fs = [
            'terrain_height_DD','terrain_height_DD_xy','terrain_height_DD_rs',
            'surface_altitude_DD','surface_altitude_DD_xy','surface_altitude_DD_rs'
        ]
        if not isinstance(ds,list):
            for u,v,f in zip(us,vs,fs):
                f_ = [f] if not isinstance(f,list) else f
                if 'surface_altitude' in f_ and 'surface_altitude' not in ds.keys():
                    self.logger.warning('surface_altitude replaced by terrain_height')
                    f_ = ['terrain_height' if x == 'surface_altitude' else x for x in f_]
                ds = self._get_DD(ds,u,v,f_)
            for wind_topo_f in wind_topo_fs:
                if wind_topo_f in ds.keys():
                    ds[wind_topo_f] *= ds['column_amount']
        else:
            for data in ds:
                for u,v,f in zip(us,vs,fs):
                    f_ = [f] if not isinstance(f,list) else f
                    if 'surface_altitude' in f_ and 'surface_altitude' not in data.keys():
                        self.logger.warning('surface_altitude replaced by terrain_height')
                        f_ = ['terrain_height' if x == 'surface_altitude' else x for x in f_]
                    data = self._get_DD(data,u,v,f_)
                for wind_topo_f in wind_topo_fs:
                    if wind_topo_f in data.keys():
                        data[wind_topo_f] *= data['column_amount']
        return ds
        
    @staticmethod
    def get_theta(data):
        thetax = np.full(data['latc'].shape,np.nan,dtype=np.float32)
        thetay = np.full_like(thetax,np.nan)
        thetar = np.full_like(thetax,np.nan)
        thetas = np.full_like(thetax,np.nan)
        det_xy = np.full_like(thetax,np.nan)
        det_rs = np.full_like(thetax,np.nan)
        m_per_lat = 111e3
        m_per_lon = m_per_lat * np.cos(np.radians(data['latc'])).astype(np.float32)
        if (data['latc'].shape[0] <= 3) or (data['latc'].shape[1] <= 3):
            return thetax,thetay,thetar,thetas,det_xy,det_rs,m_per_lat,m_per_lon
        
        thetax[:,1:-1] = np.arctan2(m_per_lat*(data['latc'][:,:-2]-data['latc'][:,2:]),
                                   m_per_lon[:,1:-1]*(data['lonc'][:,:-2]-data['lonc'][:,2:]))
        thetay[1:-1,:] = np.arctan2(m_per_lat*(data['latc'][2:,:]-data['latc'][:-2,:]),
                                   m_per_lon[1:-1,:]*(data['lonc'][2:,:]-data['lonc'][:-2,:]))
        thetar[1:-1,1:-1] = np.arctan2(m_per_lat*(data['latc'][2:,:-2]-data['latc'][:-2,2:]),
                                   m_per_lon[1:-1,1:-1]*(data['lonc'][2:,:-2]-data['lonc'][:-2,2:]))
        thetas[1:-1,1:-1] = np.arctan2(m_per_lat*(data['latc'][2:,2:]-data['latc'][:-2,:-2]),
                                   m_per_lon[1:-1,1:-1]*(data['lonc'][2:,2:]-data['lonc'][:-2,:-2]))
        det_xy = np.cos(thetax)*np.sin(thetay) - np.sin(thetax)*np.cos(thetay)
        det_rs = np.cos(thetar)*np.sin(thetas) - np.sin(thetar)*np.cos(thetas)
        return thetax,thetay,thetar,thetas,det_xy,det_rs,m_per_lat,m_per_lon
    
    def _get_DD(self,data,east_wind_field=None,north_wind_field=None,fields=None):
        '''calculate directional derivatives, (u,v) dot (dvcd/dx,dvcd/dy) using xy and rs directions
        east/north_wind_field:
            u/v wind field name in met data
        fields:
            data fields to calculate DD
        '''
        thetax,thetay,thetar,thetas,det_xy,det_rs,m_per_lat,m_per_lon = self.get_theta(data)
        if fields is None:
            fields = ['column_amount']
        east_wind_field = east_wind_field or 'era5_u300'
        north_wind_field = north_wind_field or 'era5_v300'
        # eastward and northward wind
        windu = data[east_wind_field]
        windv = data[north_wind_field]
        
        # xward and yward wind
        windx = (np.sin(thetay)*windu - np.cos(thetay)*windv)/det_xy
        windy = (-np.sin(thetax)*windu + np.cos(thetax)*windv)/det_xy
        
        # rward and sward wind
        windr = (np.sin(thetas)*windu - np.cos(thetas)*windv)/det_rs
        winds = (-np.sin(thetar)*windu + np.cos(thetar)*windv)/det_rs
        
        def latlon2m(lat1,lon1,lat2,lon2,m_per_lat,m_per_lon):
            '''function to return distance in meter using latlon matrices'''
            return np.sqrt(
                np.square((lat1-lat2)*m_per_lat)
                +np.square((lon1-lon2)*m_per_lon))
        
        for field in fields:
            f = data[field]
            # gradients
            dfdx = np.full_like(f,np.nan)
            dfdy = np.full_like(f,np.nan)
            dfdr = np.full_like(f,np.nan)
            dfds = np.full_like(f,np.nan)
            dfdx[:,1:-1] = (f[:,:-2]-f[:,2:])/ \
            latlon2m(lat1=data['latc'][:,:-2],lon1=data['lonc'][:,:-2],
                    lat2=data['latc'][:,2:],lon2=data['lonc'][:,2:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[:,1:-1]
                    )
            dfdy[1:-1,:] = (f[2:,:]-f[:-2,:])/ \
            latlon2m(lat1=data['latc'][2:,:],lon1=data['lonc'][2:,:],
                    lat2=data['latc'][:-2,:],lon2=data['lonc'][:-2,:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,:]
                    )
            dfdr[1:-1,1:-1] = (f[2:,:-2]-f[:-2,2:])/ \
            latlon2m(lat1=data['latc'][2:,:-2],lon1=data['lonc'][2:,:-2],
                    lat2=data['latc'][:-2,2:],lon2=data['lonc'][:-2,2:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,1:-1]
                    )
            dfds[1:-1,1:-1] = (f[2:,2:]-f[:-2,:-2])/ \
            latlon2m(lat1=data['latc'][2:,2:],lon1=data['lonc'][2:,2:],
                    lat2=data['latc'][:-2,:-2],lon2=data['lonc'][:-2,:-2],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,1:-1]
                    )
            
            # directional derivatives
            data[field+'_DD_xy'] = windx*dfdx + windy*dfdy
            data[field+'_DD_rs'] = windr*dfdr + winds*dfds
            data[field+'_DD'] = (data[field+'_DD_xy']+data[field+'_DD_rs'])*0.5
        return data


class GeosCfSampler():
    def __init__(self,base_dir,
                 west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.west,self.east,self.south,self.north=west,east,south,north
        
    @staticmethod
    def _interpolate(data,fetcher,out_field,margin=0.5,**kwargs):
        '''utility function for self.interpolate_*
        data:
            data dict
        fetcher:
            callable taking in start/end_dt and kwargs
        out_field:
            str field name in data to store the result
        margin:
            if west/east/south/north are not in kwargs, load data this degree
            beyond bounds of lonc/latc
        '''
        start_dt = datedev_py(np.nanmin(data['UTC_matlab_datenum']))
        end_dt = datedev_py(np.nanmax(data['UTC_matlab_datenum']))
        west = kwargs.pop('west',np.nanmin(data['lonc'])-margin)
        east = kwargs.pop('east',np.nanmax(data['lonc'])+margin)
        south = kwargs.pop('south',np.nanmin(data['latc'])-margin)
        north = kwargs.pop('north',np.nanmax(data['latc'])+margin)
        (ts,ys,xs),fs = fetcher(
            start_dt,end_dt,west=west,east=east,south=south,north=north,
            **kwargs)
        dns = np.array([datetime2datenum(t) for t in ts])
        func = RegularGridInterpolator(
            (dns,ys,xs),fs,bounds_error=False,fill_value=np.nan)
        mask = (~np.isnan(data['UTC_matlab_datenum'])) &\
        (~np.isnan(data['lonc'])) & (~np.isnan(data['latc']))
        data[out_field] = np.full(data['latc'].shape,np.nan)
        data[out_field][mask] = func((
            data['UTC_matlab_datenum'][mask],
            data['latc'][mask],
            data['lonc'][mask]
            ))
        return data
    
    def interpolate_NOxNO2_ratio(self,ds,xfield='column_amount',**kwargs):
        '''sample ratio f to a dataset, which should be dict-like with keys
        UTC_matlab_datenum, latc, lonc, or a list of such dicts
        ds:
            the dataset, e.g., TEMPOL2 or S5PL2 classes or popy.l2g_data dict
        xfield:
            if not none, sampled f will be multiplied to this field
        kwargs:
            additional inputs to self.get_NOxNO2_ratio
        '''        
        if not isinstance(ds,list):
            ds = self._interpolate(ds,self.get_NOxNO2_ratio,'f',**kwargs)
            if xfield is not None:
                ds[xfield] *= ds['f']
        else:
            for data in ds:
                data = self._interpolate(data,self.get_NOxNO2_ratio,'f',**kwargs)
                if xfield is not None:
                    data[xfield] *= data['f']
        return ds
    
    def get_NOxNO2_ratio(
            self,start_dt,end_dt,sel_lev=None,
            chm_pattern='Y%Y/M%m/D%d/GEOS-CF.v01.rpl.chm_tavg_1hr_g301x141_v23.%Y%m%d_%H%Mz.nc4',
            met_pattern='Y%Y/M%m/D%d/GEOS-CF.v01.rpl.met_tavg_1hr_g301x141_v23.%Y%m%d_%H%Mz.nc4',
            west=None,east=None,south=None,north=None
                         ):
        '''return ratio f from geoscf no and no2
        start/end_dt:
            datetime of start/end time
        sel_lev:
            tuple of upper/lower level to integrate, e.g., (68,72). none for all
        chm/met_pattern:
            file name pattern of 3D chm and met collections
        west/east/south/north:
            leave the option to adaptively load sub domain
        return:
            a tuple of coordinates (time,lat,lon) and combined f 
        '''
        west = west or self.west
        east = east or self.east
        south = south or self.south
        north = north or self.north
        start_dt = start_dt-pd.Timedelta(hours=1)
        start_dt = pd.Timestamp(
            year=start_dt.year,month=start_dt.month,day=start_dt.day,
            hour=start_dt.hour,minute=30
                                )
        end_dt = end_dt+pd.Timedelta(hours=1)
        end_dt = pd.Timestamp(
            year=end_dt.year,month=end_dt.month,day=end_dt.day,
            hour=end_dt.hour,minute=30
                                )
        timestamps = pd.date_range(start_dt,end_dt,freq='1h')
        chm_files,met_files,ts = [],[],[]
        for t in timestamps:
            chm_file = os.path.join(self.base_dir,t.strftime(chm_pattern))
            met_file = os.path.join(self.base_dir,t.strftime(met_pattern))
            if os.path.exists(chm_file) and os.path.exists(met_file):
                chm_files.append(chm_file)
                met_files.append(met_file)
                ts.append(t)
            else:
                self.logger.warning(f'{t} has no files!')
        ts = pd.DatetimeIndex(ts)
        fs = []
        for ifile,(chm_path,met_path) in enumerate(zip(chm_files,met_files)):
            with Dataset(chm_path,'r') as chem, Dataset(met_path,'r') as met:
                if ifile == 0:
                    lon = chem['lon'][:].filled(np.nan)
                    lat = chem['lat'][:].filled(np.nan)
                    lev = chem['lev'][:].filled(np.nan)
                    if sel_lev is None:
                        vmask = np.ones(len(lev),dtype=bool)
                    else:
                        vmask = (lev>=sel_lev[0]) & (lev<=sel_lev[1])
                    xmask = (lon>=west) & (lon<=east)
                    ymask = (lat>=south) & (lat<=north)
                    xs,ys = lon[xmask],lat[ymask]
                    ijkmesh = np.ix_(vmask,ymask,xmask)
                    ii = ijkmesh[0].squeeze()
                    jj = ijkmesh[1].squeeze()
                    kk = ijkmesh[2].squeeze()
                
                NO = chem['NO'][:,ii,jj,kk].filled(np.nan)
                NO2 = chem['NO2'][:,ii,jj,kk].filled(np.nan)
                Q = met['Q'][:,ii,jj,kk].filled(np.nan)
                DELP = met['DELP'][:,ii,jj,kk].filled(np.nan)
                
                VCD_A = (1-Q)*DELP/9.8/0.02897 # dry air sub col, mol/m2
                VCD_NO2 = (NO2*VCD_A).sum(axis=1) # sum along v dimension
                VCD_NO = (NO*VCD_A).sum(axis=1)
                fs.append((VCD_NO+VCD_NO2)/VCD_NO2)
        fs = np.concatenate(fs,axis=0)
        return (ts,ys,xs),fs


class MetSampler():
    def __init__(self,base_dir,
                 west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.west,self.east,self.south,self.north=west,east,south,north
        
    @staticmethod
    def _interpolate_era5(
        data,fetcher,fields_3d,fields_2d,altitudes=None,margin=0.5,**kwargs
    ):
        '''utility function for self.interpolate_era5
        data:
            data dict
        fetcher:
            callable, self.get_era5 or self.get_era5_2d
        fields_3d/2d:
            3d/2d fields to sample
        altitudes:
            a list of altitude values at which to interpret era5 3d wind, assuming scale
            height of 7500 m, default to [300]
        margin:
            if west/east/south/north are not in kwargs, load data this degree
            beyond bounds of lonc/latc
        kwargs:
            additional inputs to self.get_era5(_2d)
        '''
        start_dt = datedev_py(np.nanmin(data['UTC_matlab_datenum']))
        end_dt = datedev_py(np.nanmax(data['UTC_matlab_datenum']))
        west = kwargs.pop('west',np.nanmin(data['lonc'])-margin)
        east = kwargs.pop('east',np.nanmax(data['lonc'])+margin)
        south = kwargs.pop('south',np.nanmin(data['latc'])-margin)
        north = kwargs.pop('north',np.nanmax(data['latc'])+margin)
        mask = (~np.isnan(data['UTC_matlab_datenum'])) &\
                (~np.isnan(data['lonc'])) & (~np.isnan(data['latc']))
        coords,var_dict = fetcher(
            start_dt,end_dt,fields_3d=fields_3d,fields_2d=fields_2d,
            west=west,east=east,south=south,north=north,**kwargs
        )
        coords = list(coords)
        coords[0] = np.array([datetime2datenum(t) for t in coords[0]])
        coords = tuple(coords)
        if len(coords) == 3:#2d only, time,lat,lon
            for f in fields_2d:
                func = RegularGridInterpolator(
                    coords,var_dict[f],
                    bounds_error=False,fill_value=np.nan)
                
                data[f'era5_{f}'] = np.full(data['latc'].shape,np.nan)
                data[f'era5_{f}'][mask] = func((
                    data['UTC_matlab_datenum'][mask],
                    data['latc'][mask],
                    data['lonc'][mask]
                    ))
        elif len(coords) == 4:#3d, time,lev,lat,lon
            if 'sp' not in fields_2d:
                self.logger.warning('sp is a necessary 2d field for 3d sampling')
                fields_2d += ['sp']
            for f in fields_2d:
                func = RegularGridInterpolator(
                    (coords[0],coords[2],coords[3]),var_dict[f],
                    bounds_error=False,fill_value=np.nan)
                
                data[f'era5_{f}'] = np.full(data['latc'].shape,np.nan)
                data[f'era5_{f}'][mask] = func((
                    data['UTC_matlab_datenum'][mask],
                    data['latc'][mask],
                    data['lonc'][mask]
                    ))
            if altitudes is None:
                altitudes = [300]
            scale_height = 7500
            for altitude in altitudes:
                sounding_hPa = data['era5_sp'][mask]/100*np.exp(-altitude/scale_height)
                # trim pressure at altitude above ground within available range
                sounding_hPa[sounding_hPa>np.max(coords[1])] = np.max(coords[1])
                sounding_hPa[sounding_hPa<np.min(coords[1])] = np.min(coords[1])
                
                for f in fields_3d:
                    func = RegularGridInterpolator(
                        coords,var_dict[f],
                        bounds_error=False,fill_value=np.nan)
                    data[f'era5_{f}{altitude:.0f}'] = np.full(data['latc'].shape,np.nan)
                    data[f'era5_{f}{altitude:.0f}'][mask] = func((
                        data['UTC_matlab_datenum'][mask],
                        sounding_hPa,
                        data['latc'][mask],
                        data['lonc'][mask]
                        ))
        
        return data
    
    def interpolate_era5(
        self,ds,fields_3d=['u','v'],fields_2d=['u10','v10','sp'],altitudes=None,**kwargs
    ):
        '''sample era5 data fields to a dataset, which should be dict-like with keys
        UTC_matlab_datenum, latc, lonc, or a list of such dicts
        ds:
            the dataset, e.g., TEMPOL2 or S5PL2 classes or popy.l2g_data dict
        fields_3d:
            if not none, enables sampling 3d fields at altitudes
        fields_2d:
            2d fields to sample
        altitudes:
            a list of altitude values at which to interpret era5 3d wind, assuming scale
            height of 7500 m, default to [300]
        kwargs:
            additional inputs to self.get_era5(_2d)
        '''        
        if fields_3d is None:
            fetcher = self.get_era5_2d
        else:
            fetcher = self.get_era5
        if not isinstance(ds,list):
            ds = self._interpolate_era5(ds,fetcher,fields_3d,fields_2d,altitudes,**kwargs)
        else:
            for data in ds:
                data = self._interpolate_era5(data,fetcher,fields_3d,fields_2d,altitudes,**kwargs)
        return ds
    
    def get_era5(
        self,start_dt,end_dt,sel_lev=None,
        pattern_3d='Y%Y/M%m/D%d/CONUS_3D_%Y%m%d.nc',
        pattern_2d='Y%Y/M%m/D%d/CONUS_2D_%Y%m%d.nc',
        fields_3d=['u','v'],fields_2d=['u10','v10','sp'],
        west=None,east=None,south=None,north=None
    ):
        '''return selected 3d & 2d fields from era5
        start/end_dt:
            datetime of start/end time
        sel_lev:
            tuple of upper/lower level to output, in hPa, e.g., (700,1000). none for all
        pattern_3d/2d:
            file name pattern of 3D and 2D era5 data
        fields_3d/2d:
            lists of 3d/2d fields to return
        west/east/south/north:
            leave the option to adaptively load sub domain
        return:
            a tuple of coordinates (time,lev,lat,lon) and a dict of variables
        '''
        west = west or self.west
        east = east or self.east
        south = south or self.south
        north = north or self.north
        start_dt = start_dt-pd.Timedelta(hours=1)
        start_dt = pd.Timestamp(
            year=start_dt.year,month=start_dt.month,day=start_dt.day,
            hour=start_dt.hour
                                )
        end_dt = end_dt+pd.Timedelta(hours=1)
        end_dt = pd.Timestamp(
            year=end_dt.year,month=end_dt.month,day=end_dt.day,
            hour=end_dt.hour
                                )
        timestamps = pd.date_range(start_dt,end_dt,freq='1d')
        files_3d,files_2d = [],[]
        for t in timestamps:
            file_3d = os.path.join(self.base_dir,t.strftime(pattern_3d))
            file_2d = os.path.join(self.base_dir,t.strftime(pattern_2d))
            if os.path.exists(file_3d) and os.path.exists(file_2d):
                files_3d.append(file_3d)
                files_2d.append(file_2d)
            else:
                self.logger.info(f'{t} has no enough files!')
        
        for ifile,(path_3d,path_2d) in enumerate(zip(files_3d,files_2d)):
            with Dataset(path_3d,'r') as san, Dataset(path_2d,'r') as er:#san,er=3,2 in chinese
                if ifile == 0:
                    lon = san['longitude'][:].filled(np.nan)
                    lat = san['latitude'][:].filled(np.nan)
                    
                    if 'level' in san.dimensions.keys():
                        lev = san['level'][:].filled(np.nan)
                    elif 'pressure_level' in san.dimensions.keys():
                        lev = san['pressure_level'][:].filled(np.nan)[::-1]
                    else:
                        self.logger.error('unknown format');return
                    
                    if 'time' in san.dimensions.keys():
                        ts = pd.to_datetime(san['time'][:].filled(np.nan), unit='h', origin='1900-01-01')
                    elif 'valid_time' in san.dimensions.keys():
                        ts = pd.to_datetime(san['valid_time'][:].filled(np.nan), unit='s', origin='1970-01-01')
                    else:
                        self.logger.error('unknown format');return
                    
                    if sel_lev is None:
                        vmask = np.ones(len(lev),dtype=bool)
                    else:
                        vmask = (lev>=sel_lev[0]) & (lev<=sel_lev[1])
                    
                    xmask = (lon>=west) & (lon<=east)
                    ymask = (lat>=south) & (lat<=north)
                    xs,ys,zs = lon[xmask],lat[ymask][::-1],lev[vmask]
                    ijkmesh = np.ix_(vmask,ymask,xmask)
                    ii = ijkmesh[0].squeeze()
                    # 2024 era5 3d file has pressure descending
                    if 'pressure_level' in san.dimensions.keys():
                        ii = ii[::-1]
                    jj = ijkmesh[1].squeeze()[::-1]#flip lat to ascending
                    kk = ijkmesh[2].squeeze()
                    var_dict = {f:[] for f in fields_3d+fields_2d}
                else:
                    if 'time' in san.dimensions.keys():
                        ts = ts.append(
                            pd.to_datetime(san['time'][:].filled(np.nan), unit='h', origin='1900-01-01'))
                    elif 'valid_time' in san.dimensions.keys():
                        ts = ts.append(
                            pd.to_datetime(san['valid_time'][:].filled(np.nan), unit='s', origin='1970-01-01'))
                    else:
                        self.logger.error('unknown format');return
                for f in fields_3d:
                    var_dict[f].append(san[f][:,ii,jj,kk].filled(np.nan))
                for f in fields_2d:
                    var_dict[f].append(er[f][:,jj,kk].filled(np.nan))
        var_dict = {k:np.concatenate(v,axis=0) for k,v in var_dict.items()}
        return (ts,zs,ys,xs),var_dict
    
    def get_era5_2d(
        self,start_dt,end_dt,sel_lev=None,
        pattern_3d=None,
        pattern_2d='Y%Y/M%m/D%d/CONUS_2D_%Y%m%d.nc',
        fields_3d=None,fields_2d=['u100','v100','u10','v10','sp'],
        west=None,east=None,south=None,north=None
    ):
        '''return selected 2d fields from era5
        start/end_dt:
            datetime of start/end time
        sel_lev:
            no use
        pattern/fields_2d:
            file name pattern and and fields to load in 2D era5 data
        pattern/fields_3d:
            no use
        west/east/south/north:
            leave the option to adaptively load sub domain
        return:
            a tuple of coordinates (time,lat,lon) and a dict of variables
        '''
        west = west or self.west
        east = east or self.east
        south = south or self.south
        north = north or self.north
        start_dt = start_dt-pd.Timedelta(hours=1)
        start_dt = pd.Timestamp(
            year=start_dt.year,month=start_dt.month,day=start_dt.day,
            hour=start_dt.hour
                                )
        end_dt = end_dt+pd.Timedelta(hours=1)
        end_dt = pd.Timestamp(
            year=end_dt.year,month=end_dt.month,day=end_dt.day,
            hour=end_dt.hour
                                )
        timestamps = pd.date_range(start_dt,end_dt,freq='1d')
        files_2d = []
        for t in timestamps:
            file_2d = os.path.join(self.base_dir,t.strftime(pattern_2d))
            if os.path.exists(file_2d):
                files_2d.append(file_2d)
            else:
                self.logger.info(f'{t} has no enough files!')
        
        for ifile,path_2d in enumerate(files_2d):
            with Dataset(path_2d,'r') as er:#er=2 in chinese
                if ifile == 0:
                    lon = er['longitude'][:].filled(np.nan)
                    lat = er['latitude'][:].filled(np.nan)
                    
                    if 'time' in er.dimensions.keys():
                        ts = pd.to_datetime(er['time'][:].filled(np.nan), unit='h', origin='1900-01-01')
                    elif 'valid_time' in er.dimensions.keys():
                        ts = pd.to_datetime(er['valid_time'][:].filled(np.nan), unit='s', origin='1970-01-01')
                    else:
                        self.logger.error('unknown format');return
                    
                    xmask = (lon>=west) & (lon<=east)
                    ymask = (lat>=south) & (lat<=north)
                    xs,ys = lon[xmask],lat[ymask][::-1]
                    ijmesh = np.ix_(ymask,xmask)
                    ii = ijmesh[0].squeeze()[::-1]#flip lat to ascending
                    jj = ijmesh[1].squeeze()
                    var_dict = {f:[] for f in fields_2d}
                else:
                    if 'time' in er.dimensions.keys():
                        ts = ts.append(
                            pd.to_datetime(er['time'][:].filled(np.nan), unit='h', origin='1900-01-01'))
                    elif 'valid_time' in er.dimensions.keys():
                        ts = ts.append(
                            pd.to_datetime(er['valid_time'][:].filled(np.nan), unit='s', origin='1970-01-01'))
                    else:
                        self.logger.error('unknown format');return
                
                for f in fields_2d:
                    var_dict[f].append(er[f][:,ii,jj].filled(np.nan))
        var_dict = {k:np.concatenate(v,axis=0) for k,v in var_dict.items()}
        return (ts,ys,xs),var_dict


class S5PL2(list):
    def __init__(self,year,month,l2_dir_pattern,day=1,
                 west=-180,east=180,south=-90,north=90,
                 data_fields_l2g=None,max_cf=0.3,max_sza=70):
        self.logger = logging.getLogger(__name__)
        data_fields_l2g = data_fields_l2g or \
            ['column_amount','surface_altitude','cloud_fraction','SolarZenithAngle']
        timestamp = pd.Timestamp(year=year,month=month,day=day)
        l2_path = timestamp.strftime(l2_dir_pattern)
        self.l2_path = l2_path
        self.west,self.east,self.south,self.north=west,east,south,north
        if not os.path.exists(l2_path):
            self.logger.warning(f'{l2_path} does not exist!')
            return
        d = loadmat(l2_path)
        latc = d['output_subset']['latc'][0][0].squeeze()
        lonc = d['output_subset']['lonc'][0][0].squeeze()
        mask = (latc >= south) & (latc <= north) & \
            (lonc >= west) & (lonc <= east)
        if 'cloud_fraction' in data_fields_l2g:
            mask = mask & (d['output_subset']['cloud_fraction'][0][0].squeeze()<=max_cf)
        if 'SolarZenithAngle' in data_fields_l2g:
            mask = mask & (d['output_subset']['SolarZenithAngle'][0][0].squeeze()<=max_sza)
        if np.sum(mask) == 0:
            self.logger.warning('no pixels left!')
            return
        vs = data_fields_l2g+[
            'lonc','latc','lonr','latr','orbit','UTC_matlab_datenum',
            'across_track_position'
                              ]
        out = {}
        for v in vs:
            out[v] = d['output_subset'][v][0][0].squeeze()[mask,]
        self.out = out
        self.variables = vs
        del d
        orbits = np.unique(out['orbit'])
        orbit_masks = []
        use_orbits,start_time,end_time = [],[],[]
        
        for iorbit,orbit in enumerate(orbits):
            orbit_mask = out['orbit'] == orbit
            unique_times = np.unique(out['UTC_matlab_datenum'][orbit_mask])
            unique_xtracks = np.unique(out['across_track_position'][orbit_mask])
            nal = len(unique_times)
            nax = len(unique_xtracks)
            if nal >=3 and nax >= 3:
                orbit_masks.append(orbit_mask)
                use_orbits.append(int(orbit))
                start_time.append(datedev_py(np.nanmin(out['UTC_matlab_datenum'][orbit_mask])))
                end_time.append(datedev_py(np.nanmax(out['UTC_matlab_datenum'][orbit_mask])))
        self.orbit_masks = orbit_masks
        self.df = pd.DataFrame({
            'orbit':use_orbits,
            'start_time':pd.DatetimeIndex(start_time),
            'end_time':pd.DatetimeIndex(end_time)
        })
        # materialize the list
        for idx in range(len(orbit_masks)):
            self.append(self.get_element(idx))
        del out, self.out, self.orbit_masks
    
    def get_element(self, idx):
        orbit_mask = self.orbit_masks[idx]
        sub_out = {k:v[orbit_mask,] for k,v in self.out.items()}
        unique_times = np.sort(np.unique(sub_out['UTC_matlab_datenum']))
        values,counts = np.unique(
            np.round(
                np.diff(unique_times)*86400,4
                ),return_counts=True)
        alongtrack_dt = values[counts.argmax()]
        # warning if the along track time difference is unexpected
        if self.df['orbit'].iloc[idx] <= 9387:
            desired_dt = 1.08
        else:
            desired_dt = 0.84
        try:
            np.testing.assert_almost_equal(alongtrack_dt,desired_dt,2)
        except Exception as e:
            self.logger.warning(e)
            self.logger.warning('at orbit {}'.format(self.df['orbit'].iloc[idx]))
            alongtrack_dt = desired_dt
        alongtrack_dt /= 86400 # sec back to day
        al_idxs = np.round((unique_times-unique_times[0])/alongtrack_dt).astype(int)
        unique_xtracks = np.sort(np.unique(sub_out['across_track_position']))
        ax_idxs = (unique_xtracks-unique_xtracks[0]).astype(int)
        nal = al_idxs[-1]+1
        nax = ax_idxs[-1]+1
        granule = {}
        for k in sub_out.keys():
            if k in ['lonr','latr']:
                granule[k] = np.full((nal,nax,4),np.nan)
            else:
                granule[k] = np.full((nal,nax),np.nan)
        
        for il2 in range(len(sub_out['latc'])):
            al_i = al_idxs[
                np.where(unique_times==sub_out['UTC_matlab_datenum'][il2])[0][0]
                           ]
            ax_i = ax_idxs[
                np.where(unique_xtracks==sub_out['across_track_position'][il2])[0][0]
                ]
            for k in sub_out.keys():
                granule[k][al_i,ax_i,] = sub_out[k][il2,]
        return granule
