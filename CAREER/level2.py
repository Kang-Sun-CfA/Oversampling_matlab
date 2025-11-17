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
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from popy import datedev_py, datetime2datenum, F_ellipse
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice', category=RuntimeWarning)

class L2ToL4():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_DD(self,ds,us=None,vs=None,fs=None,oval_inflation=1.,ifov_idxs=None):
        '''get directiona derivative
        ds:
            an L2 dataset below, not popy.l2g_data dict
        us/vs:
            lists of u/v fields, default [era5_u300,era5_u10] and [era5_v300,era5_v10]
        fs:
            list of fields to get dd, default ['column_amount','surface_altitude']
        oval_inflation,ifov_idxs:
            only used in _get_DD_oval. see that function
        '''
        us = us or ['era5_u300','era5_u10']
        vs = vs or ['era5_v300','era5_v10']
        fs = fs or ['column_amount','surface_altitude']
        # possible names of u dot grad(z0)
        wind_topo_fs = [
            'terrain_height_DD','terrain_height_DD_xy','terrain_height_DD_rs',
            'surface_altitude_DD','surface_altitude_DD_xy','surface_altitude_DD_rs'
        ]
        # trompomi l2
        if isinstance(ds,S5PL2):
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
        # iasi or cris l2
        elif isinstance(ds,IASIL2) or isinstance(ds,CrISL2):
            nds = {}
            for u,v,f in zip(us,vs,fs):
                f_ = [f] if not isinstance(f,list) else f
                if 'surface_altitude' in f_ and 'surface_altitude' not in ds.keys():
                    self.logger.warning('surface_altitude replaced by terrain_height')
                    f_ = ['terrain_height' if x == 'surface_altitude' else x for x in f_]
                nds.update(self._get_DD_oval(
                    ds,u,v,f_,oval_inflation=oval_inflation,ifov_idxs=ifov_idxs))
            ds = nds
            for wind_topo_f in wind_topo_fs:
                if wind_topo_f in ds.keys():
                    ds[wind_topo_f] *= ds['column_amount']
        # tempo l2
        else:
            for u,v,f in zip(us,vs,fs):
                f_ = [f] if not isinstance(f,list) else f
                if 'surface_altitude' in f_ and 'surface_altitude' not in ds.keys():
                    self.logger.warning('surface_altitude replaced by terrain_height')
                    f_ = ['terrain_height' if x == 'surface_altitude' else x for x in f_]
                ds = self._get_DD(ds,u,v,f_)
            for wind_topo_f in wind_topo_fs:
                if wind_topo_f in ds.keys():
                    ds[wind_topo_f] *= ds['column_amount']
        
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
    
    @staticmethod
    def latlon2m(lat1,lon1,lat2,lon2,m_per_lat,m_per_lon):
        '''function to return distance in meter using latlon matrices'''
        return np.sqrt(
            np.square((lat1-lat2)*m_per_lat)
            +np.square((lon1-lon2)*m_per_lon))
    
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
                
        for field in fields:
            f = data[field]
            # gradients
            dfdx = np.full_like(f,np.nan)
            dfdy = np.full_like(f,np.nan)
            dfdr = np.full_like(f,np.nan)
            dfds = np.full_like(f,np.nan)
            dfdx[:,1:-1] = (f[:,:-2]-f[:,2:])/ \
            self.latlon2m(lat1=data['latc'][:,:-2],lon1=data['lonc'][:,:-2],
                    lat2=data['latc'][:,2:],lon2=data['lonc'][:,2:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[:,1:-1]
                    )
            dfdy[1:-1,:] = (f[2:,:]-f[:-2,:])/ \
            self.latlon2m(lat1=data['latc'][2:,:],lon1=data['lonc'][2:,:],
                    lat2=data['latc'][:-2,:],lon2=data['lonc'][:-2,:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,:]
                    )
            dfdr[1:-1,1:-1] = (f[2:,:-2]-f[:-2,2:])/ \
            self.latlon2m(lat1=data['latc'][2:,:-2],lon1=data['lonc'][2:,:-2],
                    lat2=data['latc'][:-2,2:],lon2=data['lonc'][:-2,2:],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,1:-1]
                    )
            dfds[1:-1,1:-1] = (f[2:,2:]-f[:-2,:-2])/ \
            self.latlon2m(lat1=data['latc'][2:,2:],lon1=data['lonc'][2:,2:],
                    lat2=data['latc'][:-2,:-2],lon2=data['lonc'][:-2,:-2],
                    m_per_lat=m_per_lat,m_per_lon=m_per_lon[1:-1,1:-1]
                    )
            
            # directional derivatives
            data[field+'_DD_xy'] = windx*dfdx + windy*dfdy
            data[field+'_DD_rs'] = windr*dfdr + winds*dfds
            data[field+'_DD'] = (data[field+'_DD_xy']+data[field+'_DD_rs'])*0.5
        return data
    
    def _get_DD_oval(self,data,east_wind_field=None,north_wind_field=None,fields=None,
                     oval_inflation=1.,ifov_idxs=None):
        '''calculate directional derivatives, (u,v) dot (dvcd/dx,dvcd/dy) using non-tiled, oval l2 pixels
        east/north_wind_field:
            u/v wind field name in met data
        fields:
            data fields to calculate DD
        oval_inflation:
            inflate the l4 ovals by this number
        ifov_idxs:
            a list of a list of ifov idxs (0-based). 4 ifovs should define a square-like, starting from 
            upper left, going clockwise. default values in the code
        return:
            a similar dict with the first dimension reduced (4->1 for iasi and 9-4 for cris). adding 
            the DD fields
        '''
        if len(data['latc']) == 0:
            return data.copy()
        if ifov_idxs is None:
            if isinstance(data,IASIL2):
                # iasi, starting at upper left, going clockwise
                # first to third defines x, second to fourth defines y
                ifov_idxs = [[0,1,2,3]]
            if isinstance(data,CrISL2):
                # cris
                ifov_idxs = [[8,7,4,5],[7,6,3,4],[4,3,0,1],[5,4,1,2]]
            else:
                self.logger.error('l2 not compatible');return
        
        if fields is None:
            fields = ['column_amount']
        east_wind_field = east_wind_field or 'era5_u300'
        north_wind_field = north_wind_field or 'era5_v300'
        
        latc = np.concatenate(
            [np.mean(
                data['latc'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        lonc = np.concatenate(
            [np.mean(
                data['lonc'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        UTC_matlab_datenum = np.concatenate(
            [np.mean(
                data['UTC_matlab_datenum'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        oval_u = np.concatenate(
            [np.mean(
                data['u'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        oval_v = np.concatenate(
            [np.mean(
                data['v'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        oval_t = np.concatenate(
            [np.mean(
                data['t'][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )
        
        ds = dict(
            latc=latc,lonc=lonc,UTC_matlab_datenum=UTC_matlab_datenum,
            u=oval_u,v=oval_v,t=oval_t
        )
        
        m_per_lat = 111e3
        m_per_lon = m_per_lat * np.cos(np.radians(latc)).astype(np.float32)

        thetax = np.array(
            [
                np.arctan2(
                    m_per_lat*(data['latc'][idx[2],:]-data['latc'][idx[0],:]),
                    m_per_lon[i,:]*(data['lonc'][idx[2],:]-data['lonc'][idx[0],:])
                ) for i,idx in enumerate(ifov_idxs)
            ]
        )

        thetay = np.array(
            [
                np.arctan2(
                    m_per_lat*(data['latc'][idx[3],:]-data['latc'][idx[1],:]),
                    m_per_lon[i,:]*(data['lonc'][idx[3],:]-data['lonc'][idx[1],:])
                ) for i,idx in enumerate(ifov_idxs)
            ]
        )

        det_xy = np.cos(thetax)*np.sin(thetay) - np.sin(thetax)*np.cos(thetay)
        
        # eastward and northward wind
        windu = np.concatenate(
            [np.mean(
                data[east_wind_field][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        windv = np.concatenate(
            [np.mean(
                data[north_wind_field][idx,:],axis=0,keepdims=True
            ) for idx in ifov_idxs],axis=0
        )

        # xward and yward wind
        windx = (np.sin(thetay)*windu - np.cos(thetay)*windv)/det_xy
        windy = (-np.sin(thetax)*windu + np.cos(thetax)*windv)/det_xy

        for field in fields:
            f = data[field]
            
            dfdx = np.array(
                [
                    (f[idx[2],:]-f[idx[0],:])/self.latlon2m(
                        lat1=data['latc'][idx[2],:],lon1=data['lonc'][idx[2],:],
                        lat2=data['latc'][idx[0],:],lon2=data['lonc'][idx[0],:],
                        m_per_lat=m_per_lat,m_per_lon=m_per_lon[i,:]
                    ) for i,idx in enumerate(ifov_idxs)
                ]
            )

            dfdy = np.array(
                [
                    (f[idx[3],:]-f[idx[1],:])/self.latlon2m(
                        lat1=data['latc'][idx[3],:],lon1=data['lonc'][idx[3],:],
                        lat2=data['latc'][idx[1],:],lon2=data['lonc'][idx[1],:],
                        m_per_lat=m_per_lat,m_per_lon=m_per_lon[i,:]
                    ) for i,idx in enumerate(ifov_idxs)
                ]
            )
            # directional derivative
            ds[field+'_DD'] = windx*dfdx + windy*dfdy
            # average the field to l4 pixels as well
            ds[field] = np.concatenate(
                [np.nanmean(
                    f[idx,:],axis=0,keepdims=True
                ) for idx in ifov_idxs],axis=0
            )
        
        return ds
        

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

class IASIL2(dict):
    '''iasi l2 files flattern all soundings in a day to a long list, but actual soundings are grouped
    by orbits, scanlines, instantaneous field of regard (ifors), and instantaneous field of view
    (ifovs). each scanline has 30 ifors, each ifor has 4 ifovs, so each scanline has 120 pixels.
    this class identifies/groups ifors where directional derivatives are calculated
    '''
    def __init__(
        self,year,month,day,
        west=-130,east=-63,south=23,north=52,
        ellipse_lut_path='daysss.mat'
    ):
        '''
        year, month, day:
            int, identify a date
        west,east,south,north:
            spatial boundary, ifors in this box will be included by self.load_l2
        ellipse_lut_path:
            path to a look up table storing u, v, and t data to reconstruct IASI pixel ellipsis
        '''
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(ellipse_lut_path):
            self.logger.warning(f'{ellipse_lut_path} not found. try download')
            os.system('wget https://github.com/Kang-Sun-CfA/PU_KS_share/raw/refs/heads/master/daysss.mat')
        pixel_lut = loadmat(ellipse_lut_path)
        # the following are functions - interpolating major/minor axis and rotation of ellipse
        # at given latitude and pixel number
        f_uuu = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,121)),pixel_lut['uuu4']) 
        f_vvv = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,121)),pixel_lut['vvv4']) 
        f_ttt = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,121)),pixel_lut['ttt4']) 
        self.f_uuu,self.f_vvv,self.f_ttt = f_uuu,f_vvv,f_ttt
        date = pd.Timestamp(year=year,month=month,day=day)
        self.date = date
        self.west,self.east,self.south,self.north = west,east,south,north
        # constant, 4 ifov per ifor for iasi
        self.NIFOV_PER_IFOR = 4
        
    def load_l2(
        self,l2_path_pattern,data_fields=None,am_filter=True,land_filter=True,
        pre_filter=True,post_filter=True
    ):
        '''
        l2_path_pattern:
            pattern of l2 data path, e.g., 
            '/projects/academic/kangsun/data/IASIcNH3/IASI_METOPC_L2_NH3_%Y%m%d_ULB-LATMOS_V4.0.0R.nc'
        data_fields:
            fields to read from l2 netcdf, defaults to a list code below
        am/land/pre/post_filter:
            whether to apply filter to keep only morning (am), land, and good pre/post retrieval data
        '''
        l2_list = glob.glob(self.date.strftime(l2_path_pattern))
        
        if len(l2_list) != 1:
            self.logger.warning('Number of available files is not 1 for {}'.format(
                self.date.strftime('%Y%m%d')))
            return
        
        l2_path = l2_list[0]
        NIFOV_PER_IFOR = self.NIFOV_PER_IFOR
        if data_fields is None:
            data_fields = ['AERIStime','latitude','longitude',
                           'orbit_number','scanline_number','pixel_number','ifov_number',
                           'cloud_coverage','AMPM','LS_mask',
                           'nh3_total_column','nh3_total_column_random_uncertainty',
                           'nh3_total_column_systematic_uncertainty',
                           'prefilter', 'postfilter','ground_height']
        # ncd is a dict to temporally store netcdf data. too slow to read every time
        ncd = {}
        with Dataset(l2_path) as nc:
            nsounding = nc.dimensions['time'].size
            for n in data_fields:
                ncd[n] = nc[n][:].filled(np.nan)
            with np.errstate(divide='ignore',invalid='ignore'):
                # calculate the ifor number, should be 30 per scanline
                ncd['ifor_number'] = (ncd['pixel_number']-1)//NIFOV_PER_IFOR+1
        
        inbox = (ncd['longitude']>=self.west) & \
        (ncd['longitude']<=self.east) & \
        (ncd['latitude']>=self.south) & \
        (ncd['latitude']<=self.north) 
        if am_filter:
            inbox = inbox & (ncd['AMPM']==0)

        orbits = np.unique(ncd['orbit_number'][inbox])
        scanlines = []
        ifors = []
        all_idxs = np.arange(nsounding,dtype=int)
        ifor_idxs = []
        for orbit in orbits:
            scanlines_per_orbit = np.unique(
                ncd['scanline_number'][
                    inbox & (ncd['orbit_number']==orbit)
                ]
            )
            scanlines.append(scanlines_per_orbit)
            ifors_per_orbit = []
            for scanline in scanlines_per_orbit:
                ifors_per_scanline = np.unique(
                    ncd['ifor_number'][
                        inbox & (ncd['orbit_number']==orbit) & \
                        (ncd['scanline_number']==scanline)
                    ]
                )
                for iifor,ifor in enumerate(ifors_per_scanline):
                    ifor_mask = (ncd['orbit_number'] == orbit) & \
                    (ncd['scanline_number'] == scanline) & \
                    (ncd['ifor_number'] == ifor)
                    if ifor_mask.sum() == NIFOV_PER_IFOR:
                        ifor_idxs.append(all_idxs[ifor_mask])
                    # end of ifor loop per scanline
                ifors_per_orbit.append(ifors_per_scanline)
                # end of scanline loop per orbit
            ifors.append(ifors_per_orbit)
            # end of loop over orbits
        ifor_idxs = np.array(ifor_idxs).T
        for field in data_fields+['ifor_number']:
            if field == 'longitude':
                key = 'lonc'
            elif field == 'latitude':
                key = 'latc'
            elif field == 'nh3_total_column':
                key = 'column_amount'
            else:
                key = field
            self[key] = np.array([ncd[field][ifor_idx] for ifor_idx in ifor_idxs])
        # find out elliptical parameters using lookup table            
        self['u'] = self.f_uuu((self['latc'],self['pixel_number']))
        self['v'] = self.f_vvv((self['latc'],self['pixel_number']))
        self['t'] = self.f_ttt((self['latc'],self['pixel_number']))
        # convert aeristime (seconds since 2007-1-1) to matlab datenum (days since 0000-1-1)
        self['UTC_matlab_datenum'] = self['AERIStime']/86400+733043.
        # standardize surface altitude naming
        self['surface_altitude'] = self.pop('ground_height',np.zeros_like(self['latc']))
        if land_filter:
            self['column_amount'][self['LS_mask']!=1] = np.nan
        if pre_filter:
            self['column_amount'][self['prefilter']!=1] = np.nan
        if post_filter:
            self['column_amount'][self['postfilter']!=1] = np.nan
        self['column_amount'][self['column_amount'] > 1e36] = np.nan # fill value is 9.97e36
    
    def plot(
        self,data=None,ax=None,figsize=None,if_latlon=False,npoints=20,
        plot_field='column_amount',xlim=None,ylim=None,wind_kw=None,**kwargs
    ):
        '''plot function similar to tempo.TEMPOL2.plot'''
        data = data or self
        cmap = kwargs.pop('cmap','jet')
        alpha = kwargs.pop('alpha',1)
        func = kwargs.pop('func',lambda x:x)
        ec = kwargs.pop('ec','none')
        draw_colorbar = kwargs.pop('draw_colorbar',True)
        label = kwargs.pop('label',plot_field)
        shrink = kwargs.pop('shrink',0.75)
        cartopy_scale = kwargs.pop('cartopy_scale','50m')
        if ax is None:
            figsize = kwargs.pop('figsize',(10,5))
            if if_latlon:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True,
                                     subplot_kw={"projection": ccrs.PlateCarree()})
            else:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
        else:
            fig = None
        mask = ~np.isnan(data['latc'])
        verts = [F_ellipse(
            v,u,t,npoints,lonc,latc)[0].T for v,u,t,lonc,latc in zip(
            data['v'][mask],data['u'][mask],data['t'][mask],
            data['lonc'][mask],data['latc'][mask]
        )]
        cdata = data[plot_field][mask]
        vmin = kwargs.pop('vmin',np.nanmin(cdata))
        vmax = kwargs.pop('vmax',np.nanmax(cdata))
        collection = PolyCollection(
            verts,
            array=cdata,
            cmap=cmap,edgecolors=ec
        )
        collection.set_alpha(alpha)
        collection.set_clim(vmin=vmin,vmax=vmax)
        ax.add_collection(collection)
        
        if if_latlon:
            if cartopy_scale is not None:
                ax.coastlines(resolution=cartopy_scale, color='black', linewidth=1)
                ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='k', 
                               linestyle='-',zorder=0,lw=0.5)
        else:
            ax.set_aspect('equal', adjustable='box')
        if draw_colorbar:
            cb = plt.colorbar(collection,ax=ax,label=label,shrink=shrink)
        else:
            cb = None
        
        if xlim is None:
            ax.set_xlim([np.min([v[:,0].min() for v in verts]),
                         np.max([v[:,0].max() for v in verts])])
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim([np.min([v[:,1].min() for v in verts]),
                         np.max([v[:,1].max() for v in verts])])
        else:
            ax.set_ylim(ylim)
        
        # draw wind vectors
        if wind_kw is not None:
            scale = wind_kw.pop('scale',200)
            width = wind_kw.pop('width',0.002)
            east_wind_field = wind_kw.pop('east_wind_field','era5_u100')
            north_wind_field = wind_kw.pop('north_wind_field','era5_v100')
            wind_e = data[east_wind_field][mask]
            wind_n = data[north_wind_field][mask]
            if if_latlon:
                basis_o1 = data['lonc'][mask]
                basis_o2 = data['latc'][mask]
            else:
                self.logger.warning('not implemented for wind in if_latlon=False')
                return
            ax.quiver(basis_o1,basis_o2,
                      wind_e,wind_n,scale=scale,width=width,**wind_kw)
            
        return dict(fig=fig,ax=ax,cb=cb)


class CrISL2(dict):
    '''level 2 class to handle CrIS data, similar to IASIL2
    '''
    def __init__(
        self,year,month,day,
        west=-130,east=-63,south=23,north=52,
        ellipse_lut_path='CrIS_footprint.mat'
    ):
        '''
        year, month, day:
            int, identify a date
        west,east,south,north:
            spatial boundary, ifors in this box will be included by self.load_l2
        ellipse_lut_path:
            path to a look up table storing u, v, and t data to reconstruct CrIS pixel ellipsis
        '''
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(ellipse_lut_path):
            self.logger.warning(f'{ellipse_lut_path} not found. try download')
            os.system('wget https://github.com/Kang-Sun-CfA/PU_KS_share/raw/refs/heads/master/CrIS_footprint.mat')
        pixel_lut = loadmat(ellipse_lut_path)
        # the following are functions - interpolating major/minor axis and rotation of ellipse
        # at given latitude and pixel number
        f_uuu = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['uuu4']) 
        f_vvv = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['vvv4']) 
        f_ttt = RegularGridInterpolator((np.arange(-90.,91.),np.arange(1,271)),pixel_lut['ttt4']) 
        self.f_uuu,self.f_vvv,self.f_ttt = f_uuu,f_vvv,f_ttt
        date = pd.Timestamp(year=year,month=month,day=day)
        self.date = date
        self.west,self.east,self.south,self.north = west,east,south,north
        # constant, 9 ifov per ifor for cris
        self.NIFOV_PER_IFOR = 9
    
    @staticmethod
    def download_l2(
        login,psword,dts,l2_dir,if_remove_tgz=True,
        url_pattern='https://hpfx.collab.science.gc.ca/~mas001/satellite_ext/cris/noaa20/nh3/v1_6_4/lite/%Y/%m/lite_cris_noaa20_nh3_global_v1_6_4_%Y_%m_%d.tar.gz'
    ):
        '''
        login,psword:
            username and password, see https://hpfx.collab.science.gc.ca/~mas001/satellite_ext/cris/snpp/nh3/v1_6_4/CrIS_NH3_data_usage_statement.pdf
        dts:
            an array of dates/datetimes to download l2 files
        l2_dir:
            e.g., /vscratch/grp-kangsun/kangsun/CrISNH3/L2_noaa20/
        if_remove_tgz:
            whether to remove the tgz file
        url_pattern:
            default is for noaa20. replace by snpp if wanted
        '''
        cwd = os.getcwd()
        for datetime in dts:
            os.makedirs(l2_dir,exist_ok=True)
            os.chdir(l2_dir)
            url = datetime.strftime(url_pattern)
            tgz_fn = os.path.split(url)[-1]
            run_str = f'curl -u {login}:{psword} {url} -O; tar -xzf {tgz_fn}'
            os.system(run_str)
            if if_remove_tgz:
                os.remove(tgz_fn)
        os.chdir(cwd)
            
    @staticmethod
    def _pn(number):
        return f'p{np.abs(number):03d}_0_' if number >= 0 else f'n{np.abs(number):03d}_0_'
    
    def load_l2(
        self,l2_path_pattern,data_fields=None,am_filter=True,min_LandFraction=0.,
        min_Quality_Flag=5,min_DOF=0.1,include_Cloud_Flag=[0,2,3]
    ):
        '''
        l2_path_pattern:
            pattern of l2 data path, e.g., 
            '/projects/academic/kangsun/data/CrISNH3/L2/%Y/%m/%d/Lite_Combined_NH3*%Y%m%d.nc'
        data_fields:
            fields to read from l2 netcdf, defaults to a list code below
        am_filter:
            whether to apply filter to keep only morning (am) data
        min_LandFraction/Quality_Flag/DOF:
            futher restrict data
        include_Cloud_Flag:
            which cloud flag to include
        '''
        # all file names for the day
        l2_list_day = glob.glob(self.date.strftime(l2_path_pattern))
        # find out only tiles in the bound
        lon_bound = np.arange(-180,180,20)
        lat_bound = np.arange(-90,90,15)
        lon_bound = lon_bound[np.where(lon_bound<=self.west)[0][-1]:np.min([np.where(lon_bound>=self.east)[0][0]+1,len(lon_bound)])]
        lat_bound = lat_bound[np.where(lat_bound<=self.south)[0][-1]:np.min([np.where(lat_bound>=self.north)[0][0]+1,len(lat_bound)])]
        l2_list = []
        for ilon in range(len(lon_bound[:-1])):
            for ilat in range(len(lat_bound[:-1])):
                key_str = self._pn(lon_bound[ilon])+self._pn(lon_bound[ilon+1])+\
                self._pn(lat_bound[ilat])+self._pn(lat_bound[ilat+1])
                tiles = [s for s in l2_list_day if key_str in s]
                if len(tiles) > 1:
                    self.logger.error('this should not happen')
                l2_list += tiles
        self.l2_list = l2_list
        if len(l2_list) == 0:
            self.logger.warning('no l2 files for {}'.format(self.date.strftime('%Y%m%d')))
            return
        NIFOV_PER_IFOR = self.NIFOV_PER_IFOR
        if data_fields is None:
            data_fields = ['DOF','Day_Night_Flag','LandFraction','Latitude','Longitude',
                    'Quality_Flag','Run_ID','mdate','tot_col','xretv','pressure',
                    'tot_col_total_error','Cloud_Flag']
        ncds = []
        for l2_path in l2_list:
            # ncd is a dict to temporally store netcdf data. too slow to read every time
            ncd = {}
            with Dataset(l2_path) as nc:
                for n in data_fields:
                    if pd.api.types.is_string_dtype(nc['Run_ID'].dtype):
                        ncd[n] = nc[n][:]
                    else:
                        ncd[n] = nc[n][:].filled(np.nan).data
            mask = (ncd['Latitude'] >= self.south) & (ncd['Latitude'] <= self.north) &\
            (ncd['Longitude'] >= self.west) & (ncd['Longitude'] <= self.east) &\
            (ncd['LandFraction'] >= min_LandFraction) &\
            (ncd['Quality_Flag'] >= min_Quality_Flag) &\
            np.isin(ncd['Cloud_Flag'],include_Cloud_Flag)
            if am_filter:
                mask = mask & (ncd['Day_Night_Flag'] == 1)
            ncd = {k:v[mask,] for k,v in ncd.items()}
            # get surface properties and dump profiles
            ncd['sfcvmr'] = np.zeros_like(ncd['Latitude'])
            ncd['surface_pressure'] = np.zeros_like(ncd['Latitude'])
            for io in range(len(ncd['Latitude'])):
                index = (ncd['pressure'][io,] > 0)
                pressure = ncd['pressure'][io,index]
                xretv = ncd['xretv'][io,index]
                ncd['sfcvmr'][io] = xretv[0]
                ncd['surface_pressure'][io] = pressure[0]
            # no surface altitude provided, estimate roughly
            ncd['surface_altitude'] = np.log(1013.25/ncd['surface_pressure'])*7500
            ncd.pop('xretv');
            ncd.pop('pressure');
            ncds.append(ncd)
        
        ncd = {}
        for k in ncds[0].keys():
            ncd[k] = np.concatenate([nd[k] for nd in ncds])
        ncd['ifor_number'] = np.array([int(run_id[-8:-4]) for run_id in ncd['Run_ID']])
        ncd['ifov_number'] = np.array([int(run_id[-3:]) for run_id in ncd['Run_ID']])
        ncd['scanline_number'] = np.array([int(run_id[16:21]) for run_id in ncd['Run_ID']])
        ncd['pixel_number'] = (ncd['ifor_number']-1)*9+ncd['ifov_number']
        ncd['UTC_matlab_datenum'] = ncd['mdate']+np.float64(719529.)
        ncd.pop('Run_ID');
        ncd['column_amount'] = ncd.pop('tot_col')/6.02214e19
        ncd['column_uncertainty'] = ncd.pop('tot_col_total_error')/6.02214e19
        ncd['latc'] = ncd.pop('Latitude')
        ncd['lonc'] = ncd.pop('Longitude')
        # find out elliptical parameters using lookup table            
        ncd['u'] = self.f_uuu((ncd['latc'],ncd['pixel_number']))
        ncd['v'] = self.f_vvv((ncd['latc'],ncd['pixel_number']))
        ncd['t'] = self.f_ttt((ncd['latc'],ncd['pixel_number']))
        self.ncd = ncd
        # for each unique time stamp, find unique ifor numbers
        unique_dn = np.unique(ncd['UTC_matlab_datenum'])
        uifors = []
        udns = []
        for idn,dn in enumerate(unique_dn):
            mask = ncd['UTC_matlab_datenum'] == dn
            uifor = list(np.unique(ncd['ifor_number'][mask]))
            uifors += uifor
            udns += list(np.ones(len(uifor))*dn)
        # a dataframe to identify unique ifors
        uifor_df = pd.DataFrame(data={'ifor':uifors,'dn':udns})
        # initialize nan arrays [9,number of unique ifors]
        for k in ncd.keys():
            self[k] = np.full((NIFOV_PER_IFOR,uifor_df.shape[0]),np.nan)
        # loop over each unique ifor, assign available ifovs from ncd
        for irow,row in uifor_df.iterrows():
            mask = (ncd['UTC_matlab_datenum'] == row.dn) & (ncd['ifor_number'] == row.ifor)
            ifovs = ncd['ifov_number'][mask]
            for ifov_idx,ifov in enumerate(ifovs):
                for k in self.keys():
                    self[k][ifov-1,irow] = ncd[k][mask][ifov_idx]
    
    def save_l2g(self,l2g_path_pattern=None,path=None,delete_existing=True):
        path = path or self.date.strftime(l2g_path_pattern)
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.exists(path) and delete_existing:
            os.remove(path)
        with Dataset(path,'w',format='NETCDF4') as nc:
            nc.createDimension('NIFOV_PER_IFOR',self['latc'].shape[0])
            nc.createDimension('NIFOR',self['latc'].shape[1])
            for k,v in self.items():
                var = nc.createVariable(k,'f8',('NIFOV_PER_IFOR','NIFOR'))
                var[:] = v
    
    def load_l2g(self,l2g_path_pattern=None,path=None,min_LandFraction=0.,
        min_Quality_Flag=5,min_DOF=0.1,include_Cloud_Flag=[0,2,3]):
        path = path or self.date.strftime(l2g_path_pattern)
        if not os.path.exists(path):
            self.logger.warning(f'{path} does not exist')
            return
        with Dataset(path,'r') as nc:
            for varname in nc.variables:
                variable = nc.variables[varname]
                self[varname] = variable[:].filled(np.nan)
        mask = (self['latc'] >= self.south) & (self['latc'] <= self.north) &\
            (self['lonc'] >= self.west) & (self['lonc'] <= self.east) &\
            (self['LandFraction'] >= min_LandFraction) &\
            (self['Quality_Flag'] >= min_Quality_Flag) &\
            np.isin(self['Cloud_Flag'],include_Cloud_Flag)
        for k in self.keys():
            self[k][~mask] = np.nan
        row_mask = np.sum(mask,axis=0) > 0
        for k in self.keys():
            self[k] = self[k][:,row_mask]
    
    def plot(
        self,data=None,ax=None,figsize=None,if_latlon=False,npoints=20,
        plot_field='column_amount',xlim=None,ylim=None,wind_kw=None,**kwargs
    ):
        '''plot function similar to tempo.TEMPOL2.plot'''
        data = data or self
        cmap = kwargs.pop('cmap','jet')
        alpha = kwargs.pop('alpha',1)
        func = kwargs.pop('func',lambda x:x)
        ec = kwargs.pop('ec','none')
        draw_colorbar = kwargs.pop('draw_colorbar',True)
        label = kwargs.pop('label',plot_field)
        shrink = kwargs.pop('shrink',0.75)
        cartopy_scale = kwargs.pop('cartopy_scale','50m')
        if ax is None:
            figsize = kwargs.pop('figsize',(10,5))
            if if_latlon:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True,
                                     subplot_kw={"projection": ccrs.PlateCarree()})
            else:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
        else:
            fig = None
        mask = ~np.isnan(data['latc'])
        verts = [F_ellipse(
            v,u,t,npoints,lonc,latc)[0].T for v,u,t,lonc,latc in zip(
            data['v'][mask],data['u'][mask],data['t'][mask],
            data['lonc'][mask],data['latc'][mask]
        )]
        cdata = data[plot_field][mask]
        vmin = kwargs.pop('vmin',np.nanmin(cdata))
        vmax = kwargs.pop('vmax',np.nanmax(cdata))
        collection = PolyCollection(
            verts,
            array=cdata,
            cmap=cmap,edgecolors=ec
        )
        collection.set_alpha(alpha)
        collection.set_clim(vmin=vmin,vmax=vmax)
        ax.add_collection(collection)
        
        if if_latlon:
            if cartopy_scale is not None:
                ax.coastlines(resolution=cartopy_scale, color='black', linewidth=1)
                ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='k', 
                               linestyle='-',zorder=0,lw=0.5)
        else:
            ax.set_aspect('equal', adjustable='box')
        if draw_colorbar:
            cb = plt.colorbar(collection,ax=ax,label=label,shrink=shrink)
        else:
            cb = None
        
        if xlim is None:
            ax.set_xlim([np.min([v[:,0].min() for v in verts]),
                         np.max([v[:,0].max() for v in verts])])
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim([np.min([v[:,1].min() for v in verts]),
                         np.max([v[:,1].max() for v in verts])])
        else:
            ax.set_ylim(ylim)
        
        # draw wind vectors
        if wind_kw is not None:
            scale = wind_kw.pop('scale',200)
            width = wind_kw.pop('width',0.002)
            east_wind_field = wind_kw.pop('east_wind_field','era5_u100')
            north_wind_field = wind_kw.pop('north_wind_field','era5_v100')
            wind_e = data[east_wind_field][mask]
            wind_n = data[north_wind_field][mask]
            if if_latlon:
                basis_o1 = data['lonc'][mask]
                basis_o2 = data['latc'][mask]
            else:
                self.logger.warning('not implemented for wind in if_latlon=False')
                return
            ax.quiver(basis_o1,basis_o2,
                      wind_e,wind_n,scale=scale,width=width,**wind_kw)
            
        return dict(fig=fig,ax=ax,cb=cb)