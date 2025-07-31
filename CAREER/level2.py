'''
updated classes for level 2 data of grading spectrometers
samplers of met/ctm data on level 2 pixels
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
        for iorbit,orbit in enumerate(orbits):
            orbit_mask = out['orbit'] == orbit
            unique_times = np.unique(out['UTC_matlab_datenum'][orbit_mask])
            unique_xtracks = np.unique(out['across_track_position'][orbit_mask])
            nal = len(unique_times)
            nax = len(unique_xtracks)
            if nal >=3 and nax >= 3:
                orbit_masks.append(orbit_mask)
        self.orbit_masks = orbit_masks
        self.df = pd.DataFrame({'orbit':orbits})
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
