import sys, os, glob
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.io import loadmat
from pyproj import Proj

mw_g_mol = dict(no=30,no2=46)
SCALE_HEIGHT = 7500

class CTM(dict):
    '''a class based on dictionary to handle CTM data'''
    def __init__(self,name=None,west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.name = name
    
    def load_SUSTech_CMAQ(self,time,
                          file_to_var_mapper=None,
                          file_path_pattern='/projects/academic/kangsun/data/CTM/SUSTech/*_%Y%m.nc'):
        '''load cmaq files from sustech
        time:
            a hourly pd date range
        file_to_var_mapper:
            a dict maps header of cmap file to the variables to be loaded from the file
        file_path_pattern:
            paths to cmaq files. * for different file header pattern
        '''
        self['time'] = time
        file_to_var_mapper = file_to_var_mapper or\
        {'CONC_NOx':['NO','NO2'],
        'MET':['UWIND','VWIND','PBLH'],
        'PRSFC':['PRSFC']}
        # loop over monthly files
        for imon,mon in enumerate(pd.period_range(time.min(),time.max(),freq='1M')):
            # loop over each file
            for ifile,(file_header,fields) in enumerate(file_to_var_mapper.items()):
                fn = mon.strftime(file_path_pattern.replace('*',file_header))
                self.logger.info('loading {}'.format(fn))
                with Dataset(fn,'r') as nc:
                    # assume all files in a month have the same timestamp
                    if ifile == 0:
                        nc_time = pd.to_datetime(
                            ['{0}'.format(d[0])+'{0:0>6}'.format(d[1])
                             for d in nc['TFLAG'][:,0,:]],
                            format='%Y%j%H%M%S')
                        time_mask = nc_time.isin(time)
                        # assume all files have the same spatial coordinates
                        if imon == 0:
                            # create the x and y grids
                            xgrid = np.arange(nc.NCOLS)*nc.XCELL+nc.XORIG + nc.XCELL/2
                            ygrid = np.arange(nc.NROWS)*nc.YCELL+nc.YORIG + nc.YCELL/2
                            # transform x y to lon lat
                            proj = Proj(proj='lcc',
                                       lat_1=nc.P_ALP,
                                       lat_2=nc.P_BET,
                                       lat_0=nc.YCENT,
                                       lon_0=nc.XCENT,
                                       x_0=0,y_0=0,
                                       datum='WGS84')
                            lonmesh, latmesh = proj(*np.meshgrid(xgrid, ygrid),inverse=True)
                            self.proj = proj
                            self['xgrid'] = xgrid
                            self['ygrid'] = ygrid
                            self['lonmesh'] = lonmesh
                            self['latmesh'] = latmesh
                    
                    # loop over fields in each file
                    for field in fields:
                        if imon == 0:
                            self[field] = nc[field][time_mask,]
                        else:
                            self[field] = np.concatenate(
                                (self[field],nc[field][time_mask,]),
                                axis=0)
    
    def load_NAQPMS(self,lonlat_path,pressure=None,time=None,**kwargs):
        import h5py
        if pressure is not None:
            self['pressure'] = pressure
        else:
            self['pressure'] = np.array([1000,950,925,900,850,800,
                                         750,700,650,600,550,500])
        # UTC
        if time is not None:
            self['time'] = time
        else:
            self['time'] = pd.date_range(dt.datetime(2023,5,1),
                                         dt.datetime(2023,5,1,23),
                                         freq='1h')-dt.timedelta(seconds=8*3600)
        lonlat = loadmat(lonlat_path)
        self['lonmesh'] = lonlat['lonM'].T
        self['latmesh'] = lonlat['latM'].T
        lon_center = np.mean([self.west,self.east])
        lat_center = np.mean([self.south,self.north])
        distance2center = np.sqrt(np.square(self['lonmesh']-lon_center)
                                  +np.square(self['latmesh']-lat_center))
        ind = np.unravel_index(np.argmin(distance2center),distance2center.shape)
        lon_mask = (self['lonmesh'][ind[0],] >= self.west) & \
        (self['lonmesh'][ind[0],] < self.east)
        lat_mask = (self['latmesh'][:,ind[1]] >= self.south) & \
        (self['latmesh'][:,ind[1]] < self.north)
        ijmesh = np.ix_(lat_mask,lon_mask)
        self['lonmesh'] = self['lonmesh'][ijmesh]
        self['latmesh'] = self['latmesh'][ijmesh]
        # to do: properly remove the day dimension
        for k,v in kwargs.items():
            with h5py.File(v) as f:
                self[k] = f['model_{}'.format(k)][()][...,*ijmesh][0]
            # to do: confirm missing data
            if k in ['psfc','pbl']:
                self[k][self[k]==9999.] = np.nan
    
    def get_NAQPMS_columns(self,keys,if_use_pbl=True,pressure_thickness=200.):
        
        def _get_column(number_density_layer,pressure_level,p0,p1,scale_height=7500.):
            '''calculate column amount between to pressure values
            number_density_layer:
                mol/m3 of gas species in each layer
            pressure_level:
                pressure of each level, from high pressure to low
            p0:
                higher pressure boundary
            p1:
                lower pressure boundary
            scale_height:
                atmosphere scale height in m
            '''
            sub_columns = number_density_layer * \
            scale_height * np.log(pressure_level[:-1]/pressure_level[1:])
            cum_col_toa2sfc = np.insert(np.cumsum(sub_columns[::-1]),0,0)
            p_toa2sfc = pressure_level[::-1]
            return np.interp(p0,p_toa2sfc,cum_col_toa2sfc) - np.interp(p1,p_toa2sfc,cum_col_toa2sfc)
        
        if if_use_pbl:
            p_1 = self['psfc'] * np.exp(-self['pbl']/SCALE_HEIGHT)
        else:
            p_1 = self['psfc'] - pressure_thickness
        mask = p_1 < np.min(self['pressure'])
        if np.sum(mask) >= 0:
            self.logger.warning('{} upper pressure over the boundary'.format(np.sum(mask)))
            p_1[mask] = np.min(self['pressure'])
        
        p_0 = self['psfc'].copy()
        p_0[p_0 > np.max(self['pressure'])] = np.max(self['pressure'])
        
        for key in keys:
            key_shape = self[key].shape
            profile = 0.5*(ctm[key][:,:-1,:,:]+ctm[key][:,1:,:,:])*1e-6/mw_g_mol[key] # ug/m3 to mol/m3
            self['{}_col'.format(key)] = np.array(
            [
                [
                    [_get_column(profile[itime,:,iy,ix],
                                 self['pressure'],
                                 p_0[itime,iy,ix],p_1[itime,iy,ix],
                                 scale_height=SCALE_HEIGHT
                                ) for ix in range(key_shape[3])
                    ] for iy in range(key_shape[2])
                ] for itime in range(key_shape[0])
            ])
        
    def plot(self,key,time,pressure=None,**kwargs):
        time_mask = self['time'] == time
        if pressure is not None:
            pressure_mask = self['pressure'] == pressure
        ax = kwargs.pop('ax',None)
        if ax is None:
            figsize = kwargs.pop('figsize',(10,5))
            fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=figsize,
                                  subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = None
        cmap = kwargs.pop('cmap','jet')
        vmin = kwargs.pop('vmin',None)
        vmax = kwargs.pop('vmax',None)
        func = kwargs.pop('func',lambda x:x)
        cartopy_scale = kwargs.pop('cartopy_scale','50m')
        draw_colorbar = kwargs.pop('draw_colorbar',True)
        label = kwargs.pop('label',key)
        shrink = kwargs.pop('shrink',0.75)
        extent = kwargs.pop('extent',[self.west, self.east, self.south, self.north])
        
        if pressure is not None:
            data = self[key][time_mask,pressure_mask,].squeeze()
        else:
            data = self[key][time_mask,].squeeze()
        pc = ax.pcolormesh(self['lonmesh'],self['latmesh'],
                           func(data),cmap=cmap,vmax=vmax,vmin=vmin,**kwargs)
        ax.coastlines(resolution=cartopy_scale, color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='k', 
                           linestyle='-',zorder=0,lw=0.5)
        if draw_colorbar:
            cb = plt.colorbar(pc,ax=ax,label=label,shrink=shrink)
        else:
            cb = None
        ax.set_extent(extent)
        figout = dict(fig=fig,ax=ax,pc=pc,cb=cb)
        return figout