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
from netCDF4 import Dataset

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
        self.name = name or 'CTM'
    
    def get_SUSTech_CMAQ_columns(self,nlayer_to_sum=8):
        '''calculate columns
        keys:
            3D mixing ratio fields to integrate
        nlayer_to_sum:
            first 8 layers ~ 830 hPa, 1.5 km, 9 layers ~ 670 hPa, 3 km
        '''
        p_high = self['VGTOP'] + self['VGLVLS'][nlayer_to_sum] * (self['PRSFC']-self['VGTOP'])
        h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
        self.logger.warning(
            'suming layers 0-{} for columns, roughly topping at {:.0f} hPa, {:.0f} m'.format(
                    nlayer_to_sum,np.nanmean(p_high)/100,h_high_mean))
        # pressure thickness of layers, Pa
        p_intervals = -np.diff(
            np.array(
                [self['VGTOP'] + vglvl * (self['PRSFC']-self['VGTOP']) \
                 for vglvl in self['VGLVLS']]
            ).transpose([1,0,2,3]),
            axis=1)
        
        for key in ['NO','NO2']:
            # mol/m2
            self['{}_COL'.format(key)] = np.sum(p_intervals[:,:nlayer_to_sum,:,:] \
            * self[key][:,:nlayer_to_sum,:,:]/9.8/0.02896*1e-6,axis=1)
        
        self['NOx_COL'] = self['NO_COL'] + self['NO2_COL']
        self['f'] = self['NOx_COL'] / self['NO2_COL']
        
    def get_directional_derivative(self,keys=None,wind_layer=6,ukey='UWIND',vkey='VWIND'):
        
        if keys is None:
            keys = ['NO_COL','NOx_COL','NO2_COL','f']
        if self.name.lower() in ['cmaq']:
            p_low = self['VGTOP'] + self['VGLVLS'][wind_layer] * (self['PRSFC']-self['VGTOP'])
            p_high = self['VGTOP'] + self['VGLVLS'][wind_layer+1] * (self['PRSFC']-self['VGTOP'])
            h_low_mean = np.nanmean(np.log(self['PRSFC']/p_low)*7500)
            h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
            p_low_mean = np.nanmean(p_low)
            p_high_mean = np.nanmean(p_high)
            self.logger.warning(
                'using layer {} wind, roughly at {:.0f}-{:.0f} hPa, {:.0f}-{:.0f} m'.format(
                    wind_layer,p_low_mean/100,p_high_mean/100,h_low_mean,h_high_mean))
        
        u = self[ukey][:,wind_layer,:,:]
        v = self[vkey][:,wind_layer,:,:]
        # to do: dx_vec for lon lat grid
        dx = self['XCELL']
        for key in keys:
            vcd = self[key]
            dcdx = np.full_like(vcd,np.nan)
            dcdy = np.full_like(vcd,np.nan)
            dcdr = np.full_like(vcd,np.nan)
            dcds = np.full_like(vcd,np.nan)
            dcdx[:,:,1:-1] = (vcd[:,:,2:]-vcd[:,:,:-2])/(dx*2)
            dcdy[:,1:-1,:] = (vcd[:,2:,:]-vcd[:,:-2,:])/(dx*2)
            dcdr[:,1:-1,1:-1] = (vcd[:,2:,2:]-vcd[:,:-2,:-2])/(dx*np.sqrt(2)*2)
            dcds[:,1:-1,1:-1] = (vcd[:,2:,:-2]-vcd[:,:-2,2:])/(dx*np.sqrt(2)*2)

            self[key+'_DD_XY'] = dcdx * u + dcdy * v
            self[key+'_DD_RS'] = dcdr * (u * np.cos(np.pi/4) + v * np.sin(np.pi/4)) \
            + dcds * (u * (-np.cos(np.pi/4)) + v * np.sin(np.pi/4))
            self[key+'_DD'] = 0.5 * (self[key+'_DD_XY'] + self[key+'_DD_RS'])
        if all(np.isin(['NO2_COL','NO2_COL_DD','f','f_DD'],list(self.keys()))):
            self.logger.warning('calculating contribution of f to emissions')
            self['fxNO2_COL_DD'] = self['f'] * self['NO2_COL_DD']
            self['f_DDxNO2_COL'] = self['f_DD'] * self['NO2_COL']
    
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
                            self['VGTOP'] = nc.VGTOP
                            self['VGLVLS'] = nc.VGLVLS
                            self['XCELL'] = nc.XCELL
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
                            if (self.west,self.east,self.south,self.north) == (-180,180,-90,90):
                                (self.west,self.east,self.south,self.north) = \
                                (lonmesh.min(),lonmesh.max(),latmesh.min(),latmesh.max())
                    
                    # loop over fields in each file
                    for field in fields:
                        if imon == 0:
                            self[field] = nc[field][time_mask,]
                        else:
                            self[field] = np.concatenate(
                                (self[field],nc[field][time_mask,]),
                                axis=0)
    
#     def load_NAQPMS(self,lonlat_path,pressure=None,time=None,**kwargs):
#         import h5py
#         if pressure is not None:
#             self['pressure'] = pressure
#         else:
#             self['pressure'] = np.array([1000,950,925,900,850,800,
#                                          750,700,650,600,550,500])
#         # UTC
#         if time is not None:
#             self['time'] = time
#         else:
#             self['time'] = pd.date_range(dt.datetime(2023,5,1),
#                                          dt.datetime(2023,5,1,23),
#                                          freq='1h')-dt.timedelta(seconds=8*3600)
#         lonlat = loadmat(lonlat_path)
#         self['lonmesh'] = lonlat['lonM'].T
#         self['latmesh'] = lonlat['latM'].T
#         lon_center = np.mean([self.west,self.east])
#         lat_center = np.mean([self.south,self.north])
#         distance2center = np.sqrt(np.square(self['lonmesh']-lon_center)
#                                   +np.square(self['latmesh']-lat_center))
#         ind = np.unravel_index(np.argmin(distance2center),distance2center.shape)
#         lon_mask = (self['lonmesh'][ind[0],] >= self.west) & \
#         (self['lonmesh'][ind[0],] < self.east)
#         lat_mask = (self['latmesh'][:,ind[1]] >= self.south) & \
#         (self['latmesh'][:,ind[1]] < self.north)
#         ijmesh = np.ix_(lat_mask,lon_mask)
#         self['lonmesh'] = self['lonmesh'][ijmesh]
#         self['latmesh'] = self['latmesh'][ijmesh]
#         # to do: properly remove the day dimension
#         for k,v in kwargs.items():
#             with h5py.File(v) as f:
#                 self[k] = f['model_{}'.format(k)][()][...,*ijmesh][0]
#             # to do: confirm missing data
#             if k in ['psfc','pbl']:
#                 self[k][self[k]==9999.] = np.nan
    
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
        
    def plot(self,key,time=None,layer_index=None,**kwargs):
        if time is None:
            self.logger.warning('plotting time average')
            time_mask = np.ones(self['time'].shape,dtype=bool)
        else:
            time_mask = self['time'] == time
        
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
        
        if layer_index is not None:
            data = self[key][time_mask,layer_index,].squeeze()
        else:
            data = self[key][time_mask,].squeeze()
        if time is None:
            data = np.mean(data,axis=0)
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