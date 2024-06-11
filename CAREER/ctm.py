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
import statsmodels.formula.api as smf

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
        self.logger.info(
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
        # sum layered emissions to a single one
        if all(np.isin(['NO_emis','NO2_emis'],list(self.keys()))):
            self['NO_emis_COL'] = np.nansum(self['NO_emis'],axis=1)
            self['NO2_emis_COL'] = np.nansum(self['NO2_emis'],axis=1)
            self['NOx_emis_COL'] = self['NO_emis_COL'] + self['NO2_emis_COL']
    
    def fit_topo(self,resample_rule='1M',NOxorNO2='NOx',mask=None,fit_chem=True):
        '''generate topo fit coefficients, see fit_topography in popy
        '''
        if mask is None:
            mask = np.ones((len(self['ygrid']),len(self['xgrid'])),dtype=bool)
        if 'wind_topo' not in self.keys():
            self['wind_topo'] = self['surface_altitude_DD'] * self[f'{NOxorNO2}_COL']
        resampler = self.resample(resample_rule,
                                  ['wind_topo',f'{NOxorNO2}_COL_DD',f'{NOxorNO2}_COL'])
        self['<wind_column_topo>'] = np.full_like(self['<wind_topo>'],np.nan)
        self.topo_fits = []
        for i,(k,v) in enumerate(resampler.indices.items()):
            df = pd.DataFrame(dict(y=self[f'<{NOxorNO2}_COL_DD>'][i,][mask],
                                   wt=self['<wind_topo>'][i,][mask],
                                   vcd=self[f'<{NOxorNO2}_COL>'][i,][mask])).dropna()
            if fit_chem:
                topo_fit = smf.ols('y ~ wt + vcd', data=df).fit()
            else:
                topo_fit = smf.ols('y ~ wt', data=df).fit()
            self.topo_fits.append(topo_fit)
            self['<wind_column_topo>'][i,] = self[f'<{NOxorNO2}_COL_DD>'][i,]\
            -topo_fit.params['wt']*self['<wind_topo>'][i,]
        self.df['topo_scale_height'] = [-1/topo_fit.params['wt'] for topo_fit in self.topo_fits]
        self.df['topo_rmse'] = [np.sqrt(topo_fit.mse_resid) for topo_fit in self.topo_fits]
        self.df['topo_r2'] = [topo_fit.rsquared for topo_fit in self.topo_fits]
    
    def fit_chem(self,resample_rule='1M',NOxorNO2='NOx',mask=None,chem_fit_order=1):
        '''generate chem fit coefficients, see fit_chemistry in popy
        '''
        if mask is None:
            mask = np.ones((len(self['ygrid']),len(self['xgrid'])),dtype=bool)
        if 'wind_topo' not in self.keys():
            self['wind_topo'] = self['surface_altitude_DD'] * self[f'{NOxorNO2}_COL']
        resampler = self.resample(resample_rule,
                                  ['wind_topo',f'{NOxorNO2}_COL_DD',f'{NOxorNO2}_COL'])
        if '<wind_column_topo>' not in self.keys():
            self.logger.warning('topo fit was not done. doing it now')
            self.fit_topo(resample_rule,NOxorNO2,mask)
        self['<wind_column_topo_chem>'] = self['<wind_column_topo>'].copy()
        self.chem_fits = []
        reg_formula = 'y ~'
        for i,(k,v) in enumerate(resampler.indices.items()):
            df = pd.DataFrame(dict(y=self['<wind_column_topo>'][i,][mask]))
            for order in range(1,1+chem_fit_order):
                df['vcd{}'.format(order)] = self[f'<{NOxorNO2}_COL>'][i,][mask]**order
                reg_formula += ' + vcd{}'.format(order)
            df = df.dropna()
            chem_fit = smf.ols(reg_formula, data=df).fit()
            self.chem_fits.append(chem_fit)
            for order in range(1,1+chem_fit_order):
                self['<wind_column_topo_chem>'][i,] -= \
                chem_fit.params['vcd{}'.format(order)]*self[f'<{NOxorNO2}_COL>'][i,]**order
        self.df['chem_lifetime'] = [-1/chem_fit.params['vcd1']/3600 for chem_fit in self.chem_fits]
        self.df['chem_rmse'] = [np.sqrt(chem_fit.mse_resid) for chem_fit in self.chem_fits]
        self.df['chem_r2'] = [chem_fit.rsquared for chem_fit in self.chem_fits]
    
    def resample(self,resample_rule,keys):
        '''resample data into lower freq intervals. average data fields
        resample_rule:
            freq input to df.resample
        keys:
            each key will be averaged to <key> at the freq given by resample_rule
        '''
        resampler = pd.DataFrame(
            index=self['time']).resample(
            resample_rule,label='right')
        self.df = pd.DataFrame(index=resampler.indices.keys())
        for i,(k,v) in enumerate(resampler.indices.items()):
            for key in keys:
                if i == 0:
                    self[f'<{key}>'] = np.full((self.df.shape[0],*(self[key].shape[1:])),np.nan)
                self[f'<{key}>'][i,] = np.nanmean(self[key][v,],axis=0)
        return resampler
    
    def get_directional_derivative(self,keys=None,wind_layer=6,
                                   topo_wind_layer=0,ukey='UWIND',vkey='VWIND'):
        '''get the directional derivative terms, vec{u} dot grad(key)
        keys:
            2D fields in self to calculate gradient
        (topo_)wind_layer:
            layer index in 3D wind for the 2D wind vector. separate for the wind topo term
        u/vkey:
            U/VWIND for cmaq
        '''
        if keys is None:
            keys = ['NO_COL','NOx_COL','NO2_COL','f','surface_altitude']
        if self.name.lower() in ['cmaq']:
            p_low = self['VGTOP'] + self['VGLVLS'][wind_layer] * (self['PRSFC']-self['VGTOP'])
            p_high = self['VGTOP'] + self['VGLVLS'][wind_layer+1] * (self['PRSFC']-self['VGTOP'])
            h_low_mean = np.nanmean(np.log(self['PRSFC']/p_low)*7500)
            h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
            p_low_mean = np.nanmean(p_low)
            p_high_mean = np.nanmean(p_high)
            self.logger.info(
                'using layer {} wind, roughly at {:.0f}-{:.0f} hPa, {:.0f}-{:.0f} m'.format(
                    wind_layer,p_low_mean/100,p_high_mean/100,h_low_mean,h_high_mean))
            if 'surface_altitude' in keys:
                p_low = self['VGTOP'] + self['VGLVLS'][topo_wind_layer] * (self['PRSFC']-self['VGTOP'])
                p_high = self['VGTOP'] + self['VGLVLS'][topo_wind_layer+1] * (self['PRSFC']-self['VGTOP'])
                h_low_mean = np.nanmean(np.log(self['PRSFC']/p_low)*7500)
                h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
                p_low_mean = np.nanmean(p_low)
                p_high_mean = np.nanmean(p_high)
                self.logger.info(
                    'using layer {} wind for topography, roughly at {:.0f}-{:.0f} hPa, {:.0f}-{:.0f} m'.format(
                        topo_wind_layer,p_low_mean/100,p_high_mean/100,h_low_mean,h_high_mean))
        else:
            self.logger.error('this ctm is not supported yet')
            return
        u_col = self[ukey][:,wind_layer,:,:]
        v_col = self[vkey][:,wind_layer,:,:]
        if 'surface_altitude' in keys:
            u_topo = self[ukey][:,topo_wind_layer,:,:]
            v_topo = self[vkey][:,topo_wind_layer,:,:]
        # to do: dx_vec for lon lat grid
        dx = self['XCELL']
        for key in keys:
            if key == 'surface_altitude':
                u = u_topo; v = v_topo
            else:
                u = u_col; v = v_col
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
            self.logger.info('calculating contribution of f to emissions')
            self['fxNO2_COL_DD'] = self['f'] * self['NO2_COL_DD']
            self['f_DDxNO2_COL'] = self['f_DD'] * self['NO2_COL']
    
    def get_SUSTech_CMAQ_surface_altitude(self,path,file_path_pattern=None):
        '''
        save or load surface altitude
        path:
            path to save (if file_path_pattern is not None) or load (otherwise) surface pressure data
        file_path_pattern:
            sth like '/projects/academic/kangsun/data/CTM/SUSTech/*_%Y%m.nc', see load_SUSTech_CMAQ
        '''
        if file_path_pattern is not None:
            time = pd.date_range(dt.datetime(2017,1,1),dt.datetime(2017,12,31),freq='1h')
            if 'PRSFC' in self.keys():
                self.logger.warning('PRSFC is already loaded. Temporarily replace it to get surface altitude')
                prsfc = self['PRSFC'].copy()
            file_to_var_mapper = {'PRSFC':['PRSFC']}
            self.load_SUSTech_CMAQ(time,file_to_var_mapper,file_path_pattern)
            self['surface_altitude'] = np.log(101325/np.nanmean(self['PRSFC'],axis=0)[np.newaxis,:,:])*7500
            self['PRSFC'] = prsfc.copy()
            with Dataset(path,'w') as nc:
                nc.createDimension('TSTEP',1)
                nc.createDimension('ROW',len(self['ygrid']))
                nc.createDimension('COL',len(self['xgrid']))
                vid = nc.createVariable('surface_altitude',np.float32,dimensions=('TSTEP','ROW','COL'))
                vid[:] = np.ma.masked_invalid(np.float32(self['surface_altitude']))
        else:
            with Dataset(path,'r') as nc:
                self['surface_altitude'] = nc['surface_altitude'][:].data
                
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
        'PRSFC':['PRSFC'],
        'NOx_emis':['NO','NO2']}
        # loop over monthly files
        for imon,mon in enumerate(pd.period_range(time.min(),time.max(),freq='1M')):
            initialized = False
            # loop over each file
            for ifile,(file_header,fields) in enumerate(file_to_var_mapper.items()):
                fn = mon.strftime(file_path_pattern.replace('*',file_header))
                # emis files are only available on the first day
                if file_header == 'NOx_emis':
                    fn = '01'.join(os.path.splitext(fn))
                self.logger.info('loading {}'.format(fn))
                with Dataset(fn,'r') as nc:
                    # assume all files in a month have the same timestamp, except emis
                    if file_header != 'NOx_emis' and not initialized:
                        initialized = True
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
                            self['YCELL'] = nc.YCELL
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
                    elif file_header == 'NOx_emis':
                        nc_time = pd.to_datetime(
                            ['{0}'.format(d[0])+'{0:0>6}'.format(d[1])
                             for d in nc['TFLAG'][:,0,:]],
                            format='%Y%j%H%M%S')
                        time_mask = nc_time.hour.isin(time.hour.unique())
                    
                    # loop over fields in each file
                    for field in fields:
                        if file_header == 'NOx_emis':
                            fld = field+'_emis'
                            # mol/s to mol/s/m2
                            factor = 1/(self['XCELL']*self['YCELL'])
                        else:
                            fld = field
                            factor = 1
                        if imon == 0:
                            self[fld] = nc[field][time_mask,] * factor
                        else:
                            self[fld] = np.concatenate(
                                (self[fld],nc[field][time_mask,] * factor),
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
        # remove time dimension
        if time is None:
            self.logger.warning('plotting time average')
            data = np.nanmean(self[key],axis=0)
        else:
            data = self[key][np.nonzero(self['time'] == time)[0][0]]
        if layer_index is not None:
            data = data[layer_index,]
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