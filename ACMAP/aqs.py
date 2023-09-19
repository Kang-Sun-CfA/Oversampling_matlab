import zipfile, requests, os, sys, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import kendalltau
from astropy.convolution import convolve_fft
from netCDF4 import Dataset
import statsmodels.formula.api as smf
import logging
# logging.basicConfig(level=logging.INFO)
from popy import popy,datetime2datenum,datedev_py,Level3_List

class Monitor():
    '''class for an individual aqs monitor'''
    def __init__(self,lon,lat,start_dt,end_dt,df=None,l2g_data=None):
        '''
        lon,lat,start_dt,end_dt:
            self explanatory
        df:
            a dataframe of hourly data for the monitor
        l2g_data:
            a popy l2g_data dict containing level 2 pixels overlapping with the monitor
        '''
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.lon = lon
        self.lat = lat
        self.df = df
        self.l2g_data = l2g_data
    
    def sample_l3(self,l3ds,l3_flds,min_num_samples=0.5):
        '''sample l3d fields in the main object (AQS) to each monitor. all hours in a day
        of the monitor will be filled with sampled l3 values
        l3ds:
            daily Level3_List object from the AQS object
        l3_flds:
            list of keys of l3ds to sample into Monitor.df
        min_num_samples:
            flds with num_samples below this value will be nan
        '''
        df = self.df
        for fld in l3_flds:
            df[fld] = np.full(df.shape[0],np.nan)
        for il3,l3d in enumerate(l3ds):
            datel3 = l3ds.df.index[il3]
            for fld in l3_flds:
                data = l3d[fld]
                data[l3d['num_samples']<min_num_samples] = np.nan
                func = RegularGridInterpolator((l3d['ygrid'],l3d['xgrid']),\
                                        data,bounds_error=False,fill_value=np.nan)
                df.loc[(df.index.year==datel3.year)&\
                       (df.index.month==datel3.month)&\
                       (df.index.day==datel3.day),fld] = func((self.lat,self.lon))
    
    def attach_l2g(self,l2g_data,box_km,time_hr,min_qa,if_strict_overlap):
        '''append l2g data to each monitor
        l2g_data:
            popy dict
        box_km:
            keep l2 pixel centroids this km away in a rectangular window
        time_hr:
            keep l2 pixels this hour away from an monitor measurement
        if_strict_overlap:
            if true, keep only l2 pixels overlapping with the monitor location
        '''
        df = self.df
        
        sensor_utc = df.index+dt.timedelta(seconds=1800)
        sensor_dn = np.array([datetime2datenum(t) for t in sensor_utc])
        self.df['UTC_matlab_datenum'] = sensor_dn
        lat_margin = box_km/111
        lon_margin = box_km/(111*np.cos(np.deg2rad(self.lat)))
        mask = (l2g_data['qa_value']>=min_qa)\
        &(l2g_data['lonc']>=self.lon-lon_margin) & (l2g_data['lonc']<=self.lon+lon_margin)\
        &(l2g_data['latc']>=self.lat-lat_margin) & (l2g_data['latc']<=self.lat+lat_margin)\
        &(l2g_data['UTC_matlab_datenum']>=sensor_dn.min()-time_hr/24) \
        &(l2g_data['UTC_matlab_datenum']<=sensor_dn.max()+time_hr/24)
        l2g = {k:v[mask,] for (k,v) in l2g_data.items()}
        
        if if_strict_overlap and len(l2g['latc']) > 0:
            mask = np.array([path.Path([(x,y) for x,y in zip(lonr,latr)]).contains_point([self.lon,self.lat])\
                             for lonr,latr in zip(l2g['lonr'],l2g['latr'])])
            l2g = {k:v[mask,] for (k,v) in l2g.items()}
        
        if 'n_pixel_per_monitor' in df.keys():
            n_pixel_per_monitor = df['n_pixel_per_monitor']
        else:
            n_pixel_per_monitor = np.zeros(df.shape[0])
        # mask on satellite pixels: pixels that match at least one monitor datum
        satellite_mask = np.ones(len(l2g['latc']),dtype=bool)
        for il2 in range(len(l2g['latc'])):
            # mask on monitor time series: hours that match at least one satellite pixel
            monitor_mask = (sensor_dn>=l2g['UTC_matlab_datenum'][il2]-time_hr/24)\
            & (sensor_dn<=l2g['UTC_matlab_datenum'][il2]+time_hr/24)
            n_pixel_per_monitor += monitor_mask
            satellite_mask[il2] = np.nanmin(np.abs(l2g['UTC_matlab_datenum'][il2]-sensor_dn)) <= time_hr/24
        self.df['n_pixel_per_monitor'] = n_pixel_per_monitor
        
        if not all(satellite_mask):
            self.logger.warning(
                '{} out of {} collocated l2 pixels match monitor data in time'.format(
                    sum(satellite_mask),len(satellite_mask)))
            l2g = {k:v[satellite_mask,] for (k,v) in l2g.items()}
        
        if self.l2g_data is None:
            self.l2g_data = l2g
        else:
            self.l2g_data = {k:np.concatenate((self.l2g_data[k],l2g[k]),0) for (k,v) in self.l2g_data.items()}

class AQS():
    '''class for epa ground measurements'''
    def __init__(self,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=None,east=None,south=None,north=None):
        '''
        geometry:
            a list of tuples for the polygon, e.g., [(xarray,yarray)], or geometry in a gpd row
        start/end_dt:
            datetime objects accurate to the day
        west, east, south, north:
            boundary of the region
        '''
        import shapely
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt or dt.datetime(2018,5,1)
        self.end_dt = end_dt or dt.datetime.now()
        if geometry is None and xys is not None:
            geometry = xys
        if isinstance(geometry,list):
            xys = geometry
            self.west = west or np.min([np.min(xy[0]) for xy in xys])
            self.east = east or np.max([np.max(xy[0]) for xy in xys])
            self.south = south or np.min([np.min(xy[1]) for xy in xys])
            self.north = north or np.max([np.max(xy[1]) for xy in xys])
            self.xys = xys
        elif isinstance(geometry,shapely.geometry.multipolygon.MultiPolygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [g.exterior.xy for g in geometry]
        elif isinstance(geometry,shapely.geometry.polygon.Polygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [geometry.exterior.xy]
        elif geometry is None:
            self.west = west
            self.east = east
            self.south = south
            self.north = north
            self.xys = [([west,west,east,east],[south,north,north,south])]
    
    def scatter_mdf(self,x,y,ax=None,c=None,s=None,draw_colorbar=True,**kwargs):
        '''rewrite plot_mdf for non-map scatters'''
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = None
        sc_leg_loc = kwargs.pop('sc_leg_loc','lower right')
        sc_leg_num = kwargs.pop('sc_leg_num',5)
        sc_leg_fmt = kwargs.pop('sc_leg_fmt','{x:.2f}')
        sc_leg_title = kwargs.pop('sc_leg_title',"")
        
        cb_position = kwargs.pop('cb_position',[0.75,0.3,0.2,0.02])
        cb_orientation = kwargs.pop('cb_orientation','horizontal')
        cb_label = kwargs.pop('cb_label',c)
        
        if isinstance(x,str):
            x = self.mdf[x]
        if isinstance(y,str):
            y = self.mdf[y]
        if isinstance(c,str):
            c = self.mdf[c]
        if isinstance(s,str):
            s = self.mdf[s]
        
        if s is not None:
            sdata_func = kwargs.pop('sdata_func',None)
            if sdata_func is not None:
                s = sdata_func(s)
            sdata_min = kwargs.pop('sdata_min',np.nanmin(s))
            sdata_max = kwargs.pop('sdata_max',np.nanmax(s))
            sdata_min_size = kwargs.pop('sdata_min_size',25)
            sdata_max_size = kwargs.pop('sdata_max_size',100)
            # normalize to 0-1
            s = (s-sdata_min)/(sdata_max-sdata_min)
            # normalize to sdata_min_size-sdata_max_size
            s = s*(sdata_max_size-sdata_min_size)+sdata_min_size
            
        sc = ax.scatter(x,y,c=c,s=s,**kwargs)
        leg_sc = None
        if sc_leg_loc is not None:
            handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=sc_leg_num,fmt=sc_leg_fmt,
                                                     func=lambda x:(x-sdata_min_size)\
                                                     /(sdata_max_size-sdata_min_size)\
                                                    *(sdata_max-sdata_min)+sdata_min)
            leg_sc = ax.legend(handles, labels, title=sc_leg_title,ncol=3,loc=sc_leg_loc)
            ax.add_artist(leg_sc)
        if draw_colorbar:
            cax = ax.inset_axes(cb_position)
            cb = plt.colorbar(sc,ax=ax,cax=cax,orientation=cb_orientation)
            cb.set_label(label=cb_label,size=10,color='k')
        else:
            cb = None
        return dict(fig=fig,ax=ax,sc=sc,leg_sc=leg_sc,cb=cb)
    
    def plot_mdf(self,ax=None,reset_extent=False,sdata_column=None,cdata_column='Sample Measurement',
                 draw_colorbar=True,name_region=None,**kwargs):
        '''plot monitors in self.mdf as dots on a map. dot size corresponds to self.mdf[sdata_column], 
        no size if sdata_column is not provided; dot color corresponds to self.mdf[cdata_column], 
        where cdata_column defaults to 'Sample Measurements'
        '''
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # workaround for cartopy 0.16
        from matplotlib.axes import Axes
        from cartopy.mpl.geoaxes import GeoAxes
        GeoAxes._pcolormesh_patched = Axes.pcolormesh
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = None
        cartopy_scale = kwargs.pop('cartopy_scale','110m')
        sc_leg_loc = kwargs.pop('sc_leg_loc','lower right')
        sc_leg_fmt = kwargs.pop('sc_leg_fmt','{x:.2f}')
        sc_leg_title = kwargs.pop('sc_leg_title',"")
        
        cb_position = kwargs.pop('cb_position',[0.75,0.3,0.2,0.02])
        cb_orientation = kwargs.pop('cb_orientation','horizontal')
        cb_label = kwargs.pop('cb_label',cdata_column)
        
        name_xy = kwargs.pop('name_xy',None)
        plot_xys = kwargs.pop('plot_xys',True)
        
        cdata = self.mdf[cdata_column]
        
        if sdata_column is None:
            sc = ax.scatter(self.mdf['Longitude'],self.mdf['Latitude'],
                            c=cdata,**kwargs)
            leg_sc = None
        else:
            sdata = self.mdf[sdata_column]
            sdata_func = kwargs.pop('sdata_func',None)
            if sdata_func is not None:
                sdata = sdata_func(sdata)
            sdata_min = kwargs.pop('sdata_min',np.nanmin(sdata))
            sdata_max = kwargs.pop('sdata_max',np.nanmax(sdata))
            sdata_min_size = kwargs.pop('sdata_min_size',25)
            sdata_max_size = kwargs.pop('sdata_max_size',100)
            # normalize to 0-1
            sdata = (sdata-sdata_min)/(sdata_max-sdata_min)
            # normalize to sdata_min_size-sdata_max_size
            sdata = sdata*(sdata_max_size-sdata_min_size)+sdata_min_size
            sc = ax.scatter(self.mdf['Longitude'],self.mdf['Latitude'],s=sdata,
                            c=cdata,**kwargs)
            if sc_leg_loc is None:
                leg_sc = None
            else:
                handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=7,fmt=sc_leg_fmt,
                                                     func=lambda x:(x-sdata_min_size)\
                                                     /(sdata_max_size-sdata_min_size)\
                                                    *(sdata_max-sdata_min)+sdata_min)
                leg_sc = ax.legend(handles, labels, title=sc_leg_title,ncol=3,loc=sc_leg_loc)
                ax.add_artist(leg_sc)
        if reset_extent:
            ax.set_extent([self.west,self.east,self.south,self.north])
        if cartopy_scale is not None:
            ax.coastlines(resolution=cartopy_scale, color='grey', linewidth=0.25)
            ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='grey', 
                           zorder=0,lw=0.25)
        if plot_xys:
            for xy in self.xys:
                ax.plot(*xy,color='k',zorder=0)
        if name_region is not None:
            if name_region == 'Central':
                gxy = (-87.10812672236439, 38.766609699256335)
            if name_region == 'East North Central':
                gxy = (-90.43415170737065, 45.76247395235188)
            if name_region == 'Northeast':
                gxy = (-75.25482126652217, 42.50105027796941)
            if name_region == 'Northwest':
                gxy = (-118.66368244521263, 45.104907416404906)
            if name_region == 'South':
                gxy = (-96.83087684656562, 34.416641026351996)
            if name_region == 'Southeast':
                gxy = (-84.0674647875265, 32.17560764988584)
            if name_region == 'Southwest':
                gxy = (-108.57098320279188, 37.86224553145235)
            if name_region == 'West':
                gxy = (-118.43031576775631, 38.093095290331824)
            if name_region == 'West North Central':
                gxy = (-104.72656918158656, 46.00966756024588)
        if draw_colorbar:
            cax = ax.inset_axes(cb_position)
            cb = plt.colorbar(sc,ax=ax,cax=cax,orientation=cb_orientation)
            cb.set_label(label=cb_label,size=10,color='k')
        else:
            cb = None
        return dict(fig=fig,ax=ax,sc=sc,leg_sc=leg_sc,cb=cb)
    
    def correlate_all(self,min_pblh=200,max_pblh=np.inf,l3_flds=None,local_hours=None,unit_freq=None,corr_freq='1M'):
        '''compute and plot correlations/slopes between monitor data and l3 data
        min/max_pblh:
            monitor data with era5_blh lower/higher than this value will be excluded
        l3_flds:
            list of keys of l3ds to sample in Monitor.df and correlate/regress with monitor data
        local_hours:
            a list, e.g., [12,13] if not none, local hours of monitor data to include
        unit_freq:
            sample monitor data to this frequency before correlation/regression
        corr_freq:
            time interval at which correlation/regression are done
        '''
        l3_flds = l3_flds or ['column_amount','S']
        dfs = []
        mdf_flds = ['valid_fraction','valid_n','aqs_no2']+l3_flds\
        +[f+'_r' for f in l3_flds]+[f+'_t' for f in l3_flds]+[f+'_slope' for f in l3_flds]
        for mdf_fld in mdf_flds:
            self.mdf[mdf_fld] = np.nan
        for im,monitor in enumerate(self.monitors):
            df = monitor.df
            if local_hours is not None:
                df = df.loc[pd.to_datetime(df.datetime_local).dt.hour.isin(local_hours)]
            self.mdf.loc[im,'valid_fraction'] = np.sum(~np.isnan(df['column_amount']))/np.sum(~np.isnan(df['Sample Measurement']))
            df = df.dropna(subset=l3_flds)
            if df.shape[0] == 0:
                self.logger.warning('empty monitor {}'.format(im))
                continue
            if unit_freq is not None:
                df = df.resample(unit_freq).mean()
            # add statistics to mdf
            self.mdf.loc[im,'aqs_no2'] = np.nanmean(df['Sample Measurement'])
            self.mdf.loc[im,'valid_n'] = df.shape[0]
            df['x'] = df['Sample Measurement'].values
            for f in l3_flds:
                self.mdf.loc[im,f] = np.nanmean(df[f])
                self.mdf.loc[im,f+'_r'] = df['Sample Measurement'].corr(df[f])
                self.mdf.loc[im,f+'_t'] = kendalltau(df['Sample Measurement'],df[f]).correlation
                fit = smf.ols(f+' ~ x', data=df).fit()
                self.mdf.loc[im,f+'_slope'] = fit.params['x']
            dfs.append(df)
        dfs = pd.concat(dfs)
        mask = (dfs['era5_blh']>=min_pblh) & (dfs['era5_blh']<=max_pblh)
        mask_ = (dfs['era5_blh']<min_pblh) | (dfs['era5_blh']>max_pblh)
        if np.sum(mask_) > 0:
            self.logger.warning('{}/{} data points in/out of pblh range'.format(np.sum(mask),np.sum(mask_)))
        dfs_out = dfs.loc[mask_]
        dfs = dfs.loc[mask]
        metric_dict = {}
        keys = []
        dfs['x'] = dfs['Sample Measurement'].values
        for fld in l3_flds:
            metric_dict[fld+'_r'] = dfs['Sample Measurement'].corr(dfs[fld])
            metric_dict[fld+'_t'] = kendalltau(dfs['Sample Measurement'],dfs[fld]).correlation
            
            fit = smf.ols(fld+' ~ x', data=dfs).fit()
            metric_dict[fld+'_slope'] = fit.params['x']
            for v in ['r','t','slope','slope_e']:
                keys.append(fld+'_'+v)
        resampler = dfs.resample(corr_freq)
        vdf = pd.DataFrame(np.full((resampler.count().shape[0],len(keys)),np.nan),columns=keys,index=resampler.count().index)
        for imonth,(ind,sub_df) in enumerate(resampler.__iter__()):
            tmpdf = sub_df[l3_flds].copy()
            tmpdf['x'] = sub_df['Sample Measurement'].values
            vdf.loc[ind,'Sample Measurement'] = np.nanmean(sub_df['Sample Measurement'])
            for fld in l3_flds:
                vdf.loc[ind,fld] = np.nanmean(sub_df[fld])
                vdf.loc[ind,fld+'_r'] = sub_df['Sample Measurement'].corr(sub_df[fld])
                vdf.loc[ind,fld+'_t'] = kendalltau(sub_df['Sample Measurement'],sub_df[fld]).correlation
                fit = smf.ols(fld+' ~ x', data=tmpdf).fit()
                vdf.loc[ind,fld+'_slope'] = fit.params['x']
                vdf.loc[ind,fld+'_slope_e'] = fit.bse['x']
        return dict(dfs=dfs, vdf=vdf, dfs_out=dfs_out,metric_dict=metric_dict)
    
    def load_pkl(self,filename):
        with open(filename, 'rb') as inp:
            d = pickle.load(inp)
        for fld in ['start_dt','end_dt','xys','mdf','region_mask','topo_mask']:
            if fld not in d.keys():
                self.logger.warning(f'{fld} does not exist in pkl file!')
                continue
            setattr(self,fld,d[fld])
        self.monitors = np.array([Monitor(lon=row.Longitude,lat=row.Latitude,
                                          start_dt=self.start_dt,end_dt=self.end_dt,
                                          df=d['dfs'][i].loc[(d['dfs'][i].index>=self.start_dt)&\
                                                            (d['dfs'][i].index<=self.end_dt)]
                                         ) for i,(irow,row) in enumerate(self.mdf.iterrows())])
        self.west = np.min([np.min(xy[0]) for xy in self.xys])
        self.east = np.max([np.max(xy[0]) for xy in self.xys])
        self.south = np.min([np.min(xy[1]) for xy in self.xys])
        self.north = np.max([np.max(xy[1]) for xy in self.xys])
        return self
    
    def save_pkl(self,filename):
        with open(filename, 'wb') as outp:
            pickle.dump({'dfs':[m.df for m in self.monitors],
                         'mdf':self.mdf,'xys':self.xys,
                         'start_dt':self.start_dt,'end_dt':self.end_dt,
                         'region_mask':self.region_mask,'topo_mask':self.topo_mask}, outp, pickle.HIGHEST_PROTOCOL)
    
    def match_monitor_l3(self,l3_flds=None,local_hours=None):
        '''match l3 with all monitors
        l3_flds:
            list of keys of l3ds to sample into Monitor.df
        local_hours:
            keep only relevant local hours in the monitor df, if none, keep all
        '''
        if not hasattr(self,'l3ds'):
            self.logger.error('run get_l3 first')
            return
        l3_flds = l3_flds or ['column_amount','surface_pressure','S','S_vcd','S_x','S_xgb','S_vcd_xgb',
                              'xgb_pblh','era5_blh','era5_t2m']
        for monitor in self.monitors:
            monitor.sample_l3(self.l3ds,l3_flds)
            if local_hours is not None:
                monitor.df = monitor.df.loc[pd.to_datetime(monitor.df.datetime_local).dt.hour.isin(local_hours)]
    
    def save_l3(self,l3_path_pattern,file_freq='1M',fields_name=None,min_pblh=None):
        '''save l3 objects attached to AQS
        l3_path_pattern:
            time pattern of level3 files
        file_freq:
            frequency code by which l3 files are saved
        fields_name:
            fields in l3 objects to save
        min_pblh:
            no averaging data with era5_blh < this value
        '''
        fields_name = fields_name or ['S','S_xgb','S_p','S_p_xgb','column_amount','column_amount_p',\
                                      'wind_column','wind_topo','surface_altitude','era5_blh']
        l3s = self.l3ds
        if min_pblh is not None:
            for l3 in l3s:
                l3['total_sample_weight'][l3['era5_blh']<min_pblh] = 0
                l3['num_samples'][l3['era5_blh']<min_pblh] = 0
        l3s,_ = l3s.resample(rule=file_freq)
        for l3 in l3s:
            fn = l3.start_python_datetime.strftime(l3_path_pattern)
            l3.save_nc(fn,fields_name=fields_name)
    
    def interpolate_l3(self,l3_path_pattern,file_freq='1M',fields_name=None,
                       lonlat_margin=10,attach_l3=True,topo_kw=None,chem_kw=None):
        '''lighter version of get_l3, mainly for sampling fields from popy l3 data'''
        # load level 3 files
        ewsn_dict = dict(west=self.west-lonlat_margin,east=self.east+lonlat_margin,
                        south=self.south-lonlat_margin,north=self.north+lonlat_margin)
        l3s = Level3_List(pd.period_range(self.start_dt,self.end_dt,freq=file_freq),**ewsn_dict)
        fields_name = fields_name or ['S','S_xgb','S_p','S_p_xgb','column_amount','column_amount_p',\
                                      'wind_column','wind_topo','surface_altitude','era5_blh']
        l3s.read_nc_pattern(l3_path_pattern,
                            fields_name=fields_name.copy())
        l3 = l3s.aggregate()
        region_mask = self.region_mask
        
        if topo_kw is not None:
            if 'max_iter' not in topo_kw.keys():
                topo_kw['max_iter'] = 1
            
            # create topo mask
            min_windtopo = topo_kw.pop('min_windtopo',0.001)
            max_windtopo = topo_kw.pop('max_windtopo',0.1)
            min_H = topo_kw.pop('min_H',0.1)
            max_H = topo_kw.pop('max_H',2000)
            max_wind_column = topo_kw.pop('max_wind_column',1e-9)
            topo_mask = region_mask &\
            (np.abs(l3['wind_topo']/l3['column_amount'])>=min_windtopo) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['surface_altitude']>=min_H) &\
            (l3['surface_altitude']<=max_H) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in topo_kw.keys():
                topo_kw['mask'] = topo_kw['mask'] & topo_mask
            else:
                topo_kw['mask'] = topo_mask

            l3s.fit_topography(**topo_kw)
            self.topo_mask = topo_mask
            fields_name.append('wind_column_topo')
            
        if chem_kw is not None:
            min_column_amount = chem_kw.pop('min_column_amount',)
            max_wind_column = chem_kw.pop('max_wind_column',)
            chem_mask = region_mask &\
            (l3['column_amount']>=min_column_amount) &\
            (l3['wind_column']>=max_wind_column)
            if 'mask' in chem_kw.keys():
                chem_kw['mask'] = chem_kw['mask'] & chem_mask
            else:
                chem_kw['mask'] = chem_mask
            l3s.fit_chemistry(**chem_kw)
            self.chem_mask = chem_mask
            fields_name.append('wind_column_topo_chem')
        
        if topo_kw is not None or chem_tw is not None:
            l3 = l3s.aggregate()
        
        if attach_l3:
            self.l3 = l3
            self.l3s = l3s
            self.fields_name = fields_name
        
        for fld in fields_name:
            func = RegularGridInterpolator((l3['ygrid'],l3['xgrid']),\
                                        l3[fld],bounds_error=False,fill_value=np.nan)
            self.mdf['interp_'+fld] = ''
            for irow,row in self.mdf.iterrows():
                self.mdf.loc[irow,'interp_'+fld] = func((row.Latitude,row.Longitude))
        
        
    def get_l3(self,l3_path_pattern,file_freq='1D',lonlat_margin=0.5,xgb_pblh_path_pattern=None,
               topo_kw=None,if_smooth_X=True,X_smooth_kw=None,cp_kw=None):
        '''interface popy level 3 objects. get daily clean/polluted vcd and sfc conc.
        l3_path_pattern:
            time pattern of level3 files, e.g., '/projects/academic/kangsun/data/S5PNO2/L3/ACMAP/CONUS_%Y_%m_%d.nc'
        file_freq:
            frequency code by which l3 files are saved, e.g., 1D
        lonlat_margin:
            Level3_List will be trimmed this amount broader than AQS boundaries
        xgb_pblh_path_pattern:
            add sfc ppb estimated using xgb/amdar-based pblh instead of era5_blh,
            see https://doi.org/10.5194/amt-16-563-2023
        topo_kw:
            key word arguments for fit_topo related functions, may include min/max to mask l3 and args for fit_topography
        if_smooth_X:
            when the l3 record is short, it is better not to smooth scale height
        X_smooth_kw:
            key word arguments to smooth inverse scale height (X), default using savgol, window_length=5;polyorder=3
        cp_kw:
            key word arguments to separate clean/polluted and covert from vcd to sfc conc seperately
        '''
        topo_kw = topo_kw or {}
        if 'max_iter' not in topo_kw.keys():
            topo_kw['max_iter'] = 1
        X_smooth_kw = X_smooth_kw or {}
        cp_kw = cp_kw or {}
        # load level 3 files
        ewsn_dict = dict(west=self.west-lonlat_margin,east=self.east+lonlat_margin,
                        south=self.south-lonlat_margin,north=self.north+lonlat_margin)
        l3ds = Level3_List(pd.period_range(self.start_dt,self.end_dt,freq=file_freq),**ewsn_dict)
        l3ds.read_nc_pattern(l3_path_pattern,
                            fields_name=['column_amount','surface_altitude','wind_topo',
                                         'wind_column',
                                         'era5_t2m','era5_blh','surface_pressure'])
        l3 = l3ds.aggregate()
        # create region mask
        lonmesh,latmesh = np.meshgrid(l3['xgrid'],l3['ygrid'])
        region_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        for xy in self.xys:
            boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            region_mask = region_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)
        # create topo mask
        min_windtopo = topo_kw.pop('min_windtopo',0.001)
        max_windtopo = topo_kw.pop('max_windtopo',0.1)
        min_H = topo_kw.pop('min_H',0.1)
        max_H = topo_kw.pop('max_H',2000)
        max_wind_column = topo_kw.pop('max_wind_column',1e-9)
        topo_mask = region_mask &\
        (np.abs(l3['wind_topo']/l3['column_amount'])>=min_windtopo) &\
        (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
        (l3['surface_altitude']>=min_H) &\
        (l3['surface_altitude']<=max_H) &\
        (l3['wind_column']<=max_wind_column)
        l3ms,_ = l3ds.resample(rule='1M')
        topo_kw['mask'] = topo_mask
        l3ms.fit_topography(**topo_kw)
        # smooth the monthly inverse scale height
        if if_smooth_X:
            window_length = X_smooth_kw.pop('window_length',5)
            polyorder = X_smooth_kw.pop('polyorder',3)
            X = 1/l3ms.df['topo_scale_height'].values
            Xsav = savgol_filter(X,window_length=window_length, polyorder=polyorder)
            l3ms.df['X'] = Xsav
        else:
            l3ms.df['X'] = 1/l3ms.df['topo_scale_height'].values
        # estimate clean and polluted surface concentration daily
        vcd_percentile = cp_kw.pop('vcd_percentile',10)
        vcd_percentile_window = cp_kw.pop('vcd_percentile_window',(50,50))
        vcd_smoothing_window = cp_kw.pop('vcd_smoothing_window',(50,50))
        scale_height = cp_kw.pop('scale_height',7500)
        gamma = cp_kw.pop('gamma',0.9)
        kernel=np.ones((vcd_smoothing_window))
        ymesh,xmesh = np.meshgrid(l3['ygrid'],l3['xgrid'])
        for il3,l3d in enumerate(l3ds):
            if xgb_pblh_path_pattern is not None:
                xgb_fn = l3ds.df.index[il3].strftime(xgb_pblh_path_pattern)
                if not os.path.exists(xgb_fn):
                    self.logger.warning(f'{xgb_fn} does not exist!')
                else:
                    nc = Dataset(xgb_fn)
                    xgb_pblh = nc['PBLH_estimated'][:].filled(np.nan)
                    xgb_pblh[np.isnan(xgb_pblh)] = nc['PBLH_ERA5'][:][np.isnan(xgb_pblh)]
                    func = RegularGridInterpolator((nc['latitude'][:],nc['longitude'][:]),\
                                        xgb_pblh,bounds_error=False,fill_value=np.nan)
                    l3d['xgb_pblh'] = func((ymesh.ravel(),xmesh.ravel())).reshape(xmesh.shape).T
                
            l3d['column_amount_c'] =convolve_fft(
                percentile_filter(l3d['column_amount'],
                                  percentile=vcd_percentile,
                                  size=vcd_percentile_window,mode='nearest'),kernel=kernel)
            l3d['column_amount_p'] = l3d['column_amount']-l3d['column_amount_c']
            daily_X = l3ms.df.loc[(l3ms.df.index.month==l3ds.df.index[il3].month)&\
                                 (l3ms.df.index.year==l3ds.df.index[il3].year)]['X'].squeeze()
            S_c = l3d['column_amount_c']*8.314*l3d['era5_t2m']*daily_X/l3d['surface_pressure']*1e9
            pblp = l3d['surface_pressure']*(1-np.exp(-l3d['era5_blh']/scale_height))
            # polluted sfc ppb
            l3d['S_p'] = l3d['column_amount_p']*9.8*0.02896/(gamma*pblp)*1e9
            # total (polluted + clean) sfc ppb
            l3d['S'] = l3d['S_p']+S_c
            if 'xgb_pblh' in l3d.keys():
                pblp_xgb = l3d['surface_pressure']*(1-np.exp(-l3d['xgb_pblh']/scale_height))
                # polluted sfc ppb
                l3d['S_p_xgb'] = l3d['column_amount_p']*9.8*0.02896/(gamma*pblp_xgb)*1e9
                # total (polluted + clean) sfc ppb
                l3d['S_xgb'] = l3d['S_p_xgb']+S_c
            # total sfc ppb simply from TVCD and pblp
            l3d['S_vcd'] = l3d['column_amount']*9.8*0.02896/(gamma*pblp)*1e9
            # total sfc ppb simply from TVCD and pblp
            l3d['S_vcd_xgb'] = l3d['column_amount']*9.8*0.02896/(gamma*pblp_xgb)*1e9
            # total sfc ppb from scale height
            l3d['S_x'] = l3d['column_amount']*8.314*l3d['era5_t2m']*daily_X/l3d['surface_pressure']*1e9
        # attach results to self
        self.l3ds = l3ds
        tmpdf = l3ms.df.copy()
        l3ms,_ = l3ds.resample(rule='1M')
        l3ms.df = tmpdf
        self.l3ms = l3ms
        self.l3 = l3ds.aggregate()
        self.topo_mask = topo_mask
        self.region_mask = region_mask
        
    def subset_monitors(self,monitor_csv_path=None,parameter_code=None,usecols=None):
        '''attach mdf (monitors dataframe) to the object that includes attributes of monitors
        monitor_csv_path:
            a path to aqs monitor attribute file, default to ccr location
        parameter_code:
            default 42602, NO2
        usecols:
            columns to load
        '''
        monitor_csv_path = monitor_csv_path or '/projects/academic/kangsun/data/AQS/aqs_monitors.csv'
        parameter_code = parameter_code or 42602
        usecols = usecols or ['State Code', 'County Code', 'Site Number', 'Parameter Code',
       'Parameter Name', 'POC', 'Latitude', 'Longitude','Last Method Code','Last Method','Last Sample Date']
        mdf = pd.read_csv(monitor_csv_path,usecols=usecols,
                          dtype={'State Code':object,'County Code':'Int64','Site Number':'Int64',
                                'POC':'Int64','Last Method Code':'Int64'})
        mdf = mdf.loc[(mdf['Parameter Code']==parameter_code)]
        tmask = pd.to_datetime(mdf['Last Sample Date']) >= self.start_dt
        mask = np.zeros(mdf.shape[0],dtype=bool)
        for xy in self.xys:
            boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = mdf[['Longitude','Latitude']].to_numpy()
            mask = mask | boundary_polygon.contains_points(all_points)
        mdf = mdf.loc[mask&tmask].reset_index()
        self.logger.info('located {} monitors'.format(mdf.shape[0]))
        self.mdf = mdf
    
    def load_csv(self,csv_path_pattern,usecols=None,file_freq='1Y'):
        '''attach monitors, an np array of Monitor objects. each element corresponds to one row in self.mdf.
        monitors with no measurements will be removed from self.mdf
        csv_path_pattern:
            file path pattern containing date/time code, e.g., '/projects/academic/kangsun/data/AQS/hourly_42602_%Y.csv'
        usecols:
            columns to load
        file_freq:
            aqs csv files are annual, so 1Y by default
        '''
        usecols = usecols or ['State Code', 'County Code', 'Site Num', 'POC', 'Parameter Name',\
              'Date GMT', 'Time GMT', 'Date Local', 'Time Local', 'Sample Measurement','State Name',\
              'Method Code', 'Method Name', 'Latitude', 'Longitude']
        # initialize list of dataframes, each for a monitor
        dfs = [ [] for _ in range(self.mdf.shape[0]) ]
        for p in pd.period_range(self.start_dt,self.end_dt,freq=file_freq):
            csv_path = p.strftime(csv_path_pattern)
            df = pd.read_csv(csv_path,usecols=usecols,
                             parse_dates={'datetime_UTC':['Date GMT', 'Time GMT'],
                              'datetime_local':['Date Local', 'Time Local']},
                             dtype={'State Code':object,'County Code':'Int64','Site Num':'Int64',
                                'POC':'Int64','Method Code':'Int64'})
            for irow,row in self.mdf.iterrows():
                df0 = df.loc[(df['State Code']==row['State Code'])\
                             &(df['County Code']==row['County Code'])\
                             &(df['Site Num']==row['Site Number'])\
                             &(df['POC']==row['POC'])\
                             &(df['datetime_UTC']>=self.start_dt)&(df['datetime_UTC']<=self.end_dt)]
                if df0.shape[0]==0:continue
                df0 = df0.set_index('datetime_UTC')
                dfs[irow].append(df0)
        dfs = np.array([pd.concat(df00) if len(df00)>0 else [] for df00 in dfs])
        mask = np.array([len(df) for df in dfs])>0
        self.logger.info('{} monitors contain data'.format(np.sum(mask)))
        dfs = dfs[mask]
        self.mdf = self.mdf.loc[mask].reset_index()
        self.monitors = np.array([Monitor(lon=row.Longitude,lat=row.Latitude,
                                          start_dt=self.start_dt,end_dt=self.end_dt,
                                          df=dfs[irow]) for irow,row in self.mdf.iterrows()])
    
    def match_monitor_l2g(self,l2g_path_pattern,file_freq='1M',box_km=20,time_hr=0.5,min_qa=0.75,
                         if_strict_overlap=True,l2g_fields=None):
        '''load popy l2g_data and attach overlapped l2g to Monitor objects'''
        for p in pd.period_range(self.start_dt,self.end_dt,freq=file_freq):
            l2g_path = p.strftime(l2g_path_pattern)
            s5p = popy(instrum='TROPOMI',product='NO2',
                       west=self.west,east=self.east,south=self.south,north=self.north)
            s5p.F_mat_reader(l2g_path)
            for monitor in self.monitors:
                monitor.attach_l2g(s5p.l2g_data,box_km,time_hr,min_qa,if_strict_overlap)
        l2g_fields = l2g_fields or ['column_amount']
        for monitor in self.monitors:
            tmp = np.full((monitor.df.shape[0],len(l2g_fields)),np.nan)
            for i,(irow,row) in enumerate(monitor.df.iterrows()):
                if row.n_pixel_per_monitor > 0:
                    mask = (monitor.l2g_data['UTC_matlab_datenum'] >= row.UTC_matlab_datenum-time_hr/24)\
                    &(monitor.l2g_data['UTC_matlab_datenum'] <= row.UTC_matlab_datenum+time_hr/24)
                    for ifield,field in enumerate(l2g_fields):
                        tmp[i,ifield] = np.nanmean(monitor.l2g_data[field][mask])
            for ifield,field in enumerate(l2g_fields):
                monitor.df[field] = tmp[:,ifield]
    
    def download_zip(self,raw_dir=None,parameter_code=None,
                     download_sites=False,download_monitors=True,years=None):
        '''download data from https://aqs.epa.gov/aqsweb/airdata/download_files.html'''
        def F_download_zip(file_key,raw_dir):
            aqs_file_url = 'https://aqs.epa.gov/aqsweb/airdata/{0}.zip'.format(file_key)
            requested = requests.get(aqs_file_url, verify=False)
            zip_path = os.path.join(raw_dir,'{0}.zip'.format(file_key))
            csv_path = os.path.join(raw_dir,'{0}.csv'.format(file_key))
            open(zip_path, 'wb').write(requested.content)
            zf = zipfile.ZipFile(zip_path).extractall(raw_dir)
            os.remove(zip_path)
            return csv_path
        
        raw_dir = raw_dir or '/projects/academic/kangsun/data/AQS/'
        parameter_code = parameter_code or 42602
        if years is None:
            years = range(2018,2023)
        if download_sites:
            file_key='aqs_sites'
            self.logger.info('downloading {}'.format(file_key))
            F_download_zip(file_key,raw_dir=raw_dir)
        if download_monitors:
            file_key='aqs_monitors'
            self.logger.info('downloading {}'.format(file_key))
            F_download_zip(file_key,raw_dir=raw_dir)
        for year in years:
            file_key = 'hourly_{}_{}'.format(parameter_code,year)
            self.logger.info('downloading {}'.format(file_key))
            F_download_zip(file_key=file_key,raw_dir=raw_dir)