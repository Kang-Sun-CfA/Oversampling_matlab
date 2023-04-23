import sys, os, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import logging
from popy import Level3_Data, F_center2edge, Level3_List
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from matplotlib import path 

class Inventory(dict):
    '''class based on dict, representing a gridded emission inventory'''
    def __init__(self,name=None,west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.west = west
        self.east = east
        self.south = south
        self.north = north
    
    def read_EDF(self,filename):
        from netCDF4 import Dataset
        nc = Dataset(filename)
        xgrid = nc['lon'][:].data
        ygrid = nc['lat'][:].data
        xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
        ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
        if not np.isclose(xgrid_size,ygrid_size,rtol=1e-03):
            self.logger.warning(f'x grid size {xgrid_size} does not equal to y grid size {ygrid_size}')
        self.grid_size = (xgrid_size+ygrid_size)/2
        xmask = (xgrid >= self.west) & (xgrid <= self.east)
        ymask = (ygrid >= self.south) & (ygrid <= self.north)
        self['xgrid'] = xgrid[xmask]
        self['ygrid'] = ygrid[ymask]
        xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
        grid_size_in_m2 = np.cos(np.deg2rad(ymesh/180*np.pi))*np.square(self.grid_size*111e3)
        # metric tons per year per grid to mol/s/m2: *1e6/16/(365*24*3600)/grid_size_in_m2
        self['data'] = nc['ch4_tpy'][:].T[np.ix_(ymask,xmask)].filled(np.nan)*1e6/16/(365*24*3600)/grid_size_in_m2
        self['grid_size_in_m2'] = grid_size_in_m2
        return self
    
    def read_GFEI(self,filename):
        from netCDF4 import Dataset
        nc = Dataset(filename)
        xgrid = nc['lon'][:].data
        ygrid = nc['lat'][:].data
        xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
        ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
        if not np.isclose(xgrid_size,ygrid_size,rtol=1e-03):
            self.logger.warning(f'x grid size {xgrid_size} does not equal to y grid size {ygrid_size}')
        self.grid_size = (xgrid_size+ygrid_size)/2
        xmask = (xgrid >= self.west) & (xgrid <= self.east)
        ymask = (ygrid >= self.south) & (ygrid <= self.north)
        self['xgrid'] = xgrid[xmask]
        self['ygrid'] = ygrid[ymask]
        xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
        grid_size_in_m2 = np.cos(np.deg2rad(ymesh/180*np.pi))*np.square(self.grid_size*111e3)
        # Mg a-1 km-2 to mol/s/m2: *1e6/16/(365*24*3600)/1e6
        self['data'] = nc['emis_ch4'][:][np.ix_(ymask,xmask)].filled(np.nan)/16/(365*24*3600)
        self['grid_size_in_m2'] = grid_size_in_m2
        return self
    
    def regrid(self,l3,fields_to_copy=None,method=None):
        '''regrid inventory to match a l3 data object'''
        from scipy.interpolate import RegularGridInterpolator
        if fields_to_copy is None:
            fields_to_copy = ['vcd','wind_topo','surface_altitude']
        if method is None:
            if self.grid_size < l3.grid_size/2:
                method = 'drop_in_the_box'
#             elif (self.grid_size >= l3.grid_size/2) and (self.grid_size < l3.grid_size*2):
#                 method = 'tessellate'
            else:
                method = 'interpolate'
            self.logger.warning(f'regridding from {self.grid_size} to {l3.grid_size} using {method}')
        
        inv = Inventory()
        inv['xgrid'] = l3['xgrid']
        inv['ygrid'] = l3['ygrid']
        ymesh,xmesh = np.meshgrid(inv['ygrid'],inv['xgrid'])
        inv.grid_size = l3.grid_size
        inv.west = inv['xgrid'].min()-inv.grid_size
        inv.east = inv['xgrid'].max()+inv.grid_size
        inv.south = inv['ygrid'].min()-inv.grid_size
        inv.north = inv['ygrid'].max()+inv.grid_size
        if method in ['interpolate']:
            f = RegularGridInterpolator((self['ygrid'],self['xgrid']),self['data'],bounds_error=False)
            inv['data'] = f((ymesh,xmesh)).T
        elif method in ['drop_in_the_box']:
            data = np.full((len(inv['ygrid']),len(inv['xgrid'])),np.nan)
            for iy,y in enumerate(inv['ygrid']):
                ymask = (self['ygrid']>=y-inv.grid_size/2) & (self['ygrid']<y+inv.grid_size/2)
                for ix,x in enumerate(inv['xgrid']):
                    xmask = (self['xgrid']>=x-inv.grid_size/2) & (self['xgrid']<x+inv.grid_size/2)
                    if np.sum(ymask) == 0 and np.sum(xmask) == 0:
                        continue
                    data[iy,ix] = np.nanmean(self['data'][np.ix_(ymask,xmask)])
            inv['data'] = data
        for field in fields_to_copy:
            if field in l3.keys():
                inv[field] = l3[field].copy()
            else:
                self.logger.warning(f'{field} does not exist in l3!')
        return inv
    
    def get_mask(self,max_emission=1e-9,include_nan=True,
                 min_windtopo=None,max_windtopo=None,min_z0=None,max_z0=None):
        '''get a mask based on self['data']. pixels lower than max_emission will be True.
        nan will alse be True if include_nan
        '''
        mask = self['data'] <= max_emission
        if include_nan:
            mask = mask | np.isnan(self['data'])
        if min_windtopo is not None:
            wt = np.abs(self['wind_topo']/self['vcd'])
            mask = mask & (wt >= min_windtopo)
        if max_windtopo is not None:
            wt = np.abs(self['wind_topo']/self['vcd'])
            mask = mask & (wt <= max_windtopo)
        if min_z0 is not None:
            mask = mask & (self['surface_altitude'] > min_z0)
        if max_z0 is not None:
            mask = mask & (self['surface_altitude'] <= max_z0)
        return mask
    
    def plot(self,ax=None,scale='log',**kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = plt.gcf()
        if scale == 'log':
            from matplotlib.colors import LogNorm
            if 'vmin' in kwargs:
                inputNorm = LogNorm(vmin=kwargs['vmin'],vmax=kwargs['vmax'])
                kwargs.pop('vmin');
                kwargs.pop('vmax');
            else:
                inputNorm = LogNorm()
            pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self['data'],norm=inputNorm,
                                         **kwargs)
        else:
            pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self['data'],**kwargs)
        ax.set_extent([self.west,self.east,self.south,self.north])
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None',
                       edgecolor='black', linewidth=1)
        cb = fig.colorbar(pc,ax=ax)
        figout = {'fig':fig,'pc':pc,'ax':ax,'cb':cb}
        return figout

class Basin():
    '''class based on dict, representing a basin, usually for oil & gas'''
    def __init__(self,gdf_row=None,xys=None,buffer_latlon=0.5,name=None,west=None,east=None,south=None,north=None):
        '''
        gdf_row:
            a row of a geopandas data frame
        xys:
            a list of tuples for the polygon, e.g., [(xarray,yarray)]
        name:
            name of basin, if None, infer from gdf_row
        buffer_latlon:
            buffer to extend the basin window
        west, east, south, north:
            boundary of the basin
        '''
        import shapely
        self.logger = logging.getLogger(__name__)
        if gdf_row is not None:
            if gdf_row.ndim > 1:
                self.logger.info('gdf is a data frame, taking its first row')
                if gdf_row.shape[0] > 1:
                    self.logger.warning(f'gdf has {gdf_row.shape[0]} rows!')
                gdf_row = gdf_row.head(1).squeeze()
            self.name = name or gdf_row.NAME
            bounds = gdf_row.geometry.bounds
            self.west = west or bounds[0]-buffer_latlon
            self.east = east or bounds[2]+buffer_latlon
            self.south = south or bounds[1]-buffer_latlon
            self.north = north or bounds[3]+buffer_latlon
            if isinstance(gdf_row.geometry,shapely.geometry.multipolygon.MultiPolygon):
                self.xys = [g.exterior.xy for g in gdf_row.geometry]
            else:
                self.xys = [gdf_row.geometry.exterior.xy]
        else:
            self.name = name
            self.west = west or np.min(xys[0])-buffer_latlon
            self.east = east or np.max(xys[0])+buffer_latlon
            self.south = south or np.min(xys[1])-buffer_latlon
            self.north = north or np.max(xys[1])+buffer_latlon
            self.xys = xys
    
    def get_basin_emission(self,period_range=None,l3_path_pattern=None,l3s=None,product='CH4',
                           Inventory_kw=None,
                           fit_topo_kw=None,fit_chem_kw=None,fit_alb_kw=None,
                           chem_min_column_amount=None,chem_max_wind_column=None):
        if product == 'CH4':
            fields_name=['wind_column','wind_column_xy','wind_column_rs',\
                                             'vcd','XCH4','surface_altitude','wind_topo',\
                                             'albedo','wind_albedo_0','wind_albedo_1',\
                                             'wind_albedo_2','wind_albedo_3','surface_pressure','pa']
        elif product == 'NO2':
            fields_name=['column_amount','surface_altitude','wind_topo',
                                    'wind_column','wind_column_xy','wind_column_rs']
        
        if period_range is None and l3_path_pattern is None:
            l3s = l3s.trim(west=self.west,east=self.east,south=self.south,north=self.north)
        else:
            l3s = Level3_List(dt_array=period_range,west=self.west,east=self.east,south=self.south,north=self.north)
            l3s.read_nc_pattern(l3_path_pattern=l3_path_pattern,fields_name=fields_name)
        
        self.l3 = l3s.aggregate()
        if Inventory_kw is not None and product == 'CH4':
            fit_topo_kw = fit_topo_kw or {} 
            fit_alb_kw = fit_alb_kw or {}
            inventory_native = Inventory(name=self.name+'_'+Inventory_kw['name'],
                                         west=self.west,east=self.east,south=self.south,north=self.north)
            
            if Inventory_kw['name'] == 'GFEI':
                inventory_native = inventory_native.read_GFEI(filename=Inventory_kw['filename'])
            elif Inventory_kw['name'] == 'EDF':
                inventory_native = inventory_native.read_EDF(filename=Inventory_kw['filename'])
            
            self.inventory_native = inventory_native
            inventory = inventory_native.regrid(self.l3,fields_to_copy=['vcd','wind_topo','surface_altitude'])
            
            if 'min_topo_windtopo' in Inventory_kw.keys():
                min_topo_windtopo = Inventory_kw['min_topo_windtopo']
            else:
                min_topo_windtopo = -np.inf
            if 'max_topo_windtopo' in Inventory_kw.keys():
                max_topo_windtopo = Inventory_kw['max_topo_windtopo']
            else:
                max_topo_windtopo = np.inf
            if 'min_alb_windtopo' in Inventory_kw.keys():
                min_alb_windtopo = Inventory_kw['min_alb_windtopo']
            else:
                min_alb_windtopo = -np.inf
            if 'max_alb_windtopo' in Inventory_kw.keys():
                max_alb_windtopo = Inventory_kw['max_alb_windtopo']
            else:
                max_alb_windtopo = np.inf
            if 'min_z0' in Inventory_kw.keys():
                min_z0 = Inventory_kw['min_z0']
            else:
                min_z0 = None
            topo_mask = inventory.get_mask(max_emission=Inventory_kw['max_topo_emission'],
                                           min_windtopo=min_topo_windtopo,
                                           max_windtopo=max_topo_windtopo,min_z0=min_z0)
            fit_topo_kw.update(dict(mask=topo_mask,min_windtopo=min_topo_windtopo,max_windtopo=max_topo_windtopo))
            
            alb_mask = inventory.get_mask(max_emission=Inventory_kw['max_alb_emission'],
                                          min_windtopo=min_alb_windtopo,
                                          max_windtopo=max_alb_windtopo,min_z0=min_z0)
            fit_alb_kw.update(dict(mask=alb_mask,min_windtopo=min_alb_windtopo,max_windtopo=max_alb_windtopo))
            inventory['topo_mask'] = topo_mask
            inventory['alb_mask'] = alb_mask
            self.inventory = inventory
            
        if fit_topo_kw is None and fit_chem_kw is None and fit_alb_kw is None:#for sub domain
            if hasattr(l3s[0].topo_fit,'bootstrap_params'):
                b_sum_mat = np.full((len(l3s),l3s[0].topo_fit.nbootstrap),np.nan)
                for il,l in enumerate(l3s):
                    for ib,bparam in enumerate(l.topo_fit.bootstrap_params):
                        b_field_name = f'wind_column_topo_b{ib}'
                        l[b_field_name] = l['wind_column'] - bparam['wt']*l['wind_topo']
                        #always remove intercept for subdomains
                        l[b_field_name] -= bparam['Intercept']
                        b_sum_mat[il,ib] = l.sum_by_mask(xys=self.xys,fields_to_sum=[b_field_name])[b_field_name]
                        l.pop(b_field_name)
                l3s.b_sum_mat = b_sum_mat
                for percentile in [1,2.5,5,10,25,50,75,90,95,97.5,99]:
                    l3s.df['summed_wind_column_topo_{}'.format(percentile)] = np.percentile(b_sum_mat,percentile,axis=1)
            self.l3s = l3s
            self.l3 = l3s.aggregate()
            self.l3s.sum_by_mask(xys=self.xys,
                                 fields_to_sum=['wind_column','wind_column_topo','wind_column_topo_alb',
                                                'wind_column_topo_chem','wind_column_topo_xy','wind_column_topo_rs'],
                                 fields_to_average=['num_samples'])
            return
        
        fit_topo_kw = fit_topo_kw or {}
        if 'nbootstrap' not in fit_topo_kw.keys():
            fit_topo_kw['nbootstrap'] = None
        l3s.fit_topography(**fit_topo_kw)
        if fit_topo_kw['nbootstrap'] is not None:
            b_sum_mat = np.full((len(l3s),fit_topo_kw['nbootstrap']),np.nan)
            for il,l in enumerate(l3s):
                for ib,bparam in enumerate(l.topo_fit.bootstrap_params):
                    b_field_name = f'wind_column_topo_b{ib}'
                    l[b_field_name] = l['wind_column'] - bparam['wt']*l['wind_topo']
                    if 'remove_intercept' in fit_topo_kw.keys():
                        if fit_topo_kw['remove_intercept']:
                            l[b_field_name] -= bparam['Intercept']
                    b_sum_mat[il,ib] = l.sum_by_mask(xys=self.xys,fields_to_sum=[b_field_name])[b_field_name]
                    l.pop(b_field_name)
            
            l3s.b_sum_mat = b_sum_mat
            for percentile in [1,2.5,5,10,25,50,75,90,95,97.5,99]:
                l3s.df['summed_wind_column_topo_{}'.format(percentile)] = np.percentile(b_sum_mat,percentile,axis=1)
        if product == 'CH4':
            if fit_alb_kw is None:
                self.l3s = l3s
                self.l3 = l3s.aggregate()
                self.l3s.sum_by_mask(xys=self.xys,
                                     fields_to_sum=['wind_column','wind_column_topo',
                                                    'wind_column_topo_xy','wind_column_topo_rs',
                                                    'wind_column_topo_alb'],
                                     fields_to_average=['num_samples'])
                return
            else:
                l3s.fit_albedo(**fit_alb_kw)
                self.l3s = l3s
                self.l3 = l3s.aggregate()
                self.l3s.sum_by_mask(xys=self.xys,
                                     fields_to_sum=['wind_column','wind_column_topo',
                                                    'wind_column_topo_xy','wind_column_topo_rs',
                                                    'wind_column_topo_alb'],
                                     fields_to_average=['num_samples'])
                return
        if product == 'NO2':
            if fit_chem_kw is None:
                self.l3s = l3s
                self.l3 = l3s.aggregate()
                self.l3s.sum_by_mask(xys=self.xys,
                                     fields_to_sum=['wind_column','wind_column_topo',
                                                    'wind_column_topo_xy','wind_column_topo_rs',
                                                    'wind_column_topo_chem'],
                                     fields_to_average=['num_samples'])
                return
            else:
                if chem_min_column_amount is not None or chem_max_wind_column is not None:
                    if 'mask' in fit_chem_kw.keys():
                        mask = fit_chem_kw['mask']
                    else:
                        mask = np.ones(self.l3['column_amount'].shape,dtype=bool)
                    if chem_min_column_amount is not None:
                        mask = mask & (self.l3['column_amount']>=chem_min_column_amount)
                    if chem_max_wind_column is not None:
                        mask = mask & (self.l3['wind_column']<=chem_max_wind_column)
                    fit_chem_kw['mask'] = mask
                l3s.fit_chemistry(**fit_chem_kw)
                self.l3s = l3s
                self.l3 = l3s.aggregate()
                self.l3s.sum_by_mask(xys=self.xys,
                                     fields_to_sum=['wind_column','wind_column_topo',
                                                    'wind_column_topo_xy','wind_column_topo_rs',
                                                    'wind_column_topo_chem'],
                                     fields_to_average=['num_samples'])
                return
        
    def plot(self,ax=None,reset_extent=True,**kwargs):
        '''overview subregions'''
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        
        for xy in self.xys:
            ax.plot(*xy,**kwargs)
        if reset_extent:
            ax.set_extent([self.west,self.east,self.south,self.north])
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None', edgecolor='k', 
                       linestyle='-',zorder=0,lw=0.5)
        
    def plot_emission_ts(self,ax=None,resample_rule=None,field=None,normalize_sec=None,
                         plot_xyrs=True,plot_bootstrap=True,if_plot=True,min_D=1,
                         sc_kw=None,pl_kw=None,fl_kw=None,er_kw=None,br_kw=None):
        if resample_rule is None:
            resample_rule = '1M'
        field = field or 'summed_wind_column_topo'
        
        if ax is None and if_plot:
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        
        sc_kw = sc_kw or {}
        if 'zorder' not in sc_kw.keys():
            sc_kw['zorder'] = 4
        if 'sdata_kw' not in sc_kw.keys():
            sdata_kw = dict(sdata_min=5,sdata_max=20,sdata_min_size=25,sdata_max_size=100)
        else:
            sdata_kw = sc_kw['sdata_kw']
            sc_kw.pop('sdata_kw');
        if 'sc_leg_loc' not in sc_kw.keys():
            sc_leg_loc = 'lower right'
        else:
            sc_leg_loc = sc_kw['sc_leg_loc']
            sc_kw.pop('sc_leg_loc');
        
        pl_kw = pl_kw or {}
        if 'zorder' not in pl_kw.keys():
            pl_kw['zorder'] = 3
        
        fl_kw = fl_kw or {}
        if 'zorder' not in fl_kw.keys():
            fl_kw['zorder'] = 1
        if 'facecolor' not in fl_kw.keys():
            fl_kw['facecolor'] = 'b'
        if 'alpha' not in fl_kw.keys():
            fl_kw['alpha'] = .25
        
        er_kw = er_kw or {}
        if 'zorder' not in er_kw.keys():
            er_kw['zorder'] = 2
        if 'fmt' not in er_kw.keys():
            er_kw['fmt'] = 'none'
        if 'capsize' not in er_kw.keys():
            er_kw['capsize'] = 3
        if 'elinewidth' not in er_kw.keys():
            er_kw['elinewidth'] = 2
        
        df = self.l3s.df.copy()
        if field+'_xy' not in df.keys():
            plot_xyrs = False
        if not hasattr(self.l3s,'b_sum_mat'):
            plot_bootstrap = False
        br_kw = br_kw or {}
        if 'legend' not in br_kw.keys():
            br_kw['legend'] = False
        if plot_bootstrap and 'capsize' not in br_kw.keys():
            br_kw['capsize'] = 5
        
        if plot_xyrs:
            df[field+'_upper'] = df[[field+'_xy',field+'_rs']].max(axis=1)
            df[field+'_lower'] = df[[field+'_xy',field+'_rs']].min(axis=1)
            df[field+'_ul'] = df[field+'_upper']-df[field+'_lower']
        if plot_bootstrap:
            b_sum_mat = self.l3s.b_sum_mat.copy()
            if plot_xyrs:
                for icol in range(b_sum_mat.shape[1]):
                    b_sum_mat[:,icol] = b_sum_mat[:,icol]+np.random.normal(scale=df[field+'_ul']/2)
            df = pd.concat([df,
                            pd.DataFrame(b_sum_mat,index=df.index,
                                         columns=[f'b_{i}' for i in range(b_sum_mat.shape[1])])],axis=1)
        if 'averaged_num_samples' in df.keys():
            sdata = df['averaged_num_samples'].squeeze()
        else:
            sdata = None
        month_mol = df.multiply((df.index.end_time-df.index.start_time).total_seconds(),axis=0)
        # mol to tg for each month
        mass = month_mol*16*1e-12
        mass['tsec'] = (mass.index.end_time-mass.index.start_time).total_seconds()
        if sdata is not None:
            mass['averaged_num_samples'] = sdata
            drop_idx = mass[mass['averaged_num_samples'] < min_D].index
            mass.drop(drop_idx,inplace=True)
        mass = mass.resample(resample_rule).sum().dropna()
        if normalize_sec is not None:
            mass = mass.divide(mass['tsec'],axis=0)*normalize_sec
        
        if not if_plot:
            return mass
        if resample_rule == '1M':
            br = None
            xdata = mass.index.start_time
            ydata = mass[field]
            if 'averaged_num_samples' in mass.keys():
                # normalize to 0-1
                sdata = (mass['averaged_num_samples']-sdata_kw['sdata_min'])/(sdata_kw['sdata_max']-sdata_kw['sdata_min'])
                # normalize to sdata_min_size-sdata_max_size
                sdata = sdata*(sdata_kw['sdata_max_size']-sdata_kw['sdata_min_size'])+sdata_kw['sdata_min_size']
                sc = ax.scatter(xdata,ydata,s=sdata,**sc_kw)
                handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=7,
                                                     func=lambda x:(x-sdata_kw['sdata_min_size'])\
                                                     /(sdata_kw['sdata_max_size']-sdata_kw['sdata_min_size'])\
                                                    *(sdata_kw['sdata_max']-sdata_kw['sdata_min'])+sdata_kw['sdata_min'])
                leg_sc = ax.legend(handles, labels, title="Basin satellite data coverage",ncol=3,loc=sc_leg_loc)
                ax.add_artist(leg_sc)
            else:
                sc = None;leg_sc = None
            pl = ax.plot(xdata,ydata,**pl_kw)
            if plot_xyrs:
                fl = ax.fill_between(xdata,mass[field+'_lower'],mass[field+'_upper'],**fl_kw)
            else:
                fl = None
            if plot_bootstrap:
                yerrl = mass[field]-mass[field+'_2.5']
                yerru = -mass[field]+mass[field+'_97.5']
                yerr = np.column_stack((yerrl,yerru)).T
                er = ax.errorbar(xdata,ydata,yerr=yerr,**er_kw)
            else:
                er = None
            ax.grid(which='both')
            if er is not None and fl is not None:
                leg = ax.legend([pl[0],er,fl],['Best estimate','95% CI from scale height fit','xy-rs range'])
            else:
                leg = None
        else:
            pl = None;er = None; sc = None; fl = None; leg_sc = None; leg = None
            if plot_bootstrap:
                qtl = mass[[k for k in mass.keys() if k[0:2]=='b_']].quantile([0.025,0.975],axis=1).T.to_numpy()
                yerr=np.column_stack((mass[[field]].to_numpy().squeeze()-qtl[:,0],
                                 qtl[:,1]-mass[[field]].to_numpy().squeeze()))
                br = mass.plot(kind='bar',y=field,ax=ax,yerr=yerr.T,**br_kw)
            else:
                br = mass.plot(kind='bar',y=field,ax=ax,**br_kw)
        figout = dict(ax=ax,br=br,pl=pl,sc=sc,fl=fl,er=er,leg=leg,leg_sc=leg_sc)
        return mass,figout