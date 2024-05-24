import sys, os, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import logging
from popy import Level3_Data, F_center2edge, Level3_List, popy, datedev_py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from matplotlib import path 
import shapely

class TEMPO():
    '''class for a TEMPO-observed region'''
    def __init__(self,product,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=-130,east=-65,south=23,north=51,grid_size=0.01,flux_grid_size=0.05,
                 error_model='ones'):
        '''
        geometry:
            a list of tuples for the polygon, e.g., [(xarray,yarray)], or geometry in a gpd row
        start/end_dt:
            datetime objects accurate to the day
        west, east, south, north:
            boundary of the region
        grid_size:
            grid size for level 3 data
        flux_grid_size:
            grid size for directional derivatives (level 4 data)
        '''
        self.logger = logging.getLogger(__name__)
        self.product = product
        self.grid_size = grid_size
        self.flux_grid_size = flux_grid_size
        self.error_model = error_model
        self.start_dt = start_dt or dt.datetime(2023,1,1)
        self.end_dt = end_dt or dt.datetime.now()
        if geometry is None and xys is not None:
            geometry = xys
        tmp = False
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
            self.xys = [g.exterior.xy for g in geometry.geoms] #zitong edit
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
            tmp = True
        # nudge west/south
        westmost = -180
        self.west = westmost+np.floor((self.west-westmost)/flux_grid_size)*flux_grid_size
        southmost = -90
        self.south = southmost+np.floor((self.south-southmost)/flux_grid_size)*flux_grid_size
        if tmp: 
            self.xys = [([self.west,self.west,self.east,self.east],
                         [self.south,self.north,self.north,self.south])]   
    
    def load_l3_by_local_time(self,l3_path_pattern,
                              fields_name=None,
                              local_hour_centers=None,
                              local_hour_spans=None):
        
        fields_name = fields_name or ['column_amount','local_hour','terrain_height']
        if local_hour_centers is None: 
            local_hour_centers = np.linspace(8,17,10)
        if local_hour_spans is None: 
            local_hour_spans = np.ones_like(local_hour_centers)
        nhour = len(local_hour_centers)
        
        # create an empty list of Level3_Data for each local hour value
        l3_lhs = np.array([Level3_Data() for i in range(nhour)])
        dates = pd.date_range(self.start_dt,self.end_dt,freq='1D')
        wesn_dict = dict(west=self.west,east=self.east,south=self.south,north=self.north)
        for date in dates:
            # level 3 files of the same day
            day_flist = glob.glob(date.strftime(l3_path_pattern))
            for fn in day_flist:
                l3 = Level3_Data(
                ).read_nc(fn,fields_name
                         ).trim(**wesn_dict)
                weight = l3['total_sample_weight'].copy()
                num = l3['num_samples'].copy()
                for ilh, (lh,lhs) in enumerate(zip(local_hour_centers,local_hour_spans)):
                    mask = (l3['local_hour']>= lh-lhs/2) & (l3['local_hour'] < lh+lhs/2)
                    if np.sum(mask) == 0:
                        continue
                    l3['total_sample_weight'][~mask] = 0
                    l3['num_samples'][~mask] = 0
                    l3_lhs[ilh] = l3_lhs[ilh].merge(l3)
                    l3['total_sample_weight'] = weight.copy()
                    l3['num_samples'] = num.copy()
        self.l3_lhs = l3_lhs
            
    def regrid_from_l2(self,l2_path_pattern,
                       attach_l3=False,attach_l2=False,
                       l3_path_pattern=None,
                       l4_path_pattern=None,gradient_kw=None,
                       l3_save_fields=None,l4_save_fields=None,
                       maxsza=75,maxcf=0.3,
                       ncores=0,block_length=300):
        
        if not attach_l2 and not attach_l3 and (l3_path_pattern is None) and (l4_path_pattern is None):
            self.logger.error('attach l2/l3 data or provide level3/4 paths!')
            return
        
        if gradient_kw is None:
            do_l4 = False
        else:
            do_l4 = True
        
        l3_save_fields = l3_save_fields or ['column_amount']
        l4_save_fields = l4_save_fields or \
        ['column_amount','local_hour','terrain_height','wind_topo',\
         'wind_column','wind_column_xy','wind_column_rs']
        
        if l3_path_pattern is not None:
            if 'S{0:03d}' not in l3_path_pattern:
                lst = list(os.path.splitext(l3_path_pattern))
                lst.insert(1,'S{0:03d}')
                l3_path_pattern = ''.join(lst)
                self.logger.warning('scan num is added to saved l3 file name')
        
        if l4_path_pattern is not None:
            if do_l4 and 'S{0:03d}' not in l4_path_pattern:
                lst = list(os.path.splitext(l4_path_pattern))
                lst.insert(1,'S{0:03d}')
                l4_path_pattern = ''.join(lst)
                self.logger.warning('scan num is added to saved l4 file name')
                
        dates = pd.date_range(self.start_dt,self.end_dt,freq='1D')
        wesn_dict = dict(west=self.west,east=self.east,south=self.south,north=self.north)
        
        if attach_l3:
            l3s = []
            if do_l4:
                l4s = []
        
        if attach_l2:
            l2s = []
            
        if attach_l2 or attach_l3:
            dt_array = []
        
        for date in dates:
            next_date = date+pd.DateOffset(1)
            # popy has a very old way of managing time
            start_dict = {k:v for k,v in zip(
                ['start_year','start_month','start_day','start_hour','start_minute','start_second'],
                date.timetuple()[0:6])}
            end_dict = {k:v for k,v in zip(
                ['end_year','end_month','end_day','end_hour','end_minute','end_second'],
                next_date.timetuple()[0:6])}
            # level 2 files of the same day
            day_flist = glob.glob(date.strftime(l2_path_pattern))
            
            # create tempo popy instance for the day
            tempo_l2_daily = popy(instrum='TEMPO',
                                  product=self.product,
                                  **wesn_dict,
                                  **start_dict,**end_dict,
                                  grid_size=self.grid_size,
                                  flux_grid_size=self.flux_grid_size,
                                  error_model=self.error_model,
                                  oversampling_list=[
                                      'terrain_height','column_amount','local_hour'])
            
            tempo_l2_daily.F_subset_TEMPONO2(l2_list=day_flist,maxsza=maxsza,maxcf=maxcf)
            if tempo_l2_daily.nl2 == 0:
                self.logger.info(date.strftime('%Y%m%d')+' has no l2 data, skipping')
                continue
            # datenum in local time
            local_dn = tempo_l2_daily.l2g_data['UTC_matlab_datenum']+\
            tempo_l2_daily.l2g_data['lonc']/15/24
            tempo_l2_daily.l2g_data['local_hour'] = (local_dn-np.floor(local_dn))*24
            # molec/cm2 to mol/m2
            tempo_l2_daily.l2g_data['column_amount'] = \
            tempo_l2_daily.l2g_data['column_amount']/6.02214e19 
            
            if do_l4:
                tempo_l2_daily.F_prepare_gradient(**gradient_kw)
            
            scan_nums = np.unique(tempo_l2_daily.l2g_data['scan_num'])
            for scan_num in scan_nums:
                mask = tempo_l2_daily.l2g_data['scan_num'] == scan_num
                if np.sum(mask) == 0:
                    self.logger.info('{} scan {} has no l2 data, skipping'.format(
                        date.strftime('%Y%m%d'),scan_num))
                    continue
                l2g = {k:v[mask,] for k,v in tempo_l2_daily.l2g_data.items()}
                if attach_l2:
                    l2s.append(l2g)
                l3 = tempo_l2_daily.F_parallel_regrid(
                    l2g_data=l2g,
                    ncores=ncores,
                    block_length=block_length)
                l3.start_python_datetime = datedev_py(np.nanmin(l2g['UTC_matlab_datenum']))
                l3.end_python_datetime = datedev_py(np.nanmax(l2g['UTC_matlab_datenum']))
                if attach_l3:
                    l3s.append(l3)
                if attach_l2 or attach_l3:
                    dt_array.append(l3.start_python_datetime)
                
                if l3_path_pattern is not None:
                    l3_fn = date.strftime(l3_path_pattern.format(int(scan_num)))
                    os.makedirs(os.path.split(l3_fn)[0],exist_ok=True)
                    l3.save_nc(l3_fn,l3_save_fields)
                if do_l4:
                    l4 = l3.block_reduce(self.flux_grid_size)
                    l4.calculate_gradient(**tempo_l2_daily.calculate_gradient_kw)
                    if attach_l3:
                        l4s.append(l4)
                    if l4_path_pattern is not None:
                        l4_fn = date.strftime(l4_path_pattern.format(int(scan_num)))
                        os.makedirs(os.path.split(l4_fn)[0],exist_ok=True)
                        l4.save_nc(l4_fn,l4_save_fields)
            if attach_l2 or attach_l3:
                dt_array = pd.to_datetime(dt_array)
                self.dt_array = dt_array
            if attach_l3:
                self.l3s = Level3_List(dt_array,**wesn_dict)
                for l in l3s:
                    self.l3s.add(l)
                if do_l4:
                    self.l4s = Level3_List(dt_array,**wesn_dict)
                    for l in l4s:
                        self.l4s.add(l)
            if attach_l2:
                self.l2s = l2s