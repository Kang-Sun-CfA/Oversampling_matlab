import sys, os, glob
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import logging
from popy import Level3_Data, F_center2edge, Level3_List, popy, datedev_py, datetime2datenum
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from matplotlib import path 
import shapely
from netCDF4 import Dataset
import warnings
from scipy.ndimage import percentile_filter
from astropy.convolution import convolve_fft

class TEMPO():
    '''class for a TEMPO-observed region'''
    def __init__(self,product,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=-130,east=-65,south=23,north=51,grid_size=0.01,flux_grid_size=None,
                 error_model='ones',k1=4.,k2=2.,k3=1.,inflatex=1.,inflatey=1.):
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
        error_model, k1-3, inflatex/y:
            inputs to popy class in regridding specific to TEMPO. k1 and k2 are along/across track
            super gaussian exponents (opposite from OMI/TROPOMI). inflatex/y stretch the spatial
            response in along/across track.
        '''
        self.logger = logging.getLogger(__name__)
        self.product = product
        self.grid_size = grid_size
        flux_grid_size = flux_grid_size or grid_size
        self.flux_grid_size = flux_grid_size
        self.error_model = error_model
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.inflatex = inflatex
        self.inflatey = inflatey
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
    
    def load_scans(self,l3_path_pattern,
                   min_utc_hour=None,max_utc_hour=None,
                   fields_name=None,
                   tendency=None,
                   local_hour_centers=None,
                   local_hour_spans=None,
                   blacklist=None):
        '''load l3/l4 scans
        l3_path_pattern:
            use %Y%m%d format and *
        min/max_utc_hour:
            limit scan time within these hours. infer from local hours if possible
        fields_name:
            fields to load from the l3/4 data files
        tendency:
            no tendency if None, otherwise a sublist from ['central', 'backward', 'forward']
            or ['c','b','f']
        local_hour_centers:
            if provided, calculate local hour l3/4 data each day. must be increasing. suggest
            use integer hours
        local_hour_spans:
            widths of local hour windows. a scalar will be expanded to an array
        blacklist:
            if provided, will not load the scans. should be a list of file paths
        '''
        if fields_name is None:
            fields_name = ['wind_column','wind_topo',
                           'column_amount','local_hour','terrain_height']
        if tendency is not None:
            do_tendency = True
        else:
            do_tendency = False
        
        if local_hour_centers is not None:
            do_local_hour = True
            if local_hour_spans is None: 
                local_hour_spans = np.ones_like(local_hour_centers)*\
                np.abs(np.mean(np.diff(local_hour_centers)))
            elif np.isscalar(local_hour_spans):
                local_hour_spans = np.ones_like(local_hour_centers)*local_hour_spans
            min_utc_hour = min_utc_hour or \
            local_hour_centers[0]-local_hour_spans[0]/2-self.east/15
            max_utc_hour = max_utc_hour or \
            local_hour_centers[-1]+local_hour_spans[-1]/2-self.west/15
            if do_tendency:
                min_utc_hour -= 1
                max_utc_hour += 1
        else:
            do_local_hour = False
            min_utc_hour = min_utc_hour or 0.
            max_utc_hour = max_utc_hour or 30.
        
        dates = pd.date_range(self.start_dt,self.end_dt,freq='1d')
        wesn_dict = dict(west=self.west,east=self.east,south=self.south,north=self.north)
        days_df = np.empty(len(dates),dtype=object)
        for idate,date in enumerate(dates):
            # level 3 files of the same day
            day_flist = np.array(glob.glob(date.strftime(l3_path_pattern)))
            if blacklist is not None:
                day_flist = [item for item in day_flist if item not in blacklist]
            nscan = len(day_flist)
            day_df = pd.DataFrame(dict(scan_num=np.zeros(nscan,dtype=int),
                                       time=np.zeros(nscan),
                                       start_time=np.zeros(nscan),
                                       end_time=np.zeros(nscan),
                                       mid_time=np.zeros(nscan),
                                       path=np.empty(nscan,dtype=str)))
            for iscan in range(nscan):
                with Dataset(day_flist[iscan],'r') as nc:
                    try:
                        day_df.loc[iscan,'scan_num'] = nc.scan_num
                    except Exception as e:
                        self.logger.warning(e)
                        self.logger.warning('scan num not found; tendency not possible')
                        do_tendency = False
                    day_df.loc[iscan,'path'] = day_flist[iscan]
                    t1 = dt.datetime.strptime(nc.time_coverage_start,'%Y-%m-%dT%H:%M:%SZ')
                    t2 = dt.datetime.strptime(nc.time_coverage_end,'%Y-%m-%dT%H:%M:%SZ')
                    day_df.loc[iscan,'start_time'] = t1
                    day_df.loc[iscan,'end_time'] = t2
                    day_df.loc[iscan,'mid_time'] = t1 + (t2-t1)/2
            day_df = day_df.set_index('start_time',drop=False).sort_index()
            scan_start_hour = (pd.to_datetime(day_df['start_time'])-pd.to_datetime(date))/pd.Timedelta(hours=1)
            scan_end_hour = (pd.to_datetime(day_df['end_time'])-pd.to_datetime(date))/pd.Timedelta(hours=1)
            mask = (scan_end_hour >= min_utc_hour) & (scan_start_hour <= max_utc_hour)
            days_df[idate] = day_df[mask]

        days_df = pd.concat(days_df)
        l3s = Level3_List(dt_array=days_df.index,**wesn_dict)
        l3s.df['mid_time'] = days_df['mid_time']
        l3s.df['end_time'] = days_df['end_time']
        l3s.df['scan_num'] = days_df['scan_num']
        for iscan,(irow,row) in enumerate(days_df.iterrows()):
            l3s.add(Level3_Data().read_nc(row.path,fields_name))
        self.l3s = l3s
        if do_tendency:
            self.l3s.get_storage(field='column_amount',tendency=tendency)
                
        if do_local_hour:
            self.l3_lhs = self.l3s.get_local_hour_l3s(
                local_hour_centers,local_hour_spans)
                    
    def load_l3_by_local_time(self,l3_path_pattern,
                              fields_name=None,
                              local_hour_centers=None,
                              local_hour_spans=None):
        
        if fields_name is None:
            fields_name = ['column_amount','local_hour','terrain_height']
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
    
    def regrid_from_l2(self,l2_path_pattern=None,l2_dir_pattern=None,
                       data_fields=None,data_fields_l2g=None,
                       attach_l3=False,attach_l2=False,
                       l3_path_pattern=None,
                       l4_path_pattern=None,gradient_kw=None,
                       l3_save_fields=None,l4_save_fields=None,
                       maxsza=75,maxcf=0.3,
                       ncores=0,block_length=300,
                       l3_ncattr_dict=None,l4_ncattr_dict=None,
                       use_TEMPOL2=True,fadf=None):
        
        if l2_path_pattern is None and l2_dir_pattern is not None:
            l2_path_pattern = l2_dir_pattern
        if l2_path_pattern is None and l2_dir_pattern is None:
            self.logger.error('l2 path is needed!')
            return
        
        if not attach_l2 and not attach_l3 and (l3_path_pattern is None) and (l4_path_pattern is None):
            self.logger.error('attach l2/l3 data or provide level3/4 paths!')
            return
        
        if gradient_kw is None:
            do_l4 = False
        else:
            do_l4 = True
        
        l3_save_fields = l3_save_fields or ['column_amount']
        l4_save_fields = l4_save_fields or \
        ['column_amount','local_hour','terrain_height',\
         'wind_topo','wind_topo_xy','wind_topo_rs','wind_column','wind_column_xy','wind_column_rs']
        
        if l3_path_pattern is not None:
            l3_ncattr_dict = l3_ncattr_dict or {
                'description':'Level 3 data created using physical oversampling (https://doi.org/10.5194/amt-11-6679-2018)',
                'institution':'University at Buffalo',
                'contact':'Kang Sun, kangsun@buffalo.edu'}
            if 'S{0:03d}' not in l3_path_pattern:
                lst = list(os.path.splitext(l3_path_pattern))
                lst.insert(1,'S{0:03d}')
                l3_path_pattern = ''.join(lst)
                self.logger.warning('scan num is added to saved l3 file name')
        
        if l4_path_pattern is not None:
            if do_l4:
                l4_ncattr_dict = l4_ncattr_dict or {
                    'description':'Level 4 data created using directional derivative approach (https://doi.org/10.1029/2022GL101102)',
                    'institution':'University at Buffalo',
                    'contact':'Kang Sun, kangsun@buffalo.edu',
                    'x_wind_field':gradient_kw['x_wind_field'],
                    'y_wind_field':gradient_kw['y_wind_field']}
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
        
        oversampling_list = ['terrain_height','column_amount','local_hour']
        if do_l4 and use_TEMPOL2:
            oversampling_list += ['wind_column','wind_column_xy','wind_column_rs',
                                  'wind_topo','wind_topo_xy','wind_topo_rs']
            if 'do_DIV' in gradient_kw.keys():
                do_DIV = gradient_kw['do_DIV']
            else:
                do_DIV = False
            if do_DIV:
                oversampling_list += ['column_amount_DIV','column_amount_DIV_xy','column_amount_DIV_rs']
        for date in dates:
            # DDA on tempo grid
            if use_TEMPOL2:
                # level 2 files of the same day
                day_flist = glob.glob(date.strftime(l2_path_pattern))
                if len(day_flist) == 0:
                    self.logger.info(date.strftime('%Y%m%d')+' has no l2 data, skipping')
                    continue
                if os.path.isfile(day_flist[0]):
                    self.logger.warning('Please provide only directory pattern when using TEMPOL2 class! Trimming...')
                    l2_dir_pattern = os.path.split(l2_path_pattern)[0]
                else:
                    l2_dir_pattern = l2_path_pattern
                    day_flist = glob.glob(date.strftime(os.path.join(l2_dir_pattern,'*')))
                # read scan num from file names - not the most robust solution
                scan_nums = np.sort(np.unique([int(os.path.split(f)[-1][35:38]) for f in day_flist]))
                for scan_num in scan_nums:
                    tl2 = TEMPOL2(year=date.year,month=date.month,day=date.day,scan_num=scan_num,
                                  l2_dir_pattern=l2_dir_pattern,**wesn_dict)
                    if len(tl2.l2_list) == 0:
                        self.logger.info('{} scan {} has no l2 data files, skipping'.format(
                            date.strftime('%Y%m%d'),scan_num))
                        continue
                    tl2.load_l2(maxcf=maxcf,maxsza=maxsza,
                                data_fields=data_fields,data_fields_l2g=data_fields_l2g)
                    if do_l4:
                        tl2.get_theta()
                        tl2.interp_met(**gradient_kw['interp_met_kw'])
                        tl2.get_DD(fields=['column_amount'],
                                   east_wind_field=gradient_kw['x_wind_field'],
                                   north_wind_field=gradient_kw['y_wind_field'],
                                   do_DIV=do_DIV
                                  )
                        tl2.get_DD(fields=['terrain_height'],
                                   east_wind_field=gradient_kw['x_wind_field_sfc'],
                                   north_wind_field=gradient_kw['y_wind_field_sfc'])
                    
                    if fadf is not None and do_l4:
                        for irow,row in fadf.iterrows():
                            # start of facility loop
                            if 'window_km' in row.keys():
                                window_km_x = row.window_km
                                window_km_y = row.window_km
                            elif 'window_km_x' in row.keys():
                                window_km_x = row.window_km_x
                                window_km_y = row.window_km_y
                            else:
                                window_km_x = 30;window_km_y = 30
                            km_per_lat = 111
                            km_per_lon = 111*np.cos(row.latitude/180*np.pi)
                            l2g = tl2.to_popy_l2g_data(
                                west=row.longitude-window_km_x/km_per_lon,
                                east=row.longitude+window_km_x/km_per_lon,
                                south=row.latitude-window_km_y/km_per_lat,
                                north=row.latitude+window_km_y/km_per_lat)
                            if len(l2g['latc']) == 0:
                                # escape from facility loop if no l2 pixels
                                continue
                            # create a popy object for each scan/each facility
                            fpopy = popy(
                                instrum='TEMPO',
                                product=self.product,
                                west=row.longitude-window_km_x/km_per_lon,
                                east=row.longitude+window_km_x/km_per_lon,
                                south=row.latitude-window_km_y/km_per_lat,
                                north=row.latitude+window_km_y/km_per_lat,
                                grid_size=self.grid_size,
                                error_model=self.error_model,
                                oversampling_list=oversampling_list)
                            l4 = fpopy.F_parallel_regrid(
                                l2g_data=l2g,
                                ncores=ncores,
                                block_length=block_length).block_reduce(self.flux_grid_size)
                            l4_fn = date.strftime(l4_path_pattern.format(int(scan_num)))
                            l4_fn = os.path.join(
                                os.path.split(l4_fn)[0],'facilityId{}_{}'.format(row.name,os.path.split(l4_fn)[1])
                            )
                            os.makedirs(os.path.split(l4_fn)[0],exist_ok=True)
                            l4_ncattr_dict['scan_num'] = float(scan_num)
                            l4_ncattr_dict['time_coverage_start'] = \
                            datedev_py(np.nanmin(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['time_coverage_end'] = \
                            datedev_py(np.nanmax(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            l4.save_nc(l4_fn,l4_save_fields,ncattr_dict=l4_ncattr_dict)
                            # end of facility loop
                        # finish the scan if all facilities emissions are done
                        del tl2, l2g 
                        if 'fpopy' in locals():
                            del fpopy, l4
                        continue
                        # end of facility emission saving
                    
                    l2g = tl2.to_popy_l2g_data(**wesn_dict)
                    if len(l2g['latc']) == 0:
                        self.logger.info('{} scan {} has no l2 data after filtering, skipping'.format(
                            date.strftime('%Y%m%d'),scan_num))
                        continue
                    # create a popy object for each scan
                    tempo_popy = popy(instrum='TEMPO',
                                      product=self.product,
                                      **wesn_dict,
                                      grid_size=self.grid_size,
                                      error_model=self.error_model,
                                      k1=self.k1,k2=self.k2,k3=self.k3,
                                      inflatex=self.inflatex,inflatey=self.inflatey,
                                      oversampling_list=oversampling_list)
                    if attach_l2:
                        self.logger.warning('attaching l2 is tedious when using TEMPOL2')
                        l2s.append(tl2)
                    l3 = tempo_popy.F_parallel_regrid(
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
                        l3_ncattr_dict['scan_num'] = int(scan_num)
                        l3_ncattr_dict['time_coverage_start'] = \
                        datedev_py(np.nanmin(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                        l3_ncattr_dict['time_coverage_end'] = \
                        datedev_py(np.nanmax(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                        l3_ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                        l3.save_nc(l3_fn,l3_save_fields,ncattr_dict=l3_ncattr_dict)
                    if do_l4:
                        if np.isclose(self.grid_size,self.flux_grid_size) or (self.flux_grid_size is None):
                            l4 = l3
                        else:
                            l4 = l3.block_reduce(self.flux_grid_size)
                            if attach_l3:
                                l4s.append(l4)
                        if l4_path_pattern is not None:
                            l4_fn = date.strftime(l4_path_pattern.format(int(scan_num)))
                            os.makedirs(os.path.split(l4_fn)[0],exist_ok=True)
                            l4_ncattr_dict['scan_num'] = float(scan_num)
                            l4_ncattr_dict['time_coverage_start'] = \
                            datedev_py(np.nanmin(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['time_coverage_end'] = \
                            datedev_py(np.nanmax(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            l4.save_nc(l4_fn,l4_save_fields,ncattr_dict=l4_ncattr_dict)
                    del tl2, tempo_popy, l2g, l3
                    if do_l4:
                        del l4
                    # end of scan loop
                
            # regriding/DDA on popy grid
            else:
                next_date = date+pd.DateOffset(1)
                # popy has a very old way of managing time
                # start from 8 am utc of the day
                start_dict = {k:v for k,v in zip(
                    ['start_year','start_month','start_day','start_hour','start_minute','start_second'],
                    date.timetuple()[0:3]+(8,0,0))}
                # end at 2 am utc of the next day
                end_dict = {k:v for k,v in zip(
                    ['end_year','end_month','end_day','end_hour','end_minute','end_second'],
                    next_date.timetuple()[0:3]+(2,0,0))}
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
                                      k1=self.k1,k2=self.k2,k3=self.k3,
                                      inflatex=self.inflatex,inflatey=self.inflatey,
                                      oversampling_list=oversampling_list)

                tempo_l2_daily.F_subset_TEMPONO2(l2_list=day_flist,
                                                 maxsza=maxsza,maxcf=maxcf,
                                                 data_fields=data_fields,data_fields_l2g=data_fields_l2g)
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
                        l3_ncattr_dict['scan_num'] = int(scan_num)
                        l3_ncattr_dict['time_coverage_start'] = \
                        datedev_py(np.nanmin(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                        l3_ncattr_dict['time_coverage_end'] = \
                        datedev_py(np.nanmax(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                        l3_ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                        l3.save_nc(l3_fn,l3_save_fields,ncattr_dict=l3_ncattr_dict)
                    if do_l4:
                        l4 = l3.block_reduce(self.flux_grid_size)
                        l4.calculate_gradient(**tempo_l2_daily.calculate_gradient_kw)
                        if attach_l3:
                            l4s.append(l4)
                        if l4_path_pattern is not None:
                            l4_fn = date.strftime(l4_path_pattern.format(int(scan_num)))
                            os.makedirs(os.path.split(l4_fn)[0],exist_ok=True)
                            l4_ncattr_dict['scan_num'] = float(scan_num)
                            l4_ncattr_dict['time_coverage_start'] = \
                            datedev_py(np.nanmin(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['time_coverage_end'] = \
                            datedev_py(np.nanmax(l2g['UTC_matlab_datenum'])).strftime('%Y-%m-%dT%H:%M:%SZ')
                            l4_ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            l4.save_nc(l4_fn,l4_save_fields,ncattr_dict=l4_ncattr_dict)

                tempo_l2_daily = None
            
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

class TEMPOL2(dict):
    def __init__(self,year,month,day,scan_num,l2_dir_pattern,
                 west=None,east=None,south=None,north=None):
        '''
        year, month, day:
            int, identify a date
        scan_num:
            int, tempo scan number
        l2_dir_pattern:
            pattern of tempo level 2 data directory, e.g., 
            '/projects/academic/kangsun/data/TEMPONO2/TEMPO_NO2_L2_V03/%Y/%m/%d/'
        '''
        self.logger = logging.getLogger(__name__)
        dates = pd.date_range(dt.date(year,month,day),freq='1d',periods=2)
        self.date = dates[0]
        self.scan_num = scan_num
        l2_path_pattern = os.path.join(l2_dir_pattern,f'*S{scan_num:>03}G*')
        l2_list = []
        for date in dates:
            l2_list += glob.glob(date.strftime(l2_path_pattern))
        l2_list = np.array(l2_list)
        
        if_right_date = np.zeros(len(l2_list),dtype=bool)
        earliest = dates[0]+pd.offsets.Hour(8)
        latest = dates[0]+pd.offsets.Hour(27)
        try:
            # try extracting granule time from file name
            for il2,l2_path in enumerate(l2_list):
                start_dt = dt.datetime.strptime(
                    os.path.split(l2_path)[-1][17:33],'%Y%m%dT%H%M%SZ')
                if (start_dt > earliest) & (start_dt < latest):
                    if_right_date[il2] = True
        except Exception as e:
            self.logger.warning(e)
            self.logger.warning('cannot determine granule time based on file name')
            # loop over scan num on this and next day to keep only the right granules
            for il2,l2_path in enumerate(l2_list):
                with Dataset(l2_path,'r') as nc:
                    start_dt = dt.datetime.strptime(
                        nc.time_coverage_start,'%Y-%m-%dT%H:%M:%SZ')
                    if (start_dt > earliest) & (start_dt < latest):
                        if_right_date[il2] = True
        
        l2_list = l2_list[if_right_date]
        self.l2_list = l2_list
        if len(l2_list) == 0:
            self.logger.warning('No available l2 files for {} scan {}'.format(
                self.date.strftime('%Y%m%d'),scan_num)
                               )
            return
        ngranule = len(l2_list)
        along_tracks = np.zeros(ngranule,dtype=int)
        granule_numbers = np.zeros(ngranule,dtype=int)
        granule_mask = np.ones(ngranule,dtype=bool)
        if west is not None:
            from shapely.geometry import Polygon
            polygon = Polygon(np.array([[west,west,east,east],\
                                        [south,north,north,south]]).T)
        # loop over right granules of the scan to save mirror steps
        for il2,l2_path in enumerate(l2_list):
            try:
                with Dataset(l2_path,'r') as nc:
                    if west is not None:
                        xys = nc.geospatial_bounds.split('((')[-1].split('))')[0].split(',')
                        granule_poly = Polygon(
                            np.array(
                                [xy.split(' ')[::-1] for xy in xys]).astype(float)
                        )
                        if not polygon.intersects(granule_poly):
                            self.logger.info(f'{l2_path} does not intersect with wesn bounds')
                            granule_mask[il2] = False
                    granule_numbers[il2] = nc.granule_num
                    along_tracks[il2] = nc.dimensions['mirror_step'].size
                    if il2 == 0:
                        xtrack = nc.dimensions['xtrack'].size
                    else:
                        if xtrack != nc.dimensions['xtrack'].size:
                            self.logger.error('inconsistent xtrack dimension!')
            except Exception as e:
                self.logger.warning(f'{l2_path} gives error!')
                granule_mask[il2] = False
            
        l2_list = l2_list[granule_mask]
        self.l2_list = l2_list
        if len(l2_list) == 0:
            self.logger.warning('No available l2 files for {} scan {}'.format(
                self.date.strftime('%Y%m%d'),scan_num)
                               )
            return
        self.along_tracks = along_tracks[granule_mask]
        self.xtrack = xtrack
    
    def load_l2(self,data_fields=None,data_fields_l2g=None,maxcf=0.2,maxsza=None):
        along_tracks = self.along_tracks
        l2_list = self.l2_list
        if data_fields is None:
            data_fields = ['/support_data/amf_cloud_fraction',\
                           '/geolocation/latitude',\
                           '/geolocation/longitude',\
                           '/support_data/surface_pressure',\
                           '/support_data/terrain_height',\
                           '/geolocation/latitude_bounds',\
                           '/geolocation/longitude_bounds',\
                           '/geolocation/solar_zenith_angle',\
                           '/product/main_data_quality_flag',\
                           '/product/vertical_column_troposphere']  
            data_fields_l2g = ['cloud_fraction','latc','lonc',
                               'surface_pressure','terrain_height',
                               'latr','lonr','sza','qa','column_amount']
        
        for short_name in data_fields_l2g:
            if short_name in ['lonr','latr']:
                self[short_name] = np.zeros((self.along_tracks.sum(),self.xtrack,4),
                                           dtype=np.float32)
            else:
                self[short_name] = np.zeros((self.along_tracks.sum(),self.xtrack))
        
        self['time'] = np.zeros(self.along_tracks.sum())
        # loop over granules of the same scan
        start_alongtrack_idx = 0
        ngranule = len(l2_list)
        
        for igranule in range(ngranule):
            with Dataset(l2_list[igranule],'r') as nc:
                dn_utc0 = datetime2datenum(
                    dt.datetime.strptime(nc.time_coverage_start,'%Y-%m-%dT%H:%M:%SZ'))
                dn_utc1 = datetime2datenum(
                    dt.datetime.strptime(nc.time_coverage_end,'%Y-%m-%dT%H:%M:%SZ'))
                nmirror_step = nc.dimensions['mirror_step'].size
                self['time'][
                    start_alongtrack_idx:start_alongtrack_idx+along_tracks[igranule]
                ] = np.linspace(
                    dn_utc0,dn_utc1,nmirror_step+1)[:nmirror_step]
                
                for long_name,short_name in zip(data_fields,data_fields_l2g):
                    if long_name not in ['/product/main_data_quality_flag']:
                        self[short_name][
                        start_alongtrack_idx:start_alongtrack_idx+along_tracks[igranule],
                        :] = marray = nc[long_name][:]
                        mdata = marray.data.astype(float)
                        mdata[marray.mask] = np.nan
                    else:
                        self[short_name][
                        start_alongtrack_idx:start_alongtrack_idx+along_tracks[igranule],
                        :,] = nc[long_name][:]
                    # end of field loop
                # end of nc file reading
            start_alongtrack_idx += along_tracks[igranule]

        mask = (self['qa']==0)
        if maxcf is not None:
            mask = mask & (self['cloud_fraction']<=maxcf)
        if maxsza is not None:
            mask = mask & (self['sza']<=maxsza)
        self['column_amount'][~mask] = np.nan
        self['column_amount'] /= 6.02214e19     
        self['UTC_matlab_datenum'] = np.broadcast_to(self['time'][:,np.newaxis],self['latc'].shape)
        self['across_track_position'] = np.broadcast_to(np.arange(1.,self.xtrack+1,dtype=int
                                                                 )[np.newaxis,:],
                                                        self['latc'].shape)
        m_per_lat = 111e3
        m_per_lon = m_per_lat * np.cos(np.radians(self['latc'])).astype(np.float32)
        self['m_per_lat'] = m_per_lat
        self['m_per_lon'] = m_per_lon
            
    def get_theta(self):
        if (self['latc'].shape[0] < 3) or (self['latc'].shape[1] < 3):
            self.logger.warning('matrix too small for theta/DD terms!')
            for k in ['thetax','thetay','thetar','thetas','xdoty','rdots','det_xy','det_rs']:
                self[k] = np.full(self['latc'].shape,np.nan,dtype=np.float32)
            return
        m_per_lat = self['m_per_lat']
        m_per_lon = self['m_per_lon']
        thetax = np.full(self['latc'].shape,np.nan,dtype=np.float32)
        thetay = np.full(self['latc'].shape,np.nan,dtype=np.float32)
        thetar = np.full(self['latc'].shape,np.nan,dtype=np.float32)
        thetas = np.full(self['latc'].shape,np.nan,dtype=np.float32)
        thetax[:,1:-1] = np.arctan2(m_per_lat*(self['latc'][:,:-2]-self['latc'][:,2:]),
                                   m_per_lon[:,1:-1]*(self['lonc'][:,:-2]-self['lonc'][:,2:]))
        thetay[1:-1,:] = np.arctan2(m_per_lat*(self['latc'][2:,:]-self['latc'][:-2,:]),
                                   m_per_lon[1:-1,:]*(self['lonc'][2:,:]-self['lonc'][:-2,:]))
        thetar[1:-1,1:-1] = np.arctan2(m_per_lat*(self['latc'][2:,:-2]-self['latc'][:-2,2:]),
                                   m_per_lon[1:-1,1:-1]*(self['lonc'][2:,:-2]-self['lonc'][:-2,2:]))
        thetas[1:-1,1:-1] = np.arctan2(m_per_lat*(self['latc'][2:,2:]-self['latc'][:-2,:-2]),
                                   m_per_lon[1:-1,1:-1]*(self['lonc'][2:,2:]-self['lonc'][:-2,:-2]))
        self['thetax'] = thetax
        self['thetay'] = thetay
        self['thetar'] = thetar
        self['thetas'] = thetas
        self['xdoty'] = np.cos(thetax)*np.cos(thetay) + np.sin(thetax)*np.sin(thetay)
        self['rdots'] = np.cos(thetar)*np.cos(thetas) + np.sin(thetar)*np.sin(thetas)
        self['det_xy'] = np.cos(thetax)*np.sin(thetay) - np.sin(thetax)*np.cos(thetay)
        self['det_rs'] = np.cos(thetar)*np.sin(thetas) - np.sin(thetar)*np.cos(thetas)
    
    def interp_met(self,which_met='era5',met_dir=None,interp_fields=None,
                  altitudes=None,**kwargs):
        mask = (~np.isnan(self['UTC_matlab_datenum'])) &\
        (~np.isnan(self['lonc'])) & (~np.isnan(self['latc']))
        sounding_lon = self['lonc'][mask]
        sounding_lat = self['latc'][mask]
        sounding_datenum = self['UTC_matlab_datenum'][mask]
        if which_met.lower() in ['era5']:
            if altitudes is None:
                from popy import F_interp_era5
                if interp_fields is None:
                    interp_fields = ['u10','v10','u100','v100']
                sounding_interp = F_interp_era5(
                    sounding_lon,sounding_lat,sounding_datenum,
                    met_dir,interp_fields)                
                for key in sounding_interp.keys():
                    self.logger.info(key+' from ERA5 is sampled to L2 coordinate/time')
                    self['era5_'+key] = np.full(self['latc'].shape,np.nan,dtype=np.float32)
                    self['era5_'+key][mask] = sounding_interp[key]
            else:
                from popy import F_interp_era5_uv
                era5_3d_path_pattern = met_dir or \
                '/projects/academic/kangsun/data/ERA5/Y%Y/M%m/D%d/CONUS_3D_%Y%m%d.nc'
                era5_2d_path_pattern = kwargs.pop('era5_2d_path_pattern',None)
                if interp_fields is None:
                    interp_fields = ['u10','v10']
                sounding_interp = F_interp_era5_uv(
                    sounding_lon,sounding_lat,sounding_datenum,
                    era5_3d_path_pattern,era5_2d_path_pattern,interp_fields,altitudes)
                for key in sounding_interp.keys():
                    self.logger.info(key+' from ERA5 is sampled to L2 coordinate/time')
                    self['era5_'+key] = np.full(self['latc'].shape,np.nan,dtype=np.float32)
                    self['era5_'+key][mask] = sounding_interp[key]
    
    def get_background(self,field='column_amount',
                       percentile=10,percentile_window=(50,75),
                       smoothing_window=None):
        '''too much wasted time. better done in l3 of single overpass'''
        if smoothing_window is None:
            smoothing_window = percentile_window
        
        kernel = np.ones(smoothing_window)
        self[field+'_background'] = convolve_fft(
            percentile_filter(self[field],percentile=percentile,
                              size=percentile_window,mode='nearest'),kernel=kernel)
        
    def get_wind_stripe(self,east_wind_field=None,north_wind_field=None,min_abs_udotx=1.):
        east_wind_field = east_wind_field or 'era5_u500'
        north_wind_field = north_wind_field or 'era5_v500'
        # eastward and northward wind
        windu = self[east_wind_field]
        windv = self[north_wind_field]
        # wind vector dot x vector
        self['udotx'] = windu*np.cos(self['thetax']) + windv*np.sin(self['thetax'])
        mask = (np.abs(self['udotx']) > min_abs_udotx)
        self['wind_stripe'] = np.full(self['column_amount'].shape,np.nan)
        self['wind_stripe'][mask] = (self['column_amount_DD_xy']/self['udotx'])[mask]
        
    def get_DD(self,east_wind_field=None,north_wind_field=None,fields=None,
               keep_single_xyrs=False,do_DIV=False):
        '''calculate directional derivatives, (u,v) dot (dvcd/dx,dvcd/dy) using xy and rs directions
        east/north_wind_field:
            u/v wind field name in met data
        fields:
            data fields to calculate DD
        keep_single_xyrs:
            if true, when one of xy and rs is nan, report the average as the non-nan element. 
            otherwise, nan the average
        do_DIV:
            if true, calculate flux divergence d(u*vcd)/dx+d(v*vcd)/dy
        '''
        if fields is None:
            fields = ['column_amount']
        east_wind_field = east_wind_field or 'era5_u500'
        north_wind_field = north_wind_field or 'era5_v500'
        # eastward and northward wind
        windu = self[east_wind_field]
        windv = self[north_wind_field]
        
        # xward and yward wind
        windx = (np.sin(self['thetay'])*windu - np.cos(self['thetay'])*windv)/self['det_xy']
        windy = (-np.sin(self['thetax'])*windu + np.cos(self['thetax'])*windv)/self['det_xy']
        
        # rward and sward wind
        windr = (np.sin(self['thetas'])*windu - np.cos(self['thetas'])*windv)/self['det_rs']
        winds = (-np.sin(self['thetar'])*windu + np.cos(self['thetar'])*windv)/self['det_rs']
        
        def latlon2m(lat1,lon1,lat2,lon2,m_per_lat,m_per_lon):
            '''function to return distance in meter using latlon matrices'''
            return np.sqrt(
                np.square((lat1-lat2)*m_per_lat)
                +np.square((lon1-lon2)*m_per_lon))
        
        for field in fields:
            f = self[field]
            # gradients
            dfdx = np.full_like(f,np.nan)
            dfdy = np.full_like(f,np.nan)
            dfdr = np.full_like(f,np.nan)
            dfds = np.full_like(f,np.nan)
            dfdx[:,1:-1] = (f[:,:-2]-f[:,2:])/ \
            latlon2m(lat1=self['latc'][:,:-2],lon1=self['lonc'][:,:-2],
                    lat2=self['latc'][:,2:],lon2=self['lonc'][:,2:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][:,1:-1]
                    )
            dfdy[1:-1,:] = (f[2:,:]-f[:-2,:])/ \
            latlon2m(lat1=self['latc'][2:,:],lon1=self['lonc'][2:,:],
                    lat2=self['latc'][:-2,:],lon2=self['lonc'][:-2,:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,:]
                    )
            dfdr[1:-1,1:-1] = (f[2:,:-2]-f[:-2,2:])/ \
            latlon2m(lat1=self['latc'][2:,:-2],lon1=self['lonc'][2:,:-2],
                    lat2=self['latc'][:-2,2:],lon2=self['lonc'][:-2,2:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,1:-1]
                    )
            dfds[1:-1,1:-1] = (f[2:,2:]-f[:-2,:-2])/ \
            latlon2m(lat1=self['latc'][2:,2:],lon1=self['lonc'][2:,2:],
                    lat2=self['latc'][:-2,:-2],lon2=self['lonc'][:-2,:-2],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,1:-1]
                    )
            
            self[f'd{field}dx'] = dfdx 
            self[f'd{field}dy'] = dfdy
            self[f'd{field}dr'] = dfdr
            self[f'd{field}ds'] = dfds
            
            ### proven to be equivalent to the simple formula below
#             # gradient (vec{g}) = gx*vec{x}+gy*vec{y} = gr*vec{r}+gy*vec{r}
#             gx = (dfdx-dfdy*self['xdoty'])/(1-np.square(self['xdoty']))
#             gy = (dfdy-dfdx*self['xdoty'])/(1-np.square(self['xdoty']))
            
#             gr = (dfdr-dfds*self['rdots'])/(1-np.square(self['rdots']))
#             gs = (dfds-dfdr*self['rdots'])/(1-np.square(self['rdots']))
            
#             # directional derivatives
#             self[field+'_DD_xy'] = windx*gx + windy*gy + \
#             self['xdoty']*(windx*gy + windy*gx)
#             self[field+'_DD_rs'] = windr*gr + winds*gs + \
#             self['rdots']*(windr*gs + winds*gr)
            ### proven to be equivalent to the simple formula below
            
            # directional derivatives
            self[field+'_DD_xy'] = windx*dfdx + windy*dfdy
            self[field+'_DD_rs'] = windr*dfdr + winds*dfds
            if keep_single_xyrs:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                    self[field+'_DD'] = np.nanmean(np.array([self[field+'_DD_xy'],
                                                            self[field+'_DD_rs']]),
                                                  axis=0)
                self[field+'_DD'][np.isnan(self[field+'_DD_xy']) & np.isnan(self[field+'_DD_rs'])] = np.nan
            else:
                self[field+'_DD'] = (self[field+'_DD_xy']+self[field+'_DD_rs'])*0.5
        if not do_DIV:
            return
        for field in fields:
            f = self[field]
            # gradients
            dfdx = np.full_like(f,np.nan)
            dfdy = np.full_like(f,np.nan)
            dfdr = np.full_like(f,np.nan)
            dfds = np.full_like(f,np.nan)
            dfdx[:,1:-1] = ((f*windx)[:,:-2]-(f*windx)[:,2:])/ \
            latlon2m(lat1=self['latc'][:,:-2],lon1=self['lonc'][:,:-2],
                    lat2=self['latc'][:,2:],lon2=self['lonc'][:,2:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][:,1:-1]
                    )
            dfdy[1:-1,:] = ((f*windy)[2:,:]-(f*windy)[:-2,:])/ \
            latlon2m(lat1=self['latc'][2:,:],lon1=self['lonc'][2:,:],
                    lat2=self['latc'][:-2,:],lon2=self['lonc'][:-2,:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,:]
                    )
            dfdr[1:-1,1:-1] = ((f*windr)[2:,:-2]-(f*windr)[:-2,2:])/ \
            latlon2m(lat1=self['latc'][2:,:-2],lon1=self['lonc'][2:,:-2],
                    lat2=self['latc'][:-2,2:],lon2=self['lonc'][:-2,2:],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,1:-1]
                    )
            dfds[1:-1,1:-1] = ((f*winds)[2:,2:]-(f*winds)[:-2,:-2])/ \
            latlon2m(lat1=self['latc'][2:,2:],lon1=self['lonc'][2:,2:],
                    lat2=self['latc'][:-2,:-2],lon2=self['lonc'][:-2,:-2],
                    m_per_lat=self['m_per_lat'],m_per_lon=self['m_per_lon'][1:-1,1:-1]
                    )
#             self[f'dF{field}dx'] = dfdx
#             self[f'dF{field}dy'] = dfdy
#             self[f'dF{field}dr'] = dfdr
#             self[f'dF{field}ds'] = dfds
            # flux divergences
            self[field+'_DIV_xy'] = dfdx + dfdy
            self[field+'_DIV_rs'] = dfdr + dfds
            if keep_single_xyrs:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                    self[field+'_DIV'] = np.nanmean(np.array([self[field+'_DIV_xy'],
                                                            self[field+'_DIV_rs']]),
                                                  axis=0)
                self[field+'_DIV'][np.isnan(self[field+'_DIV_xy']) & np.isnan(self[field+'_DIV_rs'])] = np.nan
            else:
                self[field+'_DIV'] = (self[field+'_DIV_xy']+self[field+'_DIV_rs'])*0.5
        
    
    def to_popy_l2g_data(self,west=-180,east=180,south=-90,north=90,additional_mapping=None):
        mask = (self['lonc'] >= west) & (self['lonc'] <= east) &\
        (self['latc'] >= south) & (self['latc'] <= north) &\
        ~np.isnan(self['column_amount']) 
        if 'column_amount_DD' in self.keys():
            mask = mask & ~np.isnan(self['column_amount_DD'])
        
        l2g_data = {k:self[k][mask] for k in ['cloud_fraction','latc','lonc',
                                               'surface_pressure','terrain_height',
                                               'column_amount','UTC_matlab_datenum',
                                               'across_track_position']}
        # datenum in local time
        local_dn = l2g_data['UTC_matlab_datenum']+l2g_data['lonc']/15/24
        l2g_data['local_hour'] = (local_dn-np.floor(local_dn))*24
        l2g_data['latr'] = np.column_stack((self['latr'][:,:,2][mask],
                                           self['latr'][:,:,1][mask],
                                           self['latr'][:,:,0][mask],
                                           self['latr'][:,:,3][mask]))
        l2g_data['lonr'] = np.column_stack((self['lonr'][:,:,2][mask],
                                           self['lonr'][:,:,1][mask],
                                           self['lonr'][:,:,0][mask],
                                           self['lonr'][:,:,3][mask]))
        if 'terrain_height_DD' in self.keys():
            l2g_data['wind_topo'] = l2g_data['column_amount']*self['terrain_height_DD'][mask]
        if 'terrain_height_DD_xy' in self.keys():
            l2g_data['wind_topo_xy'] = l2g_data['column_amount']*self['terrain_height_DD_xy'][mask]
        if 'terrain_height_DD_rs' in self.keys():
            l2g_data['wind_topo_rs'] = l2g_data['column_amount']*self['terrain_height_DD_rs'][mask]
        wind_column_mapping = {
            'column_amount_DD_xy':'wind_column_xy',
            'column_amount_DD_rs':'wind_column_rs',
            'column_amount_DD':'wind_column',
            'udotx':'udotx',
            'wind_stripe':'wind_stripe',
            'column_amount_DIV_xy':'column_amount_DIV_xy',
            'column_amount_DIV_rs':'column_amount_DIV_rs',
            'column_amount_DIV':'column_amount_DIV'
        }
        for k,v in wind_column_mapping.items():
            if k not in self.keys():
                self.logger.info(f'{k} not in TEMPOL2!')
                continue
            l2g_data[v] = self[k][mask]
        
        additional_mapping = additional_mapping or {}
        for k,v in additional_mapping.items():
            if k not in self.keys():
                self.logger.info(f'{k} not in TEMPOL2!')
                continue
            l2g_data[v] = self[k][mask]
        return l2g_data
    
    def plot(self,central_lon,central_lat,WE_ext=1,SN_ext=1,if_latlon=False,
             xlim=None,ylim=None,plot_field='column_amount',
             ax=None,figsize=None,basis_kw=None,wind_kw=None,
             grad_xy_kw=None,grad_rs_kw=None,**kwargs):
        if ax is None:
            figsize = kwargs.pop('figsize',(10,5))
            if if_latlon:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True,
                                     subplot_kw={"projection": ccrs.PlateCarree()})
            else:
                fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
        else:
            fig = None
        cmap = kwargs.pop('cmap','jet')
        alpha = kwargs.pop('alpha',1)
        func = kwargs.pop('func',lambda x:x)
        ec = kwargs.pop('ec','none')
        draw_colorbar = kwargs.pop('draw_colorbar',True)
        label = kwargs.pop('label',plot_field)
        shrink = kwargs.pop('shrink',0.75)
        cartopy_scale = kwargs.pop('cartopy_scale','50m')
        distance = np.sqrt(np.square((self['latc']-central_lat)*self['m_per_lat'])\
                           +np.square((self['lonc']-central_lon)*self['m_per_lon']))
        min_idx = np.unravel_index(np.nanargmin(distance), distance.shape)
        clon = self['lonc'][min_idx]
        clat = self['latc'][min_idx]
        WE_idx0 = np.max([min_idx[0]-WE_ext,0])
        WE_idx1 = np.min([min_idx[0]+WE_ext+1,self['latc'].shape[0]-1])

        SN_idx0 = np.max([min_idx[1]-SN_ext,0])
        SN_idx1 = np.min([min_idx[1]+SN_ext+1,self['latc'].shape[1]-1])
        
        # plot l2 pixel polygons
        if if_latlon:
            latrs = self['latr'][WE_idx0:WE_idx1,SN_idx0:SN_idx1,]
            lonrs = self['lonr'][WE_idx0:WE_idx1,SN_idx0:SN_idx1,]
        else:
            latrs = (self['latr'][WE_idx0:WE_idx1,SN_idx0:SN_idx1,]-clat)*\
            self['m_per_lat']*1e-3
            lonrs = (self['lonr'][WE_idx0:WE_idx1,SN_idx0:SN_idx1,]-clon)*\
            self['m_per_lon'][WE_idx0:WE_idx1,SN_idx0:SN_idx1,np.newaxis]*1e-3
        
        latrs = latrs.reshape(-1,latrs.shape[-1])
        lonrs = lonrs.reshape(-1,lonrs.shape[-1])
        verts = [np.array([lonr,latr]).T for lonr,latr in zip(lonrs,latrs)]
        cdata = self[plot_field][WE_idx0:WE_idx1,SN_idx0:SN_idx1].ravel()
        vmin = kwargs.pop('vmin',np.nanmin(cdata))
        vmax = kwargs.pop('vmax',np.nanmax(cdata))
        collection = PolyCollection(verts,
                     array=cdata,
                 cmap=cmap,edgecolors=ec)
        collection.set_alpha(alpha)
        collection.set_clim(vmin=vmin,vmax=vmax)
        ax.add_collection(collection)
        
        if xlim is None:
            ax.set_xlim([np.nanmin(lonrs),np.nanmax(lonrs)])
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim([np.nanmin(latrs),np.nanmax(latrs)])
        else:
            ax.set_ylim(ylim)
        
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
        
        # draw basis vectors
        if basis_kw is not None:
            if if_latlon:
                self.logger.error('not compatible!')
                return
            basis_unit_km = basis_kw.pop('basis_unit_km',3)
            basis_cc = basis_kw.pop('basis_cc',['w','w','k','k'])
            fontsize = basis_kw.pop('fontsize',15)
            width = basis_kw.pop('width',0.005)
            ha = basis_kw.pop('ha','right')
            for ibasis,basis in enumerate(['x','y','r','s']):
                basis_e = np.cos(self['theta'+basis][min_idx])*basis_unit_km
                basis_n = np.sin(self['theta'+basis][min_idx])*basis_unit_km
                basis_o1 = (self['lonc'][min_idx]-clon)*self['m_per_lon'][min_idx]*1e-3
                basis_o2 = (self['latc'][min_idx]-clat)*self['m_per_lat']*1e-3
                ax.quiver(basis_o1,basis_o2,basis_e,basis_n,
                          angles='xy', scale_units='xy', 
                          scale=1,width=0.005,color=basis_cc[ibasis])
                ax.text(basis_o1+basis_e,basis_o2+basis_n,
                        r'$\vec{'+basis+r'}$',ha=ha,color=basis_cc[ibasis],
                        fontsize=fontsize)
        
        # draw wind vectors
        if wind_kw is not None:
            step = wind_kw.pop('step',2)
            scale = wind_kw.pop('scale',100)
            width = wind_kw.pop('width',0.003)
            east_wind_field = wind_kw.pop('east_wind_field','era5_u500')
            north_wind_field = wind_kw.pop('north_wind_field','era5_v500')
            wind_e = self[east_wind_field][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
            wind_n = self[north_wind_field][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
            if if_latlon:
                basis_o1 = self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
                basis_o2 = self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
            else:
                basis_o1 = (self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clon)*\
                self['m_per_lon'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*1e-3
                basis_o2 = (self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clat)*\
                self['m_per_lat']*1e-3
            ax.quiver(basis_o1.ravel(),basis_o2.ravel(),
                      wind_e.ravel(),wind_n.ravel(),scale=scale,width=width,**wind_kw)
        
        # draw xy gradient vectors
        if grad_xy_kw is not None:
            step = grad_xy_kw.pop('step',2)
            scale = grad_xy_kw.pop('scale',2e-7)
            width = grad_xy_kw.pop('width',0.003)
            color = grad_xy_kw.pop('color','b')
            if if_latlon:
                basis_o1 = self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
                basis_o2 = self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
            else:
                basis_o1 = (self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clon)*\
                self['m_per_lon'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*1e-3
                basis_o2 = (self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clat)*\
                self['m_per_lat']*1e-3
            
            grad_e = self[f'd{plot_field}dx'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.cos(self['thetax'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])+\
            self[f'd{plot_field}dy'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.cos(self['thetay'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])

            grad_n = self[f'd{plot_field}dx'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.sin(self['thetax'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])+\
            self[f'd{plot_field}dy'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.sin(self['thetay'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])
            ax.quiver(basis_o1.ravel(),basis_o2.ravel(),
                      grad_e.ravel(),grad_n.ravel(),scale=scale,width=width,color=color,**grad_xy_kw)
        
        # draw rs gradient vectors
        if grad_rs_kw is not None:
            step = grad_rs_kw.pop('step',2)
            scale = grad_rs_kw.pop('scale',2e-7)
            width = grad_rs_kw.pop('width',0.003)
            color = grad_rs_kw.pop('color','r')
            if if_latlon:
                basis_o1 = self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
                basis_o2 = self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]
            else:
                basis_o1 = (self['lonc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clon)*\
                self['m_per_lon'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*1e-3
                basis_o2 = (self['latc'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]-clat)*\
                self['m_per_lat']*1e-3
            
            grad_e = self[f'd{plot_field}dr'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.cos(self['thetar'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])+\
            self[f'd{plot_field}ds'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.cos(self['thetas'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])

            grad_n = self[f'd{plot_field}dr'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.sin(self['thetar'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])+\
            self[f'd{plot_field}ds'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step]*\
            np.sin(self['thetas'][WE_idx0:WE_idx1:step,SN_idx0:SN_idx1:step])
            ax.quiver(basis_o1.ravel(),basis_o2.ravel(),
                      grad_e.ravel(),grad_n.ravel(),scale=scale,width=width,color=color,**grad_rs_kw)
        return dict(fig=fig,ax=ax,cb=cb,clon=clon,clat=clat)
    