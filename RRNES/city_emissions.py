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
from matplotlib.colors import LogNorm
from scipy.io import loadmat
import rasterio
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib import path 
from scipy.io import loadmat

class Geo_Raster(dict):
    '''class based on dict, representing a geospatial raster'''
    def __init__(self,name):
        self.logger = logging.getLogger(__name__)
        self.name = name
    
    def read_mat(self,mat_fn):
        
        d = loadmat(mat_fn,squeeze_me=True)
        self['xgrid'] = d['xgrid']
        self['ygrid'] = d['ygrid']
        self['xres'] = np.nanmedian(np.diff(self['xgrid']))
        self['yres'] = np.nanmedian(np.diff(self['ygrid']))
        self['data'] = d['mask'] == 1
        return self
    
    def trim(self,west,east,south,north,name=None):
        name = name or self.name
        new = Geo_Raster(name)
        xmask = (self['xgrid']>=west) & (self['xgrid']<=east)
        ymask = (self['ygrid']>=south) & (self['ygrid']<=north)
        new['xgrid'] = self['xgrid'][xmask]
        new['ygrid'] = self['ygrid'][ymask]
        new['xres'] = self['xres']
        new['yres'] = self['yres']
        new['data'] = self['data'][np.ix_(ymask,xmask)]
        return new
    
    def trim_by_polygon(self,boundary_x,boundary_y,
                        in_boundary_value=None,out_boundary_value=None,name=None):
        
        boundary_polygon = path.Path([(bx,by) for (bx,by) in zip(boundary_x,boundary_y)])
        name = name or self.name
        new = Geo_Raster(name)
        xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
        grid_points = np.column_stack((xmesh.ravel(),ymesh.ravel()))
        in_mask = boundary_polygon.contains_points(grid_points).reshape(xmesh.shape)
        new['xgrid'] = self['xgrid']
        new['ygrid'] = self['ygrid']
        new['xres'] = self['xres']
        new['yres'] = self['yres']
        new['data'] = self['data']
        if in_boundary_value is not None:
            new['data'][in_mask] = in_boundary_value
        if out_boundary_value is not None:
            new['data'][~in_mask] = out_boundary_value
        return new
        
    def read_tif(self,tif_fn):
        import rasterio
        with rasterio.open(tif_fn) as src:
            data = src.read().squeeze(axis=0)
            xres = src.transform[1]
            yres = src.transform[5]
            xorig = src.transform[0]
            yorig = src.transform[3]
            xgrid = xorig+np.arange(0,src.width)*xres
            ygrid = yorig+np.arange(0,src.height)*yres
        self['xgrid'] = xgrid
        self['ygrid'] = ygrid
        self['xres'] = xres
        self['yres'] = yres
        self['data'] = data
        return self
    
    def plot(self,ax=None,**kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        pc=ax.pcolormesh(self['xgrid'],self['ygrid'],self['data'],shading='auto',**kwargs)
        figout = {'pc':pc,'ax':ax}


class Region(dict):
    '''class based on dict, representing a large region containing one or more subregions'''
    def __init__(self,name,bounds_list,west,east,south,north):
        '''
        name:
            name of region
        bounds_list:
            a list of list of four numbers indicating boundaries for subregions, west, east, south, north
        west, east, south, north:
            boundary of the entire region
        '''
        self.logger = logging.getLogger(__name__)
        for i,rb in enumerate(bounds_list):
            self['{}{}'.format(name,i)] = {'west':rb[0],'east':rb[1],'south':rb[2],'north':rb[3]}
        self.name = name
        self.west = west
        self.east = east
        self.south = south
        self.north = north
    def plot(self,ax=None):
        '''overview subregions'''
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        for k,v in self.items():
            ax.add_patch(Rectangle((v['west'],v['south']),
                                  v['east']-v['west'],v['north']-v['south'],
                                 fc='none',color='r',linewidth=1,linestyle='--'))
        ax.set_extent([self.west,self.east,self.south,self.north])
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        for k,v in self.items():
            ax.plot(v['city_df']['lng'],v['city_df']['lat'],'o')
            for irow,row in v['city_df'].iterrows():
                ax.text(row.lng+0.1,row.lat+0.1,row.city_ascii)
    def get_city_df(self,all_city_df,region_raster,city_names=None,max_ncity=20,box_km=50):
        '''
        populate each item as one subregion
        all_city_df:
            pd.read_csv('worldcities.csv'); see https://www.kaggle.com/datasets/juanmah/world-cities
        region_raster:
            a Geo_Raster object covering the entire region and indicating city coverage
        city_names:
            specify city names to include. a list of city_ascii strings or dict for cities with non-unique names
        max_ncity:
            from the largest, max number of cities to include
        box_km:
            +/- this value will be included as the raster for the city
        '''
        lat_margin = box_km/111
        lon_margin = box_km/(111*np.cos(np.deg2rad(self.north)))
        for k,v in self.items():
            loc_filter = (all_city_df['lat']>=v['south']+lat_margin)&(all_city_df['lat']<=v['north']-lat_margin)\
            &(all_city_df['lng']>=v['west']+lon_margin)&(all_city_df['lng']<=v['east']-lon_margin)
            city_filter = np.zeros(loc_filter.shape,dtype=bool)
            if city_names is not None:
                for city_name in city_names:
                    if isinstance(city_name,str):
                        city_filter = city_filter | all_city_df['city_ascii'].isin([city_name])
                    elif isinstance(city_name,dict):
                        dict_filter = np.ones(loc_filter.shape,dtype=bool)
                        for k1,v1 in city_name.items():
                            dict_filter = dict_filter & all_city_df[k1].isin([v1])
                        city_filter = city_filter | dict_filter
                loc_filter = loc_filter & city_filter
            v['city_df'] = all_city_df.loc[loc_filter].\
            sort_values('population',ascending=False).\
            iloc[:max_ncity].reset_index(drop=True)
            v['city_df']['west'] = v['city_df']['lng']-box_km/np.cos(np.deg2rad(v['city_df']['lat']))/111
            v['city_df']['east'] = v['city_df']['lng']+box_km/np.cos(np.deg2rad(v['city_df']['lat']))/111
            v['city_df']['south'] = v['city_df']['lat']-box_km/111
            v['city_df']['north'] = v['city_df']['lat']+box_km/111
            v['city_raster_list'] = []
            for irow,row in v['city_df'].iterrows():
                v['city_raster_list'].append(
                    region_raster.trim(
                        west=row.west,east=row.east,south=row.south,north=row.north,name=row.city_ascii))
    
    def update_city_raster(self,city_name,region_raster,lon_box_km=100,lat_box_km=100,
                           min_lat=-90,max_lat=90,
                           boundary_x=None,boundary_y=None,
                           in_boundary_value=None,out_boundary_value=None):
        '''fine tuning boundary for individual cities
        city_name:
            name of the city
        lon/lat_box_km:
            adjust to rectangle box instead square one. Useful for Los Angeles
        min/max_lat:
            trim boundaries, e.g., for border cities
        boundary_x/y:
            lon/lat of a boundary shape
        in_boundary:
            True if the city should be in the boundary, False if the city should be out
        '''
        for i,(k,v) in enumerate(self.items()):
            df = v['city_df']
            # skip subregion if the city is not in it
            if not np.isin(city_name,df['city_ascii']):continue
            # update city box size
            if isinstance(lon_box_km,list):
                xleft = lon_box_km[0]
                xright = lon_box_km[1]
            else:
                xleft = lon_box_km
                xright = lon_box_km
            if isinstance(lat_box_km,list):
                ylower = lat_box_km[0]
                yupper = lat_box_km[1]
            else:
                ylower = lat_box_km
                yupper = lat_box_km
            df=df.reset_index(drop=True)
            df.loc[df['city_ascii']==city_name,'west'] = df.loc[df['city_ascii']==city_name]['lng']\
            -xleft/np.cos(np.deg2rad(df.loc[df['city_ascii']==city_name]['lat']))/111
            df.loc[df['city_ascii']==city_name,'east'] = df.loc[df['city_ascii']==city_name]['lng']\
            +xright/np.cos(np.deg2rad(df.loc[df['city_ascii']==city_name]['lat']))/111
            df.loc[df['city_ascii']==city_name,'south'] = np.max([df.loc[df['city_ascii']==city_name]['lat'].squeeze()-ylower/111,min_lat])
            df.loc[df['city_ascii']==city_name,'north'] = np.min([df.loc[df['city_ascii']==city_name]['lat'].squeeze()+yupper/111,max_lat])
            new_raster=\
            region_raster.trim(west=df.loc[df['city_ascii']==city_name]['west'].squeeze(),
                         east=df.loc[df['city_ascii']==city_name]['east'].squeeze(),
                         south=df.loc[df['city_ascii']==city_name]['south'].squeeze(),
                         north=df.loc[df['city_ascii']==city_name]['north'].squeeze(),
                         name=df.loc[df['city_ascii']==city_name]['city_ascii'].squeeze())
            if boundary_x is not None:
                new_raster=\
                new_raster.trim_by_polygon(boundary_x=boundary_x,boundary_y=boundary_y,
                                           in_boundary_value=in_boundary_value,
                                           out_boundary_value=out_boundary_value)
            self[k]['city_raster_list'][df.loc[df['city_ascii']==city_name].index[0]] = new_raster
            self[k]['city_df'] = df
        
    def get_city_emission(self,period_range,l3_path_pattern,fit_topo_kw=None,fit_chem_kw=None,
                          chem_min_column_amount=None,chem_max_wind_column=None):
        '''work with the Level3_List class to get time-resolved emissions for each city
        period_range:
            time input to Level3_List
        l3_path_pattern:
            l3 file locations 
        fit_topo_kw:
            keyword argument to fit_topography function
        fit_chem_kw:
            keyword argument to fit_chemistry function
        chem_min_column_amount:
            min column amount allowed in fit_chem
        chem_max_wind_column:
            max wind_column allowed in fit_chem
        '''
        fit_topo_kw = fit_topo_kw or {}
        fit_chem_kw = fit_chem_kw or dict(resample_rule='month_of_year',return_resampled=True)
        if 'resample_rule' not in fit_chem_kw.keys():
            fit_chem_kw['resample_rule'] = 'month_of_year'
        if 'return_resampled' not in fit_chem_kw.keys():
            fit_chem_kw['return_resampled'] = True
        for i,(k,v) in enumerate(self.items()):
            if isinstance(l3_path_pattern,(list, tuple, np.ndarray)):
                l3p = l3_path_pattern[i]
            else:
                l3p = l3_path_pattern
            l3s = Level3_List(dt_array=period_range,west=v['west'],east=v['east'],south=v['south'],north=v['north'])
            l3s.read_nc_pattern(l3_path_pattern=l3p,
                               fields_name=['column_amount','surface_altitude','wind_topo',
                                        'wind_column','wind_column_xy','wind_column_rs'])
            l3s.fit_topography(**fit_topo_kw)
            if chem_min_column_amount is not None or chem_max_wind_column is not None:
                l3 = l3s.aggregate()
                mask = np.ones(l3['column_amount'].shape,dtype=bool)
                if chem_min_column_amount is not None:
                    mask = mask & (l3['column_amount']>=chem_min_column_amount)
                if chem_max_wind_column is not None:
                    mask = mask & (l3['wind_column']<=chem_max_wind_column)
                fit_chem_kw['mask'] = mask
            l3s_m = l3s.fit_chemistry(**fit_chem_kw)
            [l3s.average_by_finerMask(tif_dict=city_raster) \
                                       for city_raster in v['city_raster_list']];
            v['l3s'] = l3s            
        
    def plot_monthly_emission(self,subregion_name,nrow=3,ncol=3,hspace=0.39,wspace=0.15,
                              figsize=None,min_D=2,xlim=None,abs_ylim=None,rel_ylim=None,
                              label_ax_idx=None):
        '''next generation of plot_city_emission, add relative value panels'''
        import matplotlib.dates as mdates
        # import matplotlib.ticker as mtick
        figsize = figsize or (12,5)
        v = self[subregion_name]
        abs_ylim = abs_ylim or [-2,30]
        rel_ylim = rel_ylim or [0,2.5]
        xlim = xlim or [dt.datetime(2018,5,1),dt.datetime(2023,2,1)]
        label_ax_idx = label_ax_idx or (0,1,0)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        outer_grid = fig.add_gridspec(nrow, ncol, hspace=hspace,wspace=wspace)
        axss = np.empty((nrow,ncol,2),dtype=np.object)
        df = v['l3s'].df

        icity = 0
        for a in range(nrow):
            for b in range(ncol):
                city_name = v['city_df']['city_ascii'].iloc[icity]
                y2019 = np.zeros_like(df['{}_wind_column'.format(city_name)])
                for month in range(1,13):
                    y2019[df.index.month==month] = \
                    df['{}_wind_column_topo_chem'.format(city_name)].loc[(df.index.month==month) & (df.index.year==2019)]
                y2019 = y2019*1.32e9
                xdata = df.index.start_time+dt.timedelta(days=0)
                ydata = df['{}_wind_column_topo_chem'.format(city_name)] *1.32e9
                mask = df['{}_num_samples'.format(city_name)]>=min_D
                # remove negative months (saw in autumn Milan)
                mask = mask & (ydata>=0)
                # if one month in 2019-22 is invalid, discard this month for all years
                for month in range(1,13):
                    if not all(mask[(df.index.month == month) & (df.index.year>2018) & (df.index.year<2023)]):
                        mask[df.index.month == month] = False
                xdata = xdata[mask]
                ydata = ydata[mask]
                y2019 = y2019[mask]

                # gridspec inside gridspec
                inner_grid = outer_grid[a, b].subgridspec(2, 1, wspace=0, hspace=0)
                axs = inner_grid.subplots(sharex=True)  # Create all subplots for the inner grid.
                lines = np.empty(2,dtype=np.object)
                ax = axs[0]
                lines[0] = ax.plot(xdata,ydata,'r-o',xdata,y2019,':',markersize=3,mec=(1,0,0,0.5),mfc=(1,0,0,0.3))
                ax.fill_between(xdata,ydata,y2019,
                                where=(ydata<=y2019),
                                facecolor='b', alpha=0.25,interpolate=True)
                ax.fill_between(xdata,ydata,y2019,
                                where=(ydata>=y2019),
                                facecolor='r', alpha=0.25,interpolate=True)
                ax.set_ylim(abs_ylim)
                ax.set_xlim(xlim)
                ax.set_title('({}) {}'.format(chr(icity+97),city_name))
                ax.grid()

                ax = axs[1]
                lines[1] = ax.plot(xdata,ydata/y2019,'r-o',xdata,y2019/y2019,':',markersize=3,mec=(1,0,0,0.5),mfc=(1,0,0,0.3))
                ax.fill_between(xdata,ydata/y2019,y2019/y2019,
                                where=(ydata<=y2019),
                                facecolor='b', alpha=0.25,interpolate=True)
                ax.fill_between(xdata,ydata/y2019,y2019/y2019,
                                where=(ydata>=y2019),
                                facecolor='r', alpha=0.25,interpolate=True)
                ax.set_ylim(rel_ylim)
                ax.set_xlim(xlim)
                trimmask = ydata/y2019>rel_ylim[1]
                for x,y in zip(xdata[trimmask],(ydata/y2019)[trimmask]):
                    self.logger.warning('{} relative emission is {} in {}'.format(city_name,y,x.strftime('%Y%m')))
                ax.grid()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        #         ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                # Rotates and right-aligns the x labels so they don't crowd each other.
                for label in ax.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')

                axs[1].yaxis.tick_right()
                axs[1].yaxis.set_label_position("right")
                if label_ax_idx is not None:
                    if a == label_ax_idx[0] and b == label_ax_idx[1]:
                        idx = label_ax_idx[2]
                        axs[idx].legend(lines[idx],['Monthly emissions','Repeated 2019 emissions'])
                axss[a,b,:] = axs

                icity += 1
        for ax in axss[:,0,0]:
            ax.set_ylabel(r'nmol m$^{-2}$ s$^{-1}$');
        for ax in axss[:,-1,1]:
            ax.set_ylabel('Relative values to 2019');
        return {'fig':fig,'axss':axss}
    
    def plot_city_emission(self,subregion_name,nrow=3,ncol=3,figsize=None,min_D=2,xlim=None,ylim=None):
        '''plot the time series of city emissions per subregion'''
        import matplotlib.dates as mdates
        figsize = figsize or (12,5)
        v = self[subregion_name]
        
        fig,axs = plt.subplots(nrow,ncol,figsize=figsize,sharex=True,sharey=True,constrained_layout=True)
        axs = axs.ravel()
        for iax,(ax,city_name) in enumerate(zip(axs,v['city_df']['city_ascii'])):
            df = v['l3s'].df

            y2019 = np.zeros_like(df['{}_wind_column'.format(city_name)])
            for month in range(1,13):
                y2019[df.index.month==month] = \
                df['{}_wind_column_topo_chem'.format(city_name)].loc[(df.index.month==month) & (df.index.year==2019)]
            y2019 = y2019*1.32e9
            xdata = df.index.start_time+dt.timedelta(days=15)
            ydata = df['{}_wind_column_topo_chem'.format(city_name)] *1.32e9
            mask = df['{}_num_samples'.format(city_name)]>=min_D
            # if one month in 2019-22 is invalid, discard this month for all years
            for month in range(1,13):
                if not all(mask[(df.index.month == month) & (df.index.year>2018)]):
                    mask[df.index.month == month] = False
            xdata = xdata[mask]
            ydata = ydata[mask]
            y2019 = y2019[mask]
            ax.plot(xdata,ydata,'r-o',xdata,y2019,':',markersize=3,mec=(1,0,0,0.5),mfc=(1,0,0,0.3))
            ax.fill_between(xdata,ydata,y2019,
                            where=(ydata<=y2019),
                            facecolor='b', alpha=0.25,interpolate=True)
            ax.fill_between(xdata,ydata,y2019,
                            where=(ydata>=y2019),
                            facecolor='r', alpha=0.25,interpolate=True)
            ax.grid()
            if xlim is not None:
                ax.set_xlim(xlim);
            if ylim is not None:
                ax.set_ylim(ylim);
            ax.set_title('({}) {}'.format(chr(iax+97),city_name))
        axs[3].set_ylabel(r'Emission [nmol m$^{-2}$ s$^{-1}$]');
        for ax in axs[-3:]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
            # Rotates and right-aligns the x labels so they don't crowd each other.
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
        return {'fig':fig,'axs':axs}
    
    def get_annual_df(self,subregion_name,min_D=2,min_wind_column_topo_chem=0):
        '''get annual df from monthly df self[subregion_name]['l3s'].df
        min_D:
            if num_samples of a city in a month is below this value, the wind_column* 
            in this month of all years for this city will be np.nan
        '''
        v = self[subregion_name]
        df = v['l3s'].df.copy()
        for irow,row in v['city_df'].iterrows():
            city_name = row.city_ascii
            mask = (df['{}_num_samples'.format(city_name)]>=min_D) &\
            (df['{}_wind_column_topo_chem'.format(city_name)]>=min_wind_column_topo_chem)
            # if one month in 2019-22 is invalid, discard this month for all years
            for month in range(1,13):
                if not all(mask[(df.index.month == month) & (df.index.year>2018) & (df.index.year<2023)]):
                    mask[df.index.month == month] = False
            df.loc[~mask,'{}_wind_column_topo_chem'.format(city_name)] = np.nan
            df.loc[~mask,'{}_wind_column_topo'.format(city_name)] = np.nan
            df.loc[~mask,'{}_wind_column'.format(city_name)] = np.nan
        annual_df = df.resample('Y').mean()
        annual_df = annual_df.loc[(annual_df.index.start_time>=dt.datetime(2019,1,1))\
                                  &(annual_df.index.start_time<dt.datetime(2023,1,1))]
        return annual_df
    
    def map_pie_annual_emission(self,subregion_name,figsize=None,size_func=None,
                                labels_city_name=None,pie_offset_df=None,min_D=2):
        '''plot annual emission as pie charts on a map
        size_func:
            callable to project mean annual emission to pie radius. default is natural log
        lables_city_name:
            one city to label the wedges
        pie_offset_df:
            a dataframe, with fields city_ascii, dx, and dy, to specify cities with offset pie to avoid overlapping
        min_D:
            input to get_annual_df
        '''
        figsize = figsize or (15,5)
        v = self[subregion_name]
        annual_df = self.get_annual_df(subregion_name,min_D)
        fig,ax = plt.subplots(1,1,figsize=figsize,subplot_kw={"projection": ccrs.PlateCarree()})
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), facecolor='None', edgecolor='blue', 
                       linestyle='-.',zorder=0,lw=0.5)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='None',
                       edgecolor='blue', zorder=0, linestyle=':')
        if size_func is None:
            def size_func(x):
                return np.log(x)
        def pct_func(pct, allvals):
            absolute = (pct/100.*np.sum(allvals))
            return "{:.1f}".format(absolute)
        for irow,row in v['city_df'].iterrows():
            annual_nmol = annual_df[f'{row.city_ascii}_wind_column_topo_chem']*1.32e9
            total_nmol = np.nanmean(annual_nmol)
            size = size_func(total_nmol)
            xc = row.lng;yc = row.lat
            ax_size = 2
            dx = 0;dy = 0
            if pie_offset_df is not None:
                dx = pie_offset_df.loc[pie_offset_df['city_ascii'].isin([row.city_ascii])].dx
                dy = pie_offset_df.loc[pie_offset_df['city_ascii'].isin([row.city_ascii])].dy
                if dx.empty:
                    dx=0
                else:
                    dx = dx.squeeze()
                if dy.empty:
                    dy=0
                else:
                    dy = dy.squeeze()
            if np.square(dx)+np.square(dy) != 0:
                ax.arrow(xc+dx, yc+dy, -dx,-dy, head_width=0.2, head_length=0.2, fc='k', ec='k',length_includes_head=True)
            ax_sub = ax.inset_axes([xc-ax_size/2+dx,yc-ax_size/2+dy,ax_size,ax_size],
                                  transform=ax.transData)
            if row['city_ascii'] == labels_city_name:
                wedges, texts, autotexts = ax_sub.pie(annual_nmol,radius=size,startangle=90,counterclock=False,
                       wedgeprops={'alpha':0.9,'edgecolor':'none'},
                      autopct=lambda pct: pct_func(pct, annual_nmol),
                      textprops={'color':'w','fontsize':8},labels=range(2019,2023))
                for t in texts:
                    t.set_color('k')
            else:
                ax_sub.pie(annual_nmol,radius=size,startangle=90,counterclock=False,
                       wedgeprops={'alpha':0.9,'edgecolor':'none'},
                      autopct=lambda pct: pct_func(pct, annual_nmol),
                      textprops={'color':'w','fontsize':8})
            ax_sub.set_aspect("equal")
            ax.text(xc+dx,yc+size+dy,row['city_ascii'],ha='center',fontsize=11,fontweight='bold',zorder=100)

        ax.set_extent([v['west'],v['east'],v['south'],v['north']]) 
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        return {'fig':fig,'ax':ax,'gl':gl}
    
    def map_city_emission(self,subregion_name,draw_colorbar=False,func=None,plot_field='wind_column_topo_chem',
                          nrow=4,ncol=4,box_km=None,**kwargs):
        '''pcolormesh the emission maps for the cities'''
        fig,axs = plt.subplots(nrow,ncol,figsize=(10,7.5),constrained_layout=True)
        axs = axs.ravel()
        v = self[subregion_name]
        l3 = v['l3s'].aggregate()
        for iax,(ax,city_raster) in enumerate(zip(axs,v['city_raster_list'])):
            df_row = v['city_df'].iloc[iax]
            l3c = l3.trim(west=df_row.west,east=df_row.east,south=df_row.south,north=df_row.north)
            if func is None:
                plot_data = l3c[plot_field]*1.32e9
            else:
                plot_data = func(l3c[plot_field])
            pc = ax.pcolormesh(*F_center2edge(l3c['xgrid'],l3c['ygrid']),plot_data,**kwargs)
            if box_km is not None:
                xlim = [df_row['lng']-box_km/np.cos(np.deg2rad(df_row['lat']))/111,
                        df_row['lng']+box_km/np.cos(np.deg2rad(df_row['lat']))/111]
                ylim = [df_row['lat']-box_km/111,
                        df_row['lat']+box_km/111]
                ax.set_xlim(xlim);ax.set_ylim(ylim)
            ax.set_title('({}) {}'.format(chr(iax+97),df_row.city_ascii))
        if draw_colorbar:
            fig.colorbar(pc,ax=axs,shrink=0.3,label=r'NO$_x$ emission [nmol m$^{-2}$ s$^{-1}$]')
        return {'fig':fig,'axs':axs,'pc':pc}
    
    def map_city(self,subregion_name,figsize=None,
                 draw_colorbar=False,func=None,plot_field='wind_column_topo_chem',
                 nrow=4,ncol=4,box_km=None,emission_kw=None,raster_kw=None):
        '''combining map_city_emission and plot_city_raster'''
        figsize = figsize or (10,7.5)
        emission_kw = emission_kw or {}
        raster_kw = raster_kw or dict(cmap='Greys_r',alpha=0.1,zorder=100)
        if np.isscalar(box_km):
            box_km = np.ones(int(nrow*ncol))*box_km
        fig,axs = plt.subplots(nrow,ncol,figsize=figsize,constrained_layout=True)
        axs = axs.ravel()
        v = self[subregion_name]
        l3 = v['l3s'].aggregate()
        for iax,(ax,city_raster) in enumerate(zip(axs,v['city_raster_list'])):
            df_row = v['city_df'].iloc[iax]
            l3c = l3.trim(west=df_row.west,east=df_row.east,south=df_row.south,north=df_row.north)
            if func is None:
                plot_data = l3c[plot_field]*1.32e9
            else:
                plot_data = func(l3c[plot_field])
            pc = ax.pcolormesh(*F_center2edge(l3c['xgrid'],l3c['ygrid']),plot_data,**emission_kw)
            city_map = city_raster['data'].copy().astype(float)
            city_map[city_map==1] = np.nan
            pc_city = ax.pcolormesh(*F_center2edge(city_raster['xgrid'],city_raster['ygrid']),city_map,antialiased=True,
                                    **raster_kw)
            if box_km is None:
                pass
            else:
                xlim = [v['city_df'].iloc[iax]['lng']-box_km[iax]/np.cos(np.deg2rad(v['city_df'].iloc[iax]['lat']))/111,
                        v['city_df'].iloc[iax]['lng']+box_km[iax]/np.cos(np.deg2rad(v['city_df'].iloc[iax]['lat']))/111]
                ylim = [v['city_df'].iloc[iax]['lat']-box_km[iax]/111,
                        v['city_df'].iloc[iax]['lat']+box_km[iax]/111]
                ax.set_xlim(xlim);ax.set_ylim(ylim)
            ax.set_title('({}) {}'.format(chr(iax+97),df_row.city_ascii))
        if draw_colorbar:
            fig.colorbar(pc,ax=axs,shrink=0.3,label=r'NO$_x$ emission [nmol m$^{-2}$ s$^{-1}$]')
        return {'fig':fig,'axs':axs,'pc':pc,'pc_raster':pc_city}
    
    def plot_city_raster(self,subregion_name,nrow=4,ncol=4,box_km=None,**kwargs):
        '''plot the city area raster per subregion'''
        fig,axs = plt.subplots(nrow,ncol,figsize=(10,7.5),constrained_layout=True)
        axs = axs.ravel()
        v = self[subregion_name]
        if np.isscalar(box_km):
            box_km = np.ones(int(nrow*ncol))*box_km
        for iax,(ax,city_raster,city_name) in enumerate(zip(axs,v['city_raster_list'],v['city_df']['city_ascii'])):
            city_raster.plot(ax=ax,**kwargs)
            if box_km is None:
                pass
            else:
                xlim = [v['city_df'].iloc[iax]['lng']-box_km[iax]/np.cos(np.deg2rad(v['city_df'].iloc[iax]['lat']))/111,
                        v['city_df'].iloc[iax]['lng']+box_km[iax]/np.cos(np.deg2rad(v['city_df'].iloc[iax]['lat']))/111]
                ylim = [v['city_df'].iloc[iax]['lat']-box_km[iax]/111,
                        v['city_df'].iloc[iax]['lat']+box_km[iax]/111]
                ax.set_xlim(xlim);ax.set_ylim(ylim)
            ax.set_title('({}) {}'.format(chr(iax+97),city_name))
        return {'fig':fig,'axs':axs}
    
    def plot_AMS_pie(self,city_names,city_order=None,
                     figsize=None,nrow=5,ncol=6,GridSpec_kw=None,
                     plot_field=None,l3_path_pattern=None,period_range=None,
                     grid_size=None,west=None,east=None,south=None,north=None,pcolormesh_kw=None,
                     labels_city_name=None,size_func=None,min_D=2,colorbar_kw=None,pctdistance_df=None):
        '''plot Aerosol Mass Spectrometry-tyle pie/map
        city_names:
            names of cities to plot as pies
        city_order:
            order of cities around the border, range(len(city_names)) by default
        figsize:
            size of figure
        nrow,ncol:
            numbers of rows and columns input to GridSpec. the edge panels (2*nrow+2*ncol-4) host the pies.
            to plot 18 cities, nrow + ncol has to be at least 11
        GridSpec_kw:
            keyword arguments to GridSpec
        plot_field:
            Level3_Data field to pcolormesh on the central map. plot bare cartopy map if None
        period_range:
            time input to Level3_List
        l3_path_pattern:
            l3 file locations
        grid_size:
            block_reduce the l3 grid size if provided
        west/east/south/north:
            trim the region if provided
        pcolormesh_kw:
            keyword arguments to pcolormesh
        size_func:
            callable to project mean annual emission to pie radius. default is natural log
        lables_city_name:
            one city to label the wedges
        min_D:
            input to get_annual_df
        colorbar_kw:
            keyword arguments related to the colorbar. if not None, has to contain bounds,
            label, color, orientation
        pctdistance_df:
            if specified, should be a df of city_ascii and pctdistance to the corresponding pie
        '''
        if plot_field is not None:
            west = west or self.west
            east = east or self.east
            south = south or self.south
            north = north or self.north
            if not hasattr(self,'l3'):
                l3s = Level3_List(period_range,west=west,east=east,south=south,north=north)
                l3s.read_nc_pattern(l3_path_pattern=l3_path_pattern,
                                    fields_name=['column_amount','surface_altitude','wind_topo',
                                                 'wind_column','wind_column_xy','wind_column_rs'])
                l3s.fit_topography()
                l3s.fit_chemistry(resample_rule='month_of_year',return_resampled=False)
                l3 = l3s.aggregate()
            else:
                self.logger.warning('use existing self.l3')
                l3 = self.l3
            if grid_size is not None:
                l3 = l3.block_reduce(grid_size)
            self.l3 = l3
        
        default_pctdistance = 0.6
        figsize= figsize or (12,10)
        size_func = size_func or (lambda x:np.log(x)/2)
        GridSpec_kw = GridSpec_kw or dict(hspace=0.5, wspace=0.5)
        city_order = city_order or range(len(city_names))
        def pct_func(pct, allvals):
            absolute = (pct/100.*np.sum(allvals))
            return "{:.1f}".format(absolute)
        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(nrow, ncol, **GridSpec_kw)
        ax = fig.add_subplot(grid[1:-1,1:-1],projection=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='black', linewidth=1)
#         ax.add_feature(cfeature.RIVERS.with_scale('50m'), facecolor='None', edgecolor='blue', 
#                        linestyle='-.',zorder=0,lw=0.5)
#         ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='None',
#                        edgecolor='blue', zorder=0, linestyle='-',lw=0.5)
        cb = None;pc = None
        if plot_field is not None:
            if 'wind_column' in plot_field:
                plotdata = l3[plot_field]*1.32e9
            else:
                plotdata = l3[plot_field]
            pc = ax.pcolormesh(*F_center2edge(l3['xgrid'],l3['ygrid']),plotdata,
                               **pcolormesh_kw)
            if colorbar_kw is not None:
                cax = ax.inset_axes(colorbar_kw['bounds'])
                cb = fig.colorbar(pc,ax=ax,cax=cax,orientation=colorbar_kw['orientation'])
                cb.set_label(label=colorbar_kw['label'],size=10,color=colorbar_kw['color'])
                cb.outline.set_edgecolor(colorbar_kw['color'])
                cb.ax.yaxis.set_tick_params(color=colorbar_kw['color'], labelcolor=colorbar_kw['color'])
                cb.ax.xaxis.set_tick_params(color=colorbar_kw['color'], labelcolor=colorbar_kw['color'])
                cb.ax.tick_params(labelsize=10)
                cb.ax.tick_params(axis=u'both', which=u'both',length=0)
        
        ax.set_xlim([west,east])
        ax.set_ylim([south,north])
        city_pie_df = pd.DataFrame({'city_ascii':city_names,
                                   'order':city_order})
        pie_rows = np.hstack((np.arange(0,nrow),np.ones(ncol-1)*nrow-1,
                              np.arange(nrow-2,-1,-1),np.zeros(ncol-2))).astype(int)
        pie_cols = np.hstack((np.zeros(nrow),np.arange(1,ncol),
                              np.ones(nrow-1)*-1,np.arange(ncol-2,0,-1))).astype(int)
        city_pie_df['row'] = pie_rows[city_pie_df['order']]
        city_pie_df['col'] = pie_cols[city_pie_df['order']]
        if pctdistance_df is not None:
            city_pie_df = city_pie_df.merge(pctdistance_df,on='city_ascii',how='outer')
            city_pie_df['pctdistance'].fillna(default_pctdistance,inplace=True)
        for k,v in self.items():
            ax.add_patch(Rectangle((v['west'],v['south']),
                                  v['east']-v['west'],v['north']-v['south'],
                                 fc='none',color='r',linewidth=1,linestyle='--'))
            annual_df = self.get_annual_df(k,min_D=min_D)
            for irow,row in city_pie_df.iterrows():
                if not np.isin(row.city_ascii,v['city_df']['city_ascii']):continue
                ax_sub = fig.add_subplot(grid[row.row,row.col])
                ax_sub.set_aspect("equal")
                annual_nmol = annual_df[f'{row.city_ascii}_wind_column_topo_chem']*1.32e9
                total_nmol = np.nanmean(annual_nmol)
                size = size_func(total_nmol)
                if hasattr(row,'pctdistance'):
                    pctdistance = row.pctdistance
                else:
                    pctdistance = default_pctdistance
                if row['city_ascii'] == labels_city_name:
                    wedges, texts, autotexts = ax_sub.pie(annual_nmol,radius=size,startangle=90,counterclock=False,
                           wedgeprops={'alpha':0.95,'edgecolor':'none'},
                          autopct=lambda pct: pct_func(pct, annual_nmol),pctdistance=pctdistance,
                          textprops={'color':'k','fontsize':8},labels=range(2019,2023))

                else:
                    wedges, texts, autotexts = ax_sub.pie(annual_nmol,radius=size,startangle=90,counterclock=False,
                           wedgeprops={'alpha':0.95,'edgecolor':'none'},pctdistance=pctdistance,
                          autopct=lambda pct: pct_func(pct, annual_nmol),
                          textprops={'color':'k','fontsize':8})
                for w in wedges:
                    w.set_zorder(2)
                for t in texts:
                    t.set_color('k')
                city_row = v['city_df'].loc[v['city_df']['city_ascii']==row.city_ascii]
                con = ConnectionPatch(xyA=(0,0), xyB=(city_row.lng.squeeze(),city_row.lat.squeeze()), coordsA="data", 
                                      coordsB=ax.transData,
                                      axesA=ax_sub, axesB=ax,
                                      arrowstyle="->",zorder=1,color='k')
                ax_sub.add_artist(con)
                if row.row in [-1,nrow-1] and row.col not in [0,ncol-1,-1]:
                    ax_sub.set_title('{}, {:.1f}'.format(row.city_ascii,total_nmol),fontsize=10,y=-0.05)
                else:
                    ax_sub.set_title('{}, {:.1f}'.format(row.city_ascii,total_nmol),fontsize=10)
        return {'fig':fig,'ax':ax}
