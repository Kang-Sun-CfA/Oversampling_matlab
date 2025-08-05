import logging
try:
    import rasterio, pyproj
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    from rasterio.transform import from_origin
except Exception as e:
    logging.warning(e)
    logging.warning('CDL class may not work without these packages')
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import datetime as dt
import time
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys,os,glob
from popy import Level3_Data, F_center2edge, Level3_List

# code hosting classes for gridded, raster like datasets that interact with popy data

class CDL:
    '''class to handle crop data layer (cdl) data'''
    def __init__(self,filename,land_cover_codes_csv=None,shape_file=None):
        self.logger = logging.getLogger(__name__)
        self.filename = filename
        if land_cover_codes_csv:
            self.lcc_df = pd.read_csv(land_cover_codes_csv)
        else:
            self.lcc_df = None
        if shape_file:
            self.shape_file = shape_file
    
    def get_fractions(self,l3=None,xgrid=None,ygrid=None,block_length=None,if_mask=True):
        '''get fractions of land types on a l3 grid
        l3:
            a popy Level3_Data object
        x/ygrid:
            if l3 is not provided, use as lon/lat grids
        result attached to self:
            fractions:
                3D array, len(ygrid),len(xgrid),len(unique_code)
            df:
                dataframe listing codes in the domain, the average fraction
                of each code, and land cover name of each code (if lcc_df is there)
        '''
        if l3 is not None:
            xgrid,ygrid = l3['xgrid'],l3['ygrid']
        
        if not hasattr(self,'shape_file') and if_mask:
            self.logger.warning('no shape file provided in the beginning, no masking')
            if_mask = False
        
        if if_mask:
            mask = self.mask(l3=l3,shape_file=self.shape_file)
        
        nrows,ncols = len(ygrid),len(xgrid)
        block_length = block_length or max(nrows,ncols)
        # open tif file
        land_cover = rasterio.open(self.filename)
        cdl_crs = land_cover.crs
        wgs84 = pyproj.CRS('EPSG:4326')
        cdl_proj = pyproj.CRS(cdl_crs)
        # transformer from wgs84 to cdl' projection
        transformer = pyproj.Transformer.from_crs(wgs84,cdl_proj,always_xy=True)
        # loop over blocks
        iblock = 0
        block_codes = []
        block_fractions = []
        nblock_row = len(range(0,nrows,block_length))
        nblock_col = len(range(0,ncols,block_length))
        nblock = nblock_row*nblock_col
        self.logger.info(f'divide l3 grid to {nblock} blocks')
        for i_start in range(0,nrows,block_length):
            for j_start in range(0,ncols,block_length):
                iblock+=1
                i_end = min(i_start + block_length, nrows)
                j_end = min(j_start + block_length, ncols)
                ygrid_block = ygrid[i_start:i_end]
                xgrid_block = xgrid[j_start:j_end]
                # l3 grid corners to cdl's crs
                xmeshr_cdl,ymeshr_cdl = transformer.transform(
                    *np.meshgrid(
                    *F_center2edge(xgrid_block,ygrid_block)))
                min_x = max(land_cover.bounds.left,np.min(xmeshr_cdl))
                max_x = min(land_cover.bounds.right,np.max(xmeshr_cdl))
                min_y = max(land_cover.bounds.bottom,np.min(ymeshr_cdl))
                max_y = min(land_cover.bounds.top,np.max(ymeshr_cdl))

                window = land_cover.window(min_x,min_y,max_x,max_y
                                        ).round_offsets().round_shape()
                window_transform = land_cover.window_transform(window)
                img_data = land_cover.read(1,window=window,fill_value=0)
                self.logger.info('CDL block {} has shape {}x{}'.format(
                    iblock,img_data.shape[0],img_data.shape[1]))
                
                unique_codes = np.sort(np.unique(img_data))
                fractions = np.zeros((len(ygrid_block),len(xgrid_block),len(unique_codes)),
                                     dtype=np.float32)
                for i in range(len(ygrid_block)):
                    for j in range(len(xgrid_block)):
                        # to make sure fractions are calculated only within the mask (shapefile)
                        if if_mask:                        
                            i_global = i_start + i
                            j_global = j_start + j
                            if i_global >= mask.shape[0] or j_global >= mask.shape[1]:
                                continue
                            if not mask[i_global, j_global]:
                                continue

                        x_min = xmeshr_cdl[i,j]
                        x_max = xmeshr_cdl[i,j+1]
                        y_min = ymeshr_cdl[i+1,j]
                        y_max = ymeshr_cdl[i,j]

                        # Convert bounds to pixel coordinates
                        col_min, row_max = ~window_transform * (x_min, y_max)
                        col_max, row_min = ~window_transform * (x_max, y_min)

                        # Round to get integer pixel indices
                        col_min = int(np.floor(col_min))
                        col_max = int(np.ceil(col_max))
                        row_min = int(np.floor(row_min))
                        row_max = int(np.ceil(row_max))

                        # Clip to the actual data window
                        col_min = max(0, col_min)
                        col_max = min(img_data.shape[1], col_max)
                        row_min = max(0, row_min)
                        row_max = min(img_data.shape[0], row_max)

                        patch = img_data[row_min:row_max,col_min:col_max]

                        counts = np.bincount(patch.ravel(),
                                             minlength=unique_codes.max()+1
                                            )
                        total_pixels = patch.size
                        if total_pixels > 0:
                            fractions[i,j,:] = counts[unique_codes]/total_pixels
                block_codes.append(unique_codes)
                block_fractions.append(fractions)
        
        land_cover.close()
        if iblock != nblock:
            self.logger.error(f'iblock {iblock} does not equal nblock {nblock}!')
        if iblock == 1:
            df = pd.DataFrame(
                dict(
                    Code=unique_codes,
                    mean_frac=fractions.mean(axis=(0,1))))
            if self.lcc_df is not None:
                df = df.merge(self.lcc_df,on='Code')
            self.fractions = fractions
            self.df = df
            self.xgrid,self.ygrid = xgrid,ygrid
            return
        # assemble fractions from blocks to one piece
        all_codes = np.sort(np.unique(np.concatenate(block_codes)))
        blocks = []
        for codes,fractions in zip(block_codes,block_fractions):
            block = np.zeros(fractions.shape[:2]+(len(all_codes),),dtype=np.float32)
            index = np.searchsorted(all_codes,codes)
            block[:,:,index] = fractions
            # transpose land index to the first for np.block later
            blocks.append(block.transpose([2,0,1]))
        all_fractions = np.block([blocks[i:i+nblock_col] for i in range(0,nblock,nblock_col)]
                                ).transpose([1,2,0]) # transpose back
        
        self.fractions = all_fractions
        df = pd.DataFrame(
            dict(
                Code=all_codes,
                mean_frac=all_fractions.mean(axis=(0,1))))
        if self.lcc_df is not None:
            df = df.merge(self.lcc_df,on='Code')
        
        self.df = df
        self.xgrid,self.ygrid = xgrid,ygrid
        return
    
    def mask(self,l3=None,shape_file=None): # to mask a shape file
        if l3 is not None:
            xgrid,ygrid = l3['xgrid'],l3['ygrid']
        lon, lat = np.meshgrid(xgrid, ygrid)
        shp = gpd.read_file(shape_file)
        # Rasterize polygons onto the grid
        res_lon = np.diff(xgrid)[0]
        res_lat = np.diff(ygrid)[0]
        west = xgrid[0] - res_lon / 2
        north = ygrid[0] + res_lat / 2
        transform = from_origin(west, north, res_lon, -res_lat)
        mask = geometry_mask([mapping(geom) for geom in shp.geometry],
                             out_shape=lon.shape,
                             transform=transform,
                             invert=True)
        return mask
    
    def select(self,code=None,land_type=None):
        if land_type == 'dominant':
            indices = np.argmax(self.fractions, axis=-1)
            dominant = self.df['Code'].values[indices]
            return dominant
        if code is None:
            if not any(self.df['Type'].isin([land_type])):
                self.logger.error(f'{land_type} not available!')
                return
            code = self.df['Code'][self.df['Type']==land_type].squeeze()
        else:
            if not any(self.df['Code'].isin([code])):
                self.logger.error(f'{code} not available!')
                return
        data = self.fractions[...,self.df['Code']==code].squeeze()
        return data
    
    def plot(self,
        code=None,land_type=None,
        ax=None,scale='linear',**kwargs
            ):
        
        data = self.select(code,land_type)
        if data is None:return
        
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
            pc = ax.pcolormesh(*F_center2edge(self.xgrid,self.ygrid),data,norm=inputNorm,
                                         **kwargs)
        else:
            pc = ax.pcolormesh(*F_center2edge(self.xgrid,self.ygrid),data,**kwargs)
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None',
                       edgecolor='black', linewidth=1)
        if land_type is not None:
            ax.set_title(land_type)
        cb = fig.colorbar(pc,ax=ax)
        figout = {'fig':fig,'pc':pc,'ax':ax,'cb':cb}
        return figout
    
    def save_nc(self,l3_filename,
                fields_name=None,
                fields_rename=None,
                fields_comment=None,
                fields_unit=None,
                ncattr_dict=None):    
        fields_name = fields_name or []
        if fields_rename is None:
            fields_rename = fields_name.copy()
        if fields_comment is None:
            fields_comment = ['' for i in range(len(fields_name))]
        if fields_unit is None:
            fields_unit = ['' for i in range(len(fields_name))]
        if 'xgrid' not in fields_name:
            fields_name.append('xgrid')
            fields_rename.append('xgrid')
            fields_comment.append('horizontal grid')
            fields_unit.append('degree')
        if 'ygrid' not in fields_name:
            fields_name.append('ygrid')
            fields_rename.append('ygrid')
            fields_comment.append('vertical grid')
            fields_unit.append('degree')
        lonmesh,latmesh = np.meshgrid(self.xgrid,self.ygrid)
        self.lonmesh = lonmesh
        self.latmesh = latmesh
        if 'lonmesh' not in fields_name:
            fields_name.append('lonmesh')
            fields_rename.append('lonmesh')
            fields_comment.append('longitude mesh')
            fields_unit.append('degree_east')
        if 'latmesh' not in fields_name:
            fields_name.append('latmesh')
            fields_rename.append('latmesh')
            fields_comment.append('latitude mesh')
            fields_unit.append('degree_north')
        if 'fractions' not in fields_name:
            fields_name.append('fractions')
            fields_rename.append('fractions')
            fields_comment.append('land cover fractions in each satellite l3 grid cell')
            fields_unit.append('')
        if 'Code' not in fields_name:
            fields_name.append('Code')
            fields_rename.append('code')
            fields_comment.append('directory for land cover codes')
            fields_unit.append('')
        if 'Type' not in fields_name:
            fields_name.append('Type')
            fields_rename.append('type')
            fields_comment.append('directory for land cover types')
            fields_unit.append('')
        if 'mean_frac' not in fields_name:
            fields_name.append('mean_frac')
            fields_rename.append('mean_frac')
            fields_comment.append('averaged land cover fractions in the spatial domain')
            fields_unit.append('')
        nc = Dataset(l3_filename,mode='w',format='NETCDF4',clobber=True)
        if not ncattr_dict:
            ncattr_dict = {'description':'Land Cover Fractions based on CDL Data',
                           'institution':'University at Buffalo',
                           'contact':'Kang Sun, kangsun@buffalo.edu'}
        if 'history' not in ncattr_dict.keys():
            ncattr_dict['history'] = 'Created '+dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')#local time
        nc.setncatts(ncattr_dict)
        nc.createDimension('ygrid',self.fractions.shape[0])
        nc.createDimension('xgrid',self.fractions.shape[1])
        nc.createDimension('land_cover', self.fractions.shape[2])
        for (i,fn) in enumerate(fields_name):
            if fn in ['xgrid']:
                vid = nc.createVariable(fields_rename[i],np.float32,dimensions=('xgrid'))
            elif fn in ['ygrid']:
                vid = nc.createVariable(fields_rename[i],np.float32,dimensions=('ygrid'))
            elif fn in ['lonmesh','latmesh']:
                vid = nc.createVariable(fields_rename[i],np.float32,dimensions=('ygrid','xgrid'))              
            elif fn in ['fractions']:
                vid = nc.createVariable(fields_rename[i],np.float32,dimensions=('ygrid','xgrid','land_cover'))
            elif fn in ['Type']:
                vid = nc.createVariable(fields_rename[i],str,dimensions=('land_cover',))
            elif fn in ['Code']:
                vid = nc.createVariable(fields_rename[i],np.int32,dimensions=('land_cover'))
            elif fn in ['mean_frac']:
                vid = nc.createVariable(fields_rename[i],np.float32,dimensions=('land_cover'))
            # use standard_name to inform lat/lon vs x/y
            if fn == 'xgrid':
                vid.standard_name = 'projection_x_coordinate'
            elif fn == 'ygrid':
                vid.standard_name = 'projection_y_coordinate'
            elif fn == 'lonmesh':
                vid.standard_name = 'longitude'
            elif fn == 'latmesh':
                vid.standard_name = 'latitude'
            vid.comment = fields_comment[i]
            vid.units = fields_unit[i]
            if fn in ['xgrid', 'ygrid', 'lonmesh', 'latmesh', 'fractions']:
                vid[:] = np.ma.masked_invalid(np.float32(getattr(self, fn)))
            elif fn in ['Type']:
                vid[:] = self.df[fn].values.astype(str)
            elif fn in ['Code']:
                vid[:] = np.ma.masked_invalid(np.int32(getattr(self.df, fn)))
            elif fn in ['mean_frac']:
                vid[:] = np.ma.masked_invalid(np.float32(getattr(self.df, fn)))      
        nc.close()
        
    def read_nc(self,l3_filename,
                fields_name=None):
        fields_name = fields_name or []
        if 'xgrid' not in fields_name:
            fields_name.append('xgrid')
        if 'ygrid' not in fields_name:
            fields_name.append('ygrid')
        if 'lonmesh' not in fields_name:
            fields_name.append('lonmesh')
        if 'latmesh' not in fields_name:
            fields_name.append('latmesh')
        if 'fractions' not in fields_name:
            fields_name.append('fractions')
        if 'code' not in fields_name:
            fields_name.append('code')
        if 'type' not in fields_name:
            fields_name.append('type')
        if 'mean_frac' not in fields_name:
            fields_name.append('mean_frac')           
        nc = Dataset(l3_filename,'r')
        for (i,varname) in enumerate(fields_name):
            # the variable names are inconsistent with Level3_Data in CF-compatible nc files
            nc_varname = varname
            if varname == 'xgrid':
                if 'projection_x_coordinate' in nc.variables.keys():
                    nc_varname = 'projection_x_coordinate'
            if varname == 'ygrid':
                if 'projection_y_coordinate' in nc.variables.keys():
                    nc_varname = 'projection_y_coordinate'
            if varname == 'lonmesh':
                if 'longitude' in nc.variables.keys():
                    nc_varname = 'longitude'
            if varname == 'latmesh':
                if 'latitude' in nc.variables.keys():
                    nc_varname = 'latitude'
            try:
                setattr(self, varname, nc[nc_varname][:].filled(np.nan))
            except:
                self.logger.debug('{} cannot be filled by nan or is not a masked array'.format(nc_varname))
                setattr(self, varname, np.array(nc[nc_varname][:]))
        nc.close()
        return self

class Inventory(dict):
    '''class based on dict, representing a gridded emission inventory'''
    def __init__(self,name='inventory',west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.west = west
        self.east = east
        self.south = south
        self.north = north
    
    def read_GFEI(self,filename):
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
    
    def read_NEINOX(self,monthly_filenames=None,nei_dir=None,unit='mol/m2/s'):
        '''read a list of monthly NEI inventory files
        monthly_filenames:
            a list of file paths
        nei_dir:
            if provided, supersedes monthly_filenames
        unit:
            emission will be converted from kg/m2/s (double check) to this unit
        '''
        field = 'NOX'
        monthly_filenames = monthly_filenames or\
            glob.glob(os.path.join(nei_dir,'2016fh_16j_merge_0pt1degree_month_*.ncf'))
        mons = []
        for i,filename in enumerate(monthly_filenames):
            nc = Dataset(filename)
            if i == 0:
                xgrid = nc['lon'][:].data
                ygrid = nc['lat'][:].data
                xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
                ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
                self.xgrid_size = xgrid_size
                self.ygrid_size = ygrid_size
                if not np.isclose(xgrid_size,ygrid_size,rtol=1e-03):
                    self.logger.warning(f'x grid size {xgrid_size} does not equal to y grid size {ygrid_size}')
                self.grid_size = (xgrid_size+ygrid_size)/2
                xmask = (xgrid >= self.west) & (xgrid <= self.east)
                ymask = (ygrid >= self.south) & (ygrid <= self.north)
                self['xgrid'] = xgrid[xmask]
                self['ygrid'] = ygrid[ymask]
#                 self.west = self['xgrid'].min()-self.grid_size
#                 self.east = self['xgrid'].max()+self.grid_size
#                 self.south = self['ygrid'].min()-self.grid_size
#                 self.north = self['ygrid'].max()+self.grid_size
                xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
                monthly_fields = np.zeros((len(self['ygrid']),
                                           len(self['xgrid']),
                                           len(monthly_filenames)))
                self['grid_size_in_m2'] = np.cos(np.deg2rad(ymesh/180*np.pi))*np.square(self.grid_size*111e3)
                nc_unit = nc[field].units
                if nc_unit == 'kg/m2/s' and unit=='nmol/m2/s':
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1e9/0.046
                elif nc_unit == 'kg/m2/s' and unit=='mol/m2/s':
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1/0.046
                else:
                    self.logger.info('no unit conversion is done')
                    self[f'{field} unit'] = nc_unit
                    unit_factor = 1.
            monthly_fields[:,:,i] = unit_factor*nc[field][:].filled(np.nan)[0,0,:,:][np.ix_(ymask,xmask)]# time and lev are singular dimensions
            mons.append(dt.datetime(int(str(nc.SDATE)[0:4]),1,1)+dt.timedelta(days=-1+int(str(nc.SDATE)[-3:])))
            nc.close()
        self[field] = monthly_fields
        self['mons'] = pd.to_datetime(mons).to_period('1M')
        self['data'] = np.mean(monthly_fields, axis=2) # mean emission over months, named "data" to match basin_emissions.py
            
        return self
    
    def regrid_to_l3(self,l3=None,xgrid=None,ygrid=None,method=None):
        '''regrid inventory to match the mesh of a l3 data object
        method:
            if none, choose from drop_in_the_box and interpolate based on the relative grid size of inventory and l3
        '''
        if method is None:
            if self.grid_size < l3.grid_size/2:
                method = 'drop_in_the_box'
#             elif (self.grid_size >= l3.grid_size/2) and (self.grid_size < l3.grid_size*2):
#                 method = 'tessellate'
            else:
                method = 'interpolate'
            self.logger.warning(f'regridding from {self.grid_size} to {l3.grid_size} using {method}')
        
        if xgrid is None:
            xgrid = l3['xgrid']
        if ygrid is None:
            ygrid = l3['ygrid']
        ymesh,xmesh = np.meshgrid(ygrid,xgrid)
        if method in ['interpolate']:
            f = RegularGridInterpolator((self['ygrid'],self['xgrid']),self['data'],bounds_error=False)
            data = f((ymesh,xmesh)).T
        elif method in ['drop_in_the_box']:
            data = np.full((len(inv['ygrid']),len(inv['xgrid'])),np.nan)
            for iy,y in enumerate(inv['ygrid']):
                ymask = (self['ygrid']>=y-inv.grid_size/2) & (self['ygrid']<y+inv.grid_size/2)
                for ix,x in enumerate(inv['xgrid']):
                    xmask = (self['xgrid']>=x-inv.grid_size/2) & (self['xgrid']<x+inv.grid_size/2)
                    if np.sum(ymask) == 0 and np.sum(xmask) == 0:
                        continue
                    data[iy,ix] = np.nanmean(self['data'][np.ix_(ymask,xmask)])
        
        return data
    
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
