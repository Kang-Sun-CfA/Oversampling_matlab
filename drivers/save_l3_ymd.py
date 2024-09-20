'''
Run as "python save_l3_ymd.py <path of config yml file>". 
This is the main driver calling the functions in popy.py 
(https://github.com/Kang-Sun-CfA/Oversampling_matlab/blob/master/popy.py). 
See regions defined within to extend to your regions of interest.
'''
import sys, os, glob
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.WARNING)
import yaml

control_txt_fn = sys.argv[1]
with open(control_txt_fn,'r') as stream:
    control = yaml.full_load(stream)
# https://github.com/Kang-Sun-CfA/Oversampling_matlab/blob/master/popy.py
sys.path.append(control['popy_dir'])
from popy import F_wrapper_l3, popy

product = control['product']
instrum = control['instrum']
# define spatial boundary
regions = str(control['region']).split('-')
region = regions[0]
if len(regions) > 1:
    subregion = regions[1]
else:
    subregion = region
if region.lower() == 'cn':
    west = 73
    east = 135
    south = 18
    north = 54
elif region.lower() == 'eu':
    west = -11
    east = 41
    south = 33
    north = 61
elif region.lower() == 'af':
    west = -18
    east = 52
    south = -35
    north = 38
    if subregion.lower() == 'westafrica':
        subwest=-18;subeast=25;subsouth=0;subnorth=15
elif region.lower() == 'conus':
    west = -128
    east = -65
    south = 24
    north = 50
    if subregion.lower() == 'cornbelt':
        subwest=-105;subeast=-80;subsouth=36;subnorth=50

if subregion == region:
    subwest=west;subeast=east;subsouth=south;subnorth=north

# define time
ps = pd.period_range(control['start'],control['end'],freq=control['freq'])
if control['if_standardize_year']:
    ps = ps[(ps.month>=ps[0].month)&(ps.month<=ps[-1].month)]

error_model = control.pop('error_model','ones')
algorithm = control.pop('algorithm',None)
oversampling_list = control.pop('oversampling_list',None) 
if_use_presaved_l2g = control.pop('if_use_presaved_l2g',True)
ncores = control.pop('ncores',None)
block_length=control.pop('block_length',300)
ellipse_lut_path = control.pop('ellipse_lut_path','/projects/academic/kangsun/data/IASIaNH3/daysss.mat')
# if those important inputs are not given, assign None and later product-specific values
for k in ['grid_size','flux_grid_size','met_path_pattern','l2_path_pattern',
          'l3_path_pattern','which_met','interp_fields','save_fields']:
    if k not in control.keys():
        logging.warning(f'{k} should be given. using product-specific defaults...')
        control[k] = None

subset_function = None    
# interpolate meteorology
interp_met_kw = {'which_met':control['which_met'] or 'ERA5',
               'met_dir':control['met_path_pattern'] or '/projects/academic/kangsun/data/ERA5/Y%Y/M%m/D%d/{}_2D_%Y%m%d.nc'.format(region.upper()),
               'interp_fields':control['interp_fields'] or ['u100','v100','u10','v10']}
if 'altitudes' in control.keys():
    interp_met_kw['altitudes'] = control['altitudes']
# instrum/product/algorithm specific inputs
if product == 'NO2':
    l2_path_pattern = control['l2_path_pattern'] or \
    '/projects/academic/kangsun/data/S5P{}/L2g_PAL/{}_%Y_%m.mat'.format(product,region.upper())
    save_fields = control['save_fields'] or ['column_amount','surface_altitude','wind_topo',\
                           'wind_column','wind_column_xy','wind_column_rs']
    l3_path_pattern = control['l3_path_pattern'] or \
    '/projects/academic/kangsun/data/S5P{}/L3/{}_%Y_%m_%d.nc'.format(product,subregion.upper())
    grid_size = control['grid_size'] or 0.02
    flux_grid_size = control['flux_grid_size'] or 0.04
    
    calculate_gradient_kw = {'write_diagnostic':True,'finite_difference_order':2,'albedo_orders':None}
    gradient_kw = {'x_wind_field':'era5_u100',
               'y_wind_field':'era5_v100',
               'x_wind_field_sfc':'era5_u10',
               'y_wind_field_sfc':'era5_v10',
               'unique_layer_identifier':'orbit',
               'func_to_get_vcd':None,
               'interp_met_kw':interp_met_kw,
               'calculate_gradient_kw':calculate_gradient_kw}
    if 'altitudes' in control.keys():
        gradient_kw['x_wind_field'] = 'era5_u{}'.format(control['altitudes'][0])
        gradient_kw['y_wind_field'] = 'era5_v{}'.format(control['altitudes'][0])
elif product == 'CH4':
    
    l2_path_pattern = control['l2_path_pattern'] or \
    '/projects/academic/kangsun/data/{}/L2/%Y%m/*%Y%m%d*.nc'.format(algorithm)
    save_fields = control['save_fields'] or ['wind_column','wind_column_xy','wind_column_rs',\
                                             'vcd','XCH4','surface_altitude','wind_topo',\
                                             'albedo','wind_albedo_0','wind_albedo_1',\
                                             'wind_albedo_2','wind_albedo_3','surface_pressure','pa']
    l3_path_pattern = control['l3_path_pattern'] or \
    '/projects/academic/kangsun/data/{}/L3/{}_%Y_%m_%d.nc'.format(algorithm,subregion.upper())
    grid_size = control['grid_size'] or 0.02
    flux_grid_size = control['flux_grid_size'] or 0.08

    calculate_gradient_kw = {'write_diagnostic':True,'finite_difference_order':2,'bc_kw':dict(keys=['albedo'],
                                     orders=[[0,1,2,3]])}
    
    if algorithm == 'WFMD':
        subset_function='F_subset_S5PCH4_WFMD'
        def F_l2g(l2g_data):   
            l2g_data['vcd'] = (l2g_data['surface_pressure']/9.8/0.02896-l2g_data['colh2o'])*l2g_data['XCH4']*1e-9
            l2g_data['pa'] = l2g_data['surface_pressure']-l2g_data['colh2o']*9.8*0.02896
            return l2g_data
    elif algorithm == 'SRON':
        calculate_gradient_kw = {'write_diagnostic':True,'finite_difference_order':2,
                                'bc_kw':dict(keys=['albedo','aerosol_size'],
                                     orders=[[0,1,2],[1,2,3]])}
        save_fields = control['save_fields'] or ['wind_column','wind_column_xy','wind_column_rs',\
                                                 'vcd','XCH4','surface_altitude','wind_topo',\
                                                 'albedo','aerosol_size','wind_albedo_0','wind_albedo_1',\
                                                 'wind_albedo_2','wind_albedo_aerosol_size_1',
                                                 'wind_aerosol_size_1','wind_aerosol_size_2','wind_aerosol_size_3',\
                                                 'surface_pressure','pa']
        subset_function='F_subset_S5PCH4_SRON'
#         def F_predict_sron_orbit(sron_datetime):
#             return int(np.round(np.polyval([ 1.41875305e+01, -1.04559459e+07],datetime2datenum(sron_datetime))))
        def F_l2g(l2g_data,min_qa_value=0.5,max_sza=65):
            from scipy.stats import binned_statistic
            xbins = np.arange(0.5,215.6)
            if 'across_track_position' not in l2g_data.keys():
                return l2g_data
            strip,_,_ = binned_statistic(l2g_data['across_track_position'],l2g_data['XCH4'],
                                     statistic=np.nanmedian,bins=xbins)
            strip -= np.nanmean(strip)
            strip[np.isnan(strip)] = 0
            l2g_data['XCH4'] -= np.interp(l2g_data['across_track_position'],range(1,216),strip)
            l2g_data['vcd'] = (l2g_data['surface_pressure']/9.8/0.02896-l2g_data['colh2o'])*l2g_data['XCH4']*1e-9
            l2g_data['pa'] = l2g_data['surface_pressure']-l2g_data['colh2o']*9.8*0.02896
            mask = (l2g_data['qa_value'] >= min_qa_value) & (l2g_data['SolarZenithAngle']<=max_sza)
            return {k:v[mask,] for k,v in l2g_data.items()} 
    elif algorithm.lower() in ['offl','offline','official','operational','s5pch4']:
        subset_function='F_subset_S5PCH4'
        def F_l2g(l2g_data,min_qa_value=0.5,max_sza=65):
            from scipy.stats import binned_statistic
            xbins = np.arange(0.5,215.6)
            if 'across_track_position' not in l2g_data.keys():
                return l2g_data
            strip,_,_ = binned_statistic(l2g_data['across_track_position'],l2g_data['XCH4'],
                                     statistic=np.nanmedian,bins=xbins)
            strip -= np.nanmean(strip)
            strip[np.isnan(strip)] = 0
            l2g_data['XCH4'] -= np.interp(l2g_data['across_track_position'],range(1,216),strip)
            l2g_data['vcd'] = (l2g_data['surface_pressure']/9.8/0.02896-l2g_data['colh2o'])*l2g_data['XCH4']*1e-9
            l2g_data['pa'] = l2g_data['surface_pressure']-l2g_data['colh2o']*9.8*0.02896
            mask = (l2g_data['qa_value'] >= min_qa_value) & (l2g_data['SolarZenithAngle']<=max_sza)
            return {k:v[mask,] for k,v in l2g_data.items()} 
    
    gradient_kw = {'x_wind_field':'era5_u100',
               'y_wind_field':'era5_v100',
               'x_wind_field_sfc':'era5_u10',
               'y_wind_field_sfc':'era5_v10',
               'unique_layer_identifier':'orbit',
               'func_to_get_vcd':F_l2g,
               'interp_met_kw':interp_met_kw,
               'calculate_gradient_kw':calculate_gradient_kw}
elif product == 'HCHO':
    l2_path_pattern = control['l2_path_pattern'] or \
    '/projects/academic/kangsun/data/S5P{}/L2g/{}_%Y_%m.mat'.format(product,region.upper())
    save_fields = control['save_fields'] or ['column_amount','surface_altitude','wind_topo',\
                           'wind_column','wind_column_xy','wind_column_rs']
    l3_path_pattern = control['l3_path_pattern'] or \
    '/projects/academic/kangsun/data/S5P{}/L3/{}_%Y_%m_%d.nc'.format(product,subregion.upper())
    grid_size = control['grid_size'] or 0.02
    flux_grid_size = control['flux_grid_size'] or 0.1
    
    calculate_gradient_kw = {'write_diagnostic':True,'finite_difference_order':2,'albedo_orders':None}
    gradient_kw = {'x_wind_field':'era5_u100',
               'y_wind_field':'era5_v100',
               'x_wind_field_sfc':'era5_u10',
               'y_wind_field_sfc':'era5_v10',
               'unique_layer_identifier':'orbit',
               'func_to_get_vcd':None,
               'interp_met_kw':interp_met_kw,
               'calculate_gradient_kw':calculate_gradient_kw}

elif product == 'NH3' and instrum == 'IASI':
    
    l2_path_pattern = control['l2_path_pattern'] or ['/projects/academic/kangsun/data/IASIaNH3/L2/IASI_METOPA_L2_NH3_%Y%m%d_ULB-LATMOS_V3R.1.0.nc',
                             '/projects/academic/kangsun/data/IASIbNH3/L2/IASI_METOPB_L2_NH3_%Y%m%d_ULB-LATMOS_V3R.1.0.nc']
    if np.isscalar(l2_path_pattern):
        l2_path_pattern = [l2_path_pattern]
    save_fields = control['save_fields'] or ['column_amount','wind_column',
                                             'wind_column_xy','wind_column_rs',
                                             'wind_topo','surface_altitude']
    l3_path_pattern = control['l3_path_pattern'] or \
    '/projects/academic/kangsun/data/IASIaNH3/L3/{}_%Y_%m_%d.nc'.format(subregion.upper())
    grid_size = control['grid_size'] or 0.05
    flux_grid_size = control['flux_grid_size'] or 0.25
    
    def F_l2g(l2g_data):
        l2g_data['day'] = np.floor(l2g_data['UTC_matlab_datenum']).astype(int)
        l2g_data['column_uncertainty'] = np.ones_like(l2g_data['column_amount'])
        return l2g_data
    calculate_gradient_kw = {'write_diagnostic':True,'finite_difference_order':2,'albedo_orders':None}
    gradient_kw = {'x_wind_field':'era5_u100',
               'y_wind_field':'era5_v100',
               'x_wind_field_sfc':'era5_u10',
               'y_wind_field_sfc':'era5_v10',
               'unique_layer_identifier':'day',
               'func_to_get_vcd':F_l2g,
               'interp_met_kw':interp_met_kw,
               'calculate_gradient_kw':calculate_gradient_kw}

if not os.path.exists(os.path.split(l3_path_pattern)[0]):
    logging.warning(f'creating {os.path.split(l3_path_pattern)[0]}')
    os.makedirs(os.path.split(l3_path_pattern)[0])

# loop over time intervals
for ip,p in enumerate(ps):
    if product == 'NH3':
        iasi_date = p.start_time
        iasi_date1 = p.end_time
        for ipath,l2_path in enumerate(l2_path_pattern):
            if ipath == 0:
                iasi0 = popy(instrum=instrum,product='NH3',
                             start_year=iasi_date.year,start_month=iasi_date.month,start_day=iasi_date.day,
                             start_hour=iasi_date.hour,start_minute=iasi_date.minute,start_second=iasi_date.second,
                             end_year=iasi_date1.year,end_month=iasi_date1.month,end_day=iasi_date1.day,
                             end_hour=iasi_date1.hour,end_minute=iasi_date1.minute,end_second=iasi_date1.second,
                             west=subwest,east=subeast,south=subsouth,north=subnorth,grid_size=grid_size,flux_grid_size=flux_grid_size,
                             inflatex=2,inflatey=2,error_model=error_model)
                if not if_use_presaved_l2g:
                    iasi0.F_subset_IASINH3(l2_path_pattern=l2_path,ellipse_lut_path=ellipse_lut_path)
                else:
                    l2g_fn = iasi_date.strftime(l2_path)
                    if not os.path.exists(l2g_fn):
                        logging.warning('{} does not exist!'.format(l2g_fn))
                        continue
                    iasi0.F_mat_reader(l2g_fn)
                if iasi0.nl2 == 0:
                    continue
            else:
                iasi = popy(instrum=instrum,product='NH3',
                             start_year=iasi_date.year,start_month=iasi_date.month,start_day=iasi_date.day,
                             start_hour=iasi_date.hour,start_minute=iasi_date.minute,start_second=iasi_date.second,
                             end_year=iasi_date1.year,end_month=iasi_date1.month,end_day=iasi_date1.day,
                             end_hour=iasi_date1.hour,end_minute=iasi_date1.minute,end_second=iasi_date1.second,
                             west=subwest,east=subeast,south=subsouth,north=subnorth,grid_size=grid_size,flux_grid_size=flux_grid_size,
                             inflatex=2,inflatey=2,error_model=error_model)
                if not if_use_presaved_l2g:
                    iasi.F_subset_IASINH3(l2_path_pattern=l2_path,ellipse_lut_path=ellipse_lut_path)
                else:
                    l2g_fn = iasi_date.strftime(l2_path)
                    if not os.path.exists(l2g_fn):
                        logging.warning('{} does not exist!'.format(l2g_fn))
                        continue
                    iasi.F_mat_reader(l2g_fn)
                if iasi.nl2 == 0:
                    continue
                iasi0.l2g_data = iasi0.F_merge_l2g_data(iasi0.l2g_data,iasi.l2g_data)
        if isinstance(iasi0.l2g_data,list):
            if np.sum([len(l['latc']) for l in iasi0.l2g_data]) == 0:
                logging.warning(iasi_date.strftime(l3_path_pattern)+' is empty')
                continue
        if isinstance(iasi0.l2g_data,dict):
            if len(iasi0.l2g_data['latc']) == 0:
                logging.warning(iasi_date.strftime(l3_path_pattern)+' is empty')
                continue
        iasi0.F_prepare_gradient(**gradient_kw)
        l3 = iasi0.F_parallel_regrid(ncores=ncores,block_length=block_length)
    else:
        try:
            l2_list = None
            l3 = F_wrapper_l3(instrum=instrum,product=product,grid_size=grid_size,
                             start_year=None,start_month=None,end_year=None,end_month=None,
                             start_day=None,end_day=None,
                             west=subwest,east=subeast,south=subsouth,north=subnorth,
                             column_unit=None,
                             if_use_presaved_l2g=if_use_presaved_l2g,
                             subset_function=subset_function,
                             l2_list=l2_list,
                             l2_path_pattern=l2_path_pattern,
                             if_plot_l3=False,existing_ax=None,
                             ncores=ncores,block_length=block_length,
                             subset_kw=None,plot_kw=None,
                             start_date_array=[p.start_time],
                             end_date_array=[p.end_time],
                             proj=None,nudge_grid_origin=None,inflatex=None,inflatey=None,
                             flux_kw=None,gradient_kw=gradient_kw,flux_grid_size=flux_grid_size,
                             error_model=error_model,oversampling_list=oversampling_list)
        except Exception as e:
            logging.warning(e)
            logging.warning(p.strftime('%Y%m%d seems to be empty!'))
            continue
    l3.save_nc(l3_filename=p.strftime(l3_path_pattern),
               fields_name=save_fields)
