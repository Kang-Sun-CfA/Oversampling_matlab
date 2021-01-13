# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:24:03 2020

@author: kangsun
"""

'''# use this section to generate a control txt file
import yaml
control = {}
control['which air basin'] = 'po'
control['which molecule'] = 'NO2'
control['which sensor'] = 'OMI'
control['level 2g directory'] = '/mnt/Data2/OMNO2/L2g'
control['level 2g file header'] = 'EU'
control['popy directory'] = '/home/kangsun/OMI/Oversampling_matlab'
control['output directory'] = '/home/kangsun/RRNES/omno2_era5'
control['auxiliary directory'] = '/home/kangsun/RRNES/auxiliary_data'
control['x wind'] = 'era5_u100'
control['y wind'] = 'era5_v100'
control['start year'] = 2020
control['start month'] = 5
control['end year'] = 2020
control['end month'] = 5
control['maximal cloud fraction'] = 0.3
control['maximal solar zenith angle'] = 70
control['minimal qa_value'] = 0.5
control['if exclude row anomaly'] = True
control['oversampling list'] = ['dx','x','column_amount','ws','sp']
control['how many cores'] = 10
control['if verbose'] = True
with open('/home/kangsun/RRNES/control.txt', 'w') as stream:
    yaml.dump(control, stream,sort_keys=False)
'''
#%%
import sys
control_path = '/home/kangsun/RRNES/control.txt'
if __name__ == "__main__":
    control_path = str(sys.argv[1])
import yaml
with open(control_path,'r') as stream:
    control = yaml.full_load(stream)
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
sys.path.append(control['popy directory'])
from popy import popy, F_collocate_l2g
if 'if verbose' not in control.keys(): control['if verbose']=False
if control['if verbose']:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
from scipy.io import loadmat
from calendar import monthrange
from matplotlib import path
from scipy.stats import trim_mean
import h5py

def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, int, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

if control['which molecule'] == 'CO':
    molecular_weight = 0.028
elif control['which molecule'] == 'NO2':
    molecular_weight = 0.046
elif control['which molecule'] == 'NH3':
    molecular_weight = 0.017
ws_bin = np.append(np.linspace(0,20,41),100)
nbin = len(ws_bin)-1
ncores = control['how many cores']
if control['which sensor'] == 'TROPOMI':
    block_length = 200
    grid_size = 0.01
elif control['which sensor'] == 'OMI':
    block_length = 100
    grid_size = 0.05
nv = len(control['oversampling list'])

if 'if exclude fire AI' in control.keys():
    if_ai = control['if exclude fire AI']
else:
    if_ai = False

if not os.path.exists(control['output directory']):
    os.makedirs(control['output directory'])
basin_boundary_fn = os.path.join(control['auxiliary directory'],
                                 control['which air basin']+'_boundary.mat')
basin_boundary = loadmat(basin_boundary_fn)
basin_boundary.pop('__globals__')
basin_boundary.pop('__header__')
basin_boundary.pop('__version__')
basin_polygon = path.Path([(basin_boundary['b1x'].squeeze()[i],
                          basin_boundary['b1y'].squeeze()[i]) 
                        for i in range(len(basin_boundary['b1x'].squeeze()))])
bg_polygon = path.Path([(basin_boundary['b3x'].squeeze()[i],
                          basin_boundary['b3y'].squeeze()[i]) 
                        for i in range(len(basin_boundary['b3x'].squeeze()))])

start_year = int(control['start year'])
end_year = int(control['end year'])
start_month = int(control['start month'])
end_month = int(control['end month'])
for year in range(start_year,end_year+1):
    for month in range(1,13):
        if year == start_year and month < start_month:
            continue
        elif year == end_year and month > end_month:
            continue
        b_struct = {}
        b_struct[control['which air basin']] = {k:v.squeeze() for (k,v) in basin_boundary.items()}
        b_struct[control['which air basin']][control['which molecule']] = {}
        # daily vectors (may have missing days)
        b_struct[control['which air basin']][control['which molecule']]['utc_vec'] = np.array([])
        b_struct[control['which air basin']][control['which molecule']]['basin_c_vec'] = np.array([])
        b_struct[control['which air basin']][control['which molecule']]['basin_x_vec'] = np.array([])
        b_struct[control['which air basin']][control['which molecule']]['bg_x_vec'] = np.array([])
        b_struct[control['which air basin']][control['which molecule']]['basin_n_vec'] = np.array([])
        p = popy(instrum=control['which sensor'],
                 product=control['which molecule'],
                 west=basin_boundary['minlon3'].squeeze(),
                 east=basin_boundary['maxlon3'].squeeze(),
                 south=basin_boundary['minlat3'].squeeze(),
                 north=basin_boundary['maxlat3'].squeeze(),
                 start_year=year,start_month=month,start_day=1,
                 end_year=year,end_month=month,end_day=monthrange(year,month)[-1],
                 grid_size=grid_size,verbose=control['if verbose'])
        if if_ai:
            ai = popy(instrum=control['which sensor'],
                      product='AI',
                      west=basin_boundary['minlon3'].squeeze(),
                      east=basin_boundary['maxlon3'].squeeze(),
                      south=basin_boundary['minlat3'].squeeze(),
                      north=basin_boundary['maxlat3'].squeeze(),
                      start_year=year,start_month=month,start_day=1,
                      end_year=year,end_month=month,end_day=monthrange(year,month)[-1],
                      grid_size=grid_size,verbose=control['if verbose'])
            ai.F_mat_reader(os.path.join(control['AI level 2g directory'],
                                    control['AI level 2g file header']
                                    +'_%04d'%year+'_%02d'%month+'.mat'))
        if 'basin_grid_mask' not in locals():
            grid_points = np.hstack((p.xmesh.flatten()[:,np.newaxis],p.ymesh.flatten()[:,np.newaxis]))
            basin_grid_mask = basin_polygon.contains_points(grid_points).reshape(p.xmesh.shape)
        p.maxcf = control['maximal cloud fraction']
        p.maxsza = control['maximal solar zenith angle']
        p.min_qa_value = control['minimal qa_value']
        p.F_mat_reader(os.path.join(control['level 2g directory'],
                                    control['level 2g file header']
                                    +'_%04d'%year+'_%02d'%month+'.mat'))
        l2g_data = p.l2g_data
        # keep only rows 5-23 for OMI if excluding row anomaly
        if control['which sensor'] == 'OMI' and control['if exclude row anomaly']:
            mask = np.isin(l2g_data['across_track_position'],np.arange(5,24))
            l2g_data = {k:v[mask,] for (k,v) in l2g_data.items()}
        # additional filter for CO
        if control['which molecule'] == 'CO':
            mask = (l2g_data['scattering_height'] < 2000.) &\
            (l2g_data['scattering_OD'] < 1.)
            l2g_data = {k:v[mask,] for (k,v) in l2g_data.items()}
        # standardize column ammount to mol/m2
        if control['which sensor'] != 'TROPOMI':
            l2g_data['column_amount'] = l2g_data['column_amount']/6.02214e19
        if 'ws' in control['oversampling list']:
            l2g_data['ws'] = np.sqrt(np.power(l2g_data[control['x wind']],2)
            +np.power(l2g_data[control['y wind']],2))
        if 'sp' in control['oversampling list']:
            if 'surface_pressure' in l2g_data.keys():
                l2g_data['sp'] = l2g_data['surface_pressure']
            else:
                l2g_data['sp'] = l2g_data['era5_sp']
            if np.nanmedian(l2g_data['sp']) < 1e3:
                l2g_data['sp'] = l2g_data['sp']*100
        # x is mixing ratio
        if 'dx' in control['oversampling list'] or 'x' in control['oversampling list']:
            if control['which molecule'] == 'NO2':
                if 'merra2_TROPPT' in l2g_data.keys():
                    dp = l2g_data['sp']-l2g_data['merra2_TROPPT']
                else:
                    print('NO Tropopause data!!!')
                    dp = l2g_data['sp']-150e2
            else:
                dp = l2g_data['sp']
            if control['which molecule'] == 'CO':
                l2g_data['x'] = l2g_data['column_amount']/(dp/9.8/0.029-l2g_data['colh2o'])
            else:
                l2g_data['x'] = l2g_data['column_amount']/dp*9.8*0.029
        if 'dx' in control['oversampling list']:
            l2g_data['dx'] = np.zeros(l2g_data['x'].shape)
        # remove fire pixels if aerosol index data are to be used
        if if_ai:
            mask = ai.l2g_data['AI'] > control['fire AI threshold']
            fire_l2g = {k:v[mask,] for (k,v) in ai.l2g_data.items()}
            # keep only pixels that don't overlap with fire AI pixels
            _,l2g_data = F_collocate_l2g(l2g_data1=l2g_data,
                                         l2g_data2=fire_l2g,
                                         hour_difference=1,field_to_average='AI')
        # this loop is like chicken rib now
        for iday in range(int(np.floor(l2g_data['UTC_matlab_datenum'].min())),
                          int(np.ceil(l2g_data['UTC_matlab_datenum'].max()))):
            mask1 = np.floor(l2g_data['UTC_matlab_datenum']) == np.float(iday)
            all_points = np.hstack((l2g_data['lonc'][:,np.newaxis],l2g_data['latc'][:,np.newaxis]))
            mask_bg = mask1 & (~basin_polygon.contains_points(all_points))\
            & (bg_polygon.contains_points(all_points))
            mask_basin = mask1 & (basin_polygon.contains_points(all_points))
            mask_all = mask1 & (bg_polygon.contains_points(all_points))
            if np.sum(mask_basin) == 0:
                continue
            b_struct[control['which air basin']][control['which molecule']]['utc_vec'] = \
            np.append(b_struct[control['which air basin']][control['which molecule']]['utc_vec'],iday)
            
            b_struct[control['which air basin']][control['which molecule']]['basin_n_vec'] = \
            np.append(b_struct[control['which air basin']][control['which molecule']]['basin_n_vec'],np.sum(mask_basin))
            
            basin_c_mean = np.nanmean(l2g_data['column_amount'][mask_basin])
            b_struct[control['which air basin']][control['which molecule']]['basin_c_vec'] = \
            np.append(b_struct[control['which air basin']][control['which molecule']]['basin_c_vec'],basin_c_mean)
            if 'dx' in control['oversampling list']:
                l2g_data['dx'][mask1] = l2g_data['x'][mask1]-trim_mean(l2g_data['x'][mask_bg],0.2)
            if 'dx' in control['oversampling list'] or 'x' in control['oversampling list']:
                basin_x_mean = np.nanmean(l2g_data['x'][mask_basin])
                b_struct[control['which air basin']][control['which molecule']]['basin_x_vec'] = \
                np.append(b_struct[control['which air basin']][control['which molecule']]['basin_x_vec'],basin_x_mean)
                
                bg_x_mean = trim_mean(l2g_data['x'][mask_bg],0.2)
                b_struct[control['which air basin']][control['which molecule']]['bg_x_vec'] = \
                np.append(b_struct[control['which air basin']][control['which molecule']]['bg_x_vec'],bg_x_mean)
        
        p.oversampling_list = control['oversampling list']
        l3_month = p.F_parallel_regrid(l2g_data=l2g_data,ncores=ncores,block_length=block_length)
        A_month = np.array([np.nansum(l3_month[key][basin_grid_mask]*l3_month['total_sample_weight'][basin_grid_mask]) for key in p.oversampling_list_final])
        B_month = np.nansum(l3_month['total_sample_weight'][basin_grid_mask])
        D_month = np.nanmean(l3_month['num_samples'][basin_grid_mask])
        
        b_struct[control['which air basin']][control['which molecule']]['year_vec'] = np.float64(year)
        b_struct[control['which air basin']][control['which molecule']]['month_vec'] = np.float64(month)
        
        b_struct[control['which air basin']][control['which molecule']]['B_vec'] = B_month
        b_struct[control['which air basin']][control['which molecule']]['D_vec'] = D_month
        b_struct[control['which air basin']][control['which molecule']]['A_vec'] = A_month
        
        ime_C = np.full(nbin,np.nan)
        ime_D = np.full(nbin,np.nan)
        ime_B = np.full(nbin,np.nan)
        ime_dx = np.full(nbin,np.nan)
        ime_sp = np.full(nbin,np.nan)
        ime_ws = np.full(nbin,np.nan)
        for ibin in range(nbin):
            mask = (l2g_data['ws'] >= ws_bin[ibin]) & (l2g_data['ws'] <= ws_bin[ibin+1])
            l2g_bin = {k:v[mask,] for (k,v) in l2g_data.items()}
            l3_bin = p.F_parallel_regrid(l2g_data=l2g_bin,ncores=ncores,block_length=block_length)
            if not l3_bin:
                continue
            ime_B[ibin] = np.nansum(l3_bin['total_sample_weight'][basin_grid_mask])
            ime_D[ibin] = np.nanmean(l3_bin['num_samples'][basin_grid_mask])
            ime_C[ibin] = np.nansum(l3_bin['column_amount'][basin_grid_mask]*l3_bin['total_sample_weight'][basin_grid_mask])/ime_B[ibin]
            ime_dx[ibin] = np.nansum(l3_bin['dx'][basin_grid_mask]*l3_bin['total_sample_weight'][basin_grid_mask])/ime_B[ibin]
            ime_sp[ibin] = np.nansum(l3_bin['sp'][basin_grid_mask]*l3_bin['total_sample_weight'][basin_grid_mask])/ime_B[ibin]
            ime_ws[ibin] = np.nansum(l3_bin['ws'][basin_grid_mask]*l3_bin['total_sample_weight'][basin_grid_mask])/ime_B[ibin]
        b_struct[control['which air basin']][control['which molecule']]['ime_C'] = ime_C
        b_struct[control['which air basin']][control['which molecule']]['ime_B'] = ime_B
        b_struct[control['which air basin']][control['which molecule']]['ime_D'] = ime_D
        b_struct[control['which air basin']][control['which molecule']]['ime_dx'] = ime_dx
        b_struct[control['which air basin']][control['which molecule']]['ime_sp'] = ime_sp
        b_struct[control['which air basin']][control['which molecule']]['ime_ws'] = ime_ws
        
        hdf5_fn = os.path.join(control['output directory'],
                               control['which sensor']+'_%04d'%year+'%02d'%month+'.h5')
        save_dict_to_hdf5(b_struct,hdf5_fn)
