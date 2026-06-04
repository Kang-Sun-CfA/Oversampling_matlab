import numpy as np
import pandas as pd
import time
import sys,os,glob
import logging
from popy import arange_
from CAREER.gridded import CDL, BUI
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import yaml
import pickle
from functools import partial


DEFAULT_CONFIG_DICT = {
    'experiment':{
        'run_id':'cv0',
        'run_dir': '/projects/academic/kangsun/kangsun/IDS/v04/cv0',
        'mkdir':True,
        'seed': 42,
        'interactive':False
    },
    'data':{
        'l3s':{
            'path_pattern': '/projects/academic/kangsun/zolalaya/data/Cornbelt_NOx/'\
            's5p_cornbelt/%Y/%m/CORNBELT_S5P_NO2_%Y_%m_%d.nc',
            'ranges':[{'start':'2020-01-01','end':'2025-12-31','freq':'1D'}],
            'var_names':['column_amount_DD','column_amount','surface_altitude_DD'],
            'var_scales':[1e9,-1e4,-1e6]# very important, vcd and wt need flipping
        },
        'bui':{
            'enabled':True,
            'which':'CAMS',
            'path_pattern':'/projects/academic/kangsun/data/Inventory/'\
            'CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v6.2_monthly/'\
            'CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v6.2_monthly_%Y.nc',
            'range':{'start':'2020-01','end':'2025-12','freq':'1M'},
            'gaussian_sigma':0.5,
            'resample':None,
            'max_threshold':1e-9,
            'time_matching_method':'nearest',
            'yield_inventory':False
        },
        'cdl':{
            'enabled':True,
            'path_pattern':'/projects/academic/kangsun/data/GIS_data/'\
            'CDL/fractions/CDL_*_%Y.nc',
            'lcc_path':'/projects/academic/kangsun/data/GIS_data/CDL/fractions/lcc.csv',
            'years':[2020,2021,2022,2023,2024,2025],
            'name_codes':[
                ['corn',[1]],['soybean',[5]],
                ['dev_o',[121]],['dev_l',[122]],
                ['dev_m',[123]],['dev_h',[124]],
                ['wetland',[190,195]],['forest',[141,143,142]]
            ],
            'time_matching_method':'year'
        },
        'nfold':1,'val_interval':'1M','val_fraction':0.2,
        'train_idxs':[0],
        'train':[
            {
                'west':-104,'east':-80.5,'south':36,'north':46,
                'grid_size':0.1,
                'crop_x':64,'crop_y':64,'stride_x':8,'stride_y':8,
                'do_group':True,'ngroup_x':7,'ngroup_y':2,
                'randomize_start_xy':True,
                'intervals':['91d','91d','91d'],
                'offsets':['0d','30d','60d'],
                'fraction_milestone':[0,1,100,1],'scheduler':'linear'
            }
        ],
        'validation':[
            {
                'west':-104,'east':-80.5,'south':36,'north':46,
                'grid_size':0.1,
                'crop_x':64,'crop_y':64,'stride_x':32,'stride_y':32,
                'intervals':['1M'],
                'offsets':['0d']
            },
            {
                'west':-104,'east':-95,'south':38,'north':46,
                'grid_size':0.1,
                'crop_x':64,'crop_y':64,'stride_x':32,'stride_y':32,
                'intervals':['1M'],
                'offsets':['0d']
            },
            {
                'west':-90,'east':-81,'south':37,'north':45,
                'grid_size':0.1,
                'crop_x':64,'crop_y':64,'stride_x':32,'stride_y':32,
                'intervals':['1M'],
                'offsets':['0d']
            }
        ],
        'test':{
            'west':-104,'east':-80.5,'south':36,'north':46,'grid_size':0.1,
            'start':'2020-1-1','end':'2025-12-31','freq':'1M',
            'crop_x':64,'crop_y':64,'stride_x':8,'stride_y':8,'trim':4
        }
    },
    'model':{
        'pretrained_model_path': None,
        'spatial':{
            'unet_depth':4,'base_channels':64,'leaky_relu_rate':0.1,'norm_groups':8,
            'force_positive':True,'in_channels':2,'out_channels':None
        },
        'temporal':{
            'temporal_dim':3,'hidden_dim':128,'leaky_relu_rate':0.1,'modulate_decoder':False
        },
        'psf':{'enabled':False,'kernel_size':5},
        'yname':'column_amount_DD',
        'xnames':['column_amount','surface_altitude_DD']
    },
    'loss':{
        'main_loss': 'L1Loss',
        'use_fit_mask': True,
        'smoothness_loss':{
            'enabled':True,
            'weight_milestone':[0,0,50,1.5e-2],
            'B_weight_milestone':[0,0,50,0.3],
            'scheduler':'linear',
            'scale_weights': [1, 0.5, 0.5, 0.25],
            'channel_weights': [2,1]
        }
    },
    'training':{
        'start_epoch':0,'end_epoch':60,
        'batch_size':256,'shuffle_level':'sample',
        'lr':{
            'milestone':[0,1e-4,50,1e-4],'scheduler':'linear',
            'multipliers':{
                'other':1,'temporal_film':0.4,'psf':20
            }
        },
        'weight_decay':{
            'initial':1e-2,
            'multipliers':{
                'other':1,'temporal_film':0,'psf':0
            }
        },
        'gradient_clipping': {'max_norm': 1.0},
        'target_clampping':{'min':None,'max':3.}
    },
    'hp_tuning':{
        'enabled':False,
        'hps':[
            'loss.smoothness_loss.weight_milestone',
            'loss.smoothness_loss.B_weight_milestone'
        ]
    },
    'saving':{
        'best_model':{
            'enabled':False,
            'criteria':{
                'name':'val_loss0',
                'mode':'min',
                'rolling_window':3
            }
        },
        'final_model':{
            'enabled':False
        },
        'unet_out':{
            'enabled':True,
            'plot_channels':[0],
            'save_data':False
        }
    }
}
  

class ConfigNode(dict):
    """A dictionary subclass that allows attribute access to nested keys."""
    def __init__(self, data=None):
        if data is None:
            data = {}
        super().__init__()
        for key, value in data.items():
            self[key] = self._wrap(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigNode' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, dict):
            return ConfigNode(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        else:
            return value

    def to_dict(self):
        """Convert the ConfigNode and its children back to a standard dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, ConfigNode) else v for v in value]
            else:
                result[key] = value
        return result

class CEConfig:
    def __init__(self, config_dict=None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default structure if no dict provided
        if config_dict is None:
            config_dict = self._get_default_config()
        
        self._config = ConfigNode(config_dict)
        self.check()
    
    @staticmethod
    def _get_default_config():
        """Defines the default configuration structure and values."""
        return DEFAULT_CONFIG_DICT
    
    def __getattr__(self, item):
        """Allow top-level attribute access to sections (e.g., config.experiment)."""
        return getattr(self._config, item)
    
    def __getitem__(self, item):
        """Maintain dictionary-like access for backward compatibility or ease of use."""
        return self._config[item]

    def to_yaml(self, rel_path='config.yml', abs_path=None):
        """Save the configuration to a YAML file."""
        path = abs_path or os.path.join(self._config.experiment.run_dir, rel_path)
        with open(path, 'w') as stream:
            yaml.dump(self._config.to_dict(), stream, sort_keys=False)
        self.logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_yaml(cls, path):
        """Load a configuration from a YAML file."""
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)
        return cls(data)

    def update(self, new_data):
        """Recursively update the configuration with new data."""
        def _recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v
        
        current_dict = self._config.to_dict()
        _recursive_update(current_dict, new_data)
        self._config = ConfigNode(current_dict)
        self.check()

    def get_nested(self, path):
        """Get a value using a dot-notated path (e.g., 'training.learning_rate.initial')."""
        parts = path.split('.')
        val = self._config
        for part in parts:
            if isinstance(val, dict):
                val = val[part]
            else:
                raise KeyError(f"Path '{path}' not found at '{part}'")
        return val

    def set_nested(self, path, value):
        """Set a value using a dot-notated path."""
        parts = path.split('.')
        data = self._config.to_dict()
        curr = data
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value
        self._config = ConfigNode(data)
#         self.check()

    def get_hp_combinations(self):
        """
        Generates a list of config objects for each hyperparameter combination.
        """
        hp_names = self._config.hp_tuning.hps
        if not hp_names:
            return [self]
        
        # Get the lists of values for each HP
        hp_values_lists = [self.get_nested(name) for name in hp_names]
        
        # Check if all lists have the same length
        n_combinations = len(hp_values_lists[0])
        if not all(len(l) == n_combinations for l in hp_values_lists):
            raise ValueError("All hyperparameter lists must have the same length!")
            
        configs = []
        for i in range(n_combinations):
            # Create a deep copy of current config
            new_config = CEConfig(self._config.to_dict())
            new_config.hp_tuning.enabled = False
            # Update the nested values to be single values instead of lists
            for name, values in zip(hp_names, hp_values_lists):
                new_config.set_nested(name, values[i])
            configs.append(new_config)
            
        return configs

    def check(self):
        """Validate the configuration parameters and structure."""
        exp = self._config.experiment
        data = self._config.data
        hp_tuning = self._config.hp_tuning
        # Ensure run directory exists
        if not os.path.exists(exp.run_dir) and exp.mkdir:
            self.logger.warning(f'{exp.run_dir} does not exist, trying to create one')
            try:
                os.makedirs(exp.run_dir, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to create run_dir: {e}")
        # Cross-validation check
        if data.nfold > 1 and data.val_fraction > 0:
            self.logger.warning('val fraction {} will be replaced by 1/nfold'.format(data.val_fraction))
        # hp lists have to be the same length
        if hp_tuning.enabled:
            hp_values_lists = [self.get_nested(name) for name in hp_tuning.hps]
            if not all([hasattr(l,'__len__') for l in hp_values_lists]):
                self.logger.error('hps must be all lists!')
            else:
                n_combinations = len(hp_values_lists[0])
                if not all(len(l) == n_combinations for l in hp_values_lists):
                    self.logger.error("All hp lists must have the same length!")

    def sbatch_ccr(
        self,slurm_path=None,config_path=None,python_script_path=None,
        git_path=None,module_list=None,submit=True,**kwargs
    ):
        run_dir = self._config.experiment.run_dir
        run_id = self._config.experiment.run_id
        if slurm_path is None:
            slurm_path = os.path.join(run_dir,'run_job.sh')
        if config_path is None:
            config_path = os.path.join(run_dir,'config.yml')
            self.logger.warning(f'saving current config dict to {config_path}')
            self.to_yaml(abs_path=config_path)
        if python_script_path is None:
            python_script_path = os.path.join(os.path.split(run_dir)[0],'train_cefit.py')
            self.logger.warning(f'assuming python script at {python_script_path}')
        if git_path is None:
            git_path = os.path.join(os.path.split(python_script_path)[0],'..')
            self.logger.warning(f'assuming git path at {git_path}')
        if module_list is None:
            module_list = [
                'gcc/11.2.0','openmpi/4.1.1', 'pytorch/1.13.1-CUDA-11.8.0', 
                'scipy-bundle/2021.10', 'matplotlib/3.4.3', 
                'netcdf4-python/1.5.7', 'cartopy', 'statsmodels', 'scikit-image'
            ]
        with open(slurm_path,'w') as fid:
            fid.write('#!/bin/bash')
            fid.write('\n')
            account = kwargs.pop('account','kangsun')
            fid.write('#SBATCH --account={}'.format(account))
            fid.write('\n')
            fid.write('#SBATCH --partition={}'.format(kwargs.pop('partition',account)))
            fid.write('\n')
            fid.write('#SBATCH --qos={}'.format(kwargs.pop('qos',account)))
            fid.write('\n')
            fid.write('#SBATCH --cluster={}'.format(kwargs.pop('cluster','faculty')))
            fid.write('\n')
            fid.write('#SBATCH --time={:.0f}:00:00'.format(kwargs.pop('time',1)))
            fid.write('\n')
            fid.write('#SBATCH --nodes={:.0f}'.format(kwargs.pop('nodes',1)))
            fid.write('\n')
            fid.write('#SBATCH --ntasks-per-node={:.0f}'.format(kwargs.pop('ntasks',3)))
            fid.write('\n')
            fid.write('#SBATCH --mem={:.0f}'.format(kwargs.pop('mem',80000)))
            fid.write('\n')
            fid.write('#SBATCH --gpus-per-node={:.0f}'.format(kwargs.pop('gpus',1)))
            fid.write('\n')
            fid.write('#SBATCH --job-name={}'.format(run_id))
            fid.write('\n')
            fid.write('#SBATCH --output={}.out'.format(run_id))
            fid.write('\n')
            if 'email' in kwargs.keys():
                fid.write('#SBATCH --mail-user={}'.format(kwargs['email']))
                fid.write('\n')
                fid.write('#SBATCH --mail-type=ALL')
                fid.write('\n')
            fid.write('#SBATCH --requeue')
            fid.write('\n')
            fid.write('ulimit -s unlimited')
            fid.write('\n')
            fid.write('module load '+' '.join(module_list))
            fid.write('\n')
            fid.write(f'python3 {python_script_path} {config_path} {git_path}')
            fid.write('\n')
        
        if submit:
            cwd = os.getcwd()
            os.chdir(run_dir)
            os.system(f'sbatch {slurm_path}')
            self.logger.warning(f'{slurm_path} submitted! check {run_dir} for results')
            os.chdir(cwd)


class CEDataset(Dataset):
    '''C (chemistry or correction) and E (emission) dataset
    trying to merge the chem and emission fits. this class should
    work with BUI and/or CDL as inputs of auxiliary rasters. it 
    should enable randomized crop start (0 to stride size-1). it
    should give a dummy ds without science images for pure inference,
    when only x/ygrid and dt_array are given. it will also optionally 
    output spatially clustered batches to reduce burden of batch-wise
    smooth loss in training.
    '''
    def __init__(
        self,xgrid=None,ygrid=None,dt_array=None,
        l3_kw=None,bui_kw=None,cdl_kw=None,
        westmost=-128,eastmost=-65,southmost=24,northmost=50,
        initial_random_state=10,base_year=2020,jitter_kw=None
    ):
        self.logger = logging.getLogger(__name__)
        
        self.initial_random_state = initial_random_state
        self.random_state = initial_random_state
        
        self.base_year = base_year
        self.do_jitter = False
        self.jitter_kw = jitter_kw
        
        if xgrid is not None and ygrid is not None and dt_array is not None:
            self.logger.warning('make a dummy space/time without data')
            do_dummy = True
        else:
            do_dummy = False
        self.do_dummy = do_dummy
        
        if do_dummy:
            grid_size = np.abs(np.median(np.diff(xgrid)))
            self.nframe = len(dt_array)
            df = pd.DataFrame(index=dt_array,
                              data=dict(count=range(self.nframe)))
            df = CEDataset.get_time_feature(df)
        else:
            # real l3s
            l3s = l3_kw['l3s']
            self.var_names = l3_kw.pop(
                'var_names',['column_amount_DD','column_amount','surface_altitude_DD'])
            self.var_scales = l3_kw.pop(
                'var_scales',[1e9,1e4,1e6])
            xgrid,ygrid = l3s[0]['xgrid'],l3s[0]['ygrid']
            grid_size = l3s[0].grid_size
            resample_rules = l3_kw.pop('resample_rules',['3M'])
            resample_offsets = l3_kw.pop('resample_offsets',['0d'])
            
            l3r = CEDataset.concat_l3r(
                l3s,resample_rules,resample_offsets) # resampled l3s, named as l3r
            self.nframe = len(l3r)
            df = l3r.df
            for l in l3r:
                for k,v in l.items():
                    l[k] = torch.tensor(v).float()
                valid_mask = l['num_samples'] > 0.7
                for vn in self.var_names:
                    valid_mask &= ~torch.isnan(l[vn])
                l['valid_mask'] = valid_mask
                for k in self.var_names+[
                    vn+'_xy' for vn in self.var_names
                ]+[
                    vn+'_rs' for vn in self.var_names
                ]:
                    if k in l.keys():
                        l[k][~valid_mask] = 0.
                l['total_sample_weight'][~valid_mask] = 0.
            self.l3r = l3r
        
        self.frame_y,self.frame_x = len(ygrid),len(xgrid)
        self.xgrid,self.ygrid = xgrid,ygrid
        lonmesh,latmesh = np.meshgrid(xgrid,ygrid)
        xmesh = (lonmesh-westmost)/(eastmost-westmost)
        ymesh = (latmesh-southmost)/(northmost-southmost)
        self.grid_size = torch.tensor(grid_size).float()
        self.westmost,self.eastmost,self.southmost,self.northmost=\
        westmost,eastmost,southmost,northmost
        self.xmesh = torch.tensor(xmesh).float()
        self.ymesh = torch.tensor(ymesh).float()
        self.lonmesh = lonmesh
        self.latmesh = latmesh
        self.df = df
        # interfacing the bottom up inventory object
        if bui_kw is not None:
            bui = bui_kw['bui']
            assert np.array_equal(
                self.xgrid,bui.xgrid) and np.array_equal(self.ygrid,bui.ygrid)
            time_matching_method = bui_kw.pop('time_matching_method','month')
            dt0 = CEDataset.convert_period_to_datetime(self.df.index)
            dt1 = CEDataset.convert_period_to_datetime(bui.df.index)
            self.df['bui_idxs'] = CEDataset.get_index_from_another_dts(
                dt0,dt1,method=time_matching_method
            )
            max_emission = bui_kw.pop('max_emission',1e-9)
            if bui.data.shape[1] < 1:
                self.logger.warning('bui data has more than one channels, use first for mask')
            self.masks = torch.tensor(bui.data[:,0:1,...]<=max_emission)
            self.inventory_scale = bui_kw.pop('scale',1e9)
            self.yield_inventory = bui_kw.pop('yield_inventory',False)
            if self.yield_inventory:
                self.bui_data = torch.tensor(
                    bui.data*self.inventory_scale,dtype=torch.float32)
        # interfacing the crop data layer object
        if cdl_kw is not None:
            cdl = cdl_kw['cdl']
            assert np.array_equal(
                self.xgrid,cdl.xgrid) and np.array_equal(self.ygrid,cdl.ygrid)
            time_matching_method = cdl_kw.pop('time_matching_method','year')
            dt0 = CEDataset.convert_period_to_datetime(self.df.index)
            dt1 = CEDataset.convert_period_to_datetime(cdl.df.index)
            self.df['cdl_idxs'] = CEDataset.get_index_from_another_dts(
                dt0,dt1,method=time_matching_method
            )
            self.cdl_data = torch.tensor(cdl.data,dtype=torch.float32)
    
    @staticmethod
    def get_index_from_another_dts(dt0,dt1,method='nearest'):
        '''get proper index in dt1 for each one in dt0'''
        dt1_idxs = dt1.get_indexer(dt0,method='nearest')
        if method in ['nearest']:
            return dt1_idxs
        if method in ['month']:
            for i,m in enumerate(dt0.month):
                mask = dt1.month==m
                if sum(mask)>0:
                    dt1_idxs[i] = np.where(mask)[0][0]
        elif method in ['year']:
            for i,y in enumerate(dt0.year):
                mask = dt1.year==y
                if sum(mask)>0:
                    dt1_idxs[i] = np.where(mask)[0][0]
        return dt1_idxs
    
    @staticmethod
    def convert_period_to_datetime(inp):
        '''return the mid point timestamp for a period index'''
        if isinstance(inp,pd.DataFrame):
            out = inp
            if isinstance(inp.index,pd.DatetimeIndex):
                pass
            else:
                start = inp.index.start_time
                end = inp.index.end_time
                out.index = start+(end-start)/2
        elif isinstance(inp,pd.PeriodIndex):
            start = inp.start_time
            end = inp.end_time
            out = start+(end-start)/2
        elif isinstance(inp,pd.DatetimeIndex):
            out = inp
        return out
        
    @staticmethod
    def concat_l3r(
        l3s,resample_rules,resample_offsets
    ):
        for ir,(rule,offset) in enumerate(zip(resample_rules,resample_offsets)):
            l3r_ = l3s.resample(rule,offset=offset)[0]
            l3r_.df = CEDataset.get_time_feature(l3r_.df)
            if ir == 0:
                l3r = l3r_
            else:
                if rule != resample_rules[0]:
                    l3r.df = CEDataset.convert_period_to_datetime(l3r.df)
                    l3r_.df = CEDataset.convert_period_to_datetime(l3r_.df)
                l3r.df = pd.concat([l3r.df,l3r_.df])
                l3r.df['count'] = range(len(l3r.df.index))
                for l3 in l3r_:
                    l3r.add(l3)
                assert len(l3r) == l3r.df.shape[0]
        return l3r
    
    @staticmethod
    def get_time_feature(df):
        '''add fractional year, sin/cos of fy, and span in fy to df, must be PeriodIndex'''
        assert isinstance(df.index,pd.PeriodIndex)
        start = df.index.start_time
        end = df.index.end_time
        mid = start+(end-start)/2
        nday_in_year = 365 + mid.is_leap_year.astype(int)
        fy = mid.year + (mid.dayofyear - 1) / nday_in_year
        df['fractional_year'] = fy
        df['fy_sin'] = np.sin(2*np.pi*(fy%1))
        df['fy_cos'] = np.cos(2*np.pi*(fy%1))
        df['fy_span'] = (end - start).total_seconds()/86400/nday_in_year
        return df
        
    @staticmethod
    def predict_ncrop(
        frame_y,frame_x,crop_y,crop_x,stride_y=1,stride_x=1,
        start_x=0,start_y=0,ensure_full_coverage=True
    ):
        assert start_x < stride_x and start_y < stride_y
        frame_y -= start_y
        frame_x -= start_x
        ny = (frame_y-crop_y)/stride_y//1+1
        nx = (frame_x-crop_x)/stride_x//1+1
        if ensure_full_coverage:
            if (frame_y-crop_y)/stride_y%1 != 0:
                ny += 1
            if (frame_x-crop_x)/stride_x%1 != 0:
                nx += 1
        return int(ny*nx)
    
    @staticmethod
    def get_random_error(xy,rs,mask,grid_size):
        sigma = 0.5*torch.std(
            xy[mask].ravel()-rs[mask].ravel()
        )
        if torch.isnan(sigma):
            sigma = torch.tensor(1e3)
        scaler = 1.02+18.21*grid_size
        return sigma*scaler
    
    def random_crop(
        self,crop_x=64,crop_y=64,stride_x=8,stride_y=8,crop_fraction=1.,
        randomize_start_xy=True,do_group=True,
        batch_size=256,ngroup_x=7,ngroup_y=3,drop_last=True
    ):
        self.crop_x,self.crop_y,self.stride_x,self.stride_y = \
        crop_x,crop_y,stride_x,stride_y
        self.crop_fraction = crop_fraction
        self.batch_size = batch_size
        self.do_group = do_group
        
        ncrop_per_frame_max = self.predict_ncrop(
            self.frame_y,self.frame_x,crop_y,crop_x,stride_y,stride_x,
            start_x=0,start_y=0,ensure_full_coverage=True
        )
        self.logger.info(f'max {ncrop_per_frame_max} crops per frame')
        self.ncrop_per_frame_max = ncrop_per_frame_max
        
        jj = np.arange(0,self.frame_x-crop_x+1,stride_x,dtype=int)
        ii = np.arange(0,self.frame_y-crop_y+1,stride_y,dtype=int)

        if not np.isin(self.frame_x-crop_x,jj):
            jj = np.append(jj,self.frame_x-crop_x)
        if not np.isin(self.frame_y-crop_y,ii):
            ii = np.append(ii,self.frame_y-crop_y)

        all_crops = np.array([(i,j) for i in ii for j in jj])
        # make sure theory matches practice for ncrop
        assert len(all_crops) == self.ncrop_per_frame_max
        
        selected_crops = []            
        # unique random state per frame, for start xy, sampling crops, and grouping
        random_states = np.arange(self.random_state,self.random_state+self.nframe,dtype=int)
        self.random_state = self.random_state+self.nframe
        for iframe in range(self.nframe):
            rng = np.random.default_rng(random_states[iframe])
            if not randomize_start_xy:
                all_crops_ = all_crops.copy()
            else:
                start_x,start_y = rng.choice(stride_x),rng.choice(stride_y)
                jj_ = np.arange(start_x,self.frame_x-crop_x+1,stride_x,dtype=int)
                ii_ = np.arange(start_y,self.frame_y-crop_y+1,stride_y,dtype=int)
                
                all_crops_ = np.array([(i,j) for i in ii_ for j in jj_])
                ncrop_per_frame_ = self.predict_ncrop(
                    self.frame_y,self.frame_x,crop_y,crop_x,stride_y,stride_x,
                    start_x=start_x,start_y=start_y,ensure_full_coverage=False
                )
                # make sure theory matches number of random start crops
                assert len(all_crops_) == ncrop_per_frame_
            selected_idx = rng.choice(
                len(all_crops_),
                size=int(len(all_crops_)*crop_fraction),
                replace=False
            )
            selected_crops.append(all_crops_[selected_idx,:])
            
        self.selected_crops = selected_crops
        self.ncrops_per_frame = np.array([len(sc) for sc in self.selected_crops])
        self.cumsum_ncrops = np.cumsum(self.ncrops_per_frame)
        
        if not self.do_group: return
        # do group based on selected crops
        self.ngroup_x,self.ngroup_y = ngroup_x,ngroup_y
        # frame minus crop, domain to make groups
        self.fmc_x,self.fmc_y = self.frame_x-self.crop_x,self.frame_y-self.crop_y
        self.ngroup = ngroup_x*ngroup_y
        group_height = self.fmc_y/ngroup_y
        group_width = self.fmc_x/ngroup_x
        self.logger.info(
            f'domain cut to {self.ngroup} groups, H ~ {group_height:.2f}, W ~ {group_width:.2f}')
        self.group_height,self.group_width = group_height,group_width
        frame_group_crops = np.empty((self.nframe,self.ngroup),dtype=object)
        ncrops_per_frame_per_group = np.empty((self.nframe,self.ngroup),dtype=int)
        for iframe in range(self.nframe):
            rng = np.random.default_rng(random_states[iframe])
            group_x_bdr = np.linspace(0,self.fmc_x+1,self.ngroup_x+1)
            group_x_bdr[1:-1] += (rng.uniform()-0.5)*self.group_width
            group_x_bdr_r = group_x_bdr[::-1]
            group_y_bdr = np.linspace(0,self.fmc_y+1,self.ngroup_y+1)
            group_y_bdr[1:-1] += (rng.uniform()-0.5)*self.group_height
            ijs = self.selected_crops[iframe]
            igroup = 0
            for igroup_y in range(self.ngroup_y):
                if igroup_y % 2 == 0:
                    lefts,rights = group_x_bdr[:-1],group_x_bdr[1:]
                else:
                    lefts,rights = group_x_bdr_r[1:],group_x_bdr_r[:-1]
                for igroup_x in range(self.ngroup_x):
                    group_ijs = [
                        (i,j,iframe,igroup) for i,j in ijs 
                        if (i >= group_y_bdr[igroup_y]) and (i < group_y_bdr[igroup_y+1])\
                        and (j >= lefts[igroup_x]) and (j < rights[igroup_x])
                    ]
                    ncrops_per_frame_per_group[iframe,igroup] = len(group_ijs)
                    frame_group_crops[iframe,igroup] = np.array(
                        group_ijs
                    ) if len(group_ijs) > 0 else np.empty((0,4),dtype=int)
                    igroup += 1
        
        rand_frame_idxs = rng.choice(self.nframe,size=self.nframe,replace=False)
        frame_group_crops = frame_group_crops[rand_frame_idxs,:]
        ncrops_per_frame_per_group = ncrops_per_frame_per_group[rand_frame_idxs,:]
        
        group_crops = np.concatenate(frame_group_crops.ravel(order='F'))
        nsample = group_crops.shape[0]
        if batch_size is not None:
            nbatch = nsample//batch_size
            if nbatch < 1:
                if drop_last:
                    group_crops = np.empty((0,4),dtype=int)
            else:
                nsample_batch = nbatch*batch_size
                if drop_last:
                    self.logger.warning(
                        f'sample size reduced from {nsample} to {nsample_batch}={nbatch}x{batch_size}'
                    )
                    keep_idx = np.sort(rng.choice(nsample,size=nsample_batch,replace=False))
                    group_crops = group_crops[keep_idx,:]
                start_idx = np.arange(0,nsample_batch,batch_size)
                start_idx = start_idx[
                    rng.choice(len(start_idx),size=len(start_idx),replace=False)
                ]
                batchs = [group_crops[i:i+batch_size,:] for i in start_idx]
                group_crops[:nsample_batch] = np.concatenate(batchs)
        self.group_crops = group_crops        
    
    def set_jitter(self,TF,jitter_kw=None):
        jitter_kw = jitter_kw or self.jitter_kw
        self.jitter_kw = jitter_kw
        if jitter_kw is None:
            TF = False
        self.do_jitter = TF
    
    def __len__(self):
        if self.do_group:
            length = self.group_crops.shape[0]
        else:
            length = np.sum(self.ncrops_per_frame) 
        return length
    
    def __getitem__(self,idx):
        if self.do_group:
            i,j,frame_idx,group_idx = self.group_crops[idx]
        else:
            # index of frame
            frame_idx = np.sum((idx-self.cumsum_ncrops)>=0).astype(int)
            # index of crop, from each frame
            crop_idx = idx if frame_idx == 0 else idx-self.cumsum_ncrops[frame_idx-1]
            i, j = self.selected_crops[frame_idx][crop_idx]
         # spatial feature as normalized lon/lat mesh
        xmesh = self.xmesh[i:i+self.crop_y,j:j+self.crop_x].unsqueeze(0)
        ymesh = self.ymesh[i:i+self.crop_y,j:j+self.crop_x].unsqueeze(0)
        # temporal feature
        df = self.df
        if 'fd_sin' not in df.keys():
            temporal = torch.tensor(
                [
                    df['fractional_year'].iloc[frame_idx]-self.base_year,
                    df['fy_sin'].iloc[frame_idx],
                    df['fy_cos'].iloc[frame_idx]
                ]
            ).float()
        else:
            pass # diurnal code will be here
        
        # temporal augmentation
        if self.do_jitter:
            fy_span = df['fy_span'].iloc[frame_idx]
            jitter_fy = torch.rand(1) < self.jitter_kw['p_jitter']
            if jitter_fy:
                temporal[0] += (torch.rand(1)[0]-0.5)*fy_span
                fy2pi = 2*np.pi*(temporal[0]%1)
                temporal[1] = torch.sin(fy2pi)
                temporal[2] = torch.cos(fy2pi)
            jitter_fd = torch.rand(1) < self.jitter_kw['p_jitter']
            if jitter_fd and 'fd_span' in self.jitter_kw.keys():
                jittered_fd = self.fd[img_idx]+(torch.rand(1)[0]-0.5)*self.jitter_kw['fd_span']
                temporal[3] = torch.sin(2*np.pi*jittered_fd)
                temporal[4] = torch.cos(2*np.pi*jittered_fd)
        
        out = dict(
                spatial=torch.cat([xmesh,ymesh],dim=0),
                temporal=temporal
            )
        if self.do_dummy:
            return out
        # crop l3r
        valid_mask = self.l3r[frame_idx]['valid_mask'][
            i:i+self.crop_y,j:j+self.crop_x
        ]
        out['valid_mask'] = valid_mask.unsqueeze(0)
        grid_size = torch.tensor(self.l3r[frame_idx].grid_size)
        out['grid_size'] = grid_size
        for ivar,(name,scale) in enumerate(zip(self.var_names,self.var_scales)):
            out[name] = self.l3r[frame_idx][name][
                i:i+self.crop_y,j:j+self.crop_x
            ].unsqueeze(0)*scale
            # assume the first var is DD and calculate random error from xy/rs
            if ivar == 0:
                if f'{name}_xy' in self.l3r[frame_idx].keys():
                    DD_xy = self.l3r[frame_idx][f'{name}_xy'][
                        i:i+self.crop_y,j:j+self.crop_x
                    ]*scale
                    DD_rs = self.l3r[frame_idx][f'{name}_rs'][
                        i:i+self.crop_y,j:j+self.crop_x
                    ]*scale
                    out['random_error'] = self.get_random_error(
                        DD_xy,DD_rs,valid_mask,grid_size
                    )
                else:
                    out['random_error'] = torch.tensor(1.)
        # bui outputs
        if 'bui_idxs' in df.keys():
            bui_idx = df['bui_idxs'].iloc[frame_idx]
        # yield mask
        if hasattr(self,'masks'):
            out['fit_mask'] = self.masks[bui_idx,:,i:i+self.crop_y,j:j+self.crop_x]
        if self.yield_inventory:
            out['bui'] = self.bui_data[bui_idx,:,i:i+self.crop_y,j:j+self.crop_x]
        # cdl outputs
        if 'cdl_idxs' in df.keys():
            cdl_idx = df['cdl_idxs'].iloc[frame_idx]
            out['cdl'] = self.cdl_data[cdl_idx,:,i:i+self.crop_y,j:j+self.crop_x]
        return out
    
    def plot_crops(
        self,ax=None,idxs=range(10),show_text=True,plot_patch=True,
        field='column_amount',channel_idx=0,**kwargs
    ):
        from matplotlib.patches import Rectangle
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(12,6))
        else:
            fig = plt.gca()
        x0 = self.lonmesh[0,0];x1 = self.lonmesh[0,-1]
        y0 = self.latmesh[0,0];y1 = self.latmesh[-1,0]
        ax.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],'-k')
        
        for idx in idxs:
            if self.do_group:
                i,j,frame_idx,group_idx = self.group_crops[idx]
            else:
                frame_idx = np.sum((idx-self.cumsum_ncrops)>=0).astype(int)
                # index of crop, from each frame
                crop_idx = idx if frame_idx == 0 else idx-self.cumsum_ncrops[frame_idx-1]
                i, j = self.selected_crops[frame_idx][crop_idx]
            xmesh = self.lonmesh[i:i+self.crop_y,j:j+self.crop_x]
            ymesh = self.latmesh[i:i+self.crop_y,j:j+self.crop_x]
            cx0 = xmesh[0,0];cx1 = xmesh[0,-1]
            cy0 = ymesh[0,0];cy1 = ymesh[-1,0]
            if plot_patch:
                rect = Rectangle(
                    (cx0, cy0), cx1-cx0, cy1-cy0, linewidth=2, 
                    edgecolor='r', facecolor='r',alpha=0.05)
                ax.add_patch(rect)
            
            ax.pcolormesh(
                ds[idx]['spatial'][0]*(ds.eastmost-ds.westmost)+ds.westmost,
                ds[idx]['spatial'][1]*(ds.northmost-ds.southmost)+ds.southmost,
                ds[idx][field][channel_idx],**kwargs
            )
            if show_text:
                ax.text(
                    cx0+(cx1-cx0)/2, cy0+(cy1-cy0)/2, 
                    f'{idx},{frame_idx},{group_idx}', color='black', fontsize=12,
                    ha='center', va='center'
                )


class TemporalFiLM(nn.Module):
    '''
    TemporalFiLM generates gamma and beta parameters for feature-wise linear modulation.
    It can generate parameters for multiple layers/scales simultaneously using a shared MLP.
    '''
    def __init__(
        self,temporal_dim,modulated_channels_list,
        hidden_dim=128,leaky_relu_rate=0.1
    ):
        super().__init__()
        self.modulated_channels_list = modulated_channels_list
        total_film_params = sum(modulated_channels_list)*2 # 2 for gamma and beta
        # Shared MLP for temporal feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_rate),
            nn.Linear(hidden_dim, total_film_params)
        )
        # Initialize weights and biases to produce identity transformation initially
        last_layer = self.mlp[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=0.01)
        # Initialize biases for gamma to 1 and beta to 0
        current_idx = 0
        for channels in modulated_channels_list:
            # Gamma biases (initialized to 1)
            nn.init.constant_(last_layer.bias[current_idx:current_idx+channels],1.0)
            current_idx += channels
            # Beta biases (initialized to 0)
            nn.init.constant_(last_layer.bias[current_idx:current_idx+channels],0.0)
            current_idx += channels
    
    def forward(self,temporal_features):
        '''
        temporal_features: [B, T]
        '''
        film_params = self.mlp(temporal_features) # (batch_size, total_film_params)
        # Split film_params into gamma and beta for each modulated layer
        gammas = []
        betas = []
        current_idx = 0
        for channels in self.modulated_channels_list:
            gamma = film_params[:,current_idx:current_idx+channels]
            current_idx += channels
            beta = film_params[:,current_idx:current_idx+channels]
            current_idx += channels
            gammas.append(gamma.unsqueeze(-1).unsqueeze(-1)) # (B, C, 1, 1)
            betas.append(beta.unsqueeze(-1).unsqueeze(-1))   # (B, C, 1, 1)
        return gammas, betas


class ConvBlock(nn.Module):
    '''a unet conv block integrating film'''
    def __init__(
        self,in_channels,out_channels,norm_groups=8,leaky_relu_rate=0.1,
        film_modulated=True
    ):
        super().__init__()
        self.film_modulated = film_modulated
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.norm1 = nn.GroupNorm(norm_groups,out_channels)
        self.act1 = nn.LeakyReLU(leaky_relu_rate)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.norm2 = nn.GroupNorm(norm_groups,out_channels)
        self.act2 = nn.LeakyReLU(leaky_relu_rate)
    
    def forward(self,x,gamma=None,beta=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.film_modulated:
            if gamma is None or beta is None:
                raise ValueError("gamma and beta must be provided")
            x = x * gamma + beta
        x = self.act2(x)
        return x


class FeatureModulatedUNet(nn.Module):
    '''
    U-Net with FiLM modulation on bottleneck and optionally on decoder blocks
    '''
    def __init__(
        self,in_channels=2,out_channels=2,base_channels=64,unet_depth=4,
        norm_groups=8,leaky_relu_rate=0.1,force_positive=True,temporal_kw=None
    ):
        super().__init__()
        temporal_kw = temporal_kw or {}
        temporal_dim = temporal_kw.pop('temporal_dim',3)
        temporal_hidden_dim = temporal_kw.pop('hidden_dim',128)
        temporal_leaky_relu_rate = temporal_kw.pop(
            'leaky_relu_rate',0.1)
        modulate_decoder = temporal_kw.pop('modulate_decoder',True)
        feat_channels = [base_channels*2**i for i in range(unet_depth)]
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.modulate_decoder = modulate_decoder
        # Encoder blocks
        self.downs = nn.ModuleList()
        self.downs.append(
            ConvBlock(
                in_channels,feat_channels[0],norm_groups=norm_groups,
                film_modulated=False
            )
        )
        for i in range(len(feat_channels)-2):
            self.downs.append(
                ConvBlock(
                    feat_channels[i],feat_channels[i+1],norm_groups=norm_groups,
                    film_modulated=False
                )
            )
        # Bottleneck (always modulated)
        self.bottleneck = ConvBlock(
            feat_channels[-2],feat_channels[-1],norm_groups=norm_groups,
            film_modulated=True
        )
        # Decoder blocks and upsampling layers
        self.ups = nn.ModuleList()
        for i in reversed(range(len(feat_channels)-1)):
            self.ups.append(
                nn.ConvTranspose2d(
                    feat_channels[i+1],feat_channels[i],kernel_size=2,stride=2
                )
            )
            # The decoder conv block takes (skip_channels+upsampled_channels) as input
            self.ups.append(
                ConvBlock(
                    feat_channels[i]*2,feat_channels[i],norm_groups=norm_groups,
                    film_modulated=modulate_decoder
                )
            )
        if force_positive:
            self.final_conv = nn.Sequential(
                nn.Conv2d(feat_channels[0],out_channels,kernel_size=1),
                nn.Softplus(),
            )
        else:
            self.final_conv = nn.Conv2d(feat_channels[0],out_channels,kernel_size=1)
        # FiLM setup
        modulated_channels = [feat_channels[-1]] # Bottleneck channels
        if modulate_decoder:
            # Add channels for skip connections in reverse order of encoder depth
            # These are the output channels of the encoder blocks that feed into the decoder
            modulated_channels.extend(list(reversed(feat_channels[:-1]))) # Exclude bottleneck channels

        self.temporal_film = TemporalFiLM(
            temporal_dim,modulated_channels,
            temporal_hidden_dim,temporal_leaky_relu_rate
        )
    
    def forward(self,x,temporal):
        # Generate all FiLM parameters
        gammas, betas = self.temporal_film(temporal)
        # Encoder
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # Store skip connection after block, before pooling
            skip_connections.append(x)
            x = self.pool(x)
        # Bottleneck
        # The first gamma/beta pair is for the bottleneck
        film_idx = 0
        x = self.bottleneck(x,gamma=gammas[film_idx],beta=betas[film_idx])
        film_idx += 1
        # Decoder
        # Iterate through upsampling and decoder blocks
        for i in range(len(self.ups)//2):
            up_layer = self.ups[i*2]
            dec_block = self.ups[i*2+1]
            x = up_layer(x)
            skip_feat = skip_connections.pop() # Get last stored skip connection
            x = torch.cat([skip_feat,x],dim=1)
            if self.modulate_decoder:
                # Apply FiLM to the decoder block (which processes the concatenated features)
                x = dec_block(x,gamma=gammas[film_idx],beta=betas[film_idx])
                film_idx += 1
            else:
                x = dec_block(x) # No FiLM applied if modulate_decoder is False
        return self.final_conv(x)


class SuperGaussianPSF(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        
        # Learnable parameters in the background
        self.log_wx = nn.Parameter(torch.tensor(.0))
        self.log_wy = nn.Parameter(torch.tensor(.0))
        self.log_kx = nn.Parameter(torch.tensor(0.5))
        self.log_ky = nn.Parameter(torch.tensor(0.5))
        
        # Base coordinate grid (unit spacing) to be scaled by grid_size
        self.register_buffer('_base_coords', torch.zeros(2,kernel_size,kernel_size))
        self._setup_base_coords()
    
    def _setup_base_coords(self):
        r = (self.kernel_size - 1) // 2
        x = torch.linspace(-r, r, self.kernel_size) # unit spacing
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self._base_coords = torch.stack([xx,yy],dim=0) #[2,kernel_size,kernel_size]
    
    def _create_kernels(self,grid_size):
        '''Create kernels for batched grid_size
        grid_size: 
            [B,] tensor of grid sizes
        Returns: [B, 1, kernel_size, kernel_size] kernel for each sample
        '''
        batch_size = grid_size.shape[0]
        # Expand base coordinates for batch: [B, 2, kernel_size, kernel_size]
        base_coords_expanded = self._base_coords.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # [B, 2, kernel_size, kernel_size]
        
        # Scale coordinates by grid_size: [B, 2, kernel_size, kernel_size]
        scaled_coords = base_coords_expanded * grid_size.view(-1, 1, 1, 1)
        xx = scaled_coords[:, 0]  # [B, kernel_size, kernel_size]
        yy = scaled_coords[:, 1]  # [B, kernel_size, kernel_size]
                
        # Get positive parameters (grid_size aware)
        wx = torch.exp(self.log_wx) + 1e-3 * grid_size.view(-1,1,1) # [B,1,1]
        wy = torch.exp(self.log_wy) + 1e-3 * grid_size.view(-1,1,1)
        kx = torch.exp(self.log_kx) + 1.0
        ky = torch.exp(self.log_ky) + 1.0
        
        # Compute super-Gaussian kernel
        kernel = torch.exp(
            -(torch.abs(xx/wx)**kx 
             + torch.abs(yy/wy)**ky)
        ) # [B, kernel_size, kernel_size]
        kernel = kernel / kernel.sum(dim=(-1,-2),keepdim=True) # normalize kernel per sample
        return kernel.unsqueeze(1) # [B,1,kernel_size,kernel_size]
    
    def forward(self, x, grid_size):
        '''
        x:
            [B,C,H,W], input tensor
        grid_size:
            [B], grid size in degree
        '''
        B, C, H, W = x.shape
        # batched kernels, [B,1,kernel_size,kernel_size]
        kernels = self._create_kernels(grid_size)
        
        # Apply standard convolution
        padding = (self.kernel_size - 1) // 2
        
        results = []
        for i in range(B):
            # Apply sample-specific kernel to all channels
            sample_result = F.conv2d(
                F.pad(x[i:i+1], (padding,)*4, mode='reflect'),
                kernels[i:i+1].expand(-1, C, -1, -1),  # [1, C, k, k]
                groups=C
            )
            results.append(sample_result)
        
        return torch.cat(results, dim=0)
    
    def get_kernel(self, grid_size):
        '''Returns current kernel for visualization'''
        with torch.no_grad():
            return self._create_kernels(grid_size)


class FluxCombiner(nn.Module):
    '''
    final function to sum up everything with optional learnable kernel
    '''
    def __init__(self,spatial_kw,temporal_kw,psf_kw=None):
        super().__init__()
        psf_kw = psf_kw or {'enabled':False}
        self.unet = FeatureModulatedUNet(temporal_kw=temporal_kw,**spatial_kw)
        self.do_psf = psf_kw['enabled']
        if self.do_psf:
            self.psf = SuperGaussianPSF(kernel_size=psf_kw['kernel_size'])
    
    def forward(self,spatial,predictors,temporal,grid_size=None,gia=None):
        '''
        spatial:
            B,C=2 (lon,lat),H=crop_y,W=crop_x
        predictors:
            B,C=outchannels (vcd, wt, and cdl),H=crop_y,W=crop_x
        temporal:
            B,C=temporal_dim, (year, sin(fy), cos(fy)) if 3, 
            add sin(fd), cos(fd) for 5
        grid_size:
            B, grid size, needed for psf
        gia:
            B,H,W, grid inverse area, needed for point sources
        '''
        unet_out = self.unet(x=spatial,temporal=temporal)
        area_flux = (unet_out*predictors).sum(dim=1,keepdim=True)
        # to be implemented: point flux
        all_flux = area_flux
        if self.do_psf:
            all_flux = self.psf(x=all_flux,grid_size=grid_size)
        
        return all_flux,unet_out


class MaskedLoss(nn.Module):
    def __init__(self,loss_func=None):
        super().__init__()
        self.loss_func = loss_func or nn.L1Loss(reduction='none')

    def forward(self,predict,target,mask=None):
        # Create a mask where target is not NaN
        if mask is None:
            mask = ~torch.isnan(target)
        
        # Compute L1 loss only for valid (non-NaN) pixels
        loss = self.loss_func(
            torch.nan_to_num(predict),
            torch.nan_to_num(target)
        ) * mask.float()
        
        # Normalize by the number of valid pixels (avoid div-by-zero)
        valid_pixels = mask.sum().float()
        return loss.sum() / (valid_pixels + 1e-6)  # Add epsilon for stability


class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], 
                                          dtype=torch.float32).view(1,1,3,3)
    
    def forward(self,unet_out,scale_weights=[1.0, 0.5, 0.25],channel_weights=1.):
        
        B, C, H, W = unet_out.shape
        if np.isscalar(channel_weights):
            channel_weights = channel_weights*torch.ones(C)
        else:
            channel_weights = torch.tensor(channel_weights).float()
        channel_weights = channel_weights.expand(B,-1)
        if channel_weights.device != unet_out.device:
            channel_weights = channel_weights.to(unet_out.device)
        
        lap_mean_loss = 0.
        lap_std_loss = 0.
        for i, weight in enumerate(scale_weights):
            # Downsample unet output
            scaled = F.avg_pool2d(unet_out, kernel_size=2**i)
            
            # Calculate Laplacian
            if self.gradient_kernel.device != scaled.device:
                self.gradient_kernel = self.gradient_kernel.to(scaled.device)
            gradients = F.conv2d(
                scaled, self.gradient_kernel.expand(C,1,3,3), 
                groups=C, padding=1)
            # [B,C] per scale
            lap_BC = gradients.abs().mean(dim=[-2,-1])*channel_weights*weight
            # std across B
            lap_B_std = lap_BC.mean(dim=-1).std()
            # mean laplacian per scale
            lap_mean = lap_BC.mean()
            lap_mean_loss += lap_mean
            lap_std_loss += lap_B_std
                        
        return lap_mean_loss,lap_std_loss


class BatchShuffledSampler(Sampler):
    """
    A sampler that shuffles the order of batches but keeps samples within each batch together.
    
    Args:
        dataset_len (int): Total number of samples in the dataset.
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the batch order.
        drop_last (bool): Whether to drop the last incomplete batch.
    """
    def __init__(self, dataset_len, batch_size, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate indices for the start of each batch
        self.batch_start_indices = np.arange(0, dataset_len, batch_size)
        
        if self.drop_last and len(self.batch_start_indices) > 0:
            # Check if the last batch is incomplete
            if self.dataset_len % self.batch_size != 0:
                self.batch_start_indices.pop()

    def __iter__(self):
        # 1. Get the list of batch start indices
        indices = self.batch_start_indices.copy()
        
        # 2. Shuffle the order of the batches
        if self.shuffle:
            np.random.shuffle(indices)
            
        # 3. For each batch, yield the sequence of indices within it
        for start_idx in indices:
            # Determine the end of the current batch
            end_idx = min(start_idx + self.batch_size, self.dataset_len)
            yield from range(start_idx, end_idx)

    def __len__(self):
        if self.drop_last:
            return (self.dataset_len // self.batch_size) * self.batch_size
        return self.dataset_len


class CETrainer:
    def __init__(
        self,model,lr_kw,wd_kw,device=None,
        L1Loss_or_MSE='L1Loss',initial_grad_scaler=None
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'device is {self.device.type}')
        
        self.model = model.to(self.device)
        param_groups,lr_lambdas = [],[]
        for pg_name,pg_multiplier in lr_kw['multipliers'].items():
            if pg_name != 'other':
                params = [
                    p for n,p in self.model.named_parameters()
                    if pg_name in n
                ]
            else: # 'other' group
                params = [
                    p for n,p in self.model.named_parameters()
                    if not any(
                        pgn in n for pgn in lr_kw['multipliers'].keys() 
                        if pgn != 'other'
                    )
                ]
            # weight decay for the group
            pg_wd = wd_kw['initial']*wd_kw['multipliers'][pg_name]
            param_groups.append(
                dict(
                    params=params,weight_decay=pg_wd,
                    lr=lr_kw['milestone'][1]*pg_multiplier
                )
            )
        self.optimizer = torch.optim.AdamW(param_groups)
        if lr_kw['scheduler'] == 'cosine':
            lr_lambda = partial(
                CETrainer.cosine_scheduler,
                milestone=lr_kw['milestone'],output_ratio=True
            )
        elif lr_kw['scheduler'] == 'linear':
            lr_lambda = partial(
                CETrainer.linear_scheduler,
                milestone=lr_kw['milestone'],output_ratio=True
            )
        else:
            raise ValueError('{} is not supported!'.format(lr_kw['scheduler']))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,lr_lambda=lr_lambda
        )
        if L1Loss_or_MSE == 'MSE':
            self.data_loss_func = MaskedLoss(loss_func=nn.MSELoss(reduction='none'))
        else:
            self.data_loss_func = MaskedLoss()
        
        # Only enable mixed precision on CUDA
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type=='cuda')
        )
       
    @staticmethod
    def cosine_scheduler(epoch,milestone,output_ratio=True):
        if epoch <= milestone[0]:
            out = 1. if output_ratio else milestone[1]
        elif epoch > milestone[2]:
            out = milestone[3]/milestone[1] if output_ratio else milestone[3]
        else:
            progress = (epoch-milestone[0])/(milestone[2]-milestone[0])
            cos_decay = 0.5*(1+np.cos(np.pi*progress))
            if output_ratio:
                final_ratio = milestone[3]/milestone[1]
                out = final_ratio+(1-final_ratio)*cos_decay
            else:
                out = milestone[3]+(milestone[1]-milestone[3])*cos_decay
        return out
    
    @staticmethod
    def linear_scheduler(epoch,milestone,output_ratio=True):
        if epoch <= milestone[0]:
            out = 1. if output_ratio else milestone[1]
        elif epoch > milestone[2]:
            out = milestone[3]/milestone[1] if output_ratio else milestone[3]
        else:
            progress = 1-(epoch-milestone[0])/(milestone[2]-milestone[0])
            if output_ratio:
                final_ratio = milestone[3]/milestone[1]
                out = final_ratio+(1-final_ratio)*progress
            else:
                out = milestone[3]+(milestone[1]-milestone[3])*progress
        return out
    
    def _to_device(self,data):
        return {
            k: v.to(
                self.device,non_blocking=(self.device.type=='cuda')
            ) for k, v in data.items()
        }
    
    def save_model(self,path,**kwargs):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
        }, path)
    
    def load_data(self,dss,batch_size,shuffle_level=None):
        if shuffle_level is None:
            self.data_loader = DataLoader(
                ConcatDataset(dss),batch_size=batch_size,
                shuffle=False,pin_memory=(self.device.type=='cuda')
            )
        elif shuffle_level == 'batch':
            ds = ConcatDataset(dss)
            sampler = BatchShuffledSampler(
                dataset_len=len(ds),batch_size=batch_size,
                shuffle=True,drop_last=False
            )
            self.data_loader = DataLoader(
                ds,batch_size=batch_size,sampler=sampler,shuffle=False,
                pin_memory=(self.device.type=='cuda')
            )
        elif shuffle_level == 'sample':
            self.data_loader = DataLoader(
                ConcatDataset(dss),batch_size=batch_size,
                shuffle=True,pin_memory=(self.device.type=='cuda')
            )
        elif isinstance(shuffle_level,int):
            if batch_size % shuffle_level != 0:
                raise ValueError(
                    'if a number, shuffle_level must divide batch_size to integer')
            ds = ConcatDataset(dss)
            sampler = BatchShuffledSampler(
                dataset_len=len(ds),batch_size=int(batch_size/shuffle_level),
                shuffle=True,drop_last=False
            )
            self.data_loader = DataLoader(
                ds,batch_size=batch_size,sampler=sampler,shuffle=False,
                pin_memory=(self.device.type=='cuda')
            )
        else:
            raise ValueError(f'{shuffle_level} is not supported!')
    
    def validate(
        self,val_loader,
        yname='column_amount_DD',
        xnames=['column_amount','surface_altitude_DD','cdl'],
        use_fit_mask=False,
        scale_random_error=True
    ):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                predict,unet_out = self.model(
                    spatial=batch['spatial'],
                    predictors=torch.cat([batch[xn] for xn in xnames],dim=1),
                    temporal=batch['temporal']
                )
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                predict = predict/error_scaler
                target = batch[yname]/error_scaler
                if use_fit_mask:
                    mask = batch['valid_mask']*batch['fit_mask'] 
                else:
                    mask = batch['valid_mask']
                loss = self.data_loss_func(
                    predict=predict,target=target,mask=mask
                )
                total_loss += loss.item()
            
        return total_loss/len(val_loader)
    
    def train_epoch(
        self,epoch,
        yname='column_amount_DD',
        xnames=['column_amount','surface_altitude_DD','cdl'],
        use_fit_mask=False,
        scale_random_error=True,
        smoothness_weight=1e-2,
        smoothness_B_weight=0.8,
        smoothness_scale_weights=[1,0.5,0.25],
        smoothness_channel_weights=1.,
        max_norm=1.,
        clamp_max_sigma=None,
        clamp_min_sigma=None,
        verbose=False
    ):
        self.model.train()
        
        if clamp_min_sigma is not None:
            clamp_min_sigma = -np.abs(clamp_min_sigma)
        
        if smoothness_weight is not None:
            smoothness_loss_func = LaplacianLoss()
        total_loss = 0
        total_smo_loss = 0
        total_smb_loss = 0
        start_time = time.time()
        num_batches = len(self.data_loader)
        norms = []
        for batch_idx, batch in enumerate(self.data_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            
            # Only use autocast on CUDA
            with torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
                predict,unet_out = self.model(
                    spatial=batch['spatial'],
                    predictors=torch.cat([batch[xn] for xn in xnames],dim=1),
                    temporal=batch['temporal']
                )
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                predict = predict/error_scaler
                target = batch[yname]/error_scaler
                target = torch.clamp(target,min=clamp_min_sigma,max=clamp_max_sigma)
                if use_fit_mask:
                    mask = batch['valid_mask']*batch['fit_mask'] 
                else:
                    mask = batch['valid_mask']
                loss = self.data_loss_func(
                    predict=predict,target=target,mask=mask
                )
                if smoothness_weight is not None:
                    smoothness_loss,smoothness_B_loss = smoothness_loss_func(
                        unet_out,
                        scale_weights=smoothness_scale_weights,
                        channel_weights=smoothness_channel_weights
                    )
                    loss += smoothness_weight*smoothness_loss
                    if smoothness_B_weight is not None:
                        loss += smoothness_B_weight*smoothness_B_loss
                else:
                    smoothness_loss,smoothness_B_loss = torch.tensor(0.),torch.tensor(0.)
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),max_norm=max_norm
                    )
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
                ).item()
                norms.append(total_norm)
                self.logger.info(f"Step Gradient Norm: {total_norm:.4f}")
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        self.logger.warning(f"No gradient for {name}")
                    elif torch.all(param.grad == 0):
                        self.logger.warning(f"Zero gradients for {name}")
                        ntotal = batch['valid_mask'].numel()
                        nfit = (batch['valid_mask']*batch['fit_mask']).sum()
                        self.logger.warning(f'good pixels {nfit}')
                        self.logger.warning(f'total {ntotal}')
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_smo_loss += smoothness_loss.item()
            total_smb_loss += smoothness_B_loss.item()
            
        # Epoch summary
        epoch_time = time.time() - start_time
        if verbose:
            self.logger.warning(
                f'\rEpoch {epoch + 1} | Time: {epoch_time:.1f}s | Loss: {total_loss/num_batches:.4f} | Sample: {self.data_loader.batch_size}x{num_batches}')
        result = dict(
            train_loss=total_loss/num_batches,
            smooth_loss=total_smo_loss/num_batches,
            smooth_B_loss=total_smb_loss/num_batches,
            epoch_time=epoch_time,
            sample_size=len(self.data_loader.dataset),
            grad_norm=np.nanmean(norms),
            nan_norm=np.sum(np.isnan(norms))
        )
        return result


class Inferencer:
    def __init__(self,models,device=None):
        '''should work for a list of models from nfold'''
        self.logger = logging.getLogger(__name__)
        if not isinstance(models,list):
            self.logger.warning('packing model to a list')
            models = [models]
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'device is {self.device.type}')
        self.models = [mdl.to(self.device) for mdl in models]
    
    def _to_device(self,data):
        return {k: v.to(self.device, non_blocking=(self.device.type=='cuda')) 
                for k, v in data.items()}
    
    def load_model(self,paths=None,path_pattern=None,load_optimizer=False):
        if paths is None:
            paths = glob.glob(path_pattern)
        for model,path in zip(self.models,paths):
            checkpoint = torch.load(path, map_location=self.device)
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model = model.to(self.device)
        
    def inference(self,ds,trim=2):
        '''perform inference on a dataset (ds)'''
        unet_out_channels = dict(
            dict(
                self.models[0].named_children()
            ).get('unet').named_children()
        ).get('final_conv')[0].out_channels
        nframe = ds.nframe
        ncrop_per_frame = int(ds.ncrops_per_frame[0])
        crop_x,crop_y,frame_x,frame_y=ds.crop_x,ds.crop_y,ds.frame_x,ds.frame_y
        # crop each frame into a batch
        loader = DataLoader(
            ds,batch_size=ncrop_per_frame,shuffle=False,drop_last=True
        )
        # mosaicked unet output to frame dimension
        unet_out = np.zeros((len(self.models),nframe,unet_out_channels,frame_y,frame_x))
        
        for imodel,model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                # exactly one batch per frame
                for iframe,batch in enumerate(loader):
                    B,C,H,W = batch['spatial'].shape
                    batch['dummy_predictors'] = torch.zeros(
                        B,unet_out_channels,H,W,dtype=torch.float32
                    )
                    batch = self._to_device(batch)
                    _,out = model(
                        spatial=batch['spatial'],
                        predictors=batch['dummy_predictors'],
                        temporal=batch['temporal']
                    )
                    # B, unet_out_channels, crop_y, crop_x
                    D_unet = np.zeros((unet_out_channels,frame_y,frame_x))
                    for icrop in range(ncrop_per_frame):
                        # unet_out_channels, crop_y, crop_x
                        unet_out_crop = out[icrop].cpu().detach().numpy()
                        i, j = ds.selected_crops[iframe][icrop]
                        if trim > 0:
                            trim_up,trim_down = trim,trim
                            trim_left,trim_right = trim,trim
                            if i == 0: 
                                trim_up = 0
                            if i == frame_y-crop_y: 
                                trim_down = 0
                            if j == 0: 
                                trim_left = 0
                            if j == frame_x-crop_x: 
                                trim_right = 0
                            unet_out_crop = unet_out_crop[
                                :,trim_up:crop_y-trim_down,
                                trim_left:crop_x-trim_right
                            ]
                        unet_out[
                            imodel,iframe,:,
                            i+trim_up:i+crop_y-trim_down,
                            j+trim_left:j+crop_x-trim_right
                        ] += unet_out_crop
                        D_unet[
                            :,i+trim_up:i+crop_y-trim_down,
                            j+trim_left:j+crop_x-trim_right
                        ] += np.ones(unet_out_crop.shape)
                    unet_out[imodel,iframe,] /= D_unet
        return unet_out