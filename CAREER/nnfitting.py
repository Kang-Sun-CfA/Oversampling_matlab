import numpy as np
import pandas as pd
import os,sys,glob
import logging
import yaml
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ChemFitConfig(dict):
    def __init__(self,run_id,**kwargs):
        self.logger = logging.getLogger(__name__)
        self['run_id'] = run_id
        self['run_dir'] = kwargs.pop(
            'run_dir',os.path.join('/projects/academic/kangsun/kangsun/IDS/v03',self['run_id']))
        self['l3_path_pattern'] = kwargs.pop(
            'l3_path_pattern',
            '/projects/academic/kangsun/zolalaya/data/Cornbelt_NOx/s5p_cornbelt/%Y/%m/CORNBELT_S5P_NO2_%Y_%m_%d.nc'
        )
        self['nei_dir'] = kwargs.pop('nei_dir','/projects/academic/kangsun/jobaerah/Data_NEI/')
        # fit_mask will be initialized using this threshold on annual NEI
        self['max_neinox'] = kwargs.pop('max_neinox',1e-9)
        self['grid_sizes'] = kwargs.pop('grid_sizes',[0.1])
        # a list of start/end for daily period index. a l3s will be loaded per start/end date
        self['l3s_ranges'] = kwargs.pop(
            'l3s_ranges',
            [[f'{y}-01-01',f'{y}-12-31'] for y in range(2020,2025)]
        )
        self['crop_x'] = kwargs.pop('crop_x',64)
        self['crop_y'] = kwargs.pop('crop_y',self['crop_x'])
        # a list of resampling rules for training dataset
        self['train_intervals'] = kwargs.pop(
            'train_intervals',
            ['1Y','21d','28d','35d']
        )
        # a list of list of resampling offsets for training dataset
        self['train_offsets'] = kwargs.pop(
            'train_offsets',
            [[None],['0d','5d','16d'],['0d','7d','14d'],['0d','18d','27d']]
        )
        # a list of stride sizes for training dataset
        self['train_stride_xs'] = kwargs.pop('train_strid_xs',[4,8,8,8])
        self['train_stride_ys'] = kwargs.pop('train_strid_ys',self['train_stride_xs'].copy())
        # a list of milestones for training data crop fraction, as [epoch1, value1, epoch2, value2]
        self['train_milestones'] = kwargs.pop(
            'train_milestones',
            [[0,1,100,0],[0,0,100,0.7],[0,0,100,0.7],[0,0,100,0.7]]
        )
        
        # a list of resampling rules for val/testing dataset
        self['eval_intervals'] = kwargs.pop(
            'eval_intervals',
            ['1M','21d','28d','35d']
        )
        # a list of list of resampling offsets for val/testing dataset
        self['eval_offsets'] = kwargs.pop(
            'eval_offsets',
            [[None],['10d'],['21d'],['9d']]
        )
        # a list of stride sizes for val/testing dataset
        self['eval_stride_xs'] = kwargs.pop('eval_strid_xs',[64,64,64,64])
        self['eval_stride_ys'] = kwargs.pop('eval_strid_ys',self['eval_stride_xs'].copy())
        # training configs
        self['lr'] = kwargs.pop('lr',1e-4)
        self['weight_decay'] = kwargs.pop('weight_decay',0.01)
        self['batch_size'] = kwargs.pop('batch_size',256)
        self['max_norm'] = kwargs.pop('max_norm',1.0)
        self['initial_grad_scaler'] = kwargs.pop('initial_grad_scaler',1024.0)
        self['start_epoch'] = kwargs.pop('start_epoch',0)
        self['end_epoch'] = kwargs.pop('end_epoch',200)
        self['L1Loss_or_MSE'] = kwargs.pop('L1Loss_or_MSE','L1Loss')
        self['smoothness_weight_milestones'] = kwargs.pop(
            'smoothness_weight_milestones',[0,0,100,5e-3])
        # smoothness on different levels of aggregation on unet_out
        self['smoothness_scale_weights'] = kwargs.pop(
            'smoothness_scale_weights',
            [1,0.5,0.5,0.25]
        )
        # smoothness on different channels of unet_out
        self['smoothness_channel_weights'] = kwargs.pop(
            'smoothness_channel_weights',
            [2,1]
        )
        self['save_best_model'] = kwargs.pop('save_best_model',True)
        self.check()
        
    def check(self):
        if not os.path.exists(self['run_dir']):
            run_dir = self['run_dir']
            self.logger.warning(f'{run_dir} does not exist, trying to create one')
            try:
                os.makedirs(self['run_dir'])
            except Exception as e:
                self.logger.warning(e)
        if not all(
            [
                len(self[k])==len(self['train_intervals']) 
                for k in ['train_stride_xs','train_stride_ys','train_offsets','train_milestones']
            ]
        ):
            self.logger.error('train config lists have to be the same size!')
        if not all(
            [
                len(self[k])==len(self['eval_intervals']) 
                for k in ['eval_stride_xs','eval_stride_ys','eval_offsets']
            ]
        ):
            self.logger.error('eval config lists have to be the same size!')
        
    def to_yaml(self,rel_path='config.yml',abs_path=None):
        path = abs_path or os.path.join(self['run_dir'],rel_path)
        with open(path,'w') as stream:
            yaml.dump({k:v for k,v in self.items()},stream,sort_keys=False)
    
    def read_yaml(self,rel_path='config.yml',abs_path=None):
        path = abs_path or os.path.join(self['run_dir'],rel_path)
        with open(path,'r') as stream:
            control = yaml.full_load(stream)
        for k,v in control.items():
            self[k] = v
        self.check()
        
    def sbatch_ccr(
        self,slurm_path=None,config_path=None,python_script_path=None,
        module_list=None,submit=True,**kwargs
    ):
        if slurm_path is None:
            slurm_path = os.path.join(self['run_dir'],'run_job.sh')
        if config_path is None:
            config_path = os.path.join(self['run_dir'],'config.yml')
            self.logger.warning(f'saving current config dict to {config_path}')
            self.to_yaml()
        else:
            self.read_yaml(abs_path=config_path)
            self.logger.warning(f'current config overwritten by {config_path}')
        if python_script_path is None:
            python_script_path = os.path.join(os.path.split(self['run_dir'])[0],'train_chemfit.py')
            self.logger.warning(f'assuming python script at {python_script_path}')
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
            fid.write('#SBATCH --mem={:.0f}'.format(kwargs.pop('mem',50000)))
            fid.write('\n')
            fid.write('#SBATCH --gpus-per-node={:.0f}'.format(kwargs.pop('gpus',1)))
            fid.write('\n')
            fid.write('#SBATCH --job-name={}'.format(self['run_id']))
            fid.write('\n')
            fid.write('#SBATCH --output={}.out'.format(self['run_id']))
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
            fid.write(f'python3 {python_script_path} {config_path}')
            fid.write('\n')
        
        if submit:
            cwd = os.getcwd()
            os.chdir(self['run_dir'])
            os.system(f'sbatch {slurm_path}')
            run_dir = self['run_dir']
            self.logger.warning(f'{slurm_path} submitted! check {run_dir} for results')
            os.chdir(cwd)


class ChemFitDataset(Dataset):
    def __init__(
        self,l3s,mask=None,
        resample_rule=pd.Timedelta(days=28),resample_offset=pd.Timedelta(days=0),min_D=2,
        crop_x=64,crop_y=64,stride_x=11,stride_y=11,
        crop_fraction=0.5,initial_random_state=10,randomize_crops_per_frame=True,
        DD_scaling=1e9,VCD_scaling=1e4,WT_scaling=1e6,
        westmost=-128,eastmost=-65,southmost=24,northmost=50,
        base_year=2018,jitter_kw=None
    ):
        '''
        min_D:
            minimum averaged num_samples allowed in a frame in l3r
        '''
        self.logger = logging.getLogger(__name__)
        
        self.initial_random_state = initial_random_state
        self.random_state = initial_random_state
        
        self.crop_x,self.crop_y,self.stride_x,self.stride_y = crop_x,crop_y,stride_x,stride_y
        self.crop_fraction = crop_fraction
        
        self.DD_scaling,self.VCD_scaling,self.WT_scaling=\
        DD_scaling,VCD_scaling,WT_scaling
        self.base_year = base_year
        self.do_jitter = False
        self.jitter_kw = jitter_kw
        
        # spatial domain
        lonmesh,latmesh = np.meshgrid(l3s[0]['xgrid'],l3s[0]['ygrid'])
        xmesh = (lonmesh-westmost)/(eastmost-westmost)
        ymesh = (latmesh-southmost)/(northmost-southmost)
        grid_size = l3s[0].grid_size
        self.grid_size = torch.tensor(grid_size).float()
        self.westmost,self.eastmost,self.southmost,self.northmost=\
        westmost,eastmost,southmost,northmost
        self.xmesh = torch.tensor(xmesh).float()
        self.ymesh = torch.tensor(ymesh).float()
        self.lonmesh = lonmesh
        self.latmesh = latmesh
        
        # temporal aggregation
        l3r = l3s.resample(
                resample_rule,
                offset=resample_offset
        )[0] # resampled l3s, named as l3r
        time_mask = (np.array([np.nanmean(l['num_samples']) for l in l3r])>min_D)
        l3r = l3r.trim(time_mask=time_mask)
        start = l3r.df.index.to_timestamp(how='start')
        end = l3r.df.index.to_timestamp(how='end')
        mid = start + (end - start) / 2
        fy = mid.year + (mid.dayofyear - 1) / (365 + mid.is_leap_year.astype(int))
        l3r.df['fractional_year'] = fy
        l3r.df['fy_sin'] = np.sin(2*np.pi*(fy%1))
        l3r.df['fy_cos'] = np.cos(2*np.pi*(fy%1))
        l3r.df['fy_span'] = (end - start).total_seconds()/86400/365.25
        for l in l3r:
            for k,v in l.items():
                l[k] = torch.tensor(v).float()
            valid_mask = (~torch.isnan(l['column_amount_DD'])) &\
            (~torch.isnan(l['column_amount'])) &\
            (~torch.isnan(l['surface_altitude_DD'])) &\
            (l['num_samples'] > 0.7)
            l['valid_mask'] = valid_mask
            for k in ['column_amount_DD','column_amount','column_amount_DD_xy','column_amount_DD_rs','surface_altitude_DD']:
                if k in l.keys():
                    l[k][~valid_mask] = 0.
                else:
                    self.logger.warning(f'{k} not found to zero nans')
            l['total_sample_weight'][~valid_mask] = 0.
        self.l3r = l3r
        
        if all(['column_amount_DD_xy' in l.keys() for l in l3r]):
            self.logger.info('xy/rs available, random error calculated')
            self.do_random_error = True
        else:
            self.do_random_error = True
        
        self.nframe = len(l3r)
        self.frame_y,self.frame_x = len(l3r[0]['ygrid']),len(l3r[0]['xgrid'])
        
        self.set_mask(mask)
        
        self.random_crop(
            crop_x,crop_y,stride_x,stride_y,crop_fraction,
            randomize_crops_per_frame
        )
        
    @staticmethod
    def predict_ncrop(
        frame_y,frame_x,crop_y,crop_x,stride_y=1,stride_x=1,
        ensure_full_coverage=True
    ):
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
        self,crop_x=None,crop_y=None,stride_x=None,stride_y=None,crop_fraction=None,
        randomize_crops_per_frame=True
    ):
        if crop_x is None:
            crop_x = self.crop_x
        else:
            self.crop_x = crop_x
        if crop_y is None:
            crop_y = self.crop_y
        else:
            self.crop_y = crop_y
        if stride_x is None:
            stride_x = self.stride_x
        else:
            self.stride_x = stride_x
        if stride_y is None:
            stride_y = self.stride_y
        else:
            self.stride_y = stride_y
        if crop_fraction is None:
            crop_fraction = self.crop_fraction
        else:
            self.crop_fraction = crop_fraction
        
        ncrop_per_frame_max = self.predict_ncrop(
            self.frame_y,self.frame_x,
            crop_y,crop_x,stride_y,stride_x
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
        
        self.ncrop_per_frame = int(len(all_crops)*crop_fraction)
        self.logger.info(f'with crop fraction {crop_fraction}, there are {self.ncrop_per_frame} crops per frame')
        self.logger.info(f'there are {self.nframe} frames, total {len(self)} crops')
        selected_indices = np.empty((self.nframe,self.ncrop_per_frame),dtype=int)
        self.logger.info(f'random state is {self.random_state} before cropping')
        if randomize_crops_per_frame:
            for iframe in range(self.nframe):
                rng = np.random.RandomState(self.random_state)
                selected_indices[iframe,:] = rng.choice(
                    self.ncrop_per_frame_max,
                    size=self.ncrop_per_frame,
                    replace=False
                )
                self.random_state += 1
        else:
            rng = np.random.RandomState(self.random_state)
            si = rng.choice(
                    self.ncrop_per_frame_max,
                    size=self.ncrop_per_frame,
                    replace=False
                )
            for iframe in range(self.nframe):
                selected_indices[iframe,:] = si
            self.random_state += 1
        
        self.logger.info(f'random state is {self.random_state} after cropping')
        self.selected_indices = selected_indices

        self.all_crops = all_crops
        self.crop_x = crop_x
        self.crop_y = crop_y
    
    def set_jitter(self,TF,jitter_kw=None):
        jitter_kw = jitter_kw or self.jitter_kw
        self.jitter_kw = jitter_kw
        if jitter_kw is None:
            TF = False
        self.do_jitter = TF
    
    def set_mask(self,mask=None):
        if mask is None:
            self.logger.warning('No low-emission mask - assuming all!')
            self.mask = torch.ones((self.frame_y,self.frame_x),dtype=torch.bool)
        else:
            self.mask = torch.tensor(mask,dtype=torch.bool)
        # mask must match dimension of l3r
        if (self.frame_y,self.frame_x) == self.mask.shape:
            self.logger.info('assuming same mask for all frames')
        elif (self.nframe,self.frame_y,self.frame_x) == self.mask.shape:
            self.logger.info('assuming separate mask for each frame')
        else:
            self.logger.error('mask dimension incompatible with frames!!!')
        
    def __len__(self):
        return self.nframe*self.ncrop_per_frame
    
    def __getitem__(self,idx):
        # index of frame, from l3r
        frame_idx = idx // self.ncrop_per_frame
        # index of crop, from each frame
        crop_idx = idx % self.ncrop_per_frame
        i, j = self.all_crops[self.selected_indices[frame_idx]][crop_idx]
        
        l3 = self.l3r[frame_idx]
        # num_samples, flux, column, and wind topo term from resampled l3s (self.l3r)
        D = l3['num_samples'][
            i:i+self.crop_y,j:j+self.crop_x
        ].unsqueeze(0)
        DD = l3['column_amount_DD'][
            i:i+self.crop_y,j:j+self.crop_x
        ].unsqueeze(0)*self.DD_scaling
        
        if self.do_random_error:
            DD_xy = l3['column_amount_DD_xy'][
                i:i+self.crop_y,j:j+self.crop_x
            ].unsqueeze(0)*self.DD_scaling
            DD_rs = l3['column_amount_DD_rs'][
                i:i+self.crop_y,j:j+self.crop_x
            ].unsqueeze(0)*self.DD_scaling
        
        VCD = l3['column_amount'][
            i:i+self.crop_y,j:j+self.crop_x
        ].unsqueeze(0)*self.VCD_scaling
        WT = l3['surface_altitude_DD'][
            i:i+self.crop_y,j:j+self.crop_x
        ].unsqueeze(0)*self.WT_scaling
        # low-emission mask
        if self.mask.ndim == 2:
            fit_mask = self.mask[i:i+self.crop_y,j:j+self.crop_x].clone().unsqueeze(0)
        else:
            fit_mask = self.mask[frame_idx][i:i+self.crop_y,j:j+self.crop_x].clone().unsqueeze(0)
        # nan mask
        valid_mask = l3['valid_mask'][
            i:i+self.crop_y,j:j+self.crop_x
        ].unsqueeze(0)
        # spatial feature as normalized lon/lat mesh
        xmesh = self.xmesh[i:i+self.crop_y,j:j+self.crop_x].clone().unsqueeze(0)
        ymesh = self.ymesh[i:i+self.crop_y,j:j+self.crop_x].clone().unsqueeze(0)
        # temporal feature
        df = self.l3r.df
        if 'fd_sin' not in df.keys():
            temporal = torch.tensor(
                [
                    df['fractional_year'].iloc[frame_idx]-self.base_year,
                    df['fy_sin'].iloc[frame_idx],
                    df['fy_cos'].iloc[frame_idx]
                ]
            ).float()
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
            spatial=torch.cat([xmesh,ymesh],dim=0),temporal=temporal,
            fit_mask=fit_mask,valid_mask=valid_mask,D=D,DD=DD,VCD=VCD,WT=WT
        )
        if self.do_random_error:
            out['DD_xy'],out['DD_rs'] = DD_xy,DD_rs
            out['random_error'] = self.get_random_error(DD_xy,DD_rs,valid_mask,l3.grid_size)
        return out


class ConvBlock(nn.Module):
    """a standard convolutional block."""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.GroupNorm(8,out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.GroupNorm(8,out_channels)
        )

    def forward(self, x):
        return self.block(x)


class TemporalFiLM(nn.Module):
    """Temporal features modulate spatial features."""
    def __init__(self,temporal_dim,spatial_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(temporal_dim,spatial_dim*2),  # Outputs [gamma, beta]
            nn.SiLU()  # Smooth activation (optional)
        )
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self._init_weights()  # Initialize MLP to ~identity transform

    def _init_weights(self):
        # Initialize MLP to near-identity (gamma≈1, beta≈0)
        nn.init.normal_(self.mlp[0].weight, mean=0, std=0.01)  # Small weights
        nn.init.constant_(self.mlp[0].bias[:self.spatial_dim], 1.0)  # gamma ≈ 1
        nn.init.constant_(self.mlp[0].bias[self.spatial_dim:], 0.0)  # beta ≈ 0

    def forward(self,temporal,spatial):
        # temporal: [B, T], spatial: [B, C, H, W]
        gamma_beta = self.mlp(temporal)  # [B, C*2]
        gamma,beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        return spatial*gamma.unsqueeze(-1).unsqueeze(-1)+beta.unsqueeze(-1).unsqueeze(-1)


class UNet(nn.Module):
    '''U-Net that maps spatial feature (xmesh,ymesh) to k and X'''
    def __init__(
        self,out_channels=2,in_channels=2,feat_channels=[64,128,256],temporal_dim=3,
        force_positive=True
                ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.out_channels = out_channels

        # Encoder (Downsampling path)
        current_in_channels = in_channels
        for channel in feat_channels:
            self.downs.append(ConvBlock(current_in_channels,channel))
            current_in_channels = channel

        # Bottleneck
        self.bottleneck = ConvBlock(feat_channels[-1],feat_channels[-1]*2)
        
        # feature-wise linear modulation from time to space feature
        self.temporal_film = TemporalFiLM(temporal_dim,feat_channels[-1]*2)
        
        # Decoder (Upsampling path)
        for channel in reversed(feat_channels):
            self.ups.append(nn.ConvTranspose2d(channel*2,channel,kernel_size=2,stride=2))
            # * 2 because of skip connection concatenation
            self.ups.append(ConvBlock(channel*2,channel))
        
        if force_positive:
            self.final_conv = nn.Sequential(
                nn.Conv2d(feat_channels[0],out_channels,kernel_size=1),
                nn.Softplus()
            )
        else:
            self.final_conv = nn.Conv2d(feat_channels[0],out_channels,kernel_size=1)
    
    def forward(self,spatial,temporal=None):
        skip_connections = []
        x = spatial
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        if temporal is not None:
            x = self.temporal_film(temporal,x)
        
        skip_connections = skip_connections[::-1] # Reverse skip connections for upsampling
        
        # contranspose2d, concat, and conv
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # ConvTranspose2d
            skip_connection = skip_connections[idx//2]
            # Adjust size if necessary (due to potential rounding errors in pooling/upsampling)
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:],mode='bilinear',align_corners=False
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # ConvBlock
        
        return self.final_conv(x)


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], 
                                          dtype=torch.float32).view(1,1,3,3)
    
    def forward(self,unet_out,scale_weights=[1.0, 0.5, 0.25],channel_weights=1.):
        var_loss = 0
        
        B, C, H, W = unet_out.shape
        if np.isscalar(channel_weights):
            channel_weights = channel_weights*torch.ones(C)
        else:
            channel_weights = torch.tensor(channel_weights).float()
        channel_weights = channel_weights.expand(B,-1)
        if channel_weights.device != unet_out.device:
            channel_weights = channel_weights.to(unet_out.device)
        
        for i, weight in enumerate(scale_weights):
            # Downsample unet output
            scaled = F.avg_pool2d(unet_out, kernel_size=2**i)
            
            # Calculate Laplacian
            if self.gradient_kernel.device != scaled.device:
                self.gradient_kernel = self.gradient_kernel.to(scaled.device)
            gradients = F.conv2d(
                scaled, self.gradient_kernel.expand(C,1,3,3), 
                groups=C, padding=1)
            abs_variance = gradients.abs().mean()
            # [B,C]
            weighted_variance = gradients.abs().mean(dim=[-2,-1])*channel_weights
            weighted_variance = weighted_variance.mean()
            var_loss += weight * weighted_variance
                        
        return var_loss


class NegativityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,unet_out):
        return F.relu(-unet_out).mean()

    
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


class ChemFitTrainer:
    def __init__(
        self,model,lr=1e-4,weight_decay=0.01,device='auto',
        L1Loss_or_MSE='L1Loss',initial_grad_scaler=None
    ):
        self.logger = logging.getLogger(__name__)
        # Device detection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.logger.info(f'device is {self.device.type}')
        
        self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),lr=lr,weight_decay=weight_decay)
        
        if L1Loss_or_MSE == 'MSE':
            self.data_loss_func = MaskedLoss(loss_func=nn.MSELoss(reduction='none'))
        else:
            self.data_loss_func = MaskedLoss()
        
        # Only enable mixed precision on CUDA
        initial_grad_scaler=initial_grad_scaler or 1024.
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=initial_grad_scaler,
            enabled=(self.device.type == 'cuda'))
        
    def inference(self,ds,unet_out_channels=2):
        '''perform inference on a dataset (ds)'''
        nframe = ds.nframe
        ncrop_per_frame = ds.ncrop_per_frame
        crop_x,crop_y,frame_x,frame_y=ds.crop_x,ds.crop_y,ds.frame_x,ds.frame_y
        # crop each frame into a batch
        loader = DataLoader(
            ds,batch_size=ncrop_per_frame,shuffle=False,drop_last=True
        )
        # mosaicked unet output to frame dimension
        unet_out = np.zeros((nframe,unet_out_channels,frame_y,frame_x))
                
        self.model.eval()
        with torch.no_grad():
            # exactly one batch per frame
            for iframe,batch in enumerate(loader):
                batch = self._to_device(batch)

                out = self.model(
                    spatial=batch['spatial'],
                    temporal=batch['temporal']
                )
                # B, unet_out_channels, crop_y, crop_x
                D_unet = np.zeros((unet_out_channels,frame_y,frame_x))
                for icrop in range(ncrop_per_frame):
                    # unet_out_channels, crop_y, crop_x
                    unet_out_crop = out[icrop].cpu().detach().numpy()
                    i, j = ds.all_crops[ds.selected_indices[iframe]][icrop]
                    unet_out[
                        iframe,:,i:i+crop_y,j:j+crop_x
                    ] += unet_out_crop
                    D_unet[
                        :,i:i+crop_y,j:j+crop_x
                    ] += np.ones(unet_out_crop.shape)
                unet_out[iframe,] /= D_unet
        return unet_out
        
    def load_data(self,dataset,batch_size=32,shuffle=True):
        self.data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=(self.device.type=='cuda')
        )
    
    def _to_device(self,data):
        return {k: v.to(self.device, non_blocking=(self.device.type == 'cuda')) 
                for k, v in data.items()}
    
    def save_model(self,path,**kwargs):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
        }, path)
    
    def validate(self,val_loader,scale_random_error=True):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                unet_out = self.model(
                    spatial=batch['spatial'],
                    temporal=batch['temporal']
                )
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                # B,1,H,W
                predict = (
                    unet_out*torch.cat([batch['VCD'],batch['WT']],dim=1)
                ).sum(dim=1,keepdim=True)/error_scaler
                target = -batch['DD']/error_scaler
                
                loss = self.data_loss_func(
                    predict=predict,target=target,
                    mask=batch['valid_mask']*batch['fit_mask']
                )
                total_loss += loss.item()
            
        return total_loss/len(val_loader)
    
    def train_epoch(
        self,
        epoch,
        smoothness_weight=1e-3,
        smoothness_scale_weights=[1,0.5,0.25],
        smoothness_channel_weights=1.,
        negativity_weight=None,
        max_norm=1.0,
        scale_random_error=True
    ):
        self.model.train()
        
        if smoothness_weight is not None:
            smoothness_loss_func = SmoothnessLoss()
        
        if negativity_weight is not None:
            negativity_loss_func = NegativityLoss()
        
        total_loss = 0
        total_var_loss = 0
        total_neg_loss = 0
        start_time = time.time()
        num_batches = len(self.data_loader)
        norms = []
        for batch_idx, batch in enumerate(self.data_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            
            # Only use autocast on CUDA
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                unet_out = self.model(
                    spatial=batch['spatial'],
                    temporal=batch['temporal']
                )
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                # B,1,H,W
                predict = (
                    unet_out*torch.cat([batch['VCD'],batch['WT']],dim=1)
                ).sum(dim=1,keepdim=True)/error_scaler
                target = -batch['DD']/error_scaler
                
                loss = self.data_loss_func(
                    predict=predict,target=target,
                    mask=batch['valid_mask']*batch['fit_mask']
                )
                if smoothness_weight is not None:
                    smoothness_loss = smoothness_loss_func(
                        unet_out,
                        scale_weights=smoothness_scale_weights,
                        channel_weights=smoothness_channel_weights
                    )
                    loss += smoothness_weight*smoothness_loss
                else:
                    smoothness_loss = torch.tensor(0.)
                    
                if negativity_weight is not None:
                    negativity_loss = negativity_loss_func(unet_out)
                    loss += negativity_weight*negativity_loss
                else:
                    negativity_loss = torch.tensor(0.)
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=max_norm)
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
                        return batch
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_var_loss += smoothness_loss.item()
            total_neg_loss += negativity_loss.item()
        
        # Epoch summary
        epoch_time = time.time() - start_time
        self.logger.warning(f'\rEpoch {epoch + 1} | Time: {epoch_time:.1f}s | Loss: {total_loss/num_batches:.4f} | Sample: {self.data_loader.batch_size}x{num_batches}')
        result = dict(
            train_loss=total_loss/num_batches,
            smooth_loss=total_var_loss/num_batches,
            epoch_time=epoch_time,
            sample_size=len(self.data_loader.dataset),
            grad_norm=np.mean(norms)
        )
        return result