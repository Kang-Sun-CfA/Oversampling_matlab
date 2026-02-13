import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ChemFitDataset(Dataset):
    def __init__(
        self,l3s,mask=None,
        resample_rule='28d',resample_offset='0d',min_D=2,
        crop_x=64,crop_y=64,stride_x=11,stride_y=11,
        crop_fraction=0.5,initial_random_state=10,randomize_crops_per_frame=True,
        DD_scaling=1e9,VCD_scaling=1e6,WT_scaling=1e6,
        westmost=-128,eastmost=-65,southmost=24,northmost=50,
        base_year=2018,jitter_kw=None
    ):
        self.logger = logging.getLogger(__name__)
        
        self.initial_random_state = initial_random_state
        self.random_state = initial_random_state
        
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
        for l in l3r:
            for k,v in l.items():
                l[k] = torch.tensor(v).float()
        self.l3r = l3r
        
        if all(['column_amount_DD_xy' in l.keys() for l in l3r]):
            self.logger.info('xy/rs available, random error calculated')
            self.do_random_error = True
        else:
            self.do_random_error = True
        
        self.nframe = len(l3r)
        self.frame_y,self.frame_x = len(l3r[0]['ygrid']),len(l3r[0]['xgrid'])
        if mask is None:
            self.logger.warning('No low-emission mask - assuming all!')
            self.mask = torch.ones((self.frame_y,self.frame_x),dtype=torch.bool)
        else:
            self.mask = torch.tensor(mask,dtype=torch.bool)
        # mask must have the same shape as all l3
        assert self.frame_y,self.frame_x == self.mask.shape
        self.ncrop_per_frame_max = self.predict_ncrop(
            len(l3r[0]['ygrid']),len(l3r[0]['xgrid']),
            crop_y,crop_x,stride_y,stride_x
        )
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
    
    def random_crop(
        self,crop_x,crop_y,stride_x,stride_y,crop_fraction,
        randomize_crops_per_frame=True
    ):
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
        selected_indices = np.empty((self.nframe,self.ncrop_per_frame),dtype=int)
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
        fit_mask = self.mask[i:i+self.crop_y,j:j+self.crop_x].clone().unsqueeze(0)
        # nan mask
        nan_mask = (~torch.isnan(DD)) & (D >= .7)
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
            jitter_fy = torch.rand(1) < self.jitter_kw['p_jitter']
            if jitter_fy and 'fy_span' in self.jitter_kw.keys():
                temporal[0] += (torch.rand(1)[0]-0.5)*self.jitter_kw['fy_span']
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
            fit_mask=fit_mask,nan_mask=nan_mask,D=D,DD=DD,VCD=VCD,WT=WT
        )
        if self.do_random_error:
            out['DD_xy'],out['DD_rs'] = DD_xy,DD_rs
        return out