import numpy as np
import pandas as pd
import random
import datetime as dt
import time
import sys,os,glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def subset_list(l,idxs):
    return [l[idx] for idx in idxs]

class EmissionDataset(Dataset):
    def __init__(
        self,emissions,years,fy,fy_sin,fy_cos,cdls,fd=None,emission_func=None,
        westmost=-128,eastmost=-65,southmost=24,northmost=50,base_year=2018,
        crop_size_x=128,crop_size_y=64,stride=16,crop_fraction=0.5,random_state=10,
        cdl_name_codes=None,epoch=0,stride_x=None,stride_y=None,grid_size=None,
        all_source_coords=None,jitter_kw=None
    ):
        '''
        emissions,years,fy,fy_sin,fy_cos:
            lists of the same length. each sample specify the emission image, year, fractional year,
            sin/cos of fractional year
        fd:
            fractional day. if provided, fd_sin and fd_cos will be added to the time feature
        cdls:
            list of CDL objects with length equal to the number of unique years
        emission_func:
            scale emissions by this. default *1e9, i.e., mol/m2/s to nmol/m2/s
        w/e/s/nmost:
            lon/lat bounds to normalize lon/lat to ~0-1
        base_year:
            years will be zeroed at this year
        crop_size_x/y:
            emissions will be cropped to samples of this W (x, lon) and H (y, lat)
        stride(_x/y):
            cropping is skipped by this step size
        crop_fraction:
            the fraction of all possible crops to be returned
        random_state:
            for reproducibility in crop selection
        cdl_name_codes:
            select and/or group cdl types, e.g., [('my_corn'),[1]] uses cdl index 1 as 'my_corn',
            1 - selected fractions are categorized as other fraction
        epoch:
            current epoch integer
        grid_size:
            if not given, infer from cdls[0].lonmesh
        all_source_coords:
            an n_source x 2 array listing the lon and lat of point sources
        jitter_kw:
            a dict of fy_span, and p_jitter. with a probability of p_jitter, the temporal
            feature will be uniformly sampled from fy+/-(fy_span/2)
        '''
        stride_x = stride_x or stride
        stride_y = stride_y or stride
        self.set_epoch(epoch)
        self.emission_func = emission_func or (lambda x:x*1e9)
        self.base_year = base_year
        self.do_jitter = False
        self.jitter_kw = jitter_kw
        
        if all_source_coords is not None:
            asc = torch.tensor(all_source_coords).float()
            asc[:,0] = (asc[:,0]-westmost)/(eastmost-westmost)
            asc[:,1] = (asc[:,1]-southmost)/(northmost-southmost)
            self.all_source_coords = asc
        
        xmesh = (cdls[0].lonmesh-westmost)/(eastmost-westmost)
        ymesh = (cdls[0].latmesh-southmost)/(northmost-southmost)
        grid_size = grid_size or np.median(np.abs(np.diff(cdls[0].lonmesh[0,:])))
        # grid inverse area, m-2 * 1e9 nmol/mol
        self.gia = torch.tensor(
            self.emission_func(
                1/(np.cos(np.deg2rad(cdls[0].latmesh))*np.square(grid_size*111e3))
            )
        ).float()
        self.grid_size = torch.tensor(grid_size).float()
        self.westmost,self.eastmost,self.southmost,self.northmost=westmost,\
        eastmost,southmost,northmost
        self.xmesh = torch.tensor(xmesh).float()
        self.ymesh = torch.tensor(ymesh).float()
        self.lonmesh = cdls[0].lonmesh
        self.latmesh = cdls[0].latmesh
        self.grid_size_in_m2 = np.cos(np.deg2rad(self.latmesh))*np.square(grid_size*111e3)
        
        if cdl_name_codes is None:
            cdl_name_codes = [
                ('corn',[1]),
                ('soybean',[5]),
                ('dev_o',[121]),
                ('dev_lm',[122,123]),
                ('dev_h',[124]),
                ('wetland',[190,195]),
                ('forest',[141,143,142])
            ]
        cdl_years = np.sort(np.unique(years))
        fracs = []
        for cdl in cdls:
            fracs_year = np.array([
                np.sum(cdl.fractions[...,np.isin(cdl.code,cc[1])],axis=-1) for cc in cdl_name_codes
            ])
            fracs_year[fracs_year<0] = 0.
            other_frac_year = 1-np.sum(fracs_year,axis=0,keepdims=True)
            other_frac_year[other_frac_year<0] = 0.
            fracs_year = np.concatenate((fracs_year,other_frac_year),axis=0)
            fracs.append(torch.tensor(fracs_year).float())
        
        self.cdl_name_codes = cdl_name_codes
        
        self.emissions = emissions
        self.fracs = fracs
        self.years = years
        self.fy = fy
        self.fy_sin = fy_sin
        self.fy_cos = fy_cos
        self.cdl_years = cdl_years
        if fd is not None:
            if np.isscalar(fd):
                fd = np.ones_like(fy)*fd
            self.fd_sin = np.sin(2*np.pi*fd)
            self.fd_cos = np.cos(2*np.pi*fd)
            self.fd = fd
        
        self.random_crop(
            crop_size_x=crop_size_x,crop_size_y=crop_size_y,stride=stride,stride_x=stride_x,stride_y=stride_y,
            crop_fraction=crop_fraction,random_state=random_state,epoch=epoch)
        
    def set_jitter(self,TF,jitter_kw=None):
        jitter_kw = jitter_kw or self.jitter_kw
        self.jitter_kw = jitter_kw
        if jitter_kw is None:
            TF = False
        self.do_jitter = TF
        
    def random_crop(
        self,crop_size_x=128,crop_size_y=64,stride=16,crop_fraction=0.5,
        random_state=10,epoch=None,stride_x=None,stride_y=None
    ):
        stride_x = stride_x or stride
        stride_y = stride_y or stride
        epoch = epoch or self.epoch
        self.set_epoch(epoch)
        if random_state is not None:
            random_state = random_state+self.epoch
        
        rng = np.random.RandomState(random_state)
        
        H,W = self.emissions[0].shape
        
        jj = np.arange(0,W-crop_size_x+1,stride_x,dtype=int)
        if not np.isin(W-crop_size_x,jj):
            jj = np.append(jj,W-crop_size_x)
            
        ii = np.arange(0,H-crop_size_y+1,stride_y,dtype=int)
        if not np.isin(H-crop_size_y,ii):
            ii = np.append(ii,H-crop_size_y)

        all_crops = [(i,j) for i in ii for j in jj]

        selected_indices = rng.choice(
            len(all_crops),
            size=int(len(all_crops)*crop_fraction),
            replace=False
        )

        self.selected_crops = [all_crops[i] for i in selected_indices]
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
    
    def set_epoch(self,epoch):
        self.epoch = epoch
    
    def __len__(self):
        return len(self.emissions)*len(self.selected_crops)
    
    def __getitem__(self,idx):
        img_idx = idx // len(self.selected_crops)
        crop_idx = idx % len(self.selected_crops)
        i, j = self.selected_crops[crop_idx]

        emission = self.emissions[img_idx][i:i+self.crop_size_y,j:j+self.crop_size_x
                                          ].unsqueeze(0)
        year = self.years[img_idx]
        fy = self.fy[img_idx]
        fy_sin = self.fy_sin[img_idx]
        fy_cos = self.fy_cos[img_idx]
        frac = self.fracs[np.where(self.cdl_years==year)[0][0]
                         ][:,i:i+self.crop_size_y,j:j+self.crop_size_x]
        if hasattr(self,'fd_sin'):
            fd_sin = self.fd_sin[img_idx]
            fd_cos = self.fd_cos[img_idx]
            temporal = torch.tensor(
                np.array(
                    [
                        year+fy-self.base_year,
                        fy_sin,
                        fy_cos,
                        fd_sin,
                        fd_cos
                    ]
                ),dtype=torch.float32
            )
        else:
            temporal = torch.tensor(
                np.array(
                    [
                        year+fy-self.base_year,
                        fy_sin,
                        fy_cos
                    ]
                ),dtype=torch.float32
            )
        
        xmesh = self.xmesh[i:i+self.crop_size_y,j:j+self.crop_size_x].clone().unsqueeze(0)
        ymesh = self.ymesh[i:i+self.crop_size_y,j:j+self.crop_size_x].clone().unsqueeze(0)
        
        # temporal/spatial augmentation
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
            jitter_xy = torch.rand(1) < self.jitter_kw['p_jitter']
            if jitter_xy and 'xy_span' in self.jitter_kw.keys():
                x_span = self.jitter_kw['xy_span']*self.grid_size
                x_span = x_span/(self.eastmost-self.westmost)
                jittered_x = torch.randn_like(xmesh)*x_span
                xmesh += jittered_x
                y_span = self.jitter_kw['xy_span']*self.grid_size
                y_span = y_span/(self.northmost-self.southmost)
                jittered_y = torch.randn_like(ymesh)*y_span
                ymesh += jittered_y
        
        gia = self.gia[i:i+self.crop_size_y,j:j+self.crop_size_x]
        
        return {
            'emission':self.emission_func(emission),
            'xmesh':xmesh,
            'ymesh':ymesh,
            'frac':frac,
            'temporal':temporal,
            'gia':gia,
            'grid':self.grid_size
        }
    
    def plot(self,sample_idxs=np.arange(5),figsize=(10,10),axis_off=False):
        nsample = len(sample_idxs)
        fig,axss = plt.subplots(nsample,4,figsize=figsize,constrained_layout=True)
        for idx,axs in zip(sample_idxs,axss):
            sample = self[idx]
            xmsh = sample['xmesh'].squeeze()*(self.eastmost-self.westmost)+self.westmost
            ymsh = sample['ymesh'].squeeze()*(self.northmost-self.southmost)+self.southmost
            for icol,(ax,b,cmap) in enumerate(zip(
                axs,
                [sample['emission'].squeeze(),sample['frac'][0],sample['frac'][2],sample['frac'][4]],
                ['rainbow','viridis','viridis','viridis'])):
                ax.pcolormesh(xmsh,ymsh,b,cmap=cmap,shading='auto')
                
                if axis_off:
                    ax.set_axis_off()


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

class LandTypeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.channels = channels
        
    def forward(self, emissions):
        """
        Returns a loss that should be MINIMIZED during training.
        Lower loss = stronger self-attention (desired)
        """
        # emissions: [B,8,H,W]
        B,C,H,W = emissions.shape
        assert C == self.channels
        q = self.query(emissions).flatten(-2)  # [B,8,H*W]
        k = self.key(emissions).flatten(-2)    # [B,8,H*W]
        
        # Attention matrix [B,8,8]
        attn = F.softmax(q @ k.transpose(-1,-2), dim=-1)
        
        # Encourage strong self-attention (diagonal) 
        # and weak cross-attention (off-diagonal)
        identity = torch.eye(C, device=emissions.device).unsqueeze(0).expand(B,-1,-1)  # [B,8,8]
        return F.mse_loss(attn, identity)  # Penalize deviation from identity matrix


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
    
    def _create_kernels(self,grid_size,batch_size=None):
        '''Create kernels for batched grid_size
        grid_size: 
            [B,] tensor of grid sizes
        batch_size:
            should be just B
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
        kernels = self._create_kernels(grid_size,B)
        
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


class PointSourceDistributor(nn.Module):
    def __init__(self,all_source_coords):
        '''
        all_source_coords:
            [n_sources,2], lon/lat of all point sources, normalized to ~0-1
        '''
        super().__init__()
        self.n_sources = len(all_source_coords)
        
        # Store the actual source coordinates
        self.register_buffer('all_source_coords', all_source_coords)  # [n_sources, 2]
        
    def forward(self,point_rates,spatial,gia):
        '''
        point_rates:
            [B, n_sources], point source emission rates, output from PointSourceMLP,
            unit mol s-1
        spatial:
            [B,C=2 (lon,lat),H,W]
        gia:
            [B, H, W], grid inverse area, returned from a dataset sample, unit 
            m-2 nmol mol-1
        returns:
            [B, 1, H, W], emission from all point sources in the view
        '''
        B,H,W = gia.shape
        
        spatial_field = torch.zeros(B,1,H,W,device=point_rates.device)
        
        for b in range(B):
            xmin,xmax = spatial[b,0].min(),spatial[b,0].max()
            ymin,ymax = spatial[b,1].min(),spatial[b,1].max()
            
            in_view_mask = (
                (self.all_source_coords[:,0] >= xmin) & 
                (self.all_source_coords[:,0] <= xmax) &
                (self.all_source_coords[:,1] >= ymin) & 
                (self.all_source_coords[:,1] <= ymax)
            )
            
            sources_in_view = torch.where(in_view_mask)[0]
            
            if len(sources_in_view) > 0:
                view_coords = self.all_source_coords[sources_in_view]
                normalized_x = (view_coords[:,0] - xmin) / (xmax-xmin)
                normalized_y = (view_coords[:,1] - ymin) / (ymax-ymin)
                
                pixel_x = (normalized_x * (W - 1)).round().long().clamp(0, W-1)
                pixel_y = (normalized_y * (H - 1)).round().long().clamp(0, H-1)
                
                # Get emissions for sources in view, nmol m-2 s-1
                emissions_in_view = point_rates[b,sources_in_view]*\
                gia[b,pixel_y,pixel_x]
                
                # Scatter to spatial field
                spatial_field[b, 0, pixel_y, pixel_x] += emissions_in_view
        
        return spatial_field


class PointSourceMLP(nn.Module):
    def __init__(self, all_source_coords, temporal_dim=3, hidden_dim=64):
        '''
        all_source_coords:
            [n_sources,2], lon/lat of all point sources, normalized to ~0-1
        '''
        super().__init__()
        self.n_sources = len(all_source_coords)
        self.temporal_dim = temporal_dim
        
        # Store the actual source coordinates
        self.register_buffer('all_source_coords', all_source_coords)  # [n_sources, 2]
        
        # MLP that processes [lon, lat, temporal_features] for any point source
        self.mlp = nn.Sequential(
            nn.Linear(2 + temporal_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Emissions must be positive
        )
    
    def forward(self, temporal):
        '''
        temporal: 
            [B, temporal_dim]
        returns: 
            [B, n_sources], emission rates for all sources
        '''
        B = temporal.shape[0]
        temporal_dim = self.temporal_dim
        # Batched processing for efficiency
        coords_expanded = self.all_source_coords.unsqueeze(0).expand(B, -1, -1)  # [B, n_sources, 2]
        temporal_expanded = temporal.unsqueeze(1).expand(-1, self.n_sources, -1)  # [B, n_sources, temporal_dim]
        mlp_input = torch.cat([coords_expanded, temporal_expanded], dim=-1)  # [B, n_sources, 2 + temporal_dim]
        
        # Process all sources in parallel
        mlp_input_flat = mlp_input.view(-1, 2 + temporal_dim)  # [B * n_sources, 2 + temporal_dim]
        emissions_flat = self.mlp(mlp_input_flat)  # [B * n_sources, 1]
        rates = emissions_flat.view(B, self.n_sources)  # [B, n_sources]
        
        return rates


class LandTypeEmissionModel(nn.Module):
    """UNet-like module for generating emission contribution for a single land type."""
    def __init__(
        self,n_land_types,in_channels=2,feat_channels=[64,128,256],
        temporal_dim=3,psf_kernel_size=5,do_ConvTranspose=True,do_attention=False,
        all_source_coords=None,point_hidden_dim=64
                ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.n_land_types = n_land_types

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
            if do_ConvTranspose:
                # basic ConvTranspose2d
                self.ups.append(nn.ConvTranspose2d(channel*2,channel,kernel_size=2,stride=2))
            else:
                # upsample+conv2d
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(channel*2, channel, kernel_size=3, padding=1)
                ))
            # * 2 because of skip connection concatenation
            self.ups.append(ConvBlock(channel*2,channel))
            
        self.final_conv = nn.Conv2d(feat_channels[0],n_land_types,kernel_size=1)
        self.psf = SuperGaussianPSF(kernel_size=psf_kernel_size)
        
        if all_source_coords is not None:
            self.do_point = True
            self.point_mlp = PointSourceMLP(
                all_source_coords,temporal_dim=temporal_dim,hidden_dim=point_hidden_dim)
            self.point_distributor = PointSourceDistributor(all_source_coords)
        else:
            self.do_point = False
        
        self.do_attention = do_attention
        if do_attention:
            self.attention = LandTypeAttention(n_land_types)
    
    def get_unet_outcome(self,spatial,temporal=None):
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
    
    def forward(self,spatial,frac,grid,temporal=None,gia=None):
        '''
        spatial:
            B,C=2 (lon,lat),H=crop_size_y,W=crop_size_x
        frac:
            B,C=8 (corn, soybean, ...),H=crop_size_y,W=crop_size_x
        grid:
            B, grid size
        temporal:
            B,C=temporal_dim, (year, sin(fy), cos(fy)) if 3, add sin(fd), cos(fd) for 5
        gia:
            B,H,W, grid inverse area
        '''
        emissions = self.get_unet_outcome(spatial,temporal)
        area_source = (emissions*frac).sum(dim=1,keepdim=True)
        
        if self.do_point:
            point_rates = self.point_mlp(temporal)
            point_source = self.point_distributor(point_rates,spatial,gia)
            all_source = area_source + point_source
        else:
            all_source = area_source
        
        predict = self.psf(x=all_source,grid_size=grid)
        out = dict(emissions=emissions,predict=predict)
        if self.do_attention:
            attn_loss = self.attention(emissions)
            out['attn_loss'] = attn_loss
        return out


class MaskedLoss(nn.Module):
    def __init__(self,loss_func=None):
        super().__init__()
        self.loss_func = loss_func or nn.L1Loss(reduction='none')  # Compute per-element L1

    def forward(self, predict, target):
        # Create a mask where target is not NaN
        mask = ~torch.isnan(target)
        
        # Compute L1 loss only for valid (non-NaN) pixels
        loss = self.loss_func(predict, torch.nan_to_num(target)) * mask.float()
        
        # Normalize by the number of valid pixels (avoid div-by-zero)
        valid_pixels = mask.sum().float()
        return loss.sum() / (valid_pixels + 1e-6)  # Add epsilon for stability


def augment_temporal(temporal_features, 
                     max_year_shift=0.3, 
                     max_doy_shift=30,
                     p_augment=0.8):
    """
    Args:
        temporal_features: [B, 3] tensor where columns are:
                          0: normalized year (e.g., 2022.5 - base_year)
                          1: sin(doy)
                          2: cos(doy)
        max_year_shift: Maximum year shift (± years)
        max_doy_shift: Maximum day-of-year shift (± days)
        p_augment: Probability of applying augmentation
    Returns:
        Augmented temporal features [B, 3]
    """
    B, _ = temporal_features.shape
    device = temporal_features.device
    
    # Initialize augmentation masks
    augment_mask = torch.rand(B, device=device) < p_augment
    
    if not augment_mask.any():
        return temporal_features
    
    # Year shift (linear)
    year_shifts = (torch.rand(B, device=device) * 2 - 1) * max_year_shift  # [-max, +max]
    year_shifts = year_shifts * augment_mask.float()
    augmented_years = temporal_features[:, 0] + year_shifts
    
    # DOY shift (circular)
    doy_shifts = (torch.rand(B, device=device) * 2 - 1) * max_doy_shift  # [-max, +max] days
    doy_shifts = doy_shifts * augment_mask.float()
    
    # Convert original doy to angle
    original_doy_rad = torch.atan2(temporal_features[:, 1], temporal_features[:, 2])  # [B]
    
    # Apply shift (convert days to radians)
    new_doy_rad = original_doy_rad + doy_shifts * (2 * torch.pi / 365.25)
    
    # Reconstruct periodic features
    augmented_sin_doy = torch.sin(new_doy_rad)
    augmented_cos_doy = torch.cos(new_doy_rad)
    
    # Combine augmented and original features
    result = temporal_features.clone()
    result[augment_mask, 0] = augmented_years[augment_mask]
    result[augment_mask, 1] = augmented_sin_doy[augment_mask]
    result[augment_mask, 2] = augmented_cos_doy[augment_mask]
    
    return result


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], 
                                          dtype=torch.float32).view(1,1,3,3)
        
    
    def forward(self,emissions,scale_weights=[1.0, 0.5, 0.25],channel_weights=1.):
        abs_loss = 0
        rel_loss = 0
        B, C, H, W = emissions.shape
        if np.isscalar(channel_weights):
            channel_weights = channel_weights*torch.ones(C)
        channel_weights = channel_weights.expand(B,-1)
        
        for i, weight in enumerate(scale_weights):
            # Downsample emission map
            scaled = F.avg_pool2d(emissions, kernel_size=2**i)
            
            # Calculate Laplacian
            if self.gradient_kernel.device != scaled.device:
                self.gradient_kernel = self.gradient_kernel.to(scaled.device)
            gradients = F.conv2d(
                scaled, self.gradient_kernel.expand(C,1,3,3), 
                groups=C, padding=1)
            abs_variance = gradients.abs().mean()
            # [B,C]
            rel_variance = (
                gradients.abs().mean(dim=[-2,-1])/(scaled.abs().mean(dim=[-2,-1])+1e-6)
            )*channel_weights
            rel_variance = rel_variance.mean()
            abs_loss += weight * abs_variance
            rel_loss += weight * rel_variance
            
        return abs_loss, rel_loss


class Trainer:
    def __init__(self,model,loss_func=MaskedLoss(),lr=1e-4,weight_decay=0.01,device='auto'):
        # Device detection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # parameters other than psf and point mlp
        other_params = [
            p for n, p in self.model.named_parameters()
            if (not any(k in n for k in ['log_wx', 'log_wy', 'log_kx', 'log_ky'])) and ('point_mlp' not in n)
        ]
        
        # parameters for point mlp
        point_params = [
            p for n,p in self.model.named_parameters()
            if 'point_mlp' in n
        ]

        # PSF-specific parameters
        psf_width_params = [model.psf.log_wx, model.psf.log_wy]
        psf_shape_params = [model.psf.log_kx, model.psf.log_ky]

        # Configure optimizer with different learning rates/weight decay
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': point_params, 'lr': lr*50, 'weight_decay': weight_decay},
            {'params': psf_width_params, 'lr': lr*50, 'weight_decay': 0},
            {'params': psf_shape_params, 'lr': lr*10, 'weight_decay': 0}
        ])
        
        self.loss_func = loss_func
        # Only enable mixed precision on CUDA
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def load_model(self, path, load_optimizer=True, strict=True):
        """
        Load a previously saved model checkpoint
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            strict: Whether to strictly enforce state_dict matching
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model = self.model.to(self.device)
        # Load training history
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        if 'val_history' in checkpoint:
            self.val_history = checkpoint['val_history']
            
#         print(f"Loaded model from epoch {self.epoch}")
        return checkpoint
    
    def inference(self,ds):
        '''perform inference on an EmissionDataset instance (ds)'''
        n_land_type = ds[0]['frac'].shape[0]
        nframe = len(ds.emissions)
        ncrop = len(ds.selected_crops)
        crop_size_x = ds.crop_size_x
        crop_size_y = ds.crop_size_y
        # make sure each batch represent a frame
        test_loader = DataLoader(ds,batch_size=ncrop,shuffle=False,drop_last=True)

        pure_emissions = np.zeros((nframe,n_land_type)+ds.xmesh.shape)
        fracs = np.zeros_like(pure_emissions)
        pred_emissions = np.zeros((nframe,)+ds.xmesh.shape)
        point_emissions = np.zeros((nframe,ds.all_source_coords.shape[0]))

        self.model.eval()
        with torch.no_grad():
            # exactly one batch per frame
            for iframe,batch in enumerate(test_loader):
                batch = self._to_device(batch)

                out = self.model(
                    spatial=torch.cat([batch['xmesh'],batch['ymesh']],dim=1),
                    frac=batch['frac'],
                    grid=batch['grid'],
                    temporal=batch['temporal'],
                    gia=batch['gia']
                )
                # populate area emissions for this frame
                D_pure = np.zeros((n_land_type,)+ds.xmesh.shape)
                D_pred = np.zeros(ds.xmesh.shape)
                for icrop in range(ncrop):
                    E_pure = out['emissions'][icrop].cpu().detach().numpy()
                    E_pred = out['predict'][icrop][0].cpu().detach().numpy()
                    F_pure = batch['frac'][icrop].cpu().detach().numpy()
                    ij = ds.selected_crops[icrop]
                    pure_emissions[
                        iframe,:,ij[0]:ij[0]+crop_size_y,ij[1]:ij[1]+crop_size_x
                    ] += E_pure
                    fracs[
                        iframe,:,ij[0]:ij[0]+crop_size_y,ij[1]:ij[1]+crop_size_x
                    ] += F_pure
                    pred_emissions[
                        iframe,ij[0]:ij[0]+crop_size_y,ij[1]:ij[1]+crop_size_x
                    ] += E_pred

                    D_pure[
                        :,ij[0]:ij[0]+crop_size_y,ij[1]:ij[1]+crop_size_x
                    ] += np.ones(E_pure.shape)
                    D_pred[
                        ij[0]:ij[0]+crop_size_y,ij[1]:ij[1]+crop_size_x
                    ] += np.ones(E_pred.shape)

                pure_emissions[iframe,] /= D_pure
                fracs[iframe,] /= D_pure
                pred_emissions[iframe,] /= D_pred

                point_rates = self.model.point_mlp(batch['temporal'])
                point_emissions[iframe,:] = point_rates[0].cpu()

        land_ER = np.sum(
            pure_emissions*fracs*np.broadcast_to(
                ds.grid_size_in_m2[np.newaxis,np.newaxis,...],fracs.shape
            ),axis=(-1,-2)
        )
        land_A = np.sum(
            fracs*np.broadcast_to(
                ds.grid_size_in_m2[np.newaxis,np.newaxis,...],fracs.shape
            ),axis=(-1,-2)
        )
        land_E = land_ER/land_A
        return dict(
            pure_emissions=pure_emissions,
            fracs=fracs,
            pred_emissions=pred_emissions,
            point_ER=point_emissions,
            land_E=land_E
        )
        
    def load_data(self,train_data,val_data=None,batch_size=32,shuffle=True):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, 
                                     pin_memory=(self.device.type == 'cuda'))
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                                   pin_memory=(self.device.type == 'cuda')) if val_data else None
    
    def _to_device(self,data):
        return {k: v.to(self.device, non_blocking=(self.device.type == 'cuda')) 
                for k, v in data.items()}
    
    def train_epoch(
        self,
        epoch,
        attn_weight=1e-3,
        temporal_kw=None,
        smooth_weight=1e-3,
        relative_variance_channel_weights=1.,
        verbose=False
    ):
        self.model.train()
        total_loss = 0
        total_absvar_loss = 0
        total_relvar_loss = 0
        start_time = time.time()
        # Simple progress tracking
        num_batches = len(self.train_loader)
        if verbose:
            print(f"Epoch {epoch + 1} [", end='', flush=True)
        if smooth_weight is not None:
            smooth_loss = SmoothnessLoss()
        for batch_idx, batch in enumerate(self.train_loader):
            if temporal_kw is not None:
                batch['temporal'] = augment_temporal(batch['temporal'],**temporal_kw)
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            
            # Only use autocast on CUDA
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                out = self.model(
                    spatial=torch.cat([batch['xmesh'],batch['ymesh']],dim=1),
                    frac=batch['frac'],
                    grid=batch['grid'],
                    temporal=batch['temporal'],
                    gia=batch['gia']
                )
                loss = self.loss_func(out['predict'],batch['emission'])
                if smooth_weight is not None:
                    absvar_loss,relvar_loss = smooth_loss(
                        out['emissions'],channel_weights=relative_variance_channel_weights)
                    loss += smooth_weight*(absvar_loss+relvar_loss)
                else:
                    absvar_loss,relvar_loss = torch.tensor(0.),torch.tensor(0.)
                if 'attn_loss' in out.keys():
                    loss += attn_weight*attn_loss
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                # After backward(), add:
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print(f"No gradient for {name}")
                    elif torch.all(param.grad == 0):
                        print(f"Zero gradients for {name}")
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_absvar_loss += absvar_loss.item()
            total_relvar_loss += relvar_loss.item()
            # Simple progress display
            progress = (batch_idx + 1) / num_batches
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + ' ' * (bar_length - filled_length)
            if verbose:
                print(f'\rEpoch {epoch + 1} [{bar}] {np.floor(progress*100)}%', end='', flush=True)
        
        # Epoch summary
        epoch_time = time.time() - start_time
        print(f'\rEpoch {epoch + 1} [{"="*20}] 100% | Time: {epoch_time:.1f}s | Loss: {total_loss/num_batches:.3e}')
        
        return total_loss/len(self.train_loader),\
    total_absvar_loss/len(self.train_loader),\
    total_relvar_loss/len(self.train_loader)
    
    def validate(self,val_loader=None):
        val_loader = val_loader or self.val_loader
        if not val_loader:
            return None
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                out = self.model(
                    spatial=torch.cat([batch['xmesh'],batch['ymesh']],dim=1),
                    frac=batch['frac'],
                    grid=batch['grid'],
                    temporal=batch['temporal'],
                    gia=batch['gia']
                )
                recon_loss = self.loss_func(out['predict'],batch['emission'])
                total_loss += recon_loss.item()
            
        return total_loss/len(val_loader)
    
    def save_model(self,path,**kwargs):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
        }, path)