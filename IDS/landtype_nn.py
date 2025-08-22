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
        self,emissions,years,fy,fy_sin,fy_cos,cdls,emission_func=None,
        westmost=-128,eastmost=-65,southmost=24,northmost=50,base_year=2018,
        crop_size_x=128,crop_size_y=64,stride=16,crop_fraction=0.5,random_state=10,
        cdl_name_codes=None,epoch=0,stride_x=None,stride_y=None
    ):
        stride_x = stride_x or stride
        stride_y = stride_y or stride
        self.set_epoch(epoch)
        self.emission_func = emission_func or (lambda x:x*1e9)
        self.base_year = base_year
        xmesh = (cdls[0].lonmesh-westmost)/(eastmost-westmost)
        ymesh = (cdls[0].latmesh-southmost)/(northmost-southmost)
        self.westmost,self.eastmost,self.southmost,self.northmost=westmost,\
        eastmost,southmost,northmost
        self.xmesh = torch.tensor(xmesh).float()
        self.ymesh = torch.tensor(ymesh).float()
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
        
        self.random_crop(
            crop_size_x=crop_size_x,crop_size_y=crop_size_y,stride=stride,stride_x=stride_x,stride_y=stride_y,
            crop_fraction=crop_fraction,random_state=random_state,epoch=epoch)
        
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
                
        temporal = torch.tensor(
            np.array(
                [
                    year+fy-self.base_year,
                    fy_sin,
                    fy_cos
                ]
            ),dtype=torch.float32
        )
        
        xmesh = self.xmesh[i:i+self.crop_size_y,j:j+self.crop_size_x].unsqueeze(0)
        ymesh = self.ymesh[i:i+self.crop_size_y,j:j+self.crop_size_x].unsqueeze(0)
        
        return {
            'emission':self.emission_func(emission),
            'xmesh':xmesh,
            'ymesh':ymesh,
            'frac':frac,
            'temporal':temporal
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
    def __init__(self, kernel_size=5, grid_size=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        
        # Learnable parameters in the background
        self.log_wx = nn.Parameter(torch.tensor(-2.))
        self.log_wy = nn.Parameter(torch.tensor(-2.))
        self.log_kx = nn.Parameter(torch.tensor(0.5))
        self.log_ky = nn.Parameter(torch.tensor(0.5))
        
        # Coordinate grid
        self.register_buffer('xx', torch.zeros(kernel_size,kernel_size))
        self.register_buffer('yy', torch.zeros(kernel_size,kernel_size))
        self._setup_coords()
    
    def _setup_coords(self):
        r = (self.kernel_size - 1) // 2
        x = torch.linspace(-r, r, self.kernel_size) * self.grid_size
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.xx.copy_(xx.contiguous())
        self.yy.copy_(yy.contiguous())
    
    def forward(self, x):
        wx = F.softplus(self.log_wx) + 1e-3*self.grid_size
        wy = F.softplus(self.log_wy) + 1e-3*self.grid_size
        kx = F.softplus(self.log_kx) + 1.0
        ky = F.softplus(self.log_ky) + 1.0

        kernel = torch.exp(
            -(torch.abs(self.xx/wx)**kx 
             + torch.abs(self.yy/wy)**ky)
        )
        kernel = (kernel / kernel.sum()).view(1, 1, self.kernel_size, self.kernel_size)
        # Apply standard convolution
        padding = (self.kernel_size - 1) // 2
        return F.conv2d(
            F.pad(x, (padding,)*4, mode='reflect'),
            kernel.expand(x.size(1), -1, -1, -1),
            groups=x.size(1)
        )
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Special handling for coordinate buffers"""
        # Load parameters first
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        
        # Ensure coordinates are properly initialized
        if prefix + 'xx' in state_dict:
            with torch.no_grad():
                self._setup_coords()
    
    @property
    def wx(self):
        return F.softplus(self.log_wx) + 1e-3*self.grid_size
    
    @property
    def wy(self):
        return F.softplus(self.log_wy) + 1e-3*self.grid_size
    
    @property
    def kx(self):
        return F.softplus(self.log_kx) + 1.0
    
    @property
    def ky(self):
        return F.softplus(self.log_ky) + 1.0


class LandTypeEmissionModel(nn.Module):
    """UNet-like module for generating emission contribution for a single land type."""
    def __init__(self,n_land_types,in_channels=2,feat_channels=[64,128,256],
                 temporal_dim=3,grid_size=0.1,psf_kernel_size=5,
                 do_ConvTranspose=True,do_attention=False
                ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.n_land_types = n_land_types
        self.grid_size = grid_size

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
        self.psf = SuperGaussianPSF(kernel_size=psf_kernel_size, grid_size=grid_size)
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
    
    def forward(self,spatial,frac,temporal=None):
        '''
        spatial:
            B,C=2 (lon,lat),H=64,W=64
        frac:
            B,C=8 (corn, soybean, ...),H=64,W=64
        temporal:
            B,C=3 (year, sin(fy), cos(fy))
        '''
        emissions = self.get_unet_outcome(spatial,temporal)
        predict = self.psf((emissions*frac).sum(dim=1,keepdim=True))
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
    
    def forward(self, emissions,n_land_types=8,scale_weights=[1.0, 0.5, 0.25]):
        total_loss = 0
        B, _, H, W = emissions.shape
        
        for i, weight in enumerate(scale_weights):
            # Downsample emission map
            scaled = F.avg_pool2d(emissions, kernel_size=2**i)
            
            # Calculate Laplacian
            if self.gradient_kernel.device != scaled.device:
                self.gradient_kernel = self.gradient_kernel.to(scaled.device)
            gradients = F.conv2d(
                scaled, self.gradient_kernel.expand(n_land_types,1,3,3), 
                groups=n_land_types, padding=1)
            total_loss += weight * gradients.abs().mean()
            
        return total_loss

class Trainer:
    def __init__(self,model,loss_func=MaskedLoss(),lr=1e-4,weight_decay=0.01,device='auto'):
        # Device detection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Get all model parameters EXCEPT the PSF's special parameters
        other_params = [
            p for n, p in self.model.named_parameters()
            if not any(k in n for k in ['log_wx', 'log_wy', 'log_kx', 'log_ky'])
        ]

        # PSF-specific parameters
        psf_width_params = [model.psf.log_wx, model.psf.log_wy]
        psf_shape_params = [model.psf.log_kx, model.psf.log_ky]

        # Configure optimizer with different learning rates/weight decay
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': psf_width_params, 'lr': lr*50, 'weight_decay': 0},
            {'params': psf_shape_params, 'lr': lr*10, 'weight_decay': 0}
        ])
        
        self.loss_func = loss_func
        # Only enable mixed precision on CUDA
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def load_data(self,train_data,val_data=None,batch_size=32,shuffle=True):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, 
                                     pin_memory=(self.device.type == 'cuda'))
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                                   pin_memory=(self.device.type == 'cuda')) if val_data else None
    
    def _to_device(self,data):
        return {k: v.to(self.device, non_blocking=(self.device.type == 'cuda')) 
                for k, v in data.items()}
    
    def train_epoch(self,epoch,attn_weight=1e-3,temporal_kw=None,smooth_weight=1e-3,verbose=False):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_attn_loss = 0
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
                    temporal=batch['temporal']
                )
                loss = self.loss_func(out['predict'],batch['emission'])
                if smooth_weight is not None:
                    loss += smooth_weight*smooth_loss(
                        out['emissions'],n_land_types=self.model.n_land_types)
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
        
        return total_loss/len(self.train_loader)
    
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
                    temporal=batch['temporal']
                )
                recon_loss = self.loss_func(out['predict'],batch['emission'])
                total_loss += recon_loss.item()
            
        return total_loss/len(self.val_loader)