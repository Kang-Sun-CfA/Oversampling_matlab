import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    '''
    encode raw time into features for the FiLM MLP and/or the point-source MLP.
    it has no learnable parameters; it lives in the model so that each consumer 
    can have its own encoding of time.
    raw time:
        [B,2] tensor, (years since base_year, local solar hour in [0,24))
    output features:
        trend: t, fractional year since base_year
        seasonal: sin/cos(2*pi*k*t), k = 1..n_seasonal, interleaved
        diurnal: sin/cos(2*pi*k*lh/24), k = 1..n_diurnal, interleaved
        whole-period: sin/cos(2*pi*k*t/whole_period), k = 1..n_whole, interleaved
    n_seasonal=1, n_diurnal=0, n_whole=0 reproduces v04 [fy-base_year, fy_sin, fy_cos]
    whole_period:
        in years, should be longer than the record. adding nonlinear trends
    '''
    def __init__(self,n_seasonal=1,n_diurnal=0,n_whole=0,whole_period=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.n_seasonal = int(n_seasonal)
        self.n_diurnal = int(n_diurnal)
        self.n_whole = int(n_whole)
        if self.n_whole > 0 and whole_period is None:
            raise ValueError('whole_period (years) is needed when n_whole > 0!')
        self.whole_period = float(whole_period) if whole_period is not None else None
        # harmonic numbers as buffers, so they move with .to(device) and are saved
        self.register_buffer(
            'seasonal_k',torch.arange(1,self.n_seasonal+1,dtype=torch.float32))
        self.register_buffer(
            'diurnal_k',torch.arange(1,self.n_diurnal+1,dtype=torch.float32))
        self.register_buffer(
            'whole_k',torch.arange(1,self.n_whole+1,dtype=torch.float32))
        self.feature_names = self.get_feature_names()
    
    @property
    def temporal_dim(self):
        '''number of output features'''
        return len(self.feature_names)
    
    def get_feature_names(self):
        names = ['trend']
        for prefix,n in zip(
            ['s','d','w'],# w stands for whole, not week
            [self.n_seasonal,self.n_diurnal,self.n_whole]
        ):
            for k in range(1,n+1):
                names += [f'{prefix}{k}_sin',f'{prefix}{k}_cos']
        return names
    
    @staticmethod
    def _harmonics(phase,ks):
        '''
        phase:
            [B,1], fraction of a cycle, in [0,1)
        ks:
            [K], harmonic numbers
        returns:
            [B,2K], interleaved as sin1,cos1,sin2,cos2,...
        '''
        angle = 2*np.pi*phase*ks # [B,K]
        return torch.stack(
            [torch.sin(angle),torch.cos(angle)],dim=-1
        ).reshape(phase.shape[0],-1)
    
    def forward(self,time):
        '''
        time:
            [B,2], (years since base_year, local solar hour)
        returns:
            [B,temporal_dim]
        '''
        t = time[:,0:1] # [B,1]
        lh = time[:,1:2] # [B,1]
        feats = [t]
        if self.n_seasonal > 0:
            # remainder keeps phase in [0,1); exact for integer k, kinder to precision
            feats.append(self._harmonics(torch.remainder(t,1.),self.seasonal_k))
        if self.n_diurnal > 0:
            feats.append(self._harmonics(torch.remainder(lh,24.)/24.,self.diurnal_k))
        if self.n_whole > 0:
            feats.append(self._harmonics(
                torch.remainder(t/self.whole_period,1.),self.whole_k))
        return torch.cat(feats,dim=1)


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
        # v05: a copy, so popping does not change the caller's dict
        temporal_kw = dict(temporal_kw or {})
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


class PointSourceDistributor(nn.Module):
    def __init__(self,pss_x,pss_y):
        super().__init__()
        self.n_sources = len(pss_x)
        
        # Store the actual source coordinates
        self.register_buffer(
            'all_source_coords', 
            torch.cat(
                [
                    pss_x.unsqueeze(1),
                    pss_y.unsqueeze(1)
                ],dim=1
            )
        )  # [n_sources, 2]
        
    def forward(self,point_rates,spatial,gia):
        '''
        point_rates:
            [B, n_sources], point source emission rates, 
            output from PointSourceMLP,unit mol s-1
        spatial:
            [B,C=2 (lon,lat),H,W]
        gia:
            [B, 1, H, W], grid inverse area, returned from a dataset sample, 
            unit m-2 nmol mol-1
        returns:
            [B, 1, H, W], emission from all point sources in the view
        '''
        B,C,H,W = gia.shape
        
        spatial_field = torch.zeros(B,1,H,W,device=point_rates.device)
        
        for b in range(B):
            xmesh,ymesh = spatial[b]
            xgs = torch.diff(xmesh[0,:]).median().abs()
            ygs = torch.diff(ymesh[:,0]).median().abs()
            xmin,xmax = xmesh.min(),xmesh.max()
            ymin,ymax = ymesh.min(),ymesh.max()
            
            in_view_mask = (
                (self.all_source_coords[:,0] >= xmin-xgs/2) & 
                (self.all_source_coords[:,0] < xmax+xgs/2) &
                (self.all_source_coords[:,1] >= ymin-ygs/2) & 
                (self.all_source_coords[:,1] < ymax+ygs/2)
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
                gia[b,0,pixel_y,pixel_x]
                
                # Scatter to spatial field. v05: accumulate=True adds sources 
                # sharing a pixel; v04's += kept only the last one
                spatial_field[b,0].index_put_(
                    (pixel_y,pixel_x),emissions_in_view.to(spatial_field.dtype),
                    accumulate=True
                )
        
        return spatial_field


class PointSourceMLP(nn.Module):
    def __init__(
        self,
        n_sources,
        temporal_dim,
        hidden_dim=128,
        n_layers=4,
        leaky_relu_rate=0.1,
        dropout=0.1
    ):
        super().__init__()
        self.n_sources = n_sources
        
        self.input_proj = nn.Linear(temporal_dim,hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(leaky_relu_rate),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)
        
        # Final layer with source-specific bias
        self.output_proj = nn.Linear(hidden_dim, n_sources)
        self.output_activation = nn.Softplus()
    
    def forward(self,temporal):
        '''
        temporal: 
            [B, temporal_dim]
        returns: 
            [B, n_sources], emission rates for all sources
        '''
        x = self.input_proj(temporal)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        out = self.output_proj(x)
        out = self.output_activation(out)
        
        return out


class PSF(nn.Module):
    '''
    super-Gaussian point spread function, one set of parameters per psf_idx, 
    e.g., per instrument or per way of calculating DD. widths are in degree; 
    the lon width is divided by cos(lat), as a fixed width in km spans more 
    degrees of lon at higher latitudes
    '''
    def __init__(self,kernel_size=5,n_psf=1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.n_psf = n_psf
        # Learnable parameters in the background
        self.log_wx = nn.Parameter(torch.ones(n_psf)*-2.3)
        self.log_wy = nn.Parameter(torch.ones(n_psf)*-2.3)
        self.log_kx = nn.Parameter(torch.ones(n_psf)*0.75)
        self.log_ky = nn.Parameter(torch.ones(n_psf)*0.75)
        
        # Base coordinate grid (shared)
        self.register_buffer('_base_xx', torch.zeros(kernel_size,kernel_size))
        self.register_buffer('_base_yy', torch.zeros(kernel_size,kernel_size))
        self._setup_base_coords()
    
    def _setup_base_coords(self):
        r = (self.kernel_size - 1) // 2
        x = torch.linspace(-r, r, self.kernel_size) # unit spacing
        # indexing='xy': xx varies along W (lon), yy along H (lat). v04 used 'ij',
        # where its x actually ran along lat
        xx, yy = torch.meshgrid(x, x,indexing='xy')
        self._base_xx.copy_(xx.contiguous())
        self._base_yy.copy_(yy.contiguous())
    
    def _create_kernel(self,grid_size,psf_idx,lat):
        '''grid_size, psf_idx, and lat as numbers,
        returns kernel [1,1,kernel_size,kernel_size]'''
        # kernel width measured by deg lon will be larger at higher lat
        wx = torch.exp(self.log_wx[psf_idx])/torch.cos(torch.deg2rad(lat))
        wy = torch.exp(self.log_wy[psf_idx])
        kx = torch.exp(self.log_kx[psf_idx])
        ky = torch.exp(self.log_ky[psf_idx])
        xx = self._base_xx*grid_size
        yy = self._base_yy*grid_size
        # super-Gaussian kernel
        kernel = torch.exp(-(torch.abs(xx/wx)**kx+torch.abs(yy/wy)**ky))
        return (kernel/kernel.sum()).view(1,1,self.kernel_size,self.kernel_size)
    
    def forward(self,x,grid_size,psf_idx,lat):
        '''
        x:
            [B,C,H,W], input tensor
        grid_size,psf_idx,lat:
            [B], grid size in degree, psf index, and lat in degree
        '''
        B, C, H, W = x.shape
        padding = (self.kernel_size - 1) // 2
        x_padded = F.pad(x,(padding,)*4,mode='reflect')
        results = []
        for b in range(B):
            kernel = self._create_kernel(grid_size[b],psf_idx[b],lat[b])
            conv = F.conv2d(x_padded[b:b+1],kernel.expand(C,-1,-1,-1),groups=C)
            results.append(conv)
        
        return torch.cat(results, dim=0)
    
    def get_kernel(self,grid_size,psf_idx,lat):
        '''Returns current kernel for visualization'''
        with torch.no_grad():
            return dict(
                kernel=self._create_kernel(grid_size,psf_idx,lat)[0,0],
                wx=torch.exp(self.log_wx[psf_idx])/torch.cos(torch.deg2rad(lat))*111,#km
                wy=torch.exp(self.log_wy[psf_idx])*111,
                kx = torch.exp(self.log_kx[psf_idx]),
                ky = torch.exp(self.log_ky[psf_idx])
            )


class FluxCombiner(nn.Module):
    '''
    final function to sum up everything with optional learnable kernel.
    v05: takes raw time [B,2] and encodes it with its own TemporalEncoders, one 
    for the FiLM MLP and one for the point-source MLP, so temporal_dim is derived
    '''
    def __init__(self,spatial_kw,temporal_kw,psf_kw=None,point_kw=None):
        super().__init__()
        psf_kw = dict(psf_kw or {'enabled':False})
        point_kw = dict(point_kw or {'enabled':False})
        temporal_kw = dict(temporal_kw)
        # time encoding for the FiLM MLP
        self.film_encoder = TemporalEncoder(**temporal_kw.pop('encoder',{}))
        temporal_kw['temporal_dim'] = self.film_encoder.temporal_dim
        self.unet = FeatureModulatedUNet(temporal_kw=temporal_kw,**spatial_kw)
        # v05: chem/topo channels are kept out of the psf; defined also without 
        # psf, where v04 raised an AttributeError
        self.unet_skip_channels = list(psf_kw.get('unet_skip_channels',[]))
        self.do_psf = psf_kw['enabled']
        if self.do_psf:
            self.psf = PSF(kernel_size=psf_kw['kernel_size'],n_psf=psf_kw.get('n_psf',1))
        self.do_point = point_kw['enabled']
        if self.do_point:
            self.point_encoder = TemporalEncoder(**point_kw.get('encoder',{}))
            self.point_mlp = PointSourceMLP(
                n_sources=len(point_kw['pss_x']),
                temporal_dim=self.point_encoder.temporal_dim,
                hidden_dim=point_kw['hidden_dim'],
                n_layers=point_kw['n_layers'],
                leaky_relu_rate=point_kw['leaky_relu_rate'],
                dropout=point_kw['dropout']
            )
            self.point_distributor = PointSourceDistributor(
                point_kw['pss_x'],point_kw['pss_y']
            )
    
    def get_point_rate(self,time):
        '''
        time:
            [B,2], raw time; e.g., to compare with CEMS at any time
        returns:
            [B,n_sources], point-source emission rates
        '''
        return self.point_mlp(self.point_encoder(time))
    
    def forward(
        self,spatial,predictors,time,grid_size=None,gia=None,psf_idx=None,lat=None
    ):
        '''
        spatial:
            B,C=2 (lon,lat),H=crop_y,W=crop_x
        predictors:
            B,C=outchannels (vcd, wt, and cdl),H=crop_y,W=crop_x
        time:
            B,2, raw (years since base_year, local hour), from the dataset
        grid_size:
            B, grid size, needed for psf
        gia:
            B,1,H,W, grid inverse area, needed for point sources
        psf_idx,lat:
            B, psf of each sample's stream and crop center latitude, needed for psf
        '''
        unet_out = self.unet(x=spatial,temporal=self.film_encoder(time))
        out = dict(unet_out=unet_out)
        C = unet_out.shape[1]
        # flux from area sources
        flux = (unet_out*predictors)[
            :,[c for c in range(C) if c not in self.unet_skip_channels],:,:
        ].sum(dim=1,keepdim=True)
        # add point source flux
        if self.do_point:
            point_rate = self.get_point_rate(time)
            point_flux = self.point_distributor(point_rate,spatial,gia)
            flux += point_flux 
            out['point_rate'] = point_rate
            out['point_flux'] = point_flux
        # psf applied only on area/point flux
        if self.do_psf:
            flux = self.psf(x=flux,grid_size=grid_size,psf_idx=psf_idx,lat=lat)
        # add chem and topo terms
        if len(self.unet_skip_channels) > 0:
            out['all_flux'] = flux+(unet_out*predictors)[
                :,self.unet_skip_channels,:,:
            ].sum(dim=1,keepdim=True)
        else:
            out['all_flux'] = flux
        return out


def build_model(config,pss_x=None,pss_y=None):
    '''
    FluxCombiner from config.model. out_channels follows v04: one per xname, 
    with cdl counting its classes (name_codes plus the remaining class). the 
    psf gets one parameter set per psf_idx used by the streams
    pss_x,pss_y:
        [n_sources] tensors of point-source lon/lat, normalized by wesnmost; 
        needed if model.point.enabled
    '''
    mc = config.model
    out_channels = len(mc.xnames)
    if 'cdl' in mc.xnames:
        n_cdl = len(config.data.cdl.name_codes)+1
        # add cafo land use types if cafo_paths provided
        if config.data.cdl.cafo.enabled:
            n_cdl += len(config.data.cdl.cafo.paths)
        out_channels += n_cdl-1
    spatial_kw = dict(mc.spatial.to_dict(),out_channels=out_channels)
    psf_kw = mc.psf.to_dict()
    psf_kw['n_psf'] = max(sc.get('psf_idx',0) for sc in config.data.streams)+1
    point_kw = mc.point.to_dict()
    if point_kw['enabled']:
        if pss_x is None or pss_y is None:
            raise ValueError('pss_x and pss_y are needed when model.point is enabled!')
        point_kw.update(pss_x=pss_x,pss_y=pss_y)
    return FluxCombiner(
        spatial_kw=spatial_kw,temporal_kw=mc.temporal.to_dict(),
        psf_kw=psf_kw,point_kw=point_kw
    )
