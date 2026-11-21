import numpy as np
import logging
import time
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset


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
        # smoothness by scale and channel
        lap_SC = np.zeros((len(scale_weights),C))
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
            lap_BC0 = gradients.abs().mean(dim=[-2,-1])
            lap_BC = lap_BC0*channel_weights*weight
            lap_SC[i] = lap_BC0.cpu().detach().numpy().mean(axis=0)
            # std across B
            lap_B_std = lap_BC.mean(dim=-1).std()
            # mean laplacian per scale
            lap_mean = lap_BC.mean()
            lap_mean_loss += lap_mean
            lap_std_loss += lap_B_std
                        
        return lap_mean_loss,lap_std_loss,lap_SC


class CorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,unet_out,cor_weights=None,use_double=True):
        B, C, H, W = unet_out.shape
        N = H*W
        eps = 1e-8
        if cor_weights is None:
            cor_weights = torch.tril(torch.ones((C,C)),diagonal=-1) # only lower tri matters
        else:
            cor_weights = torch.tril(torch.tensor(cor_weights),diagonal=-1)
        
        cor_weights = cor_weights.to(unet_out.device)
        
        X = unet_out.view(B,C,N)
        X_centered = (X-X.mean(dim=2,keepdim=True))
        if use_double:
            X_centered = X_centered.double()
         # Compute covariance matrices [B, C, C]
        cov = torch.bmm(X_centered, X_centered.transpose(1, 2)) / N
        # Compute standard deviations [B, C]
        var = (X_centered ** 2).mean(dim=2)
        std = torch.sqrt(var + eps)
        # Compute correlation matrices [B, C, C]
        corr = cov / (std.unsqueeze(2) @ std.unsqueeze(1) + eps)
        cor_loss = (corr.abs()*cor_weights.expand(B,C,C)).sum(dim=(-2,-1)).mean()
        return cor_loss, corr.mean(dim=0)


def scheduler(epoch=None,which='linear',milestone=None,output_ratio=None):
    '''
    a value by epoch from milestone [epoch0,value0,epoch1,value1]: value0 until 
    epoch0, value1 after epoch1, cosine or linear in between. output_ratio 
    returns value/value0, as LambdaLR needs
    '''
    if which.lower() in ['cosine']:
        return cosine_scheduler(
            epoch=epoch,milestone=milestone,output_ratio=output_ratio
        )
    elif which.lower() in ['linear']:
        return linear_scheduler(
            epoch=epoch,milestone=milestone,output_ratio=output_ratio
        )
    else:
        raise ValueError('{} is not supported as a scheduler!'.format(which))


def cosine_scheduler(epoch=None,milestone=None,output_ratio=True):
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


def linear_scheduler(epoch=None,milestone=None,output_ratio=True):
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


def scheduled_value(kw,epoch):
    '''
    kw['value'] if given, else the scheduled value at epoch.
    v05: 'is None', so an explicit value 0 is kept; v04's 'value or ...' 
    replaced 0 by the scheduler
    '''
    if kw['value'] is not None:
        return kw['value']
    return scheduler(
        epoch=epoch,which=kw['scheduler'],milestone=kw['milestone'],output_ratio=False
    )


def run_model(model,batch,xnames):
    '''
    the model on a batch of CropDataset samples (already on the model's device)
    returns:
        model output dict, and the predictors [B,C,H,W]
    '''
    predictors = torch.cat([batch[xn] for xn in xnames],dim=1)
    out = model(
        spatial=batch['spatial'],
        predictors=predictors,
        time=batch['time'],
        grid_size=batch['grid_size'],
        gia=batch['gia'],
        psf_idx=batch['psf_idx'],
        lat=batch['lat']
    )
    return out,predictors


def clip_grad(model,max_norm=None):
    '''
    total gradient norm before clipping; clips to max_norm if given. 
    v05: v04 recorded the norm after clipping, a flat line at max_norm
    '''
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),max_norm=max_norm if max_norm else float('inf'))
    return norm.item()


class Trainer:
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
            pg_wd = wd_kw['value']*wd_kw['multipliers'][pg_name]
            param_groups.append(
                dict(
                    params=params,weight_decay=pg_wd,
                    lr=lr_kw['milestone'][1]*pg_multiplier
                )
            )
        self.optimizer = torch.optim.AdamW(param_groups)
        lr_lambda = partial(
            scheduler,which=lr_kw['scheduler'],
            milestone=lr_kw['milestone'],output_ratio=True
        )
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
    
    def load_data(self,dss,batch_size):
        '''
        training samples from all datasets, shuffled. v05: do_group and 
        shuffle_level are gone; v04 used shuffle_level='sample' anyway
        '''
        self.data_loader = DataLoader(
            ConcatDataset(dss),batch_size=batch_size,
            shuffle=True,pin_memory=(self.device.type=='cuda')
        )
    
    def run_model(self,batch,xnames):
        '''the model on a batch, see run_model'''
        return run_model(self.model,batch,xnames)
    
    def validate(
        self,val_loader,
        yname='column_amount_DD',
        xnames=['column_amount','surface_altitude_DD','cdl'],
        scale_random_error=True
    ):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                out,_ = self.run_model(batch,xnames)
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                predict = out['all_flux']/error_scaler
                target = batch[yname]/error_scaler
                loss = self.data_loss_func(
                    predict=predict,target=target,mask=batch['valid_mask']
                )
                total_loss += loss.item()
            
        return total_loss/len(val_loader)
    
    def train_epoch(
        self,epoch,
        yname='column_amount_DD',
        xnames=['column_amount','surface_altitude_DD','cdl'],
        scale_random_error=True,
        smoothness_kw=None,
        correlation_kw=None,
        point_l1_kw=None,
        max_norm=1.,
        clamp_max_sigma=None,
        clamp_min_sigma=None,
        verbose=False
    ):
        '''
        one epoch over self.data_loader. loss weights in *_kw['weight']['value'] 
        must already be the values of this epoch, see scheduled_value
        '''
        self.model.train()
        
        if clamp_min_sigma is not None:
            clamp_min_sigma = -np.abs(clamp_min_sigma)
        
        smoothness_kw = smoothness_kw or {'enabled':False}
        correlation_kw = correlation_kw or {'enabled':False}
        point_l1_kw = point_l1_kw or {'enabled':False}
        if smoothness_kw['enabled']:
            smoothness_loss_func = LaplacianLoss()
        if correlation_kw['enabled']:
            correlation_loss_func = CorrelationLoss()
        total_loss = 0
        total_smo_loss = 0
        total_smb_loss = 0
        total_cor_loss = 0
        total_pl1_loss = 0
        corr_list = []
        ssc_list = []
        start_time = time.time()
        num_batches = len(self.data_loader)
        norms = []
        for batch_idx, batch in enumerate(self.data_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            
            # Only use autocast on CUDA
            with torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
                out,predictor = self.run_model(batch,xnames)
                if scale_random_error:
                    error_scaler = batch['random_error'].view(-1,1,1,1)
                else:
                    error_scaler = 1.
                predict = out['all_flux']/error_scaler
                target = batch[yname]/error_scaler
                if clamp_min_sigma and clamp_max_sigma:
                    target = torch.clamp(target,min=clamp_min_sigma,max=clamp_max_sigma)
                loss = self.data_loss_func(
                    predict=predict,target=target,mask=batch['valid_mask']
                )
                # penalize the mean absolute value of all point rates
                if point_l1_kw['enabled'] and 'point_rate' in out.keys():
                    point_l1_loss = out['point_rate'].abs().mean()
                    loss += point_l1_kw['weight']['value']*point_l1_loss
                else:
                    point_l1_loss = torch.tensor(0.)
                
                if smoothness_kw['enabled']:
                    smoothness_loss,smoothness_B_loss,smoothness_SC = smoothness_loss_func(
                        out['unet_out'],
                        scale_weights=smoothness_kw['scale_weights'],
                        channel_weights=smoothness_kw['channel_weights']
                    )
                    loss += smoothness_kw['weight']['value']*smoothness_loss
                    loss += smoothness_kw['B_weight']['value']*smoothness_B_loss
                    if smoothness_kw['output_ssc']:
                        ssc_list.append(smoothness_SC)
                else:
                    smoothness_loss,smoothness_B_loss = torch.tensor(0.),torch.tensor(0.)
                
                if correlation_kw['enabled']:
                    correlation_loss,corr = correlation_loss_func(
                        out['unet_out']*predictor if correlation_kw[
                            'include_predictor'
                        ] else out['unet_out'],
                        use_double=correlation_kw['use_double'],
                        cor_weights=correlation_kw['channel_weights']
                    )
                    loss += correlation_kw['weight']['value']*correlation_loss
                    if correlation_kw['output_corr']:
                        corr_list.append(corr.cpu().detach().numpy())
                else:
                    correlation_loss = torch.tensor(0.)
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                total_norm = clip_grad(self.model,max_norm)
                norms.append(total_norm)
                self.logger.info(f"Step Gradient Norm: {total_norm:.4f}")
                g0flag = False
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        self.logger.warning(f"No gradient for {name}")
                    elif torch.all(param.grad == 0):
                        g0flag = True
                if g0flag:
                    self.logger.warning(f'epoch{epoch}, batch{batch_idx} has zero grad')
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # v05: clip on cpu too; v04 clipped only on cuda
                norms.append(clip_grad(self.model,max_norm))
                self.optimizer.step()
            
            total_loss += loss.item()
            total_smo_loss += smoothness_loss.item()
            total_smb_loss += smoothness_B_loss.item()
            total_cor_loss += correlation_loss.item()
            total_pl1_loss += point_l1_loss.item()
            
        # Epoch summary
        epoch_time = time.time() - start_time
        if verbose:
            self.logger.warning(
                f'\rEpoch {epoch + 1} | Time: {epoch_time:.1f}s | Loss: {total_loss/num_batches:.4f} | Sample: {self.data_loader.batch_size}x{num_batches}')
        norms = np.array(norms) # v05: an array, so norms == 0 compares elementwise
        # overflow steps under amp give inf/nan norms; the scaler skips them
        finite = np.isfinite(norms)
        result = dict(
            train_loss=total_loss/num_batches,
            smooth_loss=total_smo_loss/num_batches,
            smooth_B_loss=total_smb_loss/num_batches,
            cor_loss=total_cor_loss/num_batches,
            point_l1_loss=total_pl1_loss/num_batches,
            epoch_time=epoch_time,
            sample_size=len(self.data_loader.dataset),
            grad_norm=norms[finite].mean() if finite.any() else np.nan,
            nan_norm=np.sum(~finite),
            zero_norm=np.sum(norms==0)
        )
        if correlation_kw['enabled']:
            if correlation_kw['output_corr']:
                corr = np.array(corr_list).mean(axis=0)# [C,C]
                for i in range(1,corr.shape[0]):
                    for j in range(0,i):
                        result[f'corr_{i}_{j}'] = corr[i,j]
        if smoothness_kw['enabled']:
            if smoothness_kw['output_ssc']:
                ssc = np.array(ssc_list).mean(axis=0)# [S,C]
                for i in range(ssc.shape[0]):
                    for j in range(ssc.shape[1]):
                        result[f'ssc_{i}_{j}'] = ssc[i,j]
        return result
