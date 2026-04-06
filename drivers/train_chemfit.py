import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import sys,os,glob
sys.path.append('/user/kangsun/Oversampling_matlab/')
import logging
# logging.basicConfig(level=logging.INFO)
from popy import Level3_List, Level3_Data
from CAREER.gridded import CDL, Inventory
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from CAREER.nnfitting import ChemFitDataset, ChemFitTrainer, UNet, ChemFitConfig

config_path = sys.argv[1]
config = ChemFitConfig(run_id=os.path.split(config_path)[-1])
config.read_yaml(abs_path=config_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nei = Inventory(
    west=-130,east=-67,south=23,north=51
).read_NEINOX(nei_dir=config['nei_dir'])

# make sure each ds's random states do not overlap
random_state_unit = 10000
random_state_count = 0

ds_dict = {}
eval_ds_dict = {}

# utility function to crop mask based on l3 wesn
def F_crop_mask(mask,xgrid,ygrid,west,east,south,north):
    xmask = (xgrid >= west) & (xgrid <= east)
    ymask = (ygrid >= south) & (ygrid <= north)
    return mask[np.ix_(ymask,xmask)]
# utility function to return value at epoch using milestones
def value_vs_epoch(epoch,epoch1,value1,epoch2,value2):
    if epoch <= epoch1:
        value = value1
    elif epoch > epoch2:
        value = value2
    else:
        value = (value2-value1)/(epoch2-epoch1)*(epoch-epoch1)+value1
    return value

##### load raw data and prepare train/val datasets
for irange,l3s_range in enumerate(config['l3s_ranges']):
    dt_array = pd.period_range(l3s_range[0],l3s_range[1],freq='1D')
    l3_fns = np.array([d.strftime(config['l3_path_pattern']) for d in dt_array])
    if_exist = np.array([os.path.exists(fn) for fn in l3_fns])
    dt_array = dt_array[if_exist]
    l3_fns = l3_fns[if_exist]
    # create train/val time masks if nfold > 1
    if config['nfold'] > 1:
        df = pd.DataFrame(index=dt_array,data={'count':range(len(dt_array))})
        val_random_state = config['val_random_state']
        resampler = df.resample(config['val_interval'])
        split_idxs = np.empty((config['nfold'],len(resampler)),dtype=object)
        for i,(k,idx) in enumerate(resampler.indices.items()):
            rng = np.random.default_rng(val_random_state)
            splits = np.array_split(rng.permutation(idx),config['nfold'])
            for ifold,random_fold in enumerate(rng.permutation(range(config['nfold']))):
                split_idxs[ifold,i] = splits[random_fold]
            val_random_state += 1
        split_idx = [np.concatenate(s) for s in split_idxs]
        val_time_masks = [np.isin(df['count'],s) for s in split_idx]
        train_time_masks = [~vm for vm in val_time_masks]
    # when nfold = 1, split train/val according to val_fraction
    elif config['nfold'] == 1 and config['val_fraction'] > 0:
        df = pd.DataFrame(index=dt_array,data={'count':range(len(dt_array))})
        val_random_state = config['val_random_state']
        resampler = df.resample(config['val_interval'])
        val_time_mask = []
        train_time_mask = []
        for k,idx in resampler.indices.items():
            rng = np.random.default_rng(val_random_state)
            val_idx = rng.choice(
                idx,
                size=int(len(idx)*config['val_fraction']),
                replace=False
            )
            tmp = np.isin(idx,val_idx)
            val_time_mask += tmp.tolist()
            train_time_mask +=(~tmp).tolist()
            val_random_state += 1
        train_time_masks = [np.array(train_time_mask)]
        val_time_masks = [np.array(val_time_mask)]
    # loop over grid sizes
    for grid_size in config['grid_sizes']:
        # create daily frames of l3s
        l3s = Level3_List(
            dt_array=dt_array,
            west=config['west'],east=config['east'],
            south=config['south'],north=config['north']
        )
        l3s.read_nc_pattern(
            l3_list=l3_fns,
            fields_name=[
                'column_amount','column_amount_DD','surface_altitude_DD',
                'column_amount_DD_xy','column_amount_DD_rs'
            ],
            block_reduce=grid_size
        )
        # interp nei mask on l3 grid, once per grid_size
        nei_l3 = nei.regrid_to_l3(l3=l3s[0])
        nei_l3 = gaussian_filter(nei_l3,.5)
        nei_mask = nei_l3 < config['max_neinox']
        # loop over validation folds
        for ifold in range(config['nfold']):
            # get training datasets
            if config['nfold'] == 1 and config['val_fraction'] == 0:
                l3s_train = l3s
            else: # nfold > 1 or one fold with train/val split
                l3s_train = l3s.trim(time_mask=train_time_masks[ifold])
                l3s_val = l3s.trim(time_mask=val_time_masks[ifold])
            # loop over training domains
            for iwesn,wesn in enumerate(config['train_wesns']):
                # separate mask for each domain
                train_mask = F_crop_mask(
                    nei_mask,l3s[0]['xgrid'],l3s[0]['ygrid'],**wesn
                )
                for i,interval in enumerate(config['train_intervals']):
                    for j,offset in enumerate(config['train_offsets'][i]):
                        ds = ChemFitDataset(
                            l3s=l3s_train.trim(**wesn),
                            mask=train_mask,
                            resample_rule=interval,
                            resample_offset=offset,
                            crop_x=config['crop_x'],crop_y=config['crop_y'],
                            stride_x=config['train_stride_xs'][i],
                            stride_y=config['train_stride_ys'][i],
                            crop_fraction=1,
                            initial_random_state=random_state_unit*random_state_count,
                            randomize_crops_per_frame=True,
                            DD_scaling=1e9,VCD_scaling=1e4,WT_scaling=1e6,
                            westmost=-128,eastmost=-65,southmost=24,northmost=50,
                            base_year=2018,jitter_kw=None
                        )
                        ds.set_jitter(True,jitter_kw={'p_jitter':1.})
                        random_state_count += 1
                        ds_dict_key = (
                            f'fold{ifold}',f'domain{iwesn}',
                            grid_size,'-'.join(l3s_range),interval,offset
                        )
                        ds_dict[ds_dict_key] = ds
            # get validation datasets if needed
            if config['nfold'] == 1 and config['val_fraction'] == 0:continue
            # loop over val domains
            for iwesn,wesn in enumerate(config['val_wesns']):
                # separate mask for each domain
                val_mask = F_crop_mask(
                    nei_mask,l3s[0]['xgrid'],l3s[0]['ygrid'],**wesn
                )
                for i,interval in enumerate(config['eval_intervals']):
                    for j,offset in enumerate(config['eval_offsets'][i]):
                        ds = ChemFitDataset(
                            l3s=l3s_val.trim(**wesn),
                            mask=val_mask,
                            resample_rule=interval,
                            resample_offset=offset,
                            crop_x=config['crop_x'],crop_y=config['crop_y'],
                            stride_x=config['eval_stride_xs'][i],
                            stride_y=config['eval_stride_ys'][i],
                            crop_fraction=1,
                            DD_scaling=1e9,VCD_scaling=1e4,WT_scaling=1e6,
                            westmost=-128,eastmost=-65,southmost=24,northmost=50,
                            base_year=2018,jitter_kw=None
                        )
                        ds_dict_key = (
                            f'fold{ifold}',f'domain{iwesn}',
                            grid_size,'-'.join(l3s_range),interval,offset
                        )
                        eval_ds_dict[ds_dict_key] = ds

##### combining eval ds to val ds used in training
if eval_ds_dict:
    val_ds_dict = {}
    for ifold in range(config['nfold']):
        for iwesn,wesn in enumerate(config['val_wesns']):
            for i,interval in enumerate(config['eval_intervals']):
                dss = []
                for grid_size in config['grid_sizes']:
                    for irange,l3s_range in enumerate(config['l3s_ranges']):
                        for j,offset in enumerate(config['eval_offsets'][i]):
                            ds_dict_key = (
                                f'fold{ifold}',f'domain{iwesn}',
                                grid_size,'-'.join(l3s_range),interval,offset
                            )
                            dss.append(eval_ds_dict[ds_dict_key])
                ds = ConcatDataset(dss)
                ds_dict_key = (f'fold{ifold}',f'domain{iwesn}',interval)
                val_ds_dict[ds_dict_key] = ds

##### training over folds and hps
if not config['hps']:
    nhp = 1
else:
    nhp = np.min([len(config[hp]) for hp in config['hps']])
# all possible hps
all_hps = [
    'lr','weight_decay','batch_size',
    'smoothness_weight_milestones','smoothness_B_weight_milestones',
    'var_weight_milestones','fft_weight_milestones'
]

epochs = np.arange(config['start_epoch'],config['end_epoch'])

result_flds = [
    'train_loss','smooth_loss','smooth_B_loss',
    'var_loss','fft_loss','epoch_time','sample_size',
    'grad_norm','nan_norm'
]
loss_dfs = []
# dict holding all hp values in config
hp_dict = {k:config[k] for k in all_hps if k not in config['hps']}
# initialize figures for monitoring
if eval_ds_dict:
    fig_val,axs_val = plt.subplots(
        np.ceil(config['nfold']/3).astype(int),3,
        figsize=(12,5),sharex=True,sharey=True,constrained_layout=True
    )
fig_lap,axs_lap = plt.subplots(
    np.ceil(config['nfold']/3).astype(int),3,
    figsize=(12,5),sharex=True,sharey=True,constrained_layout=True
)
# one model per fold per hyperparameter
for ifold in range(config['nfold']):
    # build val loader for this fold if needed
    if eval_ds_dict:
        vloader_dict = {
            k[1:]:DataLoader(
                v,batch_size=1,shuffle=False,pin_memory=(device.type == 'cuda')
            ) for k,v in val_ds_dict.items() if k[0]==f'fold{ifold}'
        }
        all_vloader = DataLoader(
            ConcatDataset(
                [v for k,v in val_ds_dict.items() if k[0]==f'fold{0}']
            ),
            batch_size=1,shuffle=False,pin_memory=(device.type == 'cuda')
        )
    
    # loop over hyperparameters
    for ihp in range(nhp):
        for hp_name in config['hps']:
            hp_dict[hp_name] = config[hp_name][ihp]
            logging.warning(f'fold{ifold}, hp{ihp}, {hp_name}:{config[hp_name][ihp]}')
        model = UNet(
            out_channels=2,
            in_channels=2,
            feat_channels=[64,128,256],
            temporal_dim=3,
            force_positive=True
        )
        trainer = ChemFitTrainer(
            model=model,lr=hp_dict['lr'],weight_decay=hp_dict['weight_decay'],
            initial_grad_scaler=config['initial_grad_scaler']
        )
        if config['pretrained_model_path'] is not None:
            trainer.load_model(config['pretrained_model_path'])
            trainer.logger.warning(
                'model loaded from {}'.format(config['pretrained_model_path'])
            )
        loss_df = pd.DataFrame(
            index=epochs,
            data={
                f'fold{ifold}_hp{ihp}_{k}':np.full(epochs.shape,np.nan) 
                for k in result_flds
            }
        )
        if eval_ds_dict:
            header = f'fold{ifold}_hp{ihp}_val_loss'
            vloss_df = pd.DataFrame(
                index=epochs,
                data={
                    f'{header}{k}':np.full(epochs.shape,np.nan) for k in ['']+[
                        '_'+'_'.join(k[1:]) 
                        for k in val_ds_dict.keys() if k[0]==f'fold{ifold}'
                    ]
                }
            )
            loss_df = pd.concat([loss_df,vloss_df],axis=1)
        
        best_val_loss = float('inf')
        best_val_epoch = 0
        # loop over epochs
        for iepoch,epoch in enumerate(epochs):
            dss = []
            # fresh random crop for each dataset for each epoch
            for iwesn,wesn in enumerate(config['train_wesns']):
                for iinterval,interval in enumerate(config['train_intervals']):
                    crop_fraction = value_vs_epoch(
                        epoch,*config['train_milestones'][iinterval]
                    )
                    for grid_size in config['grid_sizes']:
                        for irange,l3s_range in enumerate(config['l3s_ranges']):
                            for ioffset,offset in enumerate(
                                config['train_offsets'][iinterval]
                            ):
                                ds_dict_key = (
                                    f'fold{ifold}',f'domain{iwesn}',grid_size,
                                    '-'.join(l3s_range),interval,offset
                                )
                                ds = ds_dict[ds_dict_key]
                                ds.random_crop(crop_fraction=crop_fraction)
                                dss.append(ds)
            trainer.load_data(
                dataset=ConcatDataset(dss),
                batch_size=hp_dict['batch_size'],shuffle=True
            )
            result = trainer.train_epoch(
                epoch=epoch,
                smoothness_weight=value_vs_epoch(
                    epoch,*hp_dict['smoothness_weight_milestones']
                ),
                smoothness_B_weight=value_vs_epoch(
                    epoch,*hp_dict['smoothness_B_weight_milestones']
                ),
                smoothness_scale_weights=config['smoothness_scale_weights'],
                smoothness_channel_weights=config['smoothness_channel_weights'],
                var_weight=value_vs_epoch(
                    epoch,*hp_dict['var_weight_milestones']
                ),
                var_scale_weights=config['var_scale_weights'],
                var_channel_weights=config['var_channel_weights'],
                fft_weight=value_vs_epoch(
                    epoch,*hp_dict['fft_weight_milestones']
                ),
                fft_cutoff_freq=config['fft_cutoff_freq'],
                fft_schedule=config['fft_schedule'],
                fft_channel_weights=config['fft_channel_weights'],
                max_norm=config['max_norm'],
                clamp_max_sigma=config['clamp_max_sigma'],
                clamp_min_sigma=config['clamp_min_sigma']
            )
            for k in result_flds:
                loss_df[f'fold{ifold}_hp{ihp}_{k}'].iloc[iepoch] = result[k]
            if eval_ds_dict:
                val_loss = trainer.validate(all_vloader)
                header = f'fold{ifold}_hp{ihp}_val_loss'
                loss_df[header].iloc[iepoch] = val_loss
                for k,v in val_ds_dict.items(): 
                    if k[0]==f'fold{ifold}':
                        val_id = '_'.join(k[1:])
                        loss_df[
                            f'{header}_{val_id}'
                        ].iloc[iepoch] = trainer.validate(vloader_dict[k[1:]])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if config['save_best_model'] is not False:
                        model_filename = f'best_model_fold{ifold}_hp{ihp}.pt'
                        trainer.save_model(
                            path=os.path.join(config['run_dir'],model_filename),
                            epoch=epoch
                        )
        loss_dfs.append(loss_df)
        if eval_ds_dict:
            ax = axs_val.ravel()[ifold]
            loss_df.plot(
                y=[f'fold{ifold}_hp{ihp}_val_loss'],ax=ax
            )
            fig_val.savefig(os.path.join(config['run_dir'],'val_loss_plot.pdf'))
        
        ax = axs_lap.ravel()[ifold]
        loss_df.plot(
            y=[f'fold{ifold}_hp{ihp}_smooth_loss'],ax=ax
        )
        fig_lap.savefig(os.path.join(config['run_dir'],'lap_loss_plot.pdf'))
        if config['save_final_model'] is not False:
            model_filename = f'final_model_fold{ifold}_hp{ihp}.pt'
            trainer.save_model(
                path=os.path.join(config['run_dir'],model_filename),
                epoch=epoch
            )
        loss_df = pd.concat(loss_dfs,axis=1)
        loss_df.to_csv(os.path.join(config['run_dir'],'loss.csv'))