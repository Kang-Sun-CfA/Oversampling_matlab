'''
run this script in three ways:
    %run -i train_cefit.py # in ipython, assuming config is already defined
    python /user/kangsun/Oversampling_matlab/drivers/train_cefit.py config.yml # in terminal
    python train_cefit.py config.yml /user/kangsun/Oversampling_matlab # specifying git path
'''
import sys,os,glob
import logging
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

script_path = sys.argv[0]
git_path = os.path.join(os.path.split(script_path)[0],'..')

if len(sys.argv) == 1:
    logging.warning('assuming config already defined')
else:
    config_path = sys.argv[1]
    
if len(sys.argv) > 2:
    git_path = sys.argv[2]

sys.path.append(git_path)

from popy import Level3_List, arange_
from CAREER.gridded import CDL, BUI
from IDS.nnfitting import CEConfig, CEDataset, CETrainer, FluxCombiner, Inferencer

if len(sys.argv) > 1:
    config = CEConfig.from_yaml(path=config_path)

cfgs = config.get_hp_combinations() if config.hp_tuning.enabled else [config]

# prepare inclusive grid size and wesn to load datasets
grid_sizes = np.unique(
    [
        d.grid_size for d in config.data.train
    ]+[
        d.grid_size for d in config.data.validation
    ]
)
west = np.min(
    [d.west for d in config.data.train]+[d.west for d in config.data.validation]
)
east = np.max(
    [d.east for d in config.data.train]+[d.east for d in config.data.validation]
)
south = np.min(
    [d.south for d in config.data.train]+[d.south for d in config.data.validation]
)
north = np.max(
    [d.north for d in config.data.train]+[d.north for d in config.data.validation]
)
tkw = config.data.test
twest,teast = tkw.west or west, tkw.east or east
tsouth,tnorth = tkw.south or south, tkw.north or north
tgrid_size = tkw.grid_size or np.min(grid_sizes)
tstart = tkw.start or config.data.l3s.ranges[0].start
tend = tkw.end or config.data.l3s.ranges[-1].end
txgrid = arange_(twest,teast-tgrid_size/2,tgrid_size)+tgrid_size/2
tygrid = arange_(tsouth,tnorth-tgrid_size/2,tgrid_size)+tgrid_size/2

tdt_array = pd.period_range(tstart,tend,freq=tkw.freq)
if tkw.do_dummy:
    test_ds = CEDataset(xgrid=txgrid,ygrid=tygrid,dt_array=tdt_array)
    test_ds.random_crop(
        crop_x=tkw.crop_x,crop_y=tkw.crop_y,stride_x=tkw.stride_x,stride_y=tkw.stride_y,
        crop_fraction=1.,randomize_start_xy=False,do_group=False
    )


if config.data.bui.enabled and config.data.reload:
    bui = BUI(
        dt_array=pd.period_range(
            config.data.bui.range.start,
            config.data.bui.range.end,
            freq=config.data.bui.range.freq
        ),
        west=west,east=east,
        south=south,north=north
    )
    if config.data.bui.resample:
        bui.resample(config.data.bui.resample)
    if config.data.bui.which.lower() == 'cams':
        bui.read_CAMS_NOx(
            fn_pattern=config.data.bui.path_pattern,
            fields=['sum'],
            gaussian_sigma=config.data.bui.gaussian_sigma
        )
    else:
        raise ValueError('bui not implemented!')

if config.data.cdl.enabled and config.data.reload:
    cdl_dict = {}
    for grid_size in grid_sizes:
        cdl = CDL(lcc_path=config.data.cdl.lcc_path)
        pattern = config.data.cdl.path_pattern.replace('*',f'{grid_size:.2f}')
        cdl_fns = [
            pd.Timestamp(y,1,1).strftime(pattern) 
            for y in config.data.cdl.years
        ]
        assert all(os.path.exists(fn) for fn in cdl_fns)
        cdl.read_nc(cdl_fns,config.data.cdl.years)
        cdl = cdl.combine(config.data.cdl.name_codes).trim(
            west=west,east=east,south=south,north=north
        )
        cdl_dict[grid_size] = cdl


# make sure each ds's random states do not overlap
random_state_unit = 10000
random_state_count = config.experiment.seed
if config.data.nfold == 1 and config.data.val_fraction == 0:
    do_val = False
else:
    do_val = True
# arrays holding train/val/test datasets
if config.data.reload:
    train_dss = np.empty(
        (
            config.data.nfold,
            len(config.data.l3s.ranges),
            len(config.data.train)
        ),
        dtype=object
    )
    if do_val:
        val_dss = np.empty(
            (
                config.data.nfold,
                len(config.data.l3s.ranges),
                len(config.data.validation)
            ),
            dtype=object
        )
    if not tkw.do_dummy:
        test_dss = np.empty((len(config.data.l3s.ranges)),dtype=object)
##### load raw data and prepare train/val datasets
for irange,l3s_range in enumerate(
    config.data.l3s.ranges if config.data.reload else []
):
    dt_array = pd.period_range(l3s_range.start,l3s_range.end,freq=l3s_range.freq)
    l3_fns = np.array(
        [d.strftime(config.data.l3s.path_pattern) for d in dt_array]
    )
    if_exist = np.array([os.path.exists(fn) for fn in l3_fns])
    dt_array = dt_array[if_exist]
    l3_fns = l3_fns[if_exist]
    # create train/val time masks if nfold > 1
    if config.data.nfold > 1:
        df = pd.DataFrame(index=dt_array,data={'count':range(len(dt_array))})
        val_random_state = config.experiment.seed
        resampler = df.resample(config.data.val_interval)
        split_idxs = np.empty((config.data.nfold,len(resampler)),dtype=object)
        for i,(k,idx) in enumerate(resampler.indices.items()):
            rng = np.random.default_rng(val_random_state)
            splits = np.array_split(rng.permutation(idx),config.data.nfold)
            for ifold,random_fold in enumerate(
                rng.permutation(range(config.data.nfold))
            ):
                split_idxs[ifold,i] = splits[random_fold]
            val_random_state += 1
        split_idx = [np.concatenate(s) for s in split_idxs]
        val_time_masks = [np.isin(df['count'],s) for s in split_idx]
        train_time_masks = [~vm for vm in val_time_masks]
    # when nfold = 1, split train/val according to val_fraction
    elif config.data.nfold == 1 and config.data.val_fraction > 0:
        df = pd.DataFrame(index=dt_array,data={'count':range(len(dt_array))})
        val_random_state = config.experiment.seed
        resampler = df.resample(config.data.val_interval)
        val_time_mask = []
        train_time_mask = []
        for k,idx in resampler.indices.items():
            rng = np.random.default_rng(val_random_state)
            val_idx = rng.choice(
                idx,
                size=int(len(idx)*config.data.val_fraction),
                replace=False
            )
            tmp = np.isin(idx,val_idx)
            val_time_mask += tmp.tolist()
            train_time_mask +=(~tmp).tolist()
            val_random_state += 1
        train_time_masks = [np.array(train_time_mask)]
        val_time_masks = [np.array(val_time_mask)]
    # loop over grid sizes
    for grid_size in grid_sizes:
        # create daily frames of l3s
        l3s = Level3_List(
            dt_array=dt_array,
            west=west,east=east,
            south=south,north=north
        )
        l3s.read_nc_pattern(
            l3_list=l3_fns,
            fields_name=[
                'column_amount','column_amount_DD','surface_altitude_DD',
                'column_amount_DD_xy','column_amount_DD_rs'
            ],
            block_reduce=grid_size
        )
        # bui for the grid
        if config.data.bui.enabled:
            bui.regrid_to_l3(l3=l3s[0])
        # cdl for the grid
        if config.data.cdl.enabled:
            cdl = cdl_dict[grid_size]
            assert np.array_equal(cdl.xgrid,l3s[0]['xgrid'])
            assert np.array_equal(cdl.ygrid,l3s[0]['ygrid'])
        # get test data if not dummy
        if not tkw.do_dummy and grid_size == tgrid_size:
            test_dss[irange] = CEDataset(
                l3_kw=dict(
                    l3s=l3s.trim(
                        west=twest,east=teast,
                        south=tsouth,north=tnorth
                    ),
                    var_names=config.data.l3s.var_names,
                    var_scales=config.data.l3s.var_scales,
                    resample_rules=[tkw.freq],
                    resample_offsets=['0d']
                ),
                bui_kw=dict(
                    bui=bui.trim(
                        west=twest,east=teast,
                        south=tsouth,north=tnorth
                    ),
                    time_matching_method=config.data.bui.time_matching_method,
                    max_emission=config.data.bui.max_threshold,
                    scale=1e9,yield_inventory=config.data.bui.yield_inventory
                ) if config.data.bui.enabled else None,
                cdl_kw=dict(
                    cdl=cdl.trim(
                        west=twest,east=teast,
                        south=tsouth,north=tnorth
                    ),
                    time_matching_method=config.data.cdl.time_matching_method,
                ) if config.data.cdl.enabled else None,
                base_year=2020,jitter_kw=None
            )
            # crop the test ds as it only needs to be done once
            test_dss[irange].random_crop(
                crop_x=tkw.crop_x,crop_y=tkw.crop_y,
                stride_x=tkw.stride_x,stride_y=tkw.stride_y,
                crop_fraction=1.,randomize_start_xy=False,do_group=False
            )
        # loop over validation folds
        for ifold in range(config.data.nfold):
            # get training datasets
            if not do_val:
                l3s_train = l3s
            else: # nfold > 1 or one fold with train/val split
                l3s_train = l3s.trim(time_mask=train_time_masks[ifold])
                l3s_val = l3s.trim(time_mask=val_time_masks[ifold])
            # loop over training ds
            for itrain,tconfig in enumerate(config.data.train):
                if tconfig.grid_size != grid_size:continue
                ds = CEDataset(
                    l3_kw=dict(
                        l3s=l3s_train.trim(
                            west=tconfig.west,east=tconfig.east,
                            south=tconfig.south,north=tconfig.north
                        ),
                        var_names=config.data.l3s.var_names,
                        var_scales=config.data.l3s.var_scales,
                        resample_rules=tconfig.intervals,
                        resample_offsets=tconfig.offsets
                    ),
                    bui_kw=dict(
                        bui=bui.trim(
                            west=tconfig.west,east=tconfig.east,
                            south=tconfig.south,north=tconfig.north
                        ),
                        time_matching_method=config.data.bui.time_matching_method,
                        max_emission=config.data.bui.max_threshold,
                        scale=1e9,yield_inventory=config.data.bui.yield_inventory
                    ) if config.data.bui.enabled else None,
                    cdl_kw=dict(
                        cdl=cdl.trim(
                            west=tconfig.west,east=tconfig.east,
                            south=tconfig.south,north=tconfig.north
                        ),
                        time_matching_method=config.data.cdl.time_matching_method,
                    ) if config.data.bui.enabled else None,
                    initial_random_state=random_state_unit*random_state_count,
                    base_year=2020,jitter_kw=None
                )
                ds.set_jitter(True,jitter_kw={'p_jitter':1.})
                random_state_count += 1
                train_dss[ifold,irange,itrain] = ds
                  
            # get validation datasets if needed
            if not do_val:continue
            # loop over val domains
            for ival,vconfig in enumerate(config.data.validation):
                if vconfig.grid_size != grid_size:continue
                ds = CEDataset(
                    l3_kw=dict(
                        l3s=l3s_val.trim(
                            west=vconfig.west,east=vconfig.east,
                            south=vconfig.south,north=vconfig.north
                        ),
                        var_names=config.data.l3s.var_names,
                        var_scales=config.data.l3s.var_scales,
                        resample_rules=vconfig.intervals,
                        resample_offsets=vconfig.offsets
                    ),
                    bui_kw=dict(
                        bui=bui.trim(
                            west=vconfig.west,east=vconfig.east,
                            south=vconfig.south,north=vconfig.north
                        ),
                        time_matching_method=config.data.bui.time_matching_method,
                        max_emission=config.data.bui.max_threshold,
                        scale=1e9,yield_inventory=config.data.bui.yield_inventory
                    ) if config.data.bui.enabled else None,
                    cdl_kw=dict(
                        cdl=cdl.trim(
                            west=vconfig.west,east=vconfig.east,
                            south=vconfig.south,north=vconfig.north
                        ),
                        time_matching_method=config.data.cdl.time_matching_method,
                    ) if config.data.cdl.enabled else None,
                    base_year=2020,jitter_kw=None
                )
                # crop the val ds as it only needs to be done once
                ds.random_crop(
                    crop_x=vconfig.crop_x,crop_y=vconfig.crop_y,
                    stride_x=vconfig.stride_x,stride_y=vconfig.stride_y,
                    crop_fraction=1.,randomize_start_xy=False,do_group=False
                )
                val_dss[ifold,irange,ival] = ds

# combine test_dss
if not tkw.do_dummy:
    test_ds = ConcatDataset(test_dss)
    test_ds.nframe = np.sum([ds.nframe for ds in test_dss])
    test_ds.selected_crops = [sc for ds in test_dss for sc in ds.selected_crops]
    test_ds.l3r = [l3 for ds in test_dss for l3 in ds.l3r]
    test_ds.df = pd.concat([ds.df for ds in test_dss])
    for k in [
        'ncrops_per_frame','crop_x','crop_y','frame_x','frame_y',
        'latmesh','lonmesh','grid_size'
    ]:
        setattr(test_ds,k,getattr(test_dss[0],k))

epochs = np.arange(config.training.start_epoch,config.training.end_epoch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(config.experiment.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.experiment.seed)
    torch.cuda.manual_seed_all(config.experiment.seed)

vloaders = np.empty(val_dss.shape[1:],dtype=object)
result_flds = config.saving.loss_df.result_flds

loss_dfs = []
for ifold in range(config.data.nfold):
    # separate loader for each vds in this fold, irange dim concatenated
    if do_val:
        vloaders = [
            DataLoader(
                ConcatDataset(val_dss[ifold,:,ival]),batch_size=256,
                shuffle=False,pin_memory=(device.type=='cuda')
            ) for ival in range(val_dss.shape[2])
        ] # list of len(config.data.validation)
    for ihp,cfg in enumerate(cfgs):
        # reset random state of each training ds
        for ds in train_dss[ifold].ravel():
            ds.random_state = ds.initial_random_state
        # one model per fold per hp instance
        out_channels = len(cfg.model.xnames)
        if 'cdl' in cfg.model.xnames:
            nland = len(cfg.data.cdl.name_codes)+1
            assert nland == cdl.data.shape[1]
            out_channels += nland-1
        cfg.model.spatial.out_channels = out_channels
        if cfg.loss.smoothness_loss.enabled and cfg.loss.smoothness_loss.output_ssc:
            ssc_flds = [
                f'ssc_{i}_{j}' for i in range(
                    len(cfg.loss.smoothness_loss.scale_weights)
                ) for j in range(out_channels)
            ]
        else:
            ssc_flds = []
        if cfg.loss.correlation_loss.enabled and cfg.loss.correlation_loss.output_corr:
            corr_flds = [f'corr_{i}_{j}' for i in range(1,out_channels) for j in range(i)]
        else:
            corr_flds = []
        model = FluxCombiner(
            spatial_kw=cfg.model.spatial.to_dict(),
            temporal_kw=cfg.model.temporal.to_dict(),
            psf_kw=cfg.model.psf.to_dict()
        )
        trainer = CETrainer(
            model=model,lr_kw=cfg.training.lr.to_dict(),wd_kw=cfg.training.weight_decay.to_dict(),
            device=device
        )
        loss_df = pd.DataFrame(
            index=epochs,
            data={
                f'fold{ifold}_hp{ihp}_{k}':np.full(epochs.shape,np.nan) 
                for k in result_flds+corr_flds+ssc_flds
            }
        )
        if do_val:
            vloss_df = pd.DataFrame(
                index=epochs,
                data={
                    f'fold{ifold}_hp{ihp}_val_loss{ival}':np.full(epochs.shape,np.nan) 
                    for ival in range(val_dss.shape[2])
                }
            )
            loss_df = pd.concat([loss_df,vloss_df],axis=1)
        if cfg.saving.best_model.enabled:
            best_epoch = 0
            if cfg.saving.best_model.criteria.mode == 'min':
                best_value = float('inf')
            elif cfg.saving.best_model.criteria.mode == 'max':
                best_value = -float('inf')
            else:
                raise ValueError('not implemented')
        # loop over epochs
        for iepoch,epoch in enumerate(epochs):
            dss = []
            for itrain in cfg.data.train_idxs:
                crop_fraction = cfg.data.train[itrain].fraction.value or trainer.scheduler(
                    which=cfg.data.train[itrain].fraction.scheduler,
                    milestone=cfg.data.train[itrain].fraction.milestone,
                    epoch=epoch,output_ratio=False
                )
                for irange in range(train_dss.shape[1]):
                    ds = train_dss[ifold,irange,itrain]
                    ds.random_crop(
                        crop_x=cfg.data.train[itrain].crop_x,
                        crop_y=cfg.data.train[itrain].crop_y,
                        stride_x=cfg.data.train[itrain].stride_x,
                        stride_y=cfg.data.train[itrain].stride_y,
                        crop_fraction=crop_fraction,
                        randomize_start_xy=cfg.data.train[itrain].randomize_start_xy,
                        do_group=cfg.data.train[itrain].do_group,
                        batch_size=cfg.training.batch_size,
                        ngroup_x=cfg.data.train[itrain].ngroup_x,
                        ngroup_y=cfg.data.train[itrain].ngroup_y,
                        drop_last=cfg.data.train[itrain].do_group
                    )
                    dss.append(ds)
            trainer.load_data(
                dss=dss,batch_size=cfg.training.batch_size,
                shuffle_level=cfg.training.shuffle_level
            )
            smoothness_kw = cfg.loss.smoothness_loss
            if smoothness_kw.enabled:
                smoothness_kw.weight.value = smoothness_kw.weight.value or trainer.scheduler(
                    which=smoothness_kw.weight.scheduler,epoch=epoch,
                    milestone=smoothness_kw.weight.milestone,output_ratio=False
                )
                smoothness_kw.B_weight.value = smoothness_kw.B_weight.value or trainer.scheduler(
                    which=smoothness_kw.B_weight.scheduler,epoch=epoch,
                    milestone=smoothness_kw.B_weight.milestone,output_ratio=False
                )
            correlation_kw = cfg.loss.correlation_loss
            if correlation_kw.enabled:
                correlation_kw.weight.value = correlation_kw.weight.value or trainer.scheduler(
                    which=correlation_kw.weight.scheduler,epoch=epoch,
                    milestone=correlation_kw.weight.milestone,output_ratio=False
                )
            result = trainer.train_epoch(
                epoch=epoch,
                yname=cfg.model.yname,
                xnames=cfg.model.xnames,
                use_fit_mask=cfg.loss.use_fit_mask,
                scale_random_error=True,
                smoothness_kw=smoothness_kw,
                correlation_kw=correlation_kw,
                max_norm=cfg.training.gradient_clipping.max_norm,
                clamp_max_sigma=cfg.training.target_clampping.max,
                clamp_min_sigma=cfg.training.target_clampping.min,
                verbose=cfg.experiment.interactive
            )
            for k in result_flds+corr_flds+ssc_flds:
                if k == 'lr':
                    loss_df[f'fold{ifold}_hp{ihp}_lr'].iloc[iepoch] = \
                    trainer.lr_scheduler.get_last_lr()[0]
                else:
                    loss_df[f'fold{ifold}_hp{ihp}_{k}'].iloc[iepoch] = result[k]
            
            if do_val:
                if cfg.experiment.interactive:
                    val_t0 = time.time()
                for ival,vloader in enumerate(vloaders):
                    val_loss = trainer.validate(
                        vloader,
                        yname=cfg.model.yname,
                        xnames=cfg.model.xnames,
                        use_fit_mask=cfg.loss.use_fit_mask,
                        scale_random_error=True
                    )
                    loss_df[f'fold{ifold}_hp{ihp}_val_loss{ival}'].iloc[iepoch] = val_loss
                if cfg.experiment.interactive:
                    val_t = time.time()-val_t0
                    logging.warning(f'inference takes {val_t:.1f}s')
            trainer.lr_scheduler.step()
            # save metrics every epoch if interactive
            if cfg.experiment.interactive and cfg.saving.loss_df.save:
                loss_df.to_csv(os.path.join(cfg.experiment.run_dir,'loss0.csv'))
            
            if cfg.saving.best_model.enabled:
                df_criteria = 'fold{}_hp{}_{}'.format(
                    ifold,ihp,cfg.saving.best_model.criteria.name
                )
                rolling_window = cfg.saving.best_model.criteria.rolling_window
                criteria_value = loss_df[
                    df_criteria
                ].iloc[max(iepoch-rolling_window+1,0):iepoch+1].mean()
                if (
                    (
                        cfg.saving.best_model.criteria.mode == 'min'
                    ) and (
                        criteria_value < best_value
                    )
                ) or (
                    (
                        cfg.saving.best_model.criteria.mode == 'max'
                    ) and (
                        criteria_value > best_value
                    )
                ):
                    best_value = criteria_value
                    best_epoch = epoch
                    model_filename = f'best_model_fold{ifold}_hp{ihp}.pt'
                    trainer.save_model(
                        path=os.path.join(config.experiment.run_dir,model_filename),
                        epoch=epoch
                    ) 
        # end of epoch loop
        loss_dfs.append(loss_df)
        if cfg.saving.loss_df.save:
            loss_df = pd.concat(loss_dfs,axis=1)
            loss_df.to_csv(os.path.join(cfg.experiment.run_dir,'loss.csv'))
        if cfg.saving.unet_out.enabled:
            infer = Inferencer(models=[trainer.model],device=device)
            predict,unet_out,predictor = infer.inference(
                test_ds,xnames=cfg.model.xnames,trim=tkw.trim)
            dds = np.array(
                [
                    l3[cfg.model.yname].detach().numpy() 
                    for l3 in test_ds.l3r
                ]
            )*cfg.data.l3s.var_scales[0]
            dd_map = dds.mean(axis=0)
            dd_ts = dds.mean(axis=(-1,-2))
            predict_map = predict.mean(axis=(0,1,2))
            predict_ts = predict.mean(axis=(0,2,3,4))
            unet_out_map = unet_out.mean(axis=(0,1))
            component_map = (unet_out*predictor).mean(axis=(0,1))
            grid_size_in_m2 = np.cos(
                np.deg2rad(test_ds.latmesh)
            )*np.square(
                test_ds.grid_size.detach().numpy()*111e3
            )
            emission_rate = (
                (unet_out*predictor*grid_size_in_m2).sum(axis=(-1,-2))
            ).mean(axis=0)
            mean_flux = (
                emission_rate/(predictor*grid_size_in_m2).sum(axis=(-1,-2))
            ).mean(axis=0)
            for unet_c in cfg.saving.unet_out.plot_channels:
                fig,ax = plt.subplots(1,1,figsize=(12,10),constrained_layout=True)
                im = ax.imshow(
                    unet_out[0,:,unet_c].mean(axis=0),origin='lower')
                fig.colorbar(im,shrink=0.35)
                if cfg.saving.unet_out.save_plot:
                    fig.savefig(
                        os.path.join(
                            cfg.experiment.run_dir,
                            f'unet_out_{ifold}_{ihp}_{unet_c}.png'
                        ),dpi=150,bbox_inches='tight'
                    )
                    plt.close()
            if cfg.saving.unet_out.save_data:
                pkl_fn = os.path.join(
                    cfg.experiment.run_dir,
                    f'unet_out_{ifold}_{ihp}.pkl'
                )
                with open(pkl_fn, 'wb') as file:
                    pickle.dump(
                        {
                            'predict_map':predict_map,
                            'dd_map':dd_map,
                            'predict_ts':predict_ts,
                            'dd_ts':dd_ts,
                            'unet_out_map':unet_out_map,
                            'component_map':component_map,
                            'emission_rate':emission_rate,
                            'mean_flux':mean_flux
                        },file
                    )
        if cfg.saving.final_model.enabled:
            model_filename = f'final_model_fold{ifold}_hp{ihp}.pt'
            trainer.save_model(
                path=os.path.join(config.experiment.run_dir,model_filename),
                epoch=epoch
            )
    # end of hp loop
# end of fold loop
if cfg.saving.loss_df.save:
    loss_df = pd.concat(loss_dfs,axis=1)
    loss_df.to_csv(os.path.join(cfg.experiment.run_dir,'loss.csv'))