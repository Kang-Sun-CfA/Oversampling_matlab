'''
train all folds (and hp combinations) of a config. two ways to run:
    python IDS/v05/train.py config.yml [git_path]  # in terminal or sbatch
    from IDS.v05.train import run; run(config)     # in a notebook
git_path is the directory holding IDS (and CAREER, popy); by default two levels 
above this file
'''
import sys,os
import logging
import warnings

if __name__ == '__main__':
    git_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..')
    sys.path.append(git_path)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from IDS.v05.config import Config
    from IDS.v05.data import build_streams,build_cdl,build_datasets
    from IDS.v05.model import build_model
    from IDS.v05.trainer import Trainer,scheduled_value
    from IDS.v05.inference import Inferencer,r2_score
except ImportError:
    logging.warning('assuming modules are here')
    from config import Config
    from data import build_streams,build_cdl,build_datasets
    from model import build_model
    from trainer import Trainer,scheduled_value
    from inference import Inferencer, r2_score

def epoch_loss_kws(cfg,epoch):
    '''loss configs with the weights of this epoch, as Trainer.train_epoch needs'''
    loss_kws = {}
    for key,weight_names in [
        ('smoothness_loss',['weight','B_weight']),
        ('correlation_loss',['weight']),
        ('point_l1_loss',['weight'])
    ]:
        kw = cfg.loss[key].to_dict()
        if kw['enabled']:
            for name in weight_names:
                kw[name]['value'] = scheduled_value(kw[name],epoch)
        loss_kws[key] = kw
    return loss_kws


def is_better(value,best_value,mode):
    if mode == 'min':
        return value < best_value
    elif mode == 'max':
        return value > best_value
    raise ValueError(f'best model mode must be min or max, not {mode}')


def run(config,streams=None,cdl=None,device=None,return_models=False):
    '''
    train one model per fold and hp combination. saves loss.csv and best/final 
    models in experiment.run_dir
    streams,cdl:
        prebuilt by build_streams/build_cdl, so repeated runs skip the slow 
        loading; built here if None. rebuild them after changing what they 
        depend on: split, stream entries (specs, domain, fields, ...), or cdl
    returns:
        loss_df, epochs as index, columns as fold{i}_hp{j}_{metric}
        trained models if return_models
    '''
    logger = logging.getLogger(__name__)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = config.experiment.run_dir
    if config.model.point.enabled:
        raise NotImplementedError('point sources come back later; set model.point.enabled to False')
    
    torch.manual_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.experiment.seed)
        torch.cuda.manual_seed_all(config.experiment.seed)
    
    # data: streams and cdl once (unless given), datasets per fold
    do_infer = config.saving.inference.enabled
    if streams is None:
        # infer frames only if inference runs during training
        roles = ('train','val','infer') if do_infer else ('train','val')
        streams = build_streams(config,roles=roles)
    if cdl is None:
        cdl = build_cdl(config)
    # infer datasets use all dates, the same for every fold and hp
    infer_dss = build_datasets(config,streams,cdl=cdl,role='infer') if do_infer else {}
    if do_infer and not infer_dss:
        logger.warning('inference is enabled, but streams have no infer frames')
    
    epochs = np.arange(config.training.start_epoch,config.training.end_epoch)
    cfgs = config.get_hp_combinations() if config.hp_tuning.enabled else [config]
    loss_dfs = []
    if return_models:
        models = np.empty((config.data.split.nfold,len(cfgs)),dtype=object)
    if config.saving.inference.enabled:
        infer_outs = np.empty((config.data.split.nfold,len(cfgs)),dtype=object)
    for ifold in range(config.data.split.nfold):
        train_dss = build_datasets(config,streams,cdl=cdl,ifold=ifold,role='train')
        val_dss = build_datasets(config,streams,cdl=cdl,ifold=ifold,role='val')
        # validation crops are fixed, so loaders are made once per fold
        val_loaders = {
            name:DataLoader(
                ds,batch_size=config.training.batch_size,shuffle=False,
                pin_memory=(device.type=='cuda')
            ) for name,ds in val_dss.items()
        }
        for ihp,cfg in enumerate(cfgs):
            prefix = f'fold{ifold}_hp{ihp}_'
            model = build_model(cfg)
            if cfg.model.pretrained_model_path:
                checkpoint = torch.load(cfg.model.pretrained_model_path,map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            trainer = Trainer(
                model=model,lr_kw=cfg.training.lr.to_dict(),
                wd_kw=cfg.training.weight_decay.to_dict(),device=device,
                L1Loss_or_MSE=cfg.loss.main_loss
            )
            ds_cfgs = {d.name:d for d in cfg.data.datasets}
            criteria = cfg.saving.best_model.criteria
            best_value = np.inf if criteria.mode == 'min' else -np.inf
            # one dict of metrics per epoch; the frame is rebuilt from them, as 
            # adding columns one by one fragments it
            epoch_results = []
            if cfg.saving.inference.enabled: infer_epochs = []
            # loop over epochs
            for iepoch,epoch in enumerate(epochs):
                # crops of this epoch, at each dataset's scheduled fraction
                for name,ds in train_dss.items():
                    ds.set_epoch(
                        epoch,crop_fraction=scheduled_value(ds_cfgs[name].fraction,epoch))
                trainer.load_data(list(train_dss.values()),cfg.training.batch_size)
                loss_kws = epoch_loss_kws(cfg,epoch)
                result = trainer.train_epoch(
                    epoch=epoch,
                    yname=cfg.model.yname,
                    xnames=cfg.model.xnames,
                    scale_random_error=True,
                    smoothness_kw=loss_kws['smoothness_loss'],
                    correlation_kw=loss_kws['correlation_loss'],
                    point_l1_kw=loss_kws['point_l1_loss'],
                    max_norm=cfg.training.gradient_clipping.max_norm,
                    clamp_max_sigma=cfg.training.target_clampping.max,
                    clamp_min_sigma=cfg.training.target_clampping.min,
                    verbose=cfg.experiment.interactive
                )
                result['lr'] = trainer.lr_scheduler.get_last_lr()[0]
                # validation per dataset, e.g., per stream or subdomain
                for name,loader in val_loaders.items():
                    result[f'val_loss_{name}'] = trainer.validate(
                        loader,yname=cfg.model.yname,xnames=cfg.model.xnames,
                        scale_random_error=True
                    )
                if val_loaders:
                    result['val_loss_mean'] = np.mean(
                        [result[f'val_loss_{name}'] for name in val_loaders])
                epoch_results.append({prefix+k:v for k,v in result.items()})
                loss_df = pd.DataFrame(epoch_results,index=epochs[:len(epoch_results)])
                trainer.lr_scheduler.step()
                
                # inference on the infer datasets, every log_every epochs
                ic = cfg.saving.inference
                if ic.enabled and (iepoch+1) % ic.log_every == 0:
                    inferencer = Inferencer(models=[trainer.model])
                    infer_epoch = dict(epoch=epoch)
                    for name,infer_ds in infer_dss.items():
                        infer_out = inferencer.infer(ds=infer_ds,xnames=cfg.model.xnames)
                        infer_out['component'] = infer_out['predictor'][None]*infer_out['unet_out']
                        for field in ic.save_fields or []:
                            infer_epoch[f'{field}_{name}'] = infer_out[field]
                        for imons,mons in enumerate(ic.stats_months or [list(range(1,13))]):
                            time_mask = infer_out['frame_df']['mid'].dt.month.isin(mons).values
                            # mean maps over the same frames: predict only where observed
                            target = infer_out['target'][time_mask] # [nt,1,H,W]
                            predict = infer_out['predict'][:,time_mask].mean(axis=0)
                            predict = np.where(np.isfinite(target),predict,np.nan)
                            with warnings.catch_warnings(): # all-nan pixels stay nan
                                warnings.simplefilter('ignore',RuntimeWarning)
                                r2 = r2_score(np.nanmean(predict,axis=(0,1)),
                                              np.nanmean(target,axis=(0,1)))
                            infer_epoch[f'r2_{name}_mon{imons}'] = r2
                            if cfg.experiment.interactive:
                                logger.warning(f'mons {mons[0]}-{mons[-1]} mean {name} frame r2={r2:.3f}')
                    infer_epochs.append(infer_epoch)
                    # r2 also goes to loss_df, so it is saved and can pick the best model
                    epoch_results[-1].update(
                        {prefix+k:v for k,v in infer_epoch.items() if k.startswith('r2_')})
                    loss_df = pd.DataFrame(epoch_results,index=epochs[:len(epoch_results)])
                
                # save metrics every epoch if interactive
                if cfg.experiment.interactive and cfg.saving.loss_df.save:
                    loss_df.to_csv(os.path.join(run_dir,'loss0.csv'))
                
                if cfg.saving.best_model.enabled:
                    if prefix+criteria.name not in loss_df:
                        raise KeyError(
                            f'best model criteria {criteria.name} is not a recorded '
                            f'metric; choose from {[c[len(prefix):] for c in loss_df]}')
                    # mean over the last rolling_window epochs
                    criteria_value = loss_df[prefix+criteria.name].loc[:epoch].iloc[
                        -criteria.rolling_window:].mean()
                    if is_better(criteria_value,best_value,criteria.mode):
                        best_value = criteria_value
                        trainer.save_model(
                            path=os.path.join(run_dir,f'best_model_fold{ifold}_hp{ihp}.pt'),
                            epoch=int(epoch) # plain int: newer torch.load refuses numpy
                        )
                
            # end of epoch loop
            if cfg.saving.inference.enabled:
                infer_outs[ifold,ihp] = infer_epochs
            if return_models:
                models[ifold,ihp] = trainer.model
            if cfg.saving.final_model.enabled:
                trainer.save_model(
                    path=os.path.join(run_dir,f'final_model_fold{ifold}_hp{ihp}.pt'),
                    epoch=int(epoch)
                )
            loss_dfs.append(loss_df)
            if cfg.saving.loss_df.save:
                pd.concat(loss_dfs,axis=1).to_csv(os.path.join(run_dir,'loss.csv'))
        # end of hp loop
    # end of fold loop
    run_output = {'loss_df':pd.concat(loss_dfs,axis=1)}
    if return_models:
        run_output['models'] = models
    if config.saving.inference.enabled:
        run_output['infer'] = infer_outs
    return run_output


if __name__ == '__main__':
    config = Config.from_yaml(sys.argv[1])
    run_output = run(config)