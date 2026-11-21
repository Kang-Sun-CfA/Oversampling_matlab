import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

try:
    from IDS.v05.model import build_model
    from IDS.v05.trainer import run_model
except ImportError:
    logging.warning('assuming modules are here')
    from model import build_model
    from trainer import run_model


def r2_score(predict,target):
    '''
    coefficient of determination over pixels where both are finite
    predict,target:
        arrays of the same shape, e.g., [nframe,1,H,W]
    '''
    mask = np.isfinite(predict) & np.isfinite(target)
    residual = np.sum(np.square(target[mask]-predict[mask]))
    total = np.sum(np.square(target[mask]-target[mask].mean()))
    return 1-residual/total


class Inferencer:
    '''
    run trained models over a CropDataset, typically of the infer role (fixed 
    crops, min_valid_pixels=0), and mosaic the crops back into full frames. 
    overlapping crops are averaged after trimming `trim` pixels off crop edges 
    that are not frame edges, where the unet sees less context
    '''
    def __init__(self,models,device=None):
        '''models: a model or a list of models, e.g., from folds or seeds'''
        self.logger = logging.getLogger(__name__)
        if not isinstance(models,(list,tuple)):
            models = [models]
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.models = [model.to(self.device).eval() for model in models]
    
    @classmethod
    def from_checkpoints(cls,config,paths,device=None):
        '''models built from config, with weights from checkpoint files'''
        models = []
        for path in paths:
            model = build_model(config)
            checkpoint = torch.load(path,map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            models.append(model)
        return cls(models,device=device)
    
    @staticmethod
    def trim_window(i,j,crop_y,crop_x,frame_y,frame_x,trim):
        '''rows and columns of a crop at (i,j) kept after trimming, as slices'''
        top = 0 if i == 0 else trim
        bottom = 0 if i+crop_y == frame_y else trim
        left = 0 if j == 0 else trim
        right = 0 if j+crop_x == frame_x else trim
        return slice(top,crop_y-bottom),slice(left,crop_x-right)
    
    def infer(self,ds,xnames,trim=2,batch_size=256):
        '''
        mosaicked frames of ds for every model
        returns dict of:
            predict: [nmodel,nframe,1,H,W], modeled DD (all_flux), loss units
            unet_out: [nmodel,nframe,C,H,W], coefficients of each predictor
            predictor: [nframe,C,H,W], the predictors, the same for all models
            target: [nframe,1,H,W], observed DD in loss units, nan where invalid
            valid_mask: [nframe,H,W]
            point_flux: [nmodel,nframe,1,H,W] and point_rate: [nmodel,nframe,nsource],
                if the models have point sources
            frame_df,xgrid,ygrid: time and space of the frames
        '''
        sampler = ds.sampler
        crop_y,crop_x = sampler.crop_y,sampler.crop_x
        frame_y,frame_x = sampler.frame_y,sampler.frame_x
        # trimmed crops must still overlap, or pixels between them are never seen
        if sampler.stride_y > crop_y-2*trim or sampler.stride_x > crop_x-2*trim:
            raise ValueError(
                f'stride ({sampler.stride_y},{sampler.stride_x}) must not exceed '
                f'crop - 2*trim ({crop_y-2*trim},{crop_x-2*trim}), or the mosaic has holes')
        nmodel,nframe = len(self.models),ds.nframe
        nchannel = sum(ds[0][xn].shape[0] for xn in xnames)
        do_point = all(getattr(model,'do_point',False) for model in self.models)
        # sums of overlapping crops, divided by counts at the end
        sums = dict(
            predict=np.zeros((nmodel,nframe,1,frame_y,frame_x),dtype=np.float32),
            unet_out=np.zeros((nmodel,nframe,nchannel,frame_y,frame_x),dtype=np.float32)
        )
        if do_point:
            sums['point_flux'] = np.zeros((nmodel,nframe,1,frame_y,frame_x),dtype=np.float32)
            point_rate = np.zeros(
                (nmodel,nframe,self.models[0].point_mlp.n_sources),dtype=np.float32)
        predictor = np.zeros((nframe,nchannel,frame_y,frame_x),dtype=np.float32)
        count = np.zeros((nframe,1,frame_y,frame_x),dtype=np.float32)
        
        loader = DataLoader(ds,batch_size=batch_size,shuffle=False)
        for imodel,model in enumerate(self.models):
            with torch.no_grad():
                for batch in loader:
                    batch = {
                        k:v.to(self.device,non_blocking=(self.device.type=='cuda'))
                        for k,v in batch.items()
                    }
                    out,predictors = run_model(model,batch,xnames)
                    fields = dict(predict=out['all_flux'],unet_out=out['unet_out'])
                    if do_point:
                        fields['point_flux'] = out['point_flux']
                    fields = {k:v.cpu().numpy() for k,v in fields.items()}
                    if imodel == 0:
                        predictors = predictors.cpu().numpy()
                    # each sample knows where its crop is, so batches can be any size
                    for b,(iframe,i,j) in enumerate(batch['crop'].tolist()):
                        ys,xs = self.trim_window(i,j,crop_y,crop_x,frame_y,frame_x,trim)
                        fy,fx = slice(i+ys.start,i+ys.stop),slice(j+xs.start,j+xs.stop)
                        for name,value in fields.items():
                            sums[name][imodel,iframe,:,fy,fx] += value[b][:,ys,xs]
                        if do_point:
                            # rates depend on time only, the same for all crops of a frame
                            point_rate[imodel,iframe] = out['point_rate'][b].cpu().numpy()
                        if imodel == 0:
                            predictor[iframe,:,fy,fx] += predictors[b][:,ys,xs]
                            count[iframe,:,fy,fx] += 1
        
        # average overlapping crops; nan where a frame has no crop (skipped crops)
        with np.errstate(invalid='ignore',divide='ignore'):
            result = {name:value/count[None] for name,value in sums.items()}
            result['predictor'] = predictor/count
        if do_point:
            result['point_rate'] = point_rate
        # observations from the dataset frames, in loss units
        dd = ds.frames[:,ds.channel_idx[ds.var_names[0]]]*ds.var_scales[0]
        result['target'] = np.where(ds.valid_mask,dd,np.nan)[:,None].astype(np.float32)
        result['valid_mask'] = ds.valid_mask
        result.update(frame_df=ds.frame_df,xgrid=ds.xgrid,ygrid=ds.ygrid)
        return result
