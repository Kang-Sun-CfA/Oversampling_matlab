import os
import logging
import yaml


NO2_DEFAULT_CONFIG = {
    'experiment':{
        'run_id':'test',
        'run_dir':'/projects/academic/kangsun/kangsun/IDS/v05/NO2/test',
        'mkdir':True,
        'seed':42,
        'interactive':False
    },
    'data':{
        # global bounds normalizing lon/lat, the same for all streams and grids
        'wesnmost':{'westmost':-128,'eastmost':-65,'southmost':24,'northmost':50},
        # fractional year starts here as year 0.
        'base_year':2020,
        # split by date, shared by all streams, so a date is never validation 
        # in one stream and training in another. start/end should cover all time
        'split':{
            'nfold':5,'val_interval':'1M','val_fraction':0.,'seed':None,
            'start':'2020-01-01','end':'2025-12-31'
        },
        # streams: one per (instrument product, grid), accumulating unit frames to
        # intervals. start/end use split.start/end by default and cannot go beyond
        'streams':[{
            'name':'tropomi_no2_0.1',
            'path_pattern':'/projects/academic/kangsun/zolalaya/data/Cornbelt_NOx/'\
            's5p_cornbelt/%Y/%m/CORNBELT_S5P_NO2_%Y_%m_%d.nc',
            'start':None,'end':None,
            'local_hours':[13.5],
            'grid_size':0.1,# block_reduce target; native grid if equal
                # streams sharing a psf: same instrument and DD method
                'psf_idx':0,
            'domain':{'west':-104,'east':-80.5,'south':36,'north':46},
            # fields aggregated by Stream; must include var_names, and _xy/_rs 
            # if random_error is xy_rs
            'field_names':[
                'column_amount_DD','column_amount_DD_xy','column_amount_DD_rs',
                'column_amount','surface_altitude_DD'
            ],
            # fields returned by CropDataset in loss units; the first is the DD field
            'var_names':['column_amount_DD','column_amount','surface_altitude_DD'],
            'var_scales':[1e9,-1e4,-1e6],# very important, vcd and wt need flipping
            'min_num_samples':0.7,
            'random_error':{
                'method':'clt',# clt or xy_rs
                'sigma0':1e-8,# clt: error at one layer of coverage, physical units
                'xy_rs_coeffs':[1.02,18.21],# xy_rs: 0.5*std(xy-rs)*(a+b*grid_size)
                'fallback':1e3# loss units, if the estimate is not finite
            },
            # all (interval,offset) specs per role, aggregated in one pass. 
            # specs a dataset uses must be next to each other here
            'train':{
                'specs':[
                    ['91D','0D'],['91D','30D'],['91D','60D'],
                    ['35D','0D'],['35D','17D'],
                    ['28D','0D'],['28D','14D']
                ],
                'min_coverage':0.# option to remove truncated intervals
            },
            'val':{'specs':[['1M','0D']],'min_coverage':0.},
            # infer: all dates, no folds, for mosaicking and model evaluation
            'infer':{'specs':[['1M','0D']],'min_coverage':0.}
        }],
        # datasets: views of a stream, sliced by space/time/role, then crop
        # window can be None (whole stream domain) or a wesn dict. min_valid_pixels
        # should be 0 for infer ds: every crop is kept, so mosaicked frames have no holes
        'datasets':[
            {
                'name':'train_91d','enabled':True,'stream':'tropomi_no2_0.1','role':'train',
                'window':None,
                'specs':[['91D','0D'],['91D','30D'],['91D','60D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,1,35,0],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'train_35d','enabled':True,'stream':'tropomi_no2_0.1','role':'train',
                'window':None,
                'specs':[['35D','0D'],['35D','17D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,0,35,0.5],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'train_28d','enabled':True,'stream':'tropomi_no2_0.1','role':'train',
                'window':None,
                'specs':[['28D','0D'],['28D','14D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,0,35,0.5],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'val_all','enabled':True,'stream':'tropomi_no2_0.1','role':'val',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':32,'stride_x':32,'mode':'fixed'},
                'min_valid_pixels':3
            },
            {
                'name':'val_west','enabled':True,'stream':'tropomi_no2_0.1','role':'val',
                'window':{'west':-104,'east':-95,'south':38,'north':46},
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':32,'stride_x':32,'mode':'fixed'},
                'min_valid_pixels':3
            },
            {
                'name':'val_east','enabled':True,'stream':'tropomi_no2_0.1','role':'val',
                'window':{'west':-90,'east':-81,'south':37,'north':45},
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':32,'stride_x':32,'mode':'fixed'},
                'min_valid_pixels':3
            },
            {
                'name':'infer_all','enabled':True,'stream':'tropomi_no2_0.1','role':'infer',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'fixed'},
                'min_valid_pixels':0
            }
        ],
        # auxiliary data, matched to each dataset's grid and window
        'cdl':{
            'enabled':True,
            'path_pattern':'/projects/academic/kangsun/data/GIS_data/CDL/fractions/'\
            'CDL_0.10_%Y.nc',
            'lcc_path':'/projects/academic/kangsun/data/GIS_data/CDL/fractions/lcc.csv',
            'years':[2020,2021,2022,2023,2024,2025],
            'cafo':{'enabled':False,'scaling':1.,'paths':None},
            'name_codes':[
                ['corn',[1]],['soybean',[5]],['wheat',[22,23,24]],
                ['dev_o',[121]],['dev_l',[122]],['dev_m',[123]],['dev_h',[124]]
            ]
        },
        # unchanged from v04 for now; revisited with point sources
        'pss':{
            'enabled':False,
            'csv':{
                'enabled':False,
                'path':'/projects/academic/kangsun/kangsun/IDS/v02/'\
                'cornbelt_point_sources.csv'
            },
            'cems':{
                'enabled':True,
                'attributes_path_pattern':'/projects/academic/kangsun/data/CEMS/'\
                'attributes/trimmed_%Y.csv',
                'emissions_path_pattern':'/projects/academic/kangsun/data/CEMS/'\
                'emissions/%Y/%m/%d/%Y%m%d.csv',
                'n_facility_with_most_NOx':50,
                'fid_blacklist':[2876,1167],
                'local_hours':[13],
                'rolling_window':'30d'
            }
        }
    },
    'model':{
        'pretrained_model_path':None,
        'spatial':{
            'unet_depth':4,'base_channels':64,'leaky_relu_rate':0.1,'norm_groups':8,
            'force_positive':True,'in_channels':2,'out_channels':None
        },
        # temporal_dim is derived from the encoder, no longer set by hand
        'temporal':{
            'hidden_dim':128,'leaky_relu_rate':0.1,'modulate_decoder':False,
            'encoder':{'n_seasonal':1,'n_diurnal':0,'n_whole':0,'whole_period':None}
        },
        'point':{
            'enabled':False,'hidden_dim':128,'leaky_relu_rate':0.1,'n_layers':8,'dropout':0,
            'encoder':{'n_seasonal':1,'n_diurnal':0,'n_whole':0,'whole_period':12.}
        },
        'psf':{'enabled':True,'kernel_size':5,'unet_skip_channels':[0,1]},
        'yname':'column_amount_DD',
        'xnames':['column_amount','surface_altitude_DD','cdl']
    },
    'loss':{
        'main_loss':'L1Loss',
        'smoothness_loss':{
            'enabled':True,
            'weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'B_weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'scale_weights':[1,0.5,0.5,0.25],
            'channel_weights':[10,5,1,1,1,1,0.1,0.1,0.1,1],
            'output_ssc':True
        },
        'correlation_loss':{
            'enabled':True,
            'include_predictor':True,
            'weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'channel_weights':None,
            'use_double':True,
            'output_corr':True
        },
        'point_l1_loss':{
            'enabled':True,
            'weight':{'value':0.,'milestone':[0,0,50,0],'scheduler':'linear'}
        }
    },
    'training':{
        'start_epoch':0,'end_epoch':52,
        'batch_size':256,# samples are shuffled; do_group and shuffle_level are gone
        'lr':{
            'value':5e-4,
            'milestone':[0,5e-4,50,5e-4],
            'scheduler':'linear',
            'multipliers':{'other':1,'temporal_film':5,'psf':20}
        },
        'weight_decay':{
            'value':1e-2,
            'multipliers':{'other':1,'temporal_film':0,'psf':0}
        },
        'gradient_clipping':{'max_norm':1.0},
        'target_clampping':{'min':None,'max':None}
    },
    'hp_tuning':{
        'enabled':False,
        'hps':[
            'loss.smoothness_loss.weight.milestone',
            'loss.smoothness_loss.B_weight.milestone'
        ]
    },
    'saving':{
        'loss_df':{
            'save':True,
            'result_flds':[
                'train_loss','smooth_loss','smooth_B_loss',
                'epoch_time','sample_size','lr','point_l1_loss',
                'grad_norm','nan_norm','zero_norm','cor_loss'
            ]
        },
        'point':{'save_pser_df':True},
        'best_model':{
            'enabled':True,
            # val_loss_mean averages the val datasets; val_loss_{name} picks one
            'criteria':{'name':'val_loss_mean','mode':'min','rolling_window':3}
        },
        'final_model':{'enabled':True},
        'inference':{
            'enabled':True,'log_every':5,'stats_months':[[5,6,7,8,9]],
            'save_fields':None # choose from ['target','predict','predictor','component']
        }
    }
}


NH3_DEFAULT_CONFIG = {
    'experiment':{
        'run_id':'test',
        'run_dir':'/projects/academic/kangsun/kangsun/IDS/v05/NH3/test',
        'mkdir':False,
        'seed':42,
        'interactive':True
    },
    'data':{
        # global bounds normalizing lon/lat, the same for all streams and grids
        'wesnmost':{'westmost':-128,'eastmost':-65,'southmost':24,'northmost':50},
        # fractional year starts here as year 0.
        'base_year':2020,
        # split by date, shared by all streams, so a date is never validation 
        # in one stream and training in another. start/end should cover all time
        'split':{
            'nfold':1,'val_interval':'1M','val_fraction':0.,'seed':None,
            'start':'2020-01-01','end':'2020-12-31'
        },
        # streams: one per (instrument product, grid), accumulating unit frames to
        # intervals. start/end use split.start/end by default and cannot go beyond
        'streams':[
            {
                'name':'iasi',
                'path_pattern':'/projects/academic/kangsun/data/IASINH3/L4/L34/'\
                '%Y/%m/L4_%Y%m%d.nc',
                'start':None,'end':None,
                'local_hours':[9.5],
                'grid_size':0.1,# block_reduce target; native grid if equal
                # streams sharing a psf: same instrument and DD method
                'psf_idx':0,
                'domain':{'west':-105.5,'east':-85,'south':34.1,'north':43.9},
                # fields aggregated by Stream; must include var_names, and _xy/_rs 
                # if random_error is xy_rs
                'field_names':[
                    'column_amount_DD','column_amount_DD_xy','column_amount_DD_rs',
                    'column_amount','surface_altitude_DD'
                ],
                # fields returned by CropDataset in loss units; the first is the DD field
                'var_names':['column_amount_DD','column_amount','surface_altitude_DD'],
                'var_scales':[1e9,-1e4,-1e6],# very important, vcd and wt need flipping
                'min_num_samples':0.7,
                'random_error':{
                    'method':'clt',# clt or xy_rs
                    'sigma0':3.9e-8,# clt: error at one layer of coverage, physical units
                    'xy_rs_coeffs':[1.02,18.21],# xy_rs: 0.5*std(xy-rs)*(a+b*grid_size)
                    'fallback':1e3# loss units, if the estimate is not finite
                },
                # all (interval,offset) specs per role, aggregated in one pass. 
                # specs a dataset uses must be next to each other here
                'train':{
                    'specs':[
                        ['3M','0D'],['3M','30D'],['3M','60D'],
                        ['1M','3D'],['1M','17D']
                    ],
                    'min_coverage':0.# option to remove truncated intervals
                },
                'val':{'specs':[['1M','0D']],'min_coverage':0.},
                # infer: all dates, no folds, for mosaicking and model evaluation
                'infer':{'specs':[['1M','0D']],'min_coverage':0.}
            },
            {
                'name':'cris',
                'path_pattern':'/projects/academic/kangsun/data/CrISNH3/L4/L34/'\
                '%Y/%m/L4_%Y%m%d.nc',
                'start':None,'end':None,
                'local_hours':[13.5],
                'grid_size':0.1,# block_reduce target; native grid if equal
                # streams sharing a psf: same instrument and DD method
                'psf_idx':1,
                'domain':{'west':-105.5,'east':-85,'south':34.1,'north':43.9},
                # fields aggregated by Stream; must include var_names, and _xy/_rs 
                # if random_error is xy_rs
                'field_names':[
                    'column_amount_DD','column_amount_DD_xy','column_amount_DD_rs',
                    'column_amount','surface_altitude_DD'
                ],
                # fields returned by CropDataset in loss units; the first is the DD field
                'var_names':['column_amount_DD','column_amount','surface_altitude_DD'],
                'var_scales':[1e9,-1e4,-1e6],# very important, vcd and wt need flipping
                'min_num_samples':0.7,
                'random_error':{
                    'method':'clt',# clt or xy_rs
                    'sigma0':7.65e-8,# clt: error at one layer of coverage, physical units
                    'xy_rs_coeffs':[1.02,18.21],# xy_rs: 0.5*std(xy-rs)*(a+b*grid_size)
                    'fallback':1e3# loss units, if the estimate is not finite
                },
                # all (interval,offset) specs per role, aggregated in one pass. 
                # specs a dataset uses must be next to each other here
                'train':{
                    'specs':[
                        ['3M','0D'],['3M','30D'],['3M','60D'],
                        ['1M','3D'],['1M','17D']
                    ],
                    'min_coverage':0.# option to remove truncated intervals
                },
                'val':{'specs':[['1M','0D']],'min_coverage':0.},
                # infer: all dates, no folds, for mosaicking and model evaluation
                'infer':{'specs':[['1M','0D']],'min_coverage':0.}
            }
        ],
        # datasets: views of a stream, sliced by space/time/role, then crop
        # window can be None (whole stream domain) or a wesn dict. min_valid_pixels
        # should be 0 for infer ds: every crop is kept, so mosaicked frames have no holes
        'datasets':[
            {
                'name':'iasi_train_3M','enabled':True,'stream':'iasi','role':'train',
                'window':None,
                'specs':[['3M','0D'],['3M','30D'],['3M','60D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,1,35,0],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'iasi_train_1M','enabled':True,'stream':'iasi','role':'train',
                'window':None,
                'specs':[['1M','3D'],['1M','17D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,0,35,1],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'iasi_val','enabled':True,'stream':'iasi','role':'val',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':32,'stride_x':32,'mode':'fixed'},
                'min_valid_pixels':3
            },
            {
                'name':'iasi_infer','enabled':True,'stream':'iasi','role':'infer',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'fixed'},
                'min_valid_pixels':0
            },
            {
                'name':'cris_train_3M','enabled':True,'stream':'cris','role':'train',
                'window':None,
                'specs':[['3M','0D'],['3M','30D'],['3M','60D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,1,35,0],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'cris_train_1M','enabled':True,'stream':'cris','role':'train',
                'window':None,
                'specs':[['1M','3D'],['1M','17D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'random'},
                'min_valid_pixels':3,
                'fraction':{'value':None,'milestone':[15,0,35,1],'scheduler':'cosine'},
                'jitter':{'p_jitter':1.,'lh_span':0.}
            },
            {
                'name':'cris_val','enabled':True,'stream':'cris','role':'val',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':32,'stride_x':32,'mode':'fixed'},
                'min_valid_pixels':3
            },
            {
                'name':'cris_infer','enabled':True,'stream':'cris','role':'infer',
                'window':None,
                'specs':[['1M','0D']],
                'crop':{'crop_y':64,'crop_x':64,'stride_y':8,'stride_x':8,'mode':'fixed'},
                'min_valid_pixels':0
            }
        ],
        # auxiliary data, matched to each dataset's grid and window
        'cdl':{
            'enabled':True,
            'path_pattern':'/projects/academic/kangsun/data/GIS_data/CDL/fractions/'\
            'heartland_CDL_0.10_%Y.nc',
            'lcc_path':'/projects/academic/kangsun/data/GIS_data/CDL/fractions/lcc.csv',
            'years':[2019,2020,2021],
            'cafo':{
                'enabled':True,'scaling':100.,
                'paths':{
                    'cow':'/projects/academic/kangsun/data/GIS_data/CAFO/cafo_cow_midwest.csv',
                    'pig':'/projects/academic/kangsun/data/GIS_data/CAFO/cafo_pig_midwest.csv'
                }
            },
            'name_codes':[
                ['corn',[1,12,13]],['soybean',[5]],
                ['grains',[4,21,22,23,24,25,26,27,28,29,30]],
                ['hay',[36,37]],['pasture',[58,176]],
                ['suburban',[121,122]],['urban',[123,124]]
            ]
        },
        # unchanged from v04 for now; revisited with point sources
        'pss':{
            'enabled':False,
            'csv':{
                'enabled':False,
                'path':'/projects/academic/kangsun/kangsun/IDS/v02/'\
                'cornbelt_point_sources.csv'
            },
            'cems':{
                'enabled':True,
                'attributes_path_pattern':'/projects/academic/kangsun/data/CEMS/'\
                'attributes/trimmed_%Y.csv',
                'emissions_path_pattern':'/projects/academic/kangsun/data/CEMS/'\
                'emissions/%Y/%m/%d/%Y%m%d.csv',
                'n_facility_with_most_NOx':50,
                'fid_blacklist':[2876,1167],
                'local_hours':[13],
                'rolling_window':'30d'
            }
        }
    },
    'model':{
        'pretrained_model_path':None,
        'spatial':{
            'unet_depth':4,'base_channels':64,'leaky_relu_rate':0.1,'norm_groups':8,
            'force_positive':True,'in_channels':2,'out_channels':None
        },
        # temporal_dim is derived from the encoder, no longer set by hand
        'temporal':{
            'hidden_dim':128,'leaky_relu_rate':0.1,'modulate_decoder':False,
            'encoder':{'n_seasonal':1,'n_diurnal':0,'n_whole':0,'whole_period':None}
        },
        'point':{
            'enabled':False,'hidden_dim':128,'leaky_relu_rate':0.1,'n_layers':8,'dropout':0,
            'encoder':{'n_seasonal':1,'n_diurnal':0,'n_whole':0,'whole_period':12.}
        },
        'psf':{'enabled':True,'kernel_size':5,'unet_skip_channels':[0,1]},
        'yname':'column_amount_DD',
        'xnames':['column_amount','surface_altitude_DD','cdl']
    },
    'loss':{
        'main_loss':'L1Loss',
        'smoothness_loss':{
            'enabled':True,
            'weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'B_weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'scale_weights':[1,0.5,0.5,0.25],
            'channel_weights':1.,
            'output_ssc':True
        },
        'correlation_loss':{
            'enabled':True,
            'include_predictor':True,
            'weight':{'value':None,'milestone':[0,0,50,0],'scheduler':'linear'},
            'channel_weights':None,
            'use_double':True,
            'output_corr':True
        },
        'point_l1_loss':{
            'enabled':True,
            'weight':{'value':0.,'milestone':[0,0,50,0],'scheduler':'linear'}
        }
    },
    'training':{
        'start_epoch':0,'end_epoch':52,
        'batch_size':256,# samples are shuffled; do_group and shuffle_level are gone
        'lr':{
            'value':5e-4,
            'milestone':[0,5e-4,50,5e-4],
            'scheduler':'linear',
            'multipliers':{'other':1,'temporal_film':5,'psf':20}
        },
        'weight_decay':{
            'value':1e-2,
            'multipliers':{'other':1,'temporal_film':0,'psf':0}
        },
        'gradient_clipping':{'max_norm':10.},
        'target_clampping':{'min':None,'max':None}
    },
    'hp_tuning':{
        'enabled':False,
        'hps':[
            'loss.smoothness_loss.weight.milestone',
            'loss.smoothness_loss.B_weight.milestone'
        ]
    },
    'saving':{
        'loss_df':{
            'save':True,
            'result_flds':[
                'train_loss','smooth_loss','smooth_B_loss',
                'epoch_time','sample_size','lr','point_l1_loss',
                'grad_norm','nan_norm','zero_norm','cor_loss'
            ]
        },
        'point':{'save_pser_df':True},
        'best_model':{
            'enabled':False,
            # val_loss_mean averages the val datasets; val_loss_{name} picks one
            'criteria':{'name':'val_loss_mean','mode':'min','rolling_window':3}
        },
        'final_model':{'enabled':False},
        'inference':{
            'enabled':True,'log_every':5,'stats_months':[[5,6,7,8,9]],
            'save_fields':None # choose from ['target','predict','predictor','component']
        }
    }
}


# presaved defaults, selected by Config('NO2') or Config('NH3')
DEFAULT_CONFIGS = {'NO2':NO2_DEFAULT_CONFIG,'NH3':NH3_DEFAULT_CONFIG}


class ConfigNode(dict):
    """A dictionary subclass that allows attribute access to nested keys."""
    def __init__(self, data=None):
        if data is None:
            data = {}
        super().__init__()
        for key, value in data.items():
            self[key] = self._wrap(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigNode' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, dict):
            return ConfigNode(value)
        elif isinstance(value, (list, tuple)):
            # tuples become lists, so to_yaml writes plain yaml that from_yaml reads
            return [self._wrap(v) for v in value]
        else:
            return value

    def to_dict(self):
        """Convert the ConfigNode and its children back to a standard dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, ConfigNode) else v for v in value]
            else:
                result[key] = value
        return result


class Config:
    '''
    v05 configuration. in a notebook:
        config = Config('NO2')                 # presaved defaults, or 'NH3'
        config = Config(my_dict)               # a full config dict
        config = Config.from_yaml('config.yml')
        config.data.split.nfold = 3            # attribute access
        config.update({'training':{'end_epoch':10}})
        config.check()                         # after manual edits
        config.to_yaml(abs_path='config.yml')
    '''
    def __init__(self, config_dict='NO2'):
        self.logger = logging.getLogger(__name__)
        
        # a name picks presaved defaults; ConfigNode copies them, so edits to 
        # this config never change the defaults
        if isinstance(config_dict, str):
            if config_dict.upper() not in DEFAULT_CONFIGS:
                raise KeyError(
                    f'no default config named {config_dict}; '
                    f'choose from {list(DEFAULT_CONFIGS)}')
            config_dict = DEFAULT_CONFIGS[config_dict.upper()]
        
        self._config = ConfigNode(config_dict)
        self.check()
    
    def __getattr__(self, item):
        """Allow top-level attribute access to sections (e.g., config.experiment)."""
        return getattr(self._config, item)
    
    def __getitem__(self, item):
        """Maintain dictionary-like access for backward compatibility or ease of use."""
        return self._config[item]

    def to_yaml(self, rel_path='config.yml', abs_path=None):
        """Save the configuration to a YAML file."""
        path = abs_path or os.path.join(self._config.experiment.run_dir, rel_path)
        with open(path, 'w') as stream:
            yaml.dump(self._config.to_dict(),stream,sort_keys=False,default_flow_style=None)
        self.logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_yaml(cls, path):
        """Load a configuration from a YAML file."""
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)
        return cls(data)

    def update(self, new_data):
        """Recursively update the configuration with new data."""
        def _recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v
        
        current_dict = self._config.to_dict()
        _recursive_update(current_dict, new_data)
        self._config = ConfigNode(current_dict)
        self.check()

    def get_nested(self, path):
        """Get a value using a dot-notated path (e.g., 'training.lr.value')."""
        parts = path.split('.')
        val = self._config
        for part in parts:
            if isinstance(val, dict):
                val = val[part]
            else:
                raise KeyError(f"Path '{path}' not found at '{part}'")
        return val

    def set_nested(self, path, value):
        """Set a value using a dot-notated path."""
        parts = path.split('.')
        data = self._config.to_dict()
        curr = data
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value
        self._config = ConfigNode(data)

    def get_stream(self, name):
        '''the stream entry of a name'''
        for stream in self._config.data.streams:
            if stream.name == name:
                return stream
        raise KeyError(f'stream {name} is not in data.streams!')

    def get_datasets(self, stream=None, role=None, enabled_only=True):
        '''dataset entries, optionally of one stream and/or role'''
        return [
            ds for ds in self._config.data.datasets
            if (stream is None or ds.stream == stream)
            and (role is None or ds.role == role)
            and (ds.get('enabled',True) or not enabled_only)
        ]

    def get_hp_combinations(self):
        """
        Generates a list of config objects for each hyperparameter combination.
        """
        hp_names = self._config.hp_tuning.hps
        if not hp_names:
            return [self]
        
        # Get the lists of values for each HP
        hp_values_lists = [self.get_nested(name) for name in hp_names]
        
        # Check if all lists have the same length
        n_combinations = len(hp_values_lists[0])
        if not all(len(l) == n_combinations for l in hp_values_lists):
            raise ValueError("All hyperparameter lists must have the same length!")
            
        configs = []
        for i in range(n_combinations):
            # Create a deep copy of current config
            new_config = Config(self._config.to_dict())
            new_config.hp_tuning.enabled = False
            # Update the nested values to be single values instead of lists
            for name, values in zip(hp_names, hp_values_lists):
                new_config.set_nested(name, values[i])
            configs.append(new_config)
            
        return configs

    def check(self):
        """Validate the configuration parameters and structure."""
        for k in [
            'loss.smoothness_loss.weight',
            'loss.smoothness_loss.B_weight',
            'loss.correlation_loss.weight'
        ]:
            kw = self.get_nested(k)
            if kw.value is not None and kw.milestone is not None:
                self.logger.warning(
                    f'{k} receives competing information, either value or milestone may be overwritten'
                )
        if self._config.training.lr.milestone is None:
            self.logger.warning(f'assuming a constant lr of {self._config.training.lr.value}')
            self._config.training.lr.milestone = [
                0,self._config.training.lr.value,50,self._config.training.lr.value
            ]
        if not self._config.data.pss.enabled and self._config.model.point.enabled:
            self.logger.warning('point mlp cannot be enabled without pss data!')
            self._config.model.point.enabled = False
        exp = self._config.experiment
        data = self._config.data
        hp_tuning = self._config.hp_tuning
        # Ensure run directory exists
        if not os.path.exists(exp.run_dir) and exp.mkdir:
            self.logger.warning(f'{exp.run_dir} does not exist, trying to create one')
            try:
                os.makedirs(exp.run_dir, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to create run_dir: {e}")
        # Cross-validation check
        if data.split.nfold > 1 and data.split.val_fraction > 0:
            self.logger.warning(
                'val fraction {} will be replaced by 1/nfold'.format(data.split.val_fraction))
        # split seed same as master seed if not provided
        if data.split.seed is None:
            self._config.data.split.seed = self._config.experiment.seed
        self.check_streams_and_datasets()
        # hp lists have to be the same length
        if hp_tuning.enabled:
            hp_values_lists = [self.get_nested(name) for name in hp_tuning.hps]
            if not all([hasattr(l,'__len__') for l in hp_values_lists]):
                self.logger.error('hps must be all lists!')
            else:
                n_combinations = len(hp_values_lists[0])
                if not all(len(l) == n_combinations for l in hp_values_lists):
                    self.logger.error("All hp lists must have the same length!")

    def check_streams_and_datasets(self):
        '''
        cross-check data.streams and data.datasets; problems are logged as errors, 
        not raised, so partial edits in a notebook do not stop at every step
        '''
        data = self._config.data
        stream_names = [s.name for s in data.streams]
        if len(set(stream_names)) != len(stream_names):
            self.logger.error(f'stream names are not unique: {stream_names}')
        ds_names = [ds.name for ds in data.datasets]
        if len(set(ds_names)) != len(ds_names):
            self.logger.error(f'dataset names are not unique: {ds_names}')
        for s in data.streams:
            missing = [v for v in s.var_names if v not in s.field_names]
            if s.random_error.method == 'xy_rs':
                missing += [
                    f'{s.var_names[0]}_{suffix}' for suffix in ['xy','rs']
                    if f'{s.var_names[0]}_{suffix}' not in s.field_names]
            if missing:
                self.logger.error(f'stream {s.name}: {missing} are not in field_names')
            if s.random_error.method == 'clt' and s.random_error.sigma0 is None:
                self.logger.error(f'stream {s.name}: clt random error needs sigma0')
        psf_idxs = sorted({s.get('psf_idx',0) for s in data.streams})
        if any(not isinstance(i,int) or i < 0 for i in psf_idxs):
            self.logger.error(f'psf_idx must be non-negative integers, not {psf_idxs}')
        elif psf_idxs != list(range(len(psf_idxs))):
            self.logger.warning(
                f'psf_idx {psf_idxs} skips numbers; the unused psfs are never trained')
        for ds in data.datasets:
            if not ds.get('enabled',True):
                continue
            if ds.stream not in stream_names:
                self.logger.error(f'dataset {ds.name}: stream {ds.stream} is not in data.streams')
                continue
            if ds.role not in ['train','val','infer']:
                self.logger.error(
                    f'dataset {ds.name}: role must be train, val, or infer, not {ds.role}')
                continue
            if ds.role not in self.get_stream(ds.stream):
                self.logger.error(f'dataset {ds.name}: stream {ds.stream} has no {ds.role} specs')
                continue
            stream_specs = self.normalize_specs(self.get_stream(ds.stream)[ds.role].specs)
            ds_specs = self.normalize_specs(ds.specs)
            idx = [i for i,spec in enumerate(stream_specs) if spec in ds_specs]
            if len(idx) != len(ds_specs):
                self.logger.error(
                    f'dataset {ds.name}: specs {ds.specs} are not all in stream '
                    f'{ds.stream} {ds.role} specs')
            elif idx != list(range(idx[0],idx[0]+len(idx))):
                self.logger.error(
                    f'dataset {ds.name}: specs {ds.specs} are not next to each other '
                    f'in stream {ds.stream} {ds.role} specs')
            if ds.role != 'train' and ds.get('jitter') is not None:
                self.logger.warning(f'dataset {ds.name}: jitter on a {ds.role} set')
            if ds.role != 'train' and ds.crop.get('mode','fixed') != 'fixed':
                self.logger.warning(f'dataset {ds.name}: random crops on a {ds.role} set')
            if ds.role == 'infer' and ds.get('min_valid_pixels',0) > 0:
                self.logger.warning(
                    f'dataset {ds.name}: skipping crops leaves holes in mosaicked frames')

    @staticmethod
    def normalize_specs(specs):
        '''(interval,offset) pairs with upper-case units, so 91d and 91D match'''
        return [(str(i).strip().upper(),str(o).strip().upper()) for i,o in specs]

    def sbatch_ccr(
        self,slurm_path=None,config_path=None,python_script_path=None,
        git_path=None,module_list=None,submit=True,**kwargs
    ):
        run_dir = self._config.experiment.run_dir
        run_id = self._config.experiment.run_id
        if slurm_path is None:
            slurm_path = os.path.join(run_dir,'run_job.sh')
        if config_path is None:
            config_path = os.path.join(run_dir,'config.yml')
            self.logger.warning(f'saving current config dict to {config_path}')
            self.to_yaml(abs_path=config_path)
        if python_script_path is None:
            # train.py sits next to this file, in IDS/v05
            python_script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),'train.py')
            self.logger.warning(f'assuming python script at {python_script_path}')
        if git_path is None:
            # the directory holding IDS
            git_path = os.path.join(os.path.dirname(python_script_path),'..','..')
            self.logger.warning(f'assuming git path at {git_path}')
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
            fid.write('#SBATCH --mem={:.0f}'.format(kwargs.pop('mem',80000)))
            fid.write('\n')
            fid.write('#SBATCH --gpus-per-node={:.0f}'.format(kwargs.pop('gpus',1)))
            fid.write('\n')
            fid.write('#SBATCH --job-name={}'.format(run_id))
            fid.write('\n')
            fid.write('#SBATCH --output={}.out'.format(run_id))
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
            fid.write(f'python3 {python_script_path} {config_path} {git_path}')
            fid.write('\n')
        
        if submit:
            cwd = os.getcwd()
            os.chdir(run_dir)
            os.system(f'sbatch {slurm_path}')
            self.logger.warning(f'{slurm_path} submitted! check {run_dir} for results')
            os.chdir(cwd)