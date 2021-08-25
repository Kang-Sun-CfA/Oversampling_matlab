#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:03:57 2021

@author: kangsun
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import logging
from scipy.io import loadmat

class Feature2(OrderedDict):
    '''
    a class based on OrderedDict. each Feature2 object represents a collection
    of discover-aq (daq) profiles
    '''
    def __init__(self,if_remove_invalid=True,
                 min_pbl_coverage=0,
                 min_lsuntime=0,
                 max_lsuntime=24):
        ''' 
        if_remove_invalid:
            whether to mask out invalid daq profiles based on flag_invalid and
            flag_unconfident_lower/upper
        min_pbl_coverage:
            fraction (0-1) of pbl that has to be covered by the range of profile
            altitudes
        min/max_lsuntime:
            min/max of local solar hour, 0 to 24
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.debug('creating an instance of Feature2')
        self.if_remove_invalid = if_remove_invalid
        self.min_pbl_coverage = min_pbl_coverage
        self.min_lsuntime = min_lsuntime
        self.max_lsuntime = max_lsuntime
        self.nprofile = 0
        self.campaigns = []
        
    def subset(self,mask=None):
        ''' 
        subset the profiles in a Feature2 object according to criteria in __init__
        mask:
            additional mask if provided
        return:
            subset Feature2 object
        '''
        if mask is None:
            mask = np.ones(self.nprofile,dtype=np.bool_)
        subset_feature2 = Feature2(if_remove_invalid=True)
        if subset_feature2.if_remove_invalid:
            mask = mask & (self['flag_invalid'] == 0) \
                & (self['flag_unconfident_lower'] == 0) \
                & (self['flag_unconfident_upper'] == 0)
        mask = mask & (self['pbl_coverage'] >= self.min_pbl_coverage) \
            & (self['lsuntime'] >= self.min_lsuntime)\
            & (self['lsuntime'] <= self.max_lsuntime)
        self.logger.info('reducing profile number from {} to {}'.format(self.nprofile,np.sum(mask)))
        subset_feature2.nprofile = np.sum(mask)
        subset_feature2.campaigns = self.campaigns
        subset_feature2.profile_keys = self.profile_keys
        for key in self.profile_keys:
            subset_feature2[key] = self[key][mask]
        new_data_keys = []
        for key in self.data_keys:
            if int(key[3:]) in subset_feature2['allprofile']:
                subset_feature2[key] = self[key]
                new_data_keys.append(key)
        subset_feature2.data_keys = new_data_keys
        return subset_feature2
    
    def load_mat(self,mat_path):
        ''' 
        load feature2 mat files to python Feature2 object
        '''
        d = loadmat(mat_path,squeeze_me=True)
        self.campaigns = [d['whichcampaign']]
        self.nprofile = d['nprofile']
        # profiles to pd
        data_keys = []
        mat_data_keys = []
        for prof in d['allprofile']:
            if 'p{}'.format(prof) in d.keys():
                df = pd.DataFrame(data=d['p{}'.format(prof)].astype(np.float32),columns=d['table_names'])
                self.__setitem__(d['whichcampaign']+'_{}'.format(prof), df)
                data_keys.append(d['whichcampaign']+'_{}'.format(prof))
                mat_data_keys.append('p{}'.format(prof))
            else:
                self.logger.debug('p{} data do not exist'.format(prof))
        key_list = []
        for key in d.keys():
            if key in  ['saved_time','__header__', '__version__', '__globals__',\
                        'whichcampaign', 'nscalar', 'scalarlist', 'nprofile','table_names']\
                or key in mat_data_keys:
                continue
            if len(d[key]) != d['nprofile']:
                self.logger.warning(key+' dimension inconsistent')
                continue
            else:
                key_list.append(key)
                self.__setitem__(key,d[key])
        
        self.profile_keys = key_list
        self.data_keys = data_keys
        return self
    
    def merge(self,feature2_to_add):
        '''
        merge with another Feature2 object. return the merged object
        '''
        for key in feature2_to_add.data_keys:
            self.__setitem__(key,feature2_to_add[key])
        if self.nprofile == 0:
            self.data_keys = list(feature2_to_add.data_keys)
            self.profile_keys = list(feature2_to_add.profile_keys)
            for key in self.profile_keys:
                self[key] = feature2_to_add[key].copy()
        else:
            self.data_keys = self.data_keys+feature2_to_add.data_keys
            self.profile_keys = list(set(self.profile_keys).intersection(set(feature2_to_add.profile_keys)))
            for key in self.profile_keys:
                self[key] = np.concatenate((self[key],feature2_to_add[key]))
        self.nprofile = self.nprofile+feature2_to_add.nprofile
        self.campaigns = self.campaigns+feature2_to_add.campaigns
        return self
    
    def normalize_profile(self,molecules=['no2'],
                          p_n_edge=None,p_all_edge=None,
                          h_n_edge=None,h_all_edge=None,
                          if_bootstrap=False,nbootstrap=100,
                          bootstrap_percentiles=[2.5,97.5],
                          if_save_all_profiles=False,
                          pcount_threshold=100,
                          vertical_bounds=[0,3]):
        ''' 
        key function that returns a Normalized_Profile object sythesizing profiles
        in the current Feature2 object
        moledules:
            a list molecules to work with, e.g., no2, no, nox, ch2o, nh3
        p_n_edge:
            boundaries of normalized pressure bins, np.linspace(0, 5.05,102) by default
        p_all_edge:
            boundaries of pressure bins in hPa, not fully implemented.
        h_n_edge:
            boundaries of normalized altitude bins
        h_all_edge:
            boundaries of altitude bins in km
        if_bootstrap:
            if estimating confidence intervals (ci) via bootstrapping
        nbootstrap:
            number of bootstrap realizations if_bootstrap
        bootstrap_percentiles:
            list of percentiles to calculate among the bootstrap realizations
        if_save_all_profiles:
            whether to save raw points of all profiles concatenated together
        pcount_threshold:
            vertical bins that contain less raw profiles points than this are invalid
        vertical_bounds:
            gamma values will be calculated using normalized pressure (p_n_*) between 
            these bounds
        '''
        from scipy.stats import binned_statistic
        import warnings
        if p_n_edge is None:
            p_n_edge = np.linspace(0, 5.05,102)
        if p_all_edge is None:
            p_all_edge = np.linspace(0, 200,21)
        if h_n_edge is None:
            h_n_edge = np.linspace(0, 5.05,102)
        if h_all_edge is None:
            h_all_edge = np.linspace(0, 3,31)
        varnames = ['h_all','h_n','p_all','p_n']
        for molecule in molecules:
            varname = molecule+'_mixingratio'
            varnames.append(varname)
            varnames.append(varname+'_n')
        use_data_keys = self.data_keys
        profile_df = pd.concat([self[k][varnames] for k in use_data_keys])
        def F_count_notnan(x):
            return np.sum(~np.isnan(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pcount = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=F_count_notnan,bins=p_n_edge)
        pcount = pd.DataFrame(pcount[0].T,columns=varnames)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p_n_mean = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=np.nanmean,bins=p_n_edge)
        p_n_mean = pd.DataFrame(p_n_mean[0].T,columns=varnames)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p_n_median = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=np.nanmedian,bins=p_n_edge)
        p_n_median = pd.DataFrame(p_n_median[0].T,columns=varnames)
        outp = Normalized_Profile(pcount_threshold=pcount_threshold)
        outp.add('pcount',pcount)
        outp.add('p_n_mean',p_n_mean)
        outp.add('p_n_median',p_n_median)
        outp.add('molecules',molecules)
        outp.add('varnames',varnames)
        if if_save_all_profiles:
            outp.add('all_profiles',profile_df)
        outp.calculate_gamma('p_n_mean',
                             vertical_bounds=vertical_bounds)
        outp.calculate_gamma('p_n_median',
                             vertical_bounds=vertical_bounds)
        if if_bootstrap:
            bt_prof_mean = np.full((len(p_n_edge)-1,nbootstrap,len(molecules)),np.nan)
            bt_prof_median = np.full((len(p_n_edge)-1,nbootstrap,len(molecules)),np.nan)
            bt_profiles_mean = np.full((len(p_n_edge)-1,nbootstrap,len(molecules)),np.nan)
            bt_profiles_median = np.full((len(p_n_edge)-1,nbootstrap,len(molecules)),np.nan)
            bt_gamma_mean = np.full((nbootstrap,len(molecules)),np.nan)
            bt_gamma_median = np.full((nbootstrap,len(molecules)),np.nan)
            self.logger.info('start boostrapping {} realizations'.format(nbootstrap))
            count = 0
            for ib in range(nbootstrap):
                use_data_keys = np.random.choice(self.data_keys,size=len(self.data_keys))
                profile_df = pd.concat([self[k][varnames] for k in use_data_keys])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pcount = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=F_count_notnan,bins=p_n_edge)
                pcount = pd.DataFrame(pcount[0].T,columns=varnames)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p_n_mean = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=np.nanmean,bins=p_n_edge)
                p_n_mean = pd.DataFrame(p_n_mean[0].T,columns=varnames)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p_n_median = binned_statistic(profile_df['p_n'],profile_df[varnames].to_numpy().T,statistic=np.nanmedian,bins=p_n_edge)
                p_n_median = pd.DataFrame(p_n_median[0].T,columns=varnames)
                outp_b = Normalized_Profile(pcount_threshold=pcount_threshold)
                outp_b.add('pcount',pcount)
                outp_b.add('p_n_mean',p_n_mean)
                outp_b.add('p_n_median',p_n_median)
                outp_b.add('molecules',molecules)
                outp_b.add('varnames',varnames)
                outp_b.calculate_gamma('p_n_mean',
                                       vertical_bounds=vertical_bounds)
                outp_b.calculate_gamma('p_n_median',
                                       vertical_bounds=vertical_bounds)
                for (im,molecule) in enumerate(molecules):
                    bt_prof_mean[:,ib,im] = outp_b['p_n_mean'][molecule+'_mixingratio'].to_numpy()
                    bt_prof_median[:,ib,im] = outp_b['p_n_median'][molecule+'_mixingratio'].to_numpy()
                    bt_profiles_mean[:,ib,im] = outp_b['p_n_mean'][molecule+'_mixingratio_n'].to_numpy()
                    bt_profiles_median[:,ib,im] = outp_b['p_n_median'][molecule+'_mixingratio_n'].to_numpy()
                    bt_gamma_mean[ib,im] = outp_b['gamma_p_n_mean'][molecule+'_mixingratio_n']
                    bt_gamma_median[ib,im] = outp_b['gamma_p_n_median'][molecule+'_mixingratio_n']
                del outp_b
                if ib == count*np.round(nbootstrap/10):
                    self.logger.info('{:.0%} finished'.format(ib/nbootstrap))
                    count+= 1
            self.logger.info('boostrapping done')
            for (im,molecule) in enumerate(molecules):
                for prc in bootstrap_percentiles:
                    outp['p_n_mean']['{}_mixingratio_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_prof_mean[:,:,im],prc,axis=1)
                    outp['p_n_median']['{}_mixingratio_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_prof_median[:,:,im],prc,axis=1)
                    outp['p_n_mean']['{}_mixingratio_n_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_profiles_mean[:,:,im],prc,axis=1)
                    outp['p_n_median']['{}_mixingratio_n_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_profiles_median[:,:,im],prc,axis=1)
                    outp['gamma_p_n_mean']['{}_mixingratio_n_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_gamma_mean[:,im],prc,axis=0)
                    outp['gamma_p_n_median']['{}_mixingratio_n_CI{}'.format(molecule,prc)]\
                        = np.percentile(bt_gamma_median[:,im],prc,axis=0)
        return outp

class Normalized_Profile(OrderedDict):
    '''
    representing a suite of synthesized information from a collection of daq profiles
    '''
    def __init__(self,pcount_threshold=100):
        '''
        pcount_threshold:
            vertical bins that contain less raw profiles points than this are invalid
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.debug('creating an instance of Normalized_Profile')
        self.pcount_threshold = pcount_threshold
    
    def add(self,key,value):
        ''' 
        adding an item to the object, which is based on OrderedDict
        '''
        self.__setitem__(key,value)
        
    def calculate_gamma(self,field='p_n_median',
                        vertical_bounds=[0,3]):
        ''' 
        calculate gamma value from normalized profile
        field:
            p_n_median or p_n_mean, where the profiles are either median or mean
            of each normalized pressure bin
        vertical_bounds:
            gamma values will be calculated using normalized pressure (p_n_*) between 
            these bounds
        '''
        if field not in self.keys():
            self.logger.warning(field+' doesn''t exist')
            return
        pcount_threshold = self.pcount_threshold
        tmp_dict = {}
        for k in self[field].keys():
            if k[-2:] != '_n' or k == 'h_n' or k == 'p_n':
                continue
            integrate_x = self[field]['p_n'].to_numpy()
            integrate_y = self[field][k].to_numpy()
            mask = (~np.isnan(integrate_y)) & (~np.isnan(integrate_x)) &\
                    (self['pcount'][k].to_numpy() >=pcount_threshold) &\
                    (integrate_x >= np.min(vertical_bounds)) &\
                    (integrate_x <= np.max(vertical_bounds))
            if np.sum(mask) == 0:
                self.logger.warning(k+' empty')
                continue
            integrate_x = integrate_x[mask]
            integrate_y = integrate_y[mask]
            y_surf = integrate_y[np.argmin(integrate_x)]
            tmp_dict[k] = np.trapz(y=integrate_y,x=integrate_x)/y_surf
        self.add('gamma_'+field,tmp_dict)
    
    def plot_normalized_profile(self,molecule,existing_ax=None,
                                field='p_n_median',if_plot_all_profile=True,
                                if_plot_CI=True,bootstrap_percentiles=[2.5,97.5],
                                if_hexbin=False,
                                Xlim=(-0.1,1.8),Ylim=(-0.1,3.5)):
        ''' 
        plot the content of Normalized_Profile object in a predefined way
        '''
        pcount_threshold = self.pcount_threshold
        figout = {}
        if molecule not in self['molecules']:
            self.logger.warning(molecule+' does not exist')
            return figout
        import matplotlib.pyplot as plt
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout['fig'] = fig
        figout['ax'] = ax
        if if_plot_all_profile:
            if 'all_profiles' not in self.keys():
                self.warning('all_profiles field doesn''t exist')
                figout['all_profile'] = None
            else:
                if not if_hexbin:
                    figout['all_profile'] = ax.plot(self['all_profiles'][molecule+'_mixingratio_n'],
                                                    self['all_profiles']['p_n'],color='gray',
                                                    alpha=0.5,marker='o',linestyle='none',
                                                    markersize=2,zorder=0)
                else:
                    figout['all_profile'] = ax.hexbin(self['all_profiles'][molecule+'_mixingratio_n'],
                                                      self['all_profiles']['p_n'],bins='log',
                                                      zorder=0,extent=[*Xlim,*Ylim])
        if if_plot_CI:
            try:
                ci_1 = self[field][molecule+'_mixingratio_n_CI{}'.format(bootstrap_percentiles[0])]
                ci_2 = self[field][molecule+'_mixingratio_n_CI{}'.format(bootstrap_percentiles[1])]
                vdata = self[field]['p_n']
                mask = self['pcount'][molecule+'_mixingratio_n'] >= pcount_threshold
                vdata = vdata[mask]
                ci_1 = ci_1[mask]
                ci_2 = ci_2[mask]
                figout['CI'] = ax.fill_betweenx(vdata,ci_1,ci_2,zorder=1,color='c',alpha=0.75)
                figout['text'] = figout['ax'].text(.5,2,r'$\gamma$={:.2f} ({:.2f}-{:.2f})'\
                          .format(self['gamma_'+field][molecule+'_mixingratio_n'],\
                                  self['gamma_'+field][molecule+'_mixingratio_n_CI{}'.format(bootstrap_percentiles[0])],\
                                  self['gamma_'+field][molecule+'_mixingratio_n_CI{}'.format(bootstrap_percentiles[1])]))
            except Exception as e:
                self.logger.warning('CI plot did not work, the error message is:')
                self.logger.warning(e)
                figout['CI'] = None
                figout['text'] = figout['ax'].text(1,2,r'$\gamma$={:.2f}'.format(self['gamma_'+field][molecule+'_mixingratio_n']))
        else:
            figout['text'] = figout['ax'].text(1,2,r'$\gamma$={:.2f}'.format(self['gamma_'+field][molecule+'_mixingratio_n']))
            figout['CI'] = None
        # only plot normalized profile with total data points > pcount_threshold
        vdata = self[field]['p_n']
        hdata = self[field][molecule+'_mixingratio_n']
        mask = self['pcount'][molecule+'_mixingratio_n'] >= pcount_threshold
        vdata = vdata[mask]
        hdata = hdata[mask]
        figout['normalized_profile'] = ax.plot(hdata,vdata,zorder=2,color='r')
        figout['ax'].set_xlim(Xlim);
        figout['ax'].set_ylim(Ylim);
        return figout
#%% an example of basic usage
if __name__ == "__main__":
    from scipy.io import loadmat
    import numpy as np
    import datetime as dt
    import matplotlib.pyplot as plt
    import pandas as pd
    import os,sys
    from collections import OrderedDict
    import logging
    logging.basicConfig(level=logging.INFO)
    #%%
    campaigns = ['MD','CA','TX','CO']
    if sys.platform == 'linux':
        tol_color_dir = '/home/kangsun/RRNES'
        feature2_paths = ['/mnt/Data2/DAQ/{}/feature2_notable_{}.mat'.format(c.upper(),c.lower()) for c in campaigns]
    elif sys.platform == 'win32':
        tol_color_dir = r'C:\research\NIP'
        feature2_paths = [r'C:\research\ACMAP\feature2_notable_{}.mat'.format(c.upper(),c.lower()) for c in campaigns]
    min_pbl_coverage=0.7
    min_lsuntime=12.5
    max_lsuntime=14.5
    sys.path.append(tol_color_dir)
    import tol_colors
    f_all = Feature2(min_pbl_coverage=min_pbl_coverage,
                     min_lsuntime=min_lsuntime,
                     max_lsuntime=max_lsuntime)
    f_c = {}
    n_c = {}
    for campaign,feature2_path in zip(campaigns,feature2_paths):
        f_c[campaign] = Feature2(min_pbl_coverage=min_pbl_coverage,
                                           min_lsuntime=min_lsuntime,
                                           max_lsuntime=max_lsuntime).load_mat(feature2_path).subset()
        n_c[campaign] = f_c[campaign].normalize_profile(molecules=['no2','no','nox','ch2o'],
                                                        if_save_all_profiles=True,
                                                        if_bootstrap=True,
                                                        nbootstrap=100,
                                                        bootstrap_percentiles=[2.5,97.5],
                                                        vertical_bounds=[0,3])
        f_all = f_all.merge(f_c[campaign])
    n_all = f_all.normalize_profile(molecules=['no2','no','nox','ch2o'],
                                    if_save_all_profiles=True,
                                    if_bootstrap=True,
                                    nbootstrap=100,
                                    bootstrap_percentiles=[2.5,97.5],
                                    vertical_bounds=[0,3])
    #%%
    plt.close('all')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rcParams.update({'font.size': 11})
    fig,axs = plt.subplots(5,4,figsize=(9,6),constrained_layout=True,sharex=True,sharey=True)
    # gs = fig.add_gridspec(nrows=2,ncols=2)
    #plt.subplots_adjust(left=0.1,right=0.925)
    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    mngr.window.setGeometry(0,100,dx,dy)
    figout_list = []
    for (ic,c) in enumerate(campaigns):
        for (im,m) in enumerate(['no2','no','nox','ch2o']):
            figout = n_c[c].plot_normalized_profile(existing_ax=axs[ic,im],molecule=m,bootstrap_percentiles=[2.5,97.5],if_hexbin=True)
            figout_list.append(figout)
    for (im,m) in enumerate(['no2','no','nox','ch2o']):
        figout = n_all.plot_normalized_profile(existing_ax=axs[4,im],molecule=m,bootstrap_percentiles=[2.5,97.5],if_hexbin=True)
        figout_list.append(figout)
    for figout in figout_list:
        figout['all_profile'].set_cmap('gray_r')#(tol_colors.tol_cmap('sunset'))
        figout['text'].set_position((0.3,2.75))
        figout['CI'].set_color('b')
    axs[-1,0].set_xlabel(r'$\Pi_1$ (normalized mixing ratio)')
    axs[2,0].set_ylabel(r'$\Pi_2$ (normalized pressure)')
    
    for (ax,t) in zip(axs[0,],[r'NO$_2$','NO',r'NO$_x$','HCHO']):
        ax.set_title(t,fontsize=14)
    for (ax,t) in zip(axs[:,-1],campaigns+['All']):
        ax.annotate(t,xy=(1.05,0.5),xycoords='axes fraction',
                    va='center',ha='left',fontsize=14)
    # fig.savefig(r'C:\research\ACMAP\median_0-3_125-145_bt.png',dpi=150)
