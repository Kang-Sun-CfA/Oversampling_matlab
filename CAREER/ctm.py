import sys, os, glob
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from pyproj import Proj
from netCDF4 import Dataset
import statsmodels.formula.api as smf

mw_g_mol = dict(no=30,no2=46)
SCALE_HEIGHT = 7500

class CTM(dict):
    '''a class based on dictionary to handle CTM data'''
    def __init__(self,name=None,west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.name = name or 'CTM'
    
    def plot_taylor_diagram(self,xdata=None,ydata=None,xkey=None,ykey=None,
                           gaussian_filter_sigma=None,squeeze_time=True,
                           fig=None,figsize=(7,7),dia=None,**kwargs):
        '''plot taylor diagram of a data field to evaluate emission estimators
        '''
        if xdata is None:
            xdata = self[xkey]
        if ydata is None:
            ydata = self[ykey]
        if squeeze_time:
            xdata = np.nanmean(xdata,axis=0)
            ydata = np.nanmean(ydata,axis=0)
            if gaussian_filter_sigma is not None:
                xdata = gaussian_filter(xdata,sigma=gaussian_filter_sigma)
        else:
            if gaussian_filter_sigma is not None:
                xdata = np.array([gaussian_filter(d,sigma=gaussian_filter_sigma) for d in xdata])

        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        xdata = xdata[mask]
        ydata = ydata[mask]
        
        if dia is None:
            fig = fig or plt.figure(figsize=figsize)
            dia = TaylorDiagram(refstd=1,fig=fig,rect=111)
            contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
            plt.clabel(contours, inline=1, fontsize=10, fmt='%.1f')

            dia.add_grid()                                  # Add grid
            dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward
        dia.add_sample(np.std(ydata,ddof=1)/np.std(xdata,ddof=1),
                       np.corrcoef(xdata,ydata)[0,1],**kwargs)
        return dia
        
    def get_agreement_metrics(self,xdata=None,ydata=None,xkey=None,ykey=None,
                              gaussian_filter_sigma=None,squeeze_time=True):
        '''calculate agreements between to data fields to evaluate emission estimators
        return:
            a dict of pearson, mean bias, ols slope, mae, rmse
        '''
        if xdata is None:
            xdata = self[xkey]
        if ydata is None:
            ydata = self[ykey]
        if squeeze_time:
            xdata = np.nanmean(xdata,axis=0)
            ydata = np.nanmean(ydata,axis=0)
            if gaussian_filter_sigma is not None:
                xdata = gaussian_filter(xdata,sigma=gaussian_filter_sigma)
        else:
            if gaussian_filter_sigma is not None:
                xdata = np.array([gaussian_filter(d,sigma=gaussian_filter_sigma) for d in xdata])

        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        xdata = xdata[mask]
        ydata = ydata[mask]
        return dict(r=np.corrcoef(xdata,ydata)[0,1],
                    slope=np.linalg.lstsq(
                        np.vstack([xdata, np.ones(len(xdata))]).T, 
                        ydata, rcond=None)[0][0],
                    mb=np.mean(ydata-xdata),
                    mae=np.mean(np.abs(ydata-xdata)),
                    rmse=np.sqrt(np.mean(np.square(ydata-xdata)))
                   )
                    
    def get_SUSTech_CMAQ_columns(self,nlayer_to_sum=None,nlayer_for_f=None):
        '''calculate columns
        keys:
            3D mixing ratio fields to integrate
        nlayer_to_sum:
            first 8 layers ~ 830 hPa, 1.5 km, 9 layers ~ 670 hPa, 3 km
        nlayer_for_f:
            if specified, use different column definition for f = NOx col/NO2 col
        '''
        nlayer_to_sum = nlayer_to_sum or self['NO2'].shape[1]
        p_high = self['VGTOP'] + self['VGLVLS'][nlayer_to_sum] * (self['PRSFC']-self['VGTOP'])
        h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
        self.logger.info(
            'suming {} layers for columns, roughly topping at {:.0f} hPa, {:.0f} m'.format(
                    nlayer_to_sum,np.nanmean(p_high)/100,h_high_mean))
        # pressure thickness of layers, Pa
        p_intervals = -np.diff(
            np.array(
                [self['VGTOP'] + vglvl * (self['PRSFC']-self['VGTOP']) \
                 for vglvl in self['VGLVLS']]
            ).transpose([1,0,2,3]),
            axis=1)
        
        for key in ['NO','NO2']:
            # mol/m2
            self['{}_COL'.format(key)] = np.sum(p_intervals[:,:nlayer_to_sum,:,:] \
            * self[key][:,:nlayer_to_sum,:,:]/9.8/0.02896*1e-6,axis=1)
        
        self['NOx_COL'] = self['NO_COL'] + self['NO2_COL']
        
        if nlayer_for_f is None:
            self['f'] = self['NOx_COL'] / self['NO2_COL']
        else:
            tmp_no = np.sum(p_intervals[:,:nlayer_for_f,:,:] \
            * self['NO'][:,:nlayer_for_f,:,:]/9.8/0.02896*1e-6,axis=1)
            tmp_no2 = np.sum(p_intervals[:,:nlayer_for_f,:,:] \
            * self['NO2'][:,:nlayer_for_f,:,:]/9.8/0.02896*1e-6,axis=1)
            self['f'] = ( tmp_no + tmp_no2 ) / tmp_no2
        # sum layered emissions to a single one
        if all(np.isin(['NO_emis','NO2_emis'],list(self.keys()))):
            self['NO_emis_COL'] = np.nansum(self['NO_emis'],axis=1)
            self['NO2_emis_COL'] = np.nansum(self['NO2_emis'],axis=1)
            self['NOx_emis_COL'] = self['NO_emis_COL'] + self['NO2_emis_COL']
    
    def fit_topo(self,resample_rule='1M',NOxorNO2='NOx',mask=None,fit_chem=True,if_bootstrap=False):
        '''generate topo fit coefficients, see fit_topography in popy
        '''
        if mask is None:
            mask = np.ones((len(self['ygrid']),len(self['xgrid'])),dtype=bool)
        if 'wind_topo' not in self.keys():
            if NOxorNO2 == 'fNO2':
                self['wind_topo'] = self['surface_altitude_DD'] * self['NO2_COL']
            else:
                self['wind_topo'] = self['surface_altitude_DD'] * self[f'{NOxorNO2}_COL']
        if NOxorNO2 == 'fNO2':
            resampler = self.resample(resample_rule,
                                  ['wind_topo','NO2_COL_DD','NO2_COL'])
        else:
            resampler = self.resample(resample_rule,
                                  ['wind_topo',f'{NOxorNO2}_COL_DD',f'{NOxorNO2}_COL'])
        self[f'<{NOxorNO2}_COL_DD_topo>'] = np.full_like(self['<wind_topo>'],np.nan)
        self.topo_fits = []
        for i,(k,v) in enumerate(resampler.indices.items()):
            if NOxorNO2 == 'fNO2':
                df = pd.DataFrame(dict(y=self['<NO2_COL_DD>'][i,][mask]*self['<f>'][i,][mask]\
                                       +self['<NO2_COL>'][i,][mask]*self['<f_DD>'][i,][mask],
                                       wt=self['<wind_topo>'][i,][mask]*self['<f>'][i,][mask],
                                       vcd=self['<NO2_COL>'][i,][mask]*self['<f>'][i,][mask])
                                 ).dropna()
            else:
                df = pd.DataFrame(dict(y=self[f'<{NOxorNO2}_COL_DD>'][i,][mask],
                                       wt=self['<wind_topo>'][i,][mask],
                                       vcd=self[f'<{NOxorNO2}_COL>'][i,][mask])
                                 ).dropna()
            if if_bootstrap:
                df = df.sample(frac=1,replace=True)
            if fit_chem:
                topo_fit = smf.ols('y ~ wt + vcd', data=df).fit()
            else:
                topo_fit = smf.ols('y ~ wt', data=df).fit()
            self.topo_fits.append(topo_fit)
            if NOxorNO2 == 'fNO2':
                self[f'<{NOxorNO2}_COL_DD_topo>'][i,] = self['<f>'][i,]*self['<NO2_COL_DD>'][i,]\
                +self['<f_DD>'][i,]*self['<NO2_COL>'][i,]\
                -topo_fit.params['wt']*self['<wind_topo>'][i,]*self['<f>'][i,]
            else:
                self[f'<{NOxorNO2}_COL_DD_topo>'][i,] = self[f'<{NOxorNO2}_COL_DD>'][i,]\
            -topo_fit.params['wt']*self['<wind_topo>'][i,]
        self.df[f'{NOxorNO2}_topo_scale_height'] = [-1/topo_fit.params['wt'] for topo_fit in self.topo_fits]
        self.df[f'{NOxorNO2}_topo_rmse'] = [np.sqrt(topo_fit.mse_resid) for topo_fit in self.topo_fits]
        self.df[f'{NOxorNO2}_topo_r2'] = [topo_fit.rsquared for topo_fit in self.topo_fits]
    
    def fit_chem(self,resample_rule='1M',NOxorNO2='NOx',mask=None,
                 chem_fit_order=1,remove_intercept=False,if_bootstrap=False):
        '''generate chem fit coefficients, see fit_chemistry in popy
        '''
        if mask is None:
            mask = np.ones((len(self['ygrid']),len(self['xgrid'])),dtype=bool)
        if 'wind_topo' not in self.keys():
            if NOxorNO2 == 'fNO2':
                self['wind_topo'] = self['surface_altitude_DD'] * self['NO2_COL']
            else:
                self['wind_topo'] = self['surface_altitude_DD'] * self[f'{NOxorNO2}_COL']
        if NOxorNO2 == 'fNO2':
            resampler = self.resample(resample_rule,
                                  ['wind_topo','NO2_COL_DD','NO2_COL'])
        else:
            resampler = self.resample(resample_rule,
                                  ['wind_topo',f'{NOxorNO2}_COL_DD',f'{NOxorNO2}_COL'])
        if f'<{NOxorNO2}_COL_DD_topo>' not in self.keys():
            self.logger.warning('topo fit was not done. doing it now')
            self.fit_topo(resample_rule,NOxorNO2,mask)
        self[f'<{NOxorNO2}_COL_DD_chem>'] = self[f'<{NOxorNO2}_COL_DD_topo>'].copy()
        self.chem_fits = []
        reg_formula = 'y ~'
        for i,(k,v) in enumerate(resampler.indices.items()):
            df = pd.DataFrame(dict(y=self[f'<{NOxorNO2}_COL_DD_topo>'][i,][mask]))
            for order in range(1,1+chem_fit_order):
                if NOxorNO2 == 'fNO2':
                    df['vcd{}'.format(order)] = (self['<NO2_COL>'][i,][mask]*self['<f>'][i,][mask])**order
                else:
                    df['vcd{}'.format(order)] = self[f'<{NOxorNO2}_COL>'][i,][mask]**order
                reg_formula += ' + vcd{}'.format(order)
            df = df.dropna()
            if if_bootstrap:
                df = df.sample(frac=1,replace=True)
            chem_fit = smf.ols(reg_formula, data=df).fit()
            self.chem_fits.append(chem_fit)
            for order in range(1,1+chem_fit_order):
                if NOxorNO2 == 'fNO2':
                    self[f'<{NOxorNO2}_COL_DD_chem>'][i,] -= \
                chem_fit.params['vcd{}'.format(order)]*(self['<NO2_COL>'][i,]*self['<f>'][i,])**order
                else:
                    self[f'<{NOxorNO2}_COL_DD_chem>'][i,] -= \
                chem_fit.params['vcd{}'.format(order)]*self[f'<{NOxorNO2}_COL>'][i,]**order
            if remove_intercept:
                self[f'<{NOxorNO2}_COL_DD_chem>'][i,] -= chem_fit.params['Intercept']
        self.df[f'{NOxorNO2}_chem_lifetime'] = [-1/chem_fit.params['vcd1']/3600 for chem_fit in self.chem_fits]
        self.df[f'{NOxorNO2}_chem_rmse'] = [np.sqrt(chem_fit.mse_resid) for chem_fit in self.chem_fits]
        self.df[f'{NOxorNO2}_chem_r2'] = [chem_fit.rsquared for chem_fit in self.chem_fits]
    
    def resample(self,resample_rule,keys):
        '''resample data into lower freq intervals. average data fields
        resample_rule:
            freq input to df.resample
        keys:
            each key will be averaged to <key> at the freq given by resample_rule
        '''
        if all(['emis' in key for key in keys]):
            resampler = pd.DataFrame(
                index=self['emis_time']).resample(
                resample_rule,label='right')
        else:
            resampler = pd.DataFrame(
                index=self['time']).resample(
                resample_rule,label='right')
        if not hasattr(self,'df'):
            self.df = pd.DataFrame(index=resampler.indices.keys())
        for i,(k,v) in enumerate(resampler.indices.items()):
            for key in keys:
                if i == 0:
                    self[f'<{key}>'] = np.full((self.df.shape[0],*(self[key].shape[1:])),np.nan)
                self[f'<{key}>'][i,] = np.nanmean(self[key][v,],axis=0)
        return resampler
    
    def get_wind_divergence(self,wind_layer=5,nlayer_to_sum=6):
        '''test \Omega_b (\nabla \cdot \vec{u})\approx \int_{z_0}^{z_1} C(\nabla \cdot (u,v))\dif{z}'''
        # get xy and rs divergences given the xy and rs decomposition of vector, simplified from popy
        def F_divs(fe,fn,fne,fnw,dxy,drs):
            dfedx = np.full_like(fe,np.nan)
            dfedx[...,:,1:-1] = (fe[...,:,2:]-fe[...,:,0:-2])/(2*dxy)

            dfndy = np.full_like(fn,np.nan)
            dfndy[...,1:-1,:] = (fn[...,2:,:]-fn[...,0:-2,:])/(2*dxy)

            dfnedr = np.full_like(fne,np.nan)
            dfnedr[...,1:-1,1:-1] = (fne[...,2:,2:]-fne[...,:-2,:-2])/(2*drs)

            dfnwds = np.full_like(fnw,np.nan)
            dfnwds[...,1:-1,1:-1] = (fnw[...,2:,:-2]-fnw[...,:-2,2:])/(2*drs)

            div_xy = dfedx+dfndy
            div_rs = dfnedr+dfnwds
            return div_xy,div_rs
        
        # calculate \Omega_b (\nabla \cdot \vec{u})
        ux = self['UWIND'][:,wind_layer,:,:]
        uy = self['VWIND'][:,wind_layer,:,:]
        ur = ux * np.cos(np.pi/4) + uy * np.sin(np.pi/4)
        us = ux * (-np.cos(np.pi/4)) + uy * np.sin(np.pi/4)
        self.get_SUSTech_CMAQ_columns(nlayer_to_sum=nlayer_to_sum)
        ubar_div_xy,ubar_div_rs = F_divs(fe=ux,fn=uy,fne=ur,fnw=us,
            dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
        self['ubar_div'] = 0.5 * (ubar_div_xy + ubar_div_rs) * self['NOx_COL']
        # calculate \int_{z_0}^{z_1} C(\nabla \cdot (u,v))\dif{z}
        # pressure thickness of layers, Pa
        p_intervals = -np.diff(
            np.array(
                [self['VGTOP'] + vglvl * (self['PRSFC']-self['VGTOP']) \
                 for vglvl in self['VGLVLS']]
            ).transpose([1,0,2,3]),
            axis=1)[:,:nlayer_to_sum,:,:]

        ur = self['UWIND'][:,:nlayer_to_sum,:,:] * np.cos(np.pi/4) + self['VWIND'][:,:nlayer_to_sum,:,:] * np.sin(np.pi/4)
        us = self['UWIND'][:,:nlayer_to_sum,:,:] * (-np.cos(np.pi/4)) + self['VWIND'][:,:nlayer_to_sum,:,:] * np.sin(np.pi/4)

        uint_div_xy,uint_div_rs = F_divs(
            fe=self['UWIND'][:,:nlayer_to_sum,:,:],
            fn=self['VWIND'][:,:nlayer_to_sum,:,:],
            fne=ur,
            fnw=us,
            dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
        
        nox = (self['NO2'][:,:nlayer_to_sum,:,:]+self['NO'][:,:nlayer_to_sum,:,:])*p_intervals/9.8/0.02896*1e-6
        self['uint_div'] = 0.5*(np.nansum(nox*uint_div_xy,axis=1)+np.nansum(nox*uint_div_rs,axis=1))
        del ur, us, nox
            
    def get_flux_divergence(self,wind_layer=None):
        '''calculate divergence of horizontal flux'''
        # get xy and rs divergences given the xy and rs decomposition of vector, simplified from popy
        def F_divs(fe,fn,fne,fnw,dxy,drs):
            dfedx = np.full_like(fe,np.nan)
            dfedx[...,:,1:-1] = (fe[...,:,2:]-fe[...,:,0:-2])/(2*dxy)

            dfndy = np.full_like(fn,np.nan)
            dfndy[...,1:-1,:] = (fn[...,2:,:]-fn[...,0:-2,:])/(2*dxy)

            dfnedr = np.full_like(fne,np.nan)
            dfnedr[...,1:-1,1:-1] = (fne[...,2:,2:]-fne[...,:-2,:-2])/(2*drs)

            dfnwds = np.full_like(fnw,np.nan)
            dfnwds[...,1:-1,1:-1] = (fnw[...,2:,:-2]-fnw[...,:-2,2:])/(2*drs)

            div_xy = dfedx+dfndy
            div_rs = dfnedr+dfnwds
            return div_xy,div_rs
        
        if wind_layer is not None:
            ux = self['UWIND'][:,wind_layer,:,:]
            uy = self['VWIND'][:,wind_layer,:,:]
            ur = ux * np.cos(np.pi/4) + uy * np.sin(np.pi/4)
            us = ux * (-np.cos(np.pi/4)) + uy * np.sin(np.pi/4)
            if 'NO2_COL' not in self.keys():
                self.get_SUSTech_CMAQ_columns(nlayer_to_sum=None)
            self['NOx_DIV_XY'],self['NOx_DIV_RS'] = F_divs(
                fe=ux*self['NOx_COL'],fn=uy*self['NOx_COL'],
                fne=ur*self['NOx_COL'],fnw=us*self['NOx_COL'],
                dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
            self['NOx_DIV'] = 0.5 * (self['NOx_DIV_XY'] + self['NOx_DIV_RS'])
            self['NO2_DIV_XY'],self['NO2_DIV_RS'] = F_divs(
                fe=ux*self['NO2_COL'],fn=uy*self['NO2_COL'],
                fne=ur*self['NO2_COL'],fnw=us*self['NO2_COL'],
                dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
            self['NO2_DIV'] = 0.5 * (self['NO2_DIV_XY'] + self['NO2_DIV_RS'])
        else:
            # pressure thickness of layers, Pa
            p_intervals = -np.diff(
                np.array(
                    [self['VGTOP'] + vglvl * (self['PRSFC']-self['VGTOP']) \
                     for vglvl in self['VGLVLS']]
                ).transpose([1,0,2,3]),
                axis=1)
            
            ur = self['UWIND'] * np.cos(np.pi/4) + self['VWIND'] * np.sin(np.pi/4)
            us = self['UWIND'] * (-np.cos(np.pi/4)) + self['VWIND'] * np.sin(np.pi/4)
            
            self['NOx_DIV_XY'],self['NOx_DIV_RS'] = F_divs(
                fe=self['UWIND']*(self['NO2']+self['NO'])*p_intervals/9.8/0.02896*1e-6,
                fn=self['VWIND']*(self['NO2']+self['NO'])*p_intervals/9.8/0.02896*1e-6,
                fne=ur*(self['NO2']+self['NO'])*p_intervals/9.8/0.02896*1e-6,
                fnw=us*(self['NO2']+self['NO'])*p_intervals/9.8/0.02896*1e-6,
                dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
            
            self['NOx_DIV_XY'] = np.nansum(self['NOx_DIV_XY'],axis=1)
            self['NOx_DIV_RS'] = np.nansum(self['NOx_DIV_RS'],axis=1)
            self['NOx_DIV'] = 0.5 * (self['NOx_DIV_XY'] + self['NOx_DIV_RS'])
            
            self['NO2_DIV_XY'],self['NO2_DIV_RS'] = F_divs(
                fe=self['UWIND']*(self['NO2'])*p_intervals/9.8/0.02896*1e-6,
                fn=self['VWIND']*(self['NO2'])*p_intervals/9.8/0.02896*1e-6,
                fne=ur*(self['NO2'])*p_intervals/9.8/0.02896*1e-6,
                fnw=us*(self['NO2'])*p_intervals/9.8/0.02896*1e-6,
                dxy=self['XCELL'],drs=self['XCELL']*np.sqrt(2))
            
            self['NO2_DIV_XY'] = np.nansum(self['NO2_DIV_XY'],axis=1)
            self['NO2_DIV_RS'] = np.nansum(self['NO2_DIV_RS'],axis=1)
            self['NO2_DIV'] = 0.5 * (self['NO2_DIV_XY'] + self['NO2_DIV_RS'])
            
            del ur,us
            
    def get_directional_derivative(self,keys=None,wind_layer=6,
                                   topo_wind_layer=0,ukey='UWIND',vkey='VWIND'):
        '''get the directional derivative terms, vec{u} dot grad(key)
        keys:
            2D fields in self to calculate gradient
        (topo_)wind_layer:
            layer index in 3D wind for the 2D wind vector. separate for the wind topo term
        u/vkey:
            U/VWIND for cmaq
        '''
        if keys is None:
            keys = ['NO_COL','NOx_COL','NO2_COL','f','surface_altitude']
        if self.name.lower() in ['cmaq']:
            p_low = self['VGTOP'] + self['VGLVLS'][wind_layer] * (self['PRSFC']-self['VGTOP'])
            p_high = self['VGTOP'] + self['VGLVLS'][wind_layer+1] * (self['PRSFC']-self['VGTOP'])
            h_low_mean = np.nanmean(np.log(self['PRSFC']/p_low)*7500)
            h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
            p_low_mean = np.nanmean(p_low)
            p_high_mean = np.nanmean(p_high)
            self.logger.info(
                'using layer {} wind, roughly at {:.0f}-{:.0f} hPa, {:.0f}-{:.0f} m'.format(
                    wind_layer,p_low_mean/100,p_high_mean/100,h_low_mean,h_high_mean))
            if 'surface_altitude' in keys:
                p_low = self['VGTOP'] + self['VGLVLS'][topo_wind_layer] * (self['PRSFC']-self['VGTOP'])
                p_high = self['VGTOP'] + self['VGLVLS'][topo_wind_layer+1] * (self['PRSFC']-self['VGTOP'])
                h_low_mean = np.nanmean(np.log(self['PRSFC']/p_low)*7500)
                h_high_mean = np.nanmean(np.log(self['PRSFC']/p_high)*7500)
                p_low_mean = np.nanmean(p_low)
                p_high_mean = np.nanmean(p_high)
                self.logger.info(
                    'using layer {} wind for topography, roughly at {:.0f}-{:.0f} hPa, {:.0f}-{:.0f} m'.format(
                        topo_wind_layer,p_low_mean/100,p_high_mean/100,h_low_mean,h_high_mean))
        else:
            self.logger.error('this ctm is not supported yet')
            return
        u_col = self[ukey][:,wind_layer,:,:]
        v_col = self[vkey][:,wind_layer,:,:]
        if 'surface_altitude' in keys:
            u_topo = self[ukey][:,topo_wind_layer,:,:]
            v_topo = self[vkey][:,topo_wind_layer,:,:]
        # to do: dx_vec for lon lat grid
        dx = self['XCELL']
        for key in keys:
            if key == 'surface_altitude':
                u = u_topo; v = v_topo
            else:
                u = u_col; v = v_col
            vcd = self[key]
            dcdx = np.full_like(vcd,np.nan)
            dcdy = np.full_like(vcd,np.nan)
            dcdr = np.full_like(vcd,np.nan)
            dcds = np.full_like(vcd,np.nan)
            dcdx[:,:,1:-1] = (vcd[:,:,2:]-vcd[:,:,:-2])/(dx*2)
            dcdy[:,1:-1,:] = (vcd[:,2:,:]-vcd[:,:-2,:])/(dx*2)
            dcdr[:,1:-1,1:-1] = (vcd[:,2:,2:]-vcd[:,:-2,:-2])/(dx*np.sqrt(2)*2)
            dcds[:,1:-1,1:-1] = (vcd[:,2:,:-2]-vcd[:,:-2,2:])/(dx*np.sqrt(2)*2)

            self[key+'_DD_XY'] = dcdx * u + dcdy * v
            self[key+'_DD_RS'] = dcdr * (u * np.cos(np.pi/4) + v * np.sin(np.pi/4)) \
            + dcds * (u * (-np.cos(np.pi/4)) + v * np.sin(np.pi/4))
            self[key+'_DD'] = 0.5 * (self[key+'_DD_XY'] + self[key+'_DD_RS'])
        if all(np.isin(['NO2_COL','NO2_COL_DD','f','f_DD'],list(self.keys()))):
            self.logger.info('calculating contribution of f to emissions')
            self['fxNO2_COL_DD'] = self['f'] * self['NO2_COL_DD']
            self['f_DDxNO2_COL'] = self['f_DD'] * self['NO2_COL']
    
    def get_SUSTech_CMAQ_surface_altitude(self,path,file_path_pattern=None):
        '''
        save or load surface altitude
        path:
            path to save (if file_path_pattern is not None) or load (otherwise) surface pressure data
        file_path_pattern:
            sth like '/projects/academic/kangsun/data/CTM/SUSTech/*_%Y%m.nc', see load_SUSTech_CMAQ
        '''
        if file_path_pattern is not None:
            time = pd.date_range(dt.datetime(2017,1,1),dt.datetime(2017,12,31),freq='1h')
            if 'PRSFC' in self.keys():
                self.logger.warning('PRSFC is already loaded. Temporarily replace it to get surface altitude')
                prsfc = self['PRSFC'].copy()
            file_to_var_mapper = {'PRSFC':['PRSFC']}
            self.load_SUSTech_CMAQ(time,file_to_var_mapper,file_path_pattern)
            self['surface_altitude'] = np.log(101325/np.nanmean(self['PRSFC'],axis=0)[np.newaxis,:,:])*7500
            self['PRSFC'] = prsfc.copy()
            with Dataset(path,'w') as nc:
                nc.createDimension('TSTEP',1)
                nc.createDimension('ROW',len(self['ygrid']))
                nc.createDimension('COL',len(self['xgrid']))
                vid = nc.createVariable('surface_altitude',np.float32,dimensions=('TSTEP','ROW','COL'))
                vid[:] = np.ma.masked_invalid(np.float32(self['surface_altitude']))
        else:
            with Dataset(path,'r') as nc:
                self['surface_altitude'] = nc['surface_altitude'][:].data
                
    def load_SUSTech_CMAQ(self,time,
                          file_to_var_mapper=None,
                          file_path_pattern='/projects/academic/kangsun/data/CTM/SUSTech/*_%Y%m.nc'):
        '''load cmaq files from sustech
        time:
            a hourly pd date range
        file_to_var_mapper:
            a dict maps header of cmap file to the variables to be loaded from the file
        file_path_pattern:
            paths to cmaq files. * for different file header pattern
        '''
        self['time'] = time
        file_to_var_mapper = file_to_var_mapper or\
        {'CONC_NOx':['NO','NO2'],
        'MET':['UWIND','VWIND','PBLH'],
        'PRSFC':['PRSFC'],
        'NOx_emis':['NO','NO2']}
        # loop over monthly files
        for imon,mon in enumerate(pd.period_range(time.min(),time.max(),freq='1M')):
            initialized = False
            # loop over each file
            for ifile,(file_header,fields) in enumerate(file_to_var_mapper.items()):
                fn = mon.strftime(file_path_pattern.replace('*',file_header))
                # emis files are only available on the first day
                if file_header == 'NOx_emis':
                    fn = '01'.join(os.path.splitext(fn))
                self.logger.info('loading {}'.format(fn))
                with Dataset(fn,'r') as nc:
                    # assume all files in a month have the same timestamp, except emis
                    if file_header != 'NOx_emis' and not initialized:
                        initialized = True
                        nc_time = pd.to_datetime(
                            ['{0}'.format(d[0])+'{0:0>6}'.format(d[1])
                             for d in nc['TFLAG'][:,0,:]],
                            format='%Y%j%H%M%S')
                        time_mask = nc_time.isin(time)
                        # assume all files have the same spatial coordinates
                        if imon == 0:
                            self['VGTOP'] = nc.VGTOP
                            self['VGLVLS'] = nc.VGLVLS
                            self['XCELL'] = nc.XCELL
                            self['YCELL'] = nc.YCELL
                            # create the x and y grids
                            xgrid = np.arange(nc.NCOLS)*nc.XCELL+nc.XORIG + nc.XCELL/2
                            ygrid = np.arange(nc.NROWS)*nc.YCELL+nc.YORIG + nc.YCELL/2
                            # transform x y to lon lat
                            proj = Proj(proj='lcc',
                                       lat_1=nc.P_ALP,
                                       lat_2=nc.P_BET,
                                       lat_0=nc.YCENT,
                                       lon_0=nc.XCENT,
                                       x_0=0,y_0=0,
                                       datum='WGS84')
                            lonmesh, latmesh = proj(*np.meshgrid(xgrid, ygrid),inverse=True)
                            self.proj = proj
                            self['xgrid'] = xgrid
                            self['ygrid'] = ygrid
                            self['lonmesh'] = lonmesh
                            self['latmesh'] = latmesh
                            if (self.west,self.east,self.south,self.north) == (-180,180,-90,90):
                                (self.west,self.east,self.south,self.north) = \
                                (lonmesh.min(),lonmesh.max(),latmesh.min(),latmesh.max())
                    elif file_header == 'NOx_emis':
                        nc_time = pd.to_datetime(
                            ['{0}'.format(d[0])+'{0:0>6}'.format(d[1])
                             for d in nc['TFLAG'][:,0,:]],
                            format='%Y%j%H%M%S')
                        time_mask = nc_time.hour.isin(time.hour.unique())
                        emis_time = nc_time[time_mask]
                    
                    if file_header == 'NOx_emis':
                        if imon == 0:
                            self['emis_time'] = emis_time
                        else:
                            self['emis_time'] = np.concatenate(
                                (self['emis_time'],emis_time),axis=0)
                    # loop over fields in each file
                    for field in fields:
                        if file_header == 'NOx_emis':
                            fld = field+'_emis'
                            # mol/s to mol/s/m2
                            factor = 1/(self['XCELL']*self['YCELL'])
                        else:
                            fld = field
                            factor = 1
                        if imon == 0:
                            self[fld] = nc[field][time_mask,] * factor
                        else:
                            self[fld] = np.concatenate(
                                (self[fld],nc[field][time_mask,] * factor),
                                axis=0)
    
#     def load_NAQPMS(self,lonlat_path,pressure=None,time=None,**kwargs):
#         import h5py
#         if pressure is not None:
#             self['pressure'] = pressure
#         else:
#             self['pressure'] = np.array([1000,950,925,900,850,800,
#                                          750,700,650,600,550,500])
#         # UTC
#         if time is not None:
#             self['time'] = time
#         else:
#             self['time'] = pd.date_range(dt.datetime(2023,5,1),
#                                          dt.datetime(2023,5,1,23),
#                                          freq='1h')-dt.timedelta(seconds=8*3600)
#         lonlat = loadmat(lonlat_path)
#         self['lonmesh'] = lonlat['lonM'].T
#         self['latmesh'] = lonlat['latM'].T
#         lon_center = np.mean([self.west,self.east])
#         lat_center = np.mean([self.south,self.north])
#         distance2center = np.sqrt(np.square(self['lonmesh']-lon_center)
#                                   +np.square(self['latmesh']-lat_center))
#         ind = np.unravel_index(np.argmin(distance2center),distance2center.shape)
#         lon_mask = (self['lonmesh'][ind[0],] >= self.west) & \
#         (self['lonmesh'][ind[0],] < self.east)
#         lat_mask = (self['latmesh'][:,ind[1]] >= self.south) & \
#         (self['latmesh'][:,ind[1]] < self.north)
#         ijmesh = np.ix_(lat_mask,lon_mask)
#         self['lonmesh'] = self['lonmesh'][ijmesh]
#         self['latmesh'] = self['latmesh'][ijmesh]
#         # to do: properly remove the day dimension
#         for k,v in kwargs.items():
#             with h5py.File(v) as f:
#                 self[k] = f['model_{}'.format(k)][()][...,*ijmesh][0]
#             # to do: confirm missing data
#             if k in ['psfc','pbl']:
#                 self[k][self[k]==9999.] = np.nan
    
    def get_NAQPMS_columns(self,keys,if_use_pbl=True,pressure_thickness=200.):
        
        def _get_column(number_density_layer,pressure_level,p0,p1,scale_height=7500.):
            '''calculate column amount between to pressure values
            number_density_layer:
                mol/m3 of gas species in each layer
            pressure_level:
                pressure of each level, from high pressure to low
            p0:
                higher pressure boundary
            p1:
                lower pressure boundary
            scale_height:
                atmosphere scale height in m
            '''
            sub_columns = number_density_layer * \
            scale_height * np.log(pressure_level[:-1]/pressure_level[1:])
            cum_col_toa2sfc = np.insert(np.cumsum(sub_columns[::-1]),0,0)
            p_toa2sfc = pressure_level[::-1]
            return np.interp(p0,p_toa2sfc,cum_col_toa2sfc) - np.interp(p1,p_toa2sfc,cum_col_toa2sfc)
        
        if if_use_pbl:
            p_1 = self['psfc'] * np.exp(-self['pbl']/SCALE_HEIGHT)
        else:
            p_1 = self['psfc'] - pressure_thickness
        mask = p_1 < np.min(self['pressure'])
        if np.sum(mask) >= 0:
            self.logger.warning('{} upper pressure over the boundary'.format(np.sum(mask)))
            p_1[mask] = np.min(self['pressure'])
        
        p_0 = self['psfc'].copy()
        p_0[p_0 > np.max(self['pressure'])] = np.max(self['pressure'])
        
        for key in keys:
            key_shape = self[key].shape
            profile = 0.5*(ctm[key][:,:-1,:,:]+ctm[key][:,1:,:,:])*1e-6/mw_g_mol[key] # ug/m3 to mol/m3
            self['{}_col'.format(key)] = np.array(
            [
                [
                    [_get_column(profile[itime,:,iy,ix],
                                 self['pressure'],
                                 p_0[itime,iy,ix],p_1[itime,iy,ix],
                                 scale_height=SCALE_HEIGHT
                                ) for ix in range(key_shape[3])
                    ] for iy in range(key_shape[2])
                ] for itime in range(key_shape[0])
            ])
        
    def plot(self,key=None,data=None,time=None,layer_index=None,**kwargs):
        
        if data is None:
            # remove time dimension
            if time is None:
                self.logger.warning('plotting time average')
                data = np.nanmean(self[key],axis=0)
            else:
                data = self[key][np.nonzero(self['time'] == time)[0][0]]
            if layer_index is not None:
                data = data[layer_index,]
        
        ax = kwargs.pop('ax',None)
        if ax is None:
            figsize = kwargs.pop('figsize',(10,5))
            fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=figsize,
                                  subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = None
        cmap = kwargs.pop('cmap','jet')
        
        scale = kwargs.pop('scale','linear')
        if scale == 'log':
            from matplotlib.colors import LogNorm
            vmin = kwargs.pop('vmin',None)
            vmax = kwargs.pop('vmax',None)
            kwargs['norm'] = LogNorm(vmin=vmin,vmax=vmax)
        func = kwargs.pop('func',lambda x:x)
        cartopy_scale = kwargs.pop('cartopy_scale','50m')
        draw_colorbar = kwargs.pop('draw_colorbar',True)
        label = kwargs.pop('label',key)
        shrink = kwargs.pop('shrink',0.75)
        extent = kwargs.pop('extent',[self.west, self.east, self.south, self.north])
        
        pc = ax.pcolormesh(self['lonmesh'],self['latmesh'],
                           func(data),cmap=cmap,**kwargs)
        ax.coastlines(resolution=cartopy_scale, color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='k', 
                           linestyle='-',lw=0.5)
        if draw_colorbar:
            cb = plt.colorbar(pc,ax=ax,label=label,shrink=shrink)
        else:
            cb = None
        ax.set_extent(extent)
        figout = dict(fig=fig,ax=ax,pc=pc,cb=cb)
        return figout

class TaylorDiagram(object):
    """
    Taylor diagram for python/matplotlib [ 10.5281/zenodo.5548061 ] 
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """



        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l
    def add_scatter(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.scatter` command.
        """

        l = self.ax.scatter(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  

        return l
    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours