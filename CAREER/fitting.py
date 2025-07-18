import numpy as np
import pandas as pd
import logging

# class to handle chemistry, topography, and bias fits
class BinFit():
    def __init__(self,y_field='wind_column',x_fields=['column_amount','wind_topo'],
                 bin_field=None,no_neg_fields=None,add_field=None,
                 q_bins=None,v_bins=None,regressor=None,reg_formula=None,
                 min_rsquared=None,min_y=None,max_y=None,remove_intercept=False):
        '''initialize with common inputs
        y_field:
            name of predicted variable
        x_fields:
            a list of names of predictors
        bin_field:
            df constructed by y and xs will be binned based on this field. default x_fields[0]
        no_neg_fields:
            list of predictors that should not be negative (e.g., X for topo, k for chem), bins
            will be merged if negative value in fit appears
        add_field:
            values of y_field will be updated to this name. default y_field+'_fit'
        q_bins:
            quantile bins to cut bin_field into. input to pd.qcut. preferred if both q/v_bins exist
        v_bins:
            value bins to cut bin_field into. input to pd.cut. if q/v_bins are not provided,
            default to q_bins=[0,1], i.e., no cut and everything into a single bin
        regressor:
            sth like statsmodels.formula.api.ols
        reg_formula:
            formula input to regressor. if not given, use y_field ~ x_fields[0] + ...
        min_rsquared:
            risky zone, R2 of a bin should not go below this value, otherwise will be merged
        min/max_y:
            trim l3[y_field] optionally
        remove_intercept:
            whether to remove the fitted offset from y
        '''
        self.logger = logging.getLogger(__name__)
        self.y_field,self.x_fields = y_field,x_fields
        self.bin_field = bin_field or x_fields[0]
        self.no_neg_fields = no_neg_fields or []
        self.add_field = add_field or f'{y_field}_fit'
        
        if q_bins is None and v_bins is None:
            q_bins = [0,1]
        if regressor is None:
            import statsmodels.formula.api as smf
            regressor = smf.ols
        self.q_bins,self.v_bins = q_bins,v_bins
        self.regressor = regressor
        
        if reg_formula is None:
            reg_formula = f'{y_field} ~'
            for i,x_field in enumerate(x_fields):
                if i == 0:
                    reg_formula += f' {x_field}'
                else:
                    reg_formula += f' + {x_field}'
        self.reg_formula = reg_formula
        self.min_rsquared = min_rsquared
        self.min_y = min_y or -np.inf
        self.max_y = max_y or np.inf
        self.remove_intercept = remove_intercept
    
    def fit_l3s(self,l3s,resample_rule=None,mask=None,
                nbootstrap=None,random_state=10):
        '''
        l3s:
            a popy.Level3_List object
        resample_rule:
            rule input to l3s.resample (e.g., 1Y, 3M), month_of_year, or None (native resolution)
        mask:
            bool mask on which to conduct the fit, fit all grid cells if no mask provided
        nbootstrap:
            number of bootstrap realizations. None for no bootstrapping
        random_state:
            integer for reproducibility
        return:
            updated l3s
            fits_list: list of lists of fit outcomes per bin after all the merging
            bin_names_list: list of lists of names of these bins
            df: for time stamps, from l3s if resample_rule is None, otherwise from resampled l3s
            bootstrap_dicts_list: list of dicts of inputs to self.predict_l3. each dict has
            items of fit_params and bin_names
        '''
        fits_list = []
        bin_names_list = []
        bootstrap_dicts_list = []
        
        if resample_rule is None:# fit at native time resolution
            df = l3s.df
            for l3 in l3s:
                # run bootstrap, outcome is a list of dicts for self.predict_l3
                if nbootstrap is not None:
                    bootstrap_dicts = []
                    for i in range(nbootstrap):
                        if random_state is None:
                            l3_random_state = None
                        else:
                            l3_random_state = random_state+i
                        l3,fits,bin_names = self.fit_l3(
                            l3=l3,mask=mask,if_bootstrap=True,
                            random_state=l3_random_state
                        )
                        fit_params = [fit.params for fit in fits]
                        bootstrap_dict = dict(
                            fit_params=fit_params,
                            bin_names=bin_names
                        )
                        bootstrap_dicts.append(bootstrap_dict)
                    bootstrap_dicts_list.append(bootstrap_dicts)
                l3,fits,bin_names = self.fit_l3(l3=l3,mask=mask,if_bootstrap=False)
                fits_list.append(fits)
                bin_names_list.append(bin_names)
        else:
            l3s_resampled,resampler = l3s.resample(rule=resample_rule)
            df = l3s_resampled.df
            for l3,(k,v) in zip(l3s_resampled,resampler.indices.items()):
                if nbootstrap is not None:
                    bootstrap_dicts = []
                    for i in range(nbootstrap):
                        if random_state is None:
                            l3_random_state = None
                        else:
                            l3_random_state = random_state+i
                        l3,fits,bin_names = self.fit_l3(
                            l3=l3,mask=mask,if_bootstrap=True,
                            random_state=l3_random_state
                        )
                        fit_params = [fit.params for fit in fits]
                        bootstrap_dict = dict(
                            fit_params=fit_params,
                            bin_names=bin_names
                        )
                        bootstrap_dicts.append(bootstrap_dict)
                    bootstrap_dicts_list.append(bootstrap_dicts)
                l3,fits,bin_names = self.fit_l3(l3=l3,mask=mask,if_bootstrap=False)
                fits_list.append(fits)
                bin_names_list.append(bin_names)
                # populate predicted add_field to original l3s
                for irow,row in l3s.df.iloc[v].iterrows():
                    l3s[int(row['count'])] = self.predict_l3(
                        l3=l3s[int(row['count'])],
                        fit_params=[fit.params for fit in fits],
                        bin_names=bin_names
                    )
            
        return l3s,fits_list,bin_names_list,df,bootstrap_dicts_list
        
    def fit_l3(self,l3,mask=None,if_bootstrap=False,random_state=None):
        '''
        l3:
            a popy.Level3_Data object
        mask:
            bool mask on which to conduct the fit, fit all grid cells if no mask provided
        if_bootstrap:
            if true, this will be a bootstrap run where df is randomly sampled before fitting
        random_state:
            input to df.sample for reproducibility
        return:
            updated l3, where l3[add_field] will be added as updated l3[y_field]
            fits: list of fit outcomes per bin after all the merging
            bin_names: list of names of these bins
        '''
        y_field,x_fields,no_neg_fields,bin_field,add_field = \
        self.y_field,self.x_fields,self.no_neg_fields,self.bin_field,self.add_field
        min_y,max_y = self.min_y,self.max_y
        regressor = self.regressor
        min_rsquared = self.min_rsquared
        reg_formula = self.reg_formula
        
        if mask is None:
            mask = np.ones(l3[y_field].shape,dtype=bool)
        mask = mask & (l3[y_field] >= min_y) & (l3[y_field] <= max_y)
        
        df = {}
        df[y_field] = l3[y_field][mask]
        for x_field in x_fields:
            df[x_field] = l3[x_field][mask]
        
        df = pd.DataFrame(df).dropna()
        self.logger.info(f'l3 has {l3[y_field].size} grids, {len(df)} to be fitted')
        if df.shape[0] == 0:
            self.logger.warning('empty dataframe, returning l3 and empty lists')
            return l3,[],[]
        if if_bootstrap:
            df = df.sample(frac=1,replace=True,random_state=random_state)
        if self.v_bins is not None:
            cut_func = pd.cut
            bins = self.v_bins
        if self.q_bins is not None:
            cut_func = pd.qcut
            bins = self.q_bins
        df['x_bins'] = cut_func(df[bin_field],bins)
        
        bin_names,fits,good_fits,bin_counts = self.fit_df_by_bins(
            df,reg_formula,no_neg_fields,regressor,min_rsquared,'x_bins')
        self.logger.info(f'there are {bin_counts} samples in {len(bin_names)} bins')
        while not all(good_fits) and len(bin_names) > 1:
            self.logger.warning(f'out of {len(bin_names)} bins, {len(bin_names)-sum(good_fits)} are no good, merging')
            bin_names = self.merge_bins_for_good(bin_names,good_fits)
            df['x_bins'] = pd.cut(df[bin_field],pd.IntervalIndex(bin_names))
            bin_names,fits,good_fits,bin_counts = self.fit_df_by_bins(
                df,reg_formula,no_neg_fields,regressor,min_rsquared,'x_bins')
            self.logger.info(f'there are {bin_counts} samples in {len(bin_names)} bins')
        
        fit_params = [fit.params for fit in fits]
        l3 = self.predict_l3(l3,fit_params,bin_names)
        
        return l3,fits,bin_names
        
    def predict_l3(self,l3,fit_params,bin_names,add_field=None):
        '''util function for fit_l3 to create l3[add_field] by updating l3[y_field]
        using fitting results
        l3:
            a popy.Level3_Data object
        fit_params:
            a list of params from a list of fits, i.e., fit_params = [fit.params for fit in fits]
        bin_names: 
            list of names of these bins
        '''
        add_field = add_field or self.add_field
        if add_field in l3.keys():
            self.logger.debug(f'{add_field} exists in l3 and will be overwritten')
        l3[add_field] = l3[self.y_field].copy()
        
        min_bin_edge = np.min([b.left for b in bin_names])
        max_bin_edge = np.max([b.right for b in bin_names])

        for b,params in zip(bin_names,fit_params):
            if b.left == min_bin_edge:
                left = -np.inf
            else:
                left = b.left
            if b.right == max_bin_edge:
                right = np.inf
            else:
                right = b.right
            bin_mask = (l3[self.bin_field] > left) & (l3[self.bin_field] <= right)
            for x_field in self.x_fields:
                l3[add_field][bin_mask] -= params[x_field]*l3[x_field][bin_mask]
            if self.remove_intercept:
                l3[add_field][bin_mask] -= params['Intercept']
        return l3
                
    @staticmethod
    def fit_df_by_bins(df,reg_formula,no_neg_fields=None,regressor=None,
                       min_rsquared=None,groupby_key='x_bins'
                      ):
        '''util function for fit_l3. cut df based on the groupby_key,
        fit each subdf using regressor and reg_formula
        df:
            dataframe to be cut by grouby_key
        reg_formula:
            formula input to the regressor, e.g., wind_column ~ column_amount
        no_neg_fields:
            a list of features that should not be negative, if so, the fit is labeled no good
        min_rsquared:
            label a fit no good if R2 is lower than this
        regressor:
            sth like statsmodels.formula.api.ols
        groupby_key:
            out put of pd.cut or pd.qcut
        return:
            bin_names: list of intervals, i.e., df[groupby_key].unique()
            fits: list of fits per interval
            good_fits: list of true/false whether a fit in fits is good
            bin_counts: number of samples per bin
        '''
        no_neg_fields = no_neg_fields or []
        if regressor is None:
            import statsmodels.formula.api as smf
            regressor = smf.ols
        bin_names, fits, good_fits, bin_counts = [],[],[],[]
        for name,subdf in df.groupby(groupby_key):
            bin_names.append(name)
            bin_counts.append(len(subdf))
            good_fit = True
            if len(subdf) == 0:
                fit = None
                logging.warning(f'bin {name} is empty!')
                good_fit = False
            else:
                fit = regressor(reg_formula, data=subdf).fit()
                good_fit = all([fit.params[no_neg_field] < 0 for no_neg_field in no_neg_fields])
                if min_rsquared is not None:
                    good_fit = good_fit & (fit.rsquared >= min_rsquared)
            fits.append(fit)
            good_fits.append(good_fit)
        return bin_names,fits,good_fits,bin_counts
    
    @staticmethod
    def merge_bins_for_good(bin_names,good_fits):
        '''util function for fit_l3. merge a no good bin with the nearest good bin
        bin_names:
            list of intervals, each interval has a fit
        good_fits: 
            list of true/false whether a fit in fits is good
        return:
            new_bin_names: list of intervals after merging
        '''
        n = len(bin_names)
        new_bin_names = []
        mergeall = False
        i = 0
        while i < n:
            if good_fits[i]:
                new_bin_names.append(bin_names[i])
                i += 1
            else:
                # Find nearest good fit/inverval (left and right)
                left = i - 1
                while left >= 0 and not good_fits[left]:
                    left -= 1

                right = i + 1
                while right < n and not good_fits[right]:
                    right += 1
                # Determine which side to merge with
                if left >= 0 and right < n:
                    # Merge with closer side (prefer left if equidistant)
                    merge_left = (i - left) <= (right - i)
                else:
                    merge_left = left >= 0  # If one side is out of bounds, choose the other

                if merge_left and left >= 0:
                    # Merge current None interval with left non-None interval
                    new_bin_names[-1] = pd.Interval(
                        new_bin_names[-1].left,
                        bin_names[i].right,
                        closed=bin_names[i].closed
                    )
                elif not merge_left and right < n:
                    bin_names[right] = pd.Interval(
                        bin_names[i].left,
                        bin_names[right].right,
                        closed=bin_names[right].closed
                    )
                    i = right-1
                else:
                    # Edge case where all are None - merge all
                    mergeall=True
                i += 1
        if mergeall:
            new_bin_names = [pd.Interval(
                min([b.left for b in bin_names]),
                max([b.right for b in bin_names]),
                closed=bin_names[0].closed
            )]
        return new_bin_names
    