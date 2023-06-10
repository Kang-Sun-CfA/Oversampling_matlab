import sys,os,glob
import io
import requests
import pandas as pd
import json
import numpy as np
import datetime as dt
import logging
import matplotlib.pyplot as plt

class CEMS():
    '''this class builts upon EPA's clean air markets program data portal: https://campd.epa.gov/
    the downloading part was inspired by its GitHup repository: https://github.com/USEPA/cam-api-examples
    it focuses on NOx emissions from energy generating units/facilities'''
    def __init__(self,API_key=None,start_dt=None,end_dt=None,
                 attributes_path_pattern=None,
                 emissions_path_pattern=None,
                 west=None,east=None,south=None,north=None):
        '''
        API_key:
            need one from https://www.epa.gov/airmarkets/cam-api-portal#/api-key-signup to download data
        start/end_dt:
            datetime objects accurate to the day
        attributes/emissions_path_pattern:
            path patterns for annual attributes and daily emission files. provided as default here. 
            can be updated later during data saving/loading
        west/east/south/north:
            lon/lat boundary. default to the CONUS
        '''
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt or dt.datetime(2018,5,1)
        self.end_dt = end_dt or dt.datetime.now()
        self.west = west or -130.
        self.east = east or -63.
        self.south = south or 23.
        self.north = north or 51
        self.API_key = API_key
        self.attributes_path_pattern = attributes_path_pattern or \
        '/projects/academic/kangsun/data/CEMS/attributes/trimmed_%Y.csv'
        self.emissions_path_pattern = emissions_path_pattern or \
        '/projects/academic/kangsun/data/CEMS/emissions/%Y/%m/%d/%Y%m%d.csv'
    
    def plot_facility_map(self,fadf=None,max_nfacility=None,ax=None,reset_extent=False,add_text=False,**kwargs):
        '''plot facilities as dots on a map. dot size corresponds to self.fadf[sdata_column], 
        where sdata_column defaults to 'noxMassLbs'
        '''
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # workaround for cartopy 0.16
        from matplotlib.axes import Axes
        from cartopy.mpl.geoaxes import GeoAxes
        GeoAxes._pcolormesh_patched = Axes.pcolormesh
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = None
        cartopy_scale = kwargs.pop('cartopy_scale','110m')
        sc_leg_loc = kwargs.pop('sc_leg_loc','lower right')
        sc_leg_fmt = kwargs.pop('sc_leg_fmt','{x:.2f}')
        sc_leg_title = kwargs.pop('sc_leg_title',"Emitted NOx [lbs]")
        if fadf is None:
            fadf = self.fadf
        # assume fadf is sorted by noxMassLbs
        df = fadf.iloc[0:max_nfacility]
        sdata_column = kwargs.pop('sdata_column','noxMassLbs')
        sdata = df[sdata_column]
        sdata_func = kwargs.pop('sdata_func',None)
        if sdata_func is not None:
            sdata = sdata_func(sdata)
        sdata_min = kwargs.pop('sdata_min',np.nanmin(sdata))
        sdata_max = kwargs.pop('sdata_max',np.nanmax(sdata))
        sdata_min_size = kwargs.pop('sdata_min_size',25)
        sdata_max_size = kwargs.pop('sdata_max_size',100)
        # normalize to 0-1
        sdata = (sdata-sdata_min)/(sdata_max-sdata_min)
        # normalize to sdata_min_size-sdata_max_size
        sdata = sdata*(sdata_max_size-sdata_min_size)+sdata_min_size
        sc = ax.scatter(df['longitude'],df['latitude'],s=sdata,**kwargs)
        if sc_leg_loc is None:
            leg_sc = None
        else:
            handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=7,fmt=sc_leg_fmt,
                                                 func=lambda x:(x-sdata_min_size)\
                                                 /(sdata_max_size-sdata_min_size)\
                                                *(sdata_max-sdata_min)+sdata_min)
            leg_sc = ax.legend(handles, labels, title=sc_leg_title,ncol=3,loc=sc_leg_loc)
            ax.add_artist(leg_sc)
        if reset_extent:
            ax.set_extent([self.west,self.east,self.south,self.north])
        if cartopy_scale is not None:
            ax.coastlines(resolution=cartopy_scale, color='black', linewidth=1)
            ax.add_feature(cfeature.STATES.with_scale(cartopy_scale), facecolor='None', edgecolor='k', 
                           linestyle='-',zorder=0,lw=0.5)
        if add_text:
            from adjustText import adjust_text
            texts = [ax.text(row.longitude,row.latitude,row.facilityName,fontsize=10)\
                     for irow,row in df.iterrows()]
            adjust_text(texts,
                        x=df['longitude'].to_numpy(),
                        y=df['longitude'].to_numpy(),ax=ax,
                        expand_text=(1.1, 1.2))
        return dict(fig=fig,ax=ax,sc=sc,leg_sc=leg_sc)
    
    def trim_unit_attributes(self,new_attributes_pattern,load_emissions_kw=None):
        '''load_emissions becomes too slow with large spatiotemporal windows
        run this once to get annual NOx for unit/facility to easily remove small ones.
        see ub ccr:/projects/academic/kangsun/data/CEMS/trim_attributes.py for example
        new_attributes_pattern:
            path pattern to save the trimmed/noxMass-added attributes table
        load_emissions_kw:
            keyword arguments to self.load_emissions
        '''
        load_emissions_kw = load_emissions_kw or {}
        load_emissions_kw['if_unit_emissions'] = True
        self.load_emissions(**load_emissions_kw)
        tuadf = []
        for year in pd.period_range(self.start_dt,self.end_dt,freq='1Y'):
            left_df = self.uadf[['noxMassLbs', 'index', 'stateCode', 'facilityName', 'facilityId',
       'unitId', 'latitude', 'longitude', 'year','primaryFuelInfo',
       'secondaryFuelInfo','maxHourlyHIRate']]
            left_df = left_df.loc[left_df['year']==year.year]
            right_df = self.fadf[['noxMassLbs','year']]
            right_df = right_df.loc[right_df['year']==year.year][['noxMassLbs']]
            df = pd.merge(left_df,right_df,left_on='facilityId',
                          right_index=True,suffixes=('','Facility'),
                          sort=False)#.sort_index()
            cols = list(df)
            cols.insert(0, cols.pop(cols.index('noxMassLbsFacility')))
            cols.insert(0, cols.pop(cols.index('index')))
            df = df.loc[:, cols]
            df.to_csv(year.strftime(new_attributes_pattern),
                      index=False,index_label='index')
            tuadf.append(df)
        return pd.concat(tuadf)
    
    def get_facilities_emission_rate(self,fadf=None,emissions_path_pattern=None,states=None,
                                     local_hours=None):
        emissions_path_pattern = emissions_path_pattern or self.emissions_path_pattern
        if fadf is None:
            fadf = self.fadf
        uedf = []
        for date in pd.period_range(self.start_dt,self.end_dt,freq='1D'):
            filename = date.strftime(emissions_path_pattern)
            if not os.path.exists(filename):
                self.logger.warning('{} does not exist!'.format(filename))
                continue
            if date.day == 1:
                logging.info('loading emission file {}'.format(filename))
            edf = pd.read_csv(filename)
            edf = edf.loc[edf['Facility ID'].isin(fadf.index)]
            if local_hours is not None:
                edf = edf.loc[pd.to_datetime(edf['local_dt']).dt.hour.isin(local_hours)]
            edf['local_dt'] = pd.to_datetime(edf['local_dt'])
            uedf.append(edf)
        uedf = pd.concat(uedf).reset_index()
        fedf = uedf.groupby(['Facility ID','local_dt']
                           ).aggregate({'NOx Mass (lbs)':lambda x:np.sum(x)*0.453592/0.046/3600,#lbs to mol/s
                                        'SO2 Mass (lbs)':lambda x:np.sum(x)*0.453592/0.064/3600,#lbs to mol/s
                                        'CO2 Mass (short tons)':lambda x:np.sum(x)*907.185/0.044/3600,#short tons to mol/s
                                        'Facility Name':lambda x:x.iloc[0],
                                        'State':lambda x:x.iloc[0],
                                        'Gross Load (MW)':'sum',
                                        'Heat Input (mmBtu)':'sum'}
                                      ).rename(columns={'NOx Mass (lbs)':'NOx (mol/s)',
                                                       'SO2 Mass (lbs)':'SO2 (mol/s)',
                                                       'CO2 Mass (short tons)':'CO2 (mol/s)'})
        self.uedf = uedf
        self.fedf = fedf
        
    def subset_facilities(self,n_facility_with_most_NOx=10,enforce_presence_all_years=True):
        if not hasattr(self,'uadf'):
            self.logger.info('run load_emissions to get uadf first')
            self.load_emissions(if_unit_emissions=False,n_facility_with_most_NOx=None)
        fadf = self.uadf.groupby('facilityId'
                                ).aggregate({'noxMassLbs':'sum',
                                            'year':lambda x:len(x.unique()),
                                            'primaryFuelInfo':lambda x:'/'.join(x.dropna().unique()),
                                            'facilityName':lambda x:x.iloc[0],
                                            'stateCode':lambda x:x.iloc[0],
                                            'latitude':'mean',
                                            'longitude':'mean'}
                                           ).sort_values('noxMassLbs',ascending=False)
        if enforce_presence_all_years:
            mask = fadf['year']==len(pd.period_range(self.start_dt,self.end_dt,freq='1Y'))
            if np.sum(mask) != fadf.shape[0]:
                self.logger.warning('only {} facilities have attributes in all years out of {}'.format(
                    np.sum(mask),fadf.shape[0]))
            fadf = fadf.loc[mask]
        fadf['rank'] = np.arange(fadf.shape[0],dtype=int)+1
        self.fadf = fadf.iloc[0:n_facility_with_most_NOx]
    
    def load_emissions(self,attributes_path_pattern=None,emissions_path_pattern=None,states=None,
                      local_hours=None,if_unit_emissions=True,if_facility_emissions=False,
                      n_facility_with_most_NOx=None):
        '''
        attributes/emissions_path_pattern:
            path patterns for annual attributes and daily emission files. 
            good practice is to save trimmed attributes using self.trim_unit_attributes (takes hours), 
            then only load a small number of largest facilities, where one can turn off emissions file loading
        states:
            if provided, should be a list of state codes, e.g., ['TX']
        local_hours:
            if_provided, should be a list of int hours, e.g., [13]
        if_unit_emissions:
            if load emissions files. slow if space*time*number of units is large. can be off if only looking at unit/facility
            attributes when trimmed files are already saved
        if_facility_emissions:
            if groupby facility and calculate facility level emissions. may need a separate function
        n_facility_with_most_NOx:
            number of largest facilites (not units) to include
        if emissions are all on, adds the following to the object:
            uadf = unit attributes data frame; fadf = facility attributes data frame
            uedf = unit emissions data frame; fadf = facility emissions data frame
        '''
        attributes_path_pattern = attributes_path_pattern or self.attributes_path_pattern
        emissions_path_pattern = emissions_path_pattern or self.emissions_path_pattern
        func_1st = lambda x:x.iloc[0]
        
        uadf = []
        if if_unit_emissions:
            uedf = []
        else:
            if_facility_emissions=False# facility level emissions impossible without unit level emissions
        
        for year in pd.period_range(self.start_dt,self.end_dt,freq='1Y'):
            csv_name = year.strftime(attributes_path_pattern)
            self.logger.info('loading attribute file {}'.format(csv_name))
            adf = pd.read_csv(csv_name,index_col=0)
            mask = (adf['longitude']>=self.west)&(adf['longitude']<=self.east)&\
            (adf['latitude']>=self.south)&(adf['latitude']<=self.north)
            if states is not None:
                mask = mask & (adf['stateCode'].isin(states))
            adf = adf.loc[mask]
            # keep only units in the n_facility_with_most_NOx largest facilities
            if 'noxMassLbsFacility' in adf.keys():
                nfac = n_facility_with_most_NOx
            else:
                nfac = None
            if nfac is not None:
                gadf = adf.groupby('facilityId'
                                  ).aggregate({'noxMassLbsFacility':func_1st}
                                             ).sort_values('noxMassLbsFacility',ascending=False
                                                          ).iloc[0:nfac,:]
                adf = adf.loc[adf['facilityId'].isin(gadf.index)]
            uadf.append(adf)
            if not if_unit_emissions:
                continue# load emission files otherwise
            for date in pd.period_range(np.max([self.start_dt,year.start_time]),
                                       np.min([self.end_dt,year.end_time]),freq='1D'):
                filename = date.strftime(emissions_path_pattern)
                if not os.path.exists(filename):
                    self.logger.warning('{} does not exist!'.format(filename))
                    continue
                if date.day == 1:
                    self.logger.info('loading emission file {}'.format(filename))
                edf = pd.read_csv(filename)
                edf = edf.loc[edf['Facility ID'].isin(adf['facilityId'])]
                if local_hours is not None:
                    edf = edf.loc[pd.to_datetime(edf['local_dt']).dt.hour.isin(local_hours)]
                uedf.append(edf)
        
        self.uadf = pd.concat(uadf).reset_index()
        if if_unit_emissions:
            self.uedf = pd.concat(uedf).reset_index()
        if 'noxMassLbs' not in self.uadf.keys():
            if not if_unit_emissions:
                self.logging.warning('Please turn on if_unit_emissions')
                return
            # add column for nox emission in attribute df
            noxMassLbs = np.zeros(self.uadf.shape[0])
            for i,(irow,row) in enumerate(self.uadf.iterrows()):
                noxMassLbs[i] = self.uedf.loc[(self.uedf['Facility ID']==row.facilityId)&\
                                     (self.uedf['Unit ID']==row.unitId)]['NOx Mass (lbs)'].sum()
            self.uadf.insert(loc=0,column='noxMassLbs',value=noxMassLbs)
            self.uadf = self.uadf.sort_values('noxMassLbs',ascending=False).reset_index(drop=True)
        
        self.fadf = self.uadf.groupby('facilityId').aggregate({
            'noxMassLbs':'sum',
            'year':'mean',
            'facilityName':func_1st,
            'stateCode':func_1st,
            'latitude':'mean',
            'longitude':'mean'}).sort_values('noxMassLbs',ascending=False)
        if if_facility_emissions:
            self.fedf = self.uedf.groupby(['Facility ID','local_dt']).aggregate({
                'NOx Mass (lbs)':'sum',
                'SO2 Mass (lbs)':'sum',
                'CO2 Mass (short tons)':'sum',
                'Facility Name':func_1st,
                'State':func_1st,
                'Operating Time':'sum',
                'Gross Load (MW)':'sum',
                'Heat Input (mmBtu)':'sum'})
    
    def download_attributes(self,attributes_path_pattern=None,API_key=None):
        self.attributes_path_pattern = attributes_path_pattern
        API_key = API_key or self.API_key
        if API_key is None:
            self.logger.error('you need API key, see https://www.epa.gov/airmarkets/cam-api-portal#/api-key-signup')
            return
        # making get request using the facilities/attributes endpoint
        streamingUrl = "https://api.epa.gov/easey/streaming-services/facilities/attributes"
        for year in pd.period_range(self.start_dt,self.end_dt,freq='1Y'):
            parameters = {
                'api_key': API_key,
                'year': year.year
            }
            self.logger.info('fetching year {}'.format(year.year))
            streamingResponse = requests.get(streamingUrl, params=parameters,timeout=5)
            self.logger.info("Status code: "+str(streamingResponse.status_code))
            # collecting data as a data frame
            df = pd.DataFrame(streamingResponse.json())
            csv_name = year.strftime(attributes_path_pattern)
            self.logger.info('saving to {}'.format(csv_name))
            df.to_csv(csv_name,index=False)
    
    def resave_emissions(self,emissions_path_pattern=None,API_key=None,cols_to_keep=None):
        emissions_path_pattern = self.emissions_path_pattern or emissions_path_pattern
        API_key = API_key or self.API_key
        if API_key is None:
            self.logger.error('you need API key, see https://www.epa.gov/airmarkets/cam-api-portal#/api-key-signup')
            return
        cols_to_keep = cols_to_keep or ['State','Facility Name','Facility ID',
                                        'Unit ID','Operating Time','Gross Load (MW)','Heat Input (mmBtu)',
                                        'SO2 Mass (lbs)','CO2 Mass (short tons)','NOx Mass (lbs)']
        # S3 bucket url base + s3Path (in get request) = the full path to the files
        BUCKET_URL_BASE = 'https://api.epa.gov/easey/bulk-files/'
        parameters = {
            'api_key': API_key
        }
        self.logger.info('getting bulk file lists...')
        # executing get request
        response = requests.get("https://api.epa.gov/easey/camd-services/bulk-files", params=parameters)
        # printing status code
        self.logger.info("Status code: "+str(response.status_code))
        # converting the content from json format to a data frame
        resjson = response.content.decode('utf8').replace("'", '"')
        data = json.loads(resjson)
        s = json.dumps(data, indent=4)
        jsonread = pd.read_json(s)
        pddf = pd.DataFrame(jsonread)
        bulkFiles = pd.concat([pddf.drop(['metadata'], axis=1), pddf['metadata'].apply(pd.Series)], axis=1)
        for year in pd.period_range(self.start_dt,self.end_dt,freq='1Y'):
            # year-quarter bulkFiles, yqbf
            yqbf = bulkFiles.loc[(bulkFiles['dataType']=='Emissions') &\
                                 (bulkFiles['filename'].str.contains('emissions-hourly-{}-q'.format(year.year)))]
            # loop over quarter, save daily
            for irow, row in yqbf.iterrows():
                url = BUCKET_URL_BASE+row.s3Path
                self.logger.info('retrieving data from {}'.format(url))
                res = requests.get(url).content
                # dataframe for the quarter
                df = pd.read_csv(io.StringIO(res.decode('utf-8')))
                col_date = pd.to_datetime(df['Date'])
                col_dt = col_date+pd.to_timedelta(df['Hour'],unit='h')
                # loop over dates
                for date in col_date.unique():
                    mask = col_date == date
                    daily_df = pd.concat([pd.Series(data=col_dt[mask],name='local_dt'),df.loc[mask][cols_to_keep]],axis=1)
                    filename = pd.to_datetime(date).strftime(emissions_path_pattern)
                    os.makedirs(os.path.split(filename)[0],exist_ok=True)
                    daily_df.to_csv(filename,index=False)